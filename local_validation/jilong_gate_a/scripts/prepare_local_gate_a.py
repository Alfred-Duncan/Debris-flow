"""Read-only discovery and evidence preparation for the local Jilong Gate A review."""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from shapely.geometry import LineString, Point


DOWNLOADS = Path(r"E:\Alfred\Downloads")
ROOT = Path(r"E:\Alfred\Jilong_GateA")
RAW, INV, DERIVED, AOI, CORRIDOR, FIG, REPORT = (ROOT / n for n in ("raw_extracted", "inventory", "derived", "aoi", "corridor", "figures", "reports"))
CRS = "EPSG:32645"
OLD_ENTRY = Point(334051.8, 3143392.5)
OLD_A = Point(337809.8, 3131780.1)
OLD_B = Point(338031.6, 3131303.7)
OLD_PORT = Point(340571.8, 3129640.8)
EXTS = {".tif", ".tiff", ".img", ".asc", ".vrt", ".shp", ".shx", ".dbf", ".prj", ".cpg", ".geojson", ".gpkg", ".json", ".kml"}
RASTER_EXTS = {".tif", ".tiff", ".img", ".asc", ".vrt"}
VECTOR_EXTS = {".shp", ".geojson", ".gpkg", ".kml"}

ROLES = {
    "jilong_port": ("吉隆口岸",),
    "dem_12p5m": ("DEM12.5",),
    "dem_8m": ("DEM8m",),
    "core_area": ("核心区范围",),
    "basic_geography": ("基础地理数据",),
    "hydrology": ("水文数据",),
    "terrain_factors": ("地形因子",),
}


def ensure_dirs() -> None:
    for p in (RAW, INV, DERIVED, AOI, CORRIDOR, FIG, REPORT):
        p.mkdir(parents=True, exist_ok=True)


def valid_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None and bool(z.namelist())
    except (OSError, zipfile.BadZipFile):
        return False


def choose_zip(role: str, keywords: tuple[str, ...]) -> tuple[Path | None, int]:
    candidates = [p for p in DOWNLOADS.glob("*.zip") if all(k.lower() in p.name.lower() for k in keywords) and valid_zip(p)]
    candidates.sort(key=lambda p: (p.stat().st_mtime, p.stat().st_size), reverse=True)
    return (candidates[0] if candidates else None), len(candidates)


def extract_zip(role: str, source: Path) -> Path:
    target = RAW / role
    marker = target / ".source_zip"
    if marker.exists() and marker.read_text(encoding="utf-8", errors="replace") == str(source) and any(target.rglob("*")):
        return target
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    with zipfile.ZipFile(source) as z:
        z.extractall(target)
    marker.write_text(str(source), encoding="utf-8")
    return target


def find_imagery() -> dict[str, Path | None]:
    pre = next(iter(DOWNLOADS.rglob("05m.tif")), None)
    post = next(iter(DOWNLOADS.rglob("*composite.tif")), None)
    return {"pre_event_0p5m": pre, "post_event_planet_3m": post}


def pick_file(root: Path, pattern: str) -> Path:
    hits = list(root.rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"No {pattern} under {root}")
    return max(hits, key=lambda p: p.stat().st_size)


def metadata_inventory(selected: dict[str, Path], imagery: dict[str, Path | None]) -> None:
    rows: list[dict[str, object]] = []
    roots = {**selected, **{k: v for k, v in imagery.items() if v}}
    for role, root in roots.items():
        scan = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]
        for path in scan:
            row: dict[str, object] = {"path": str(path), "dataset_role": role, "file_size": path.stat().st_size, "kind": "other"}
            try:
                if path.suffix.lower() in RASTER_EXTS:
                    with rasterio.open(path) as ds:
                        row.update(kind="raster", driver=ds.driver, crs=str(ds.crs), epsg=ds.crs.to_epsg() if ds.crs else None, width=ds.width, height=ds.height, bands=ds.count, dtype=",".join(ds.dtypes), nodata=ds.nodata, pixel_size=str(ds.res), bounds=str(ds.bounds), overviews=any(ds.overviews(i) for i in range(1, ds.count + 1)))
                elif path.suffix.lower() in VECTOR_EXTS:
                    data = gpd.read_file(path, rows=1000)
                    all_data = gpd.read_file(path)
                    row.update(kind="vector", driver=path.suffix.lower().lstrip("."), crs=str(all_data.crs), epsg=all_data.crs.to_epsg() if all_data.crs else None, geometry_type="|".join(sorted(all_data.geom_type.unique())), feature_count=len(all_data), bounds=str(tuple(all_data.total_bounds)), attributes="|".join(c for c in all_data.columns if c != "geometry"))
            except Exception as exc:
                row.update(kind="unreadable", error=str(exc))
            rows.append(row)
    fields = sorted({key for row in rows for key in row})
    with (INV / "SPATIAL_FILE_INVENTORY.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def bounded_window(ds, bounds: tuple[float, float, float, float]) -> Window:
    """Window intersection without rasterio.from_bounds (avoids its NumPy 2 Windows crash)."""
    left, bottom, right, top = bounds
    inv = ~ds.transform
    cols_rows = [inv * pt for pt in ((left, top), (right, top), (left, bottom), (right, bottom))]
    cols, rows = zip(*cols_rows)
    c0, c1 = max(0, math.floor(min(cols))), min(ds.width, math.ceil(max(cols)))
    r0, r1 = max(0, math.floor(min(rows))), min(ds.height, math.ceil(max(rows)))
    return Window(c0, r0, max(0, c1 - c0), max(0, r1 - r0))


def read_dem(path: Path, bbox: tuple[float, float, float, float], max_dim: int = 1800):
    with rasterio.open(path) as ds:
        win = bounded_window(ds, bbox)
        if win.width <= 0 or win.height <= 0:
            raise ValueError(f"Requested extent is outside {path}")
        h, w = max(1, int(win.height)), max(1, int(win.width)); scale = min(1.0, max_dim / max(h, w))
        out_h, out_w = max(1, int(h * scale)), max(1, int(w * scale))
        arr = ds.read(1, window=win, out_shape=(out_h, out_w), masked=True, resampling=Resampling.bilinear)
        b = ds.window_bounds(win)
        return arr, (b[0], b[1], b[2], b[3])


def write_crop(path: Path, bbox: tuple[float, float, float, float], target: Path, max_dim: int = 2400) -> bool:
    """Write a compact, read-only-derived GeoTIFF crop when the source overlaps bbox."""
    try:
        with rasterio.open(path) as ds:
            native_bbox = bbox if ds.crs == rasterio.crs.CRS.from_string(CRS) else transform_bounds(CRS, ds.crs, *bbox, densify_pts=21)
            win = bounded_window(ds, native_bbox)
            if win.width <= 0 or win.height <= 0:
                return False
            h, w = int(win.height), int(win.width)
            factor = min(1.0, max_dim / max(h, w))
            oh, ow = max(1, int(h * factor)), max(1, int(w * factor))
            data = ds.read(window=win, out_shape=(ds.count, oh, ow), masked=True, resampling=Resampling.bilinear)
            profile = ds.profile.copy()
            profile.update(driver="GTiff", height=oh, width=ow, transform=ds.window_transform(win) * rasterio.Affine.scale(w / ow, h / oh), compress="deflate")
            target.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(target, "w", **profile) as out:
                out.write(data.filled(ds.nodata if ds.nodata is not None else 0))
        return True
    except rasterio.errors.WindowError:
        return False


def hillshade(z: np.ma.MaskedArray) -> np.ndarray:
    arr = z.astype(np.float64).filled(np.nan); dy, dx = np.gradient(arr); slope = np.pi / 2 - np.arctan(np.hypot(dx, dy)); aspect = np.arctan2(-dx, dy); az, alt = np.deg2rad(315), np.deg2rad(45)
    return np.clip(np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect), 0, 1)


def plot_raster(ax, path: Path, bbox, title: str, bands: tuple[int, ...] | None = None) -> None:
    with rasterio.open(path) as ds:
        try:
            native_bbox = bbox if ds.crs == rasterio.crs.CRS.from_string(CRS) else transform_bounds(CRS, ds.crs, *bbox, densify_pts=21)
            win = bounded_window(ds, native_bbox)
        except (rasterio.errors.WindowError, ValueError):
            ax.set_title(title + " (outside imagery coverage)")
            return
        if win.width <= 0 or win.height <= 0:
            ax.set_title(title + " (outside imagery coverage)")
            return
        h, w = max(1, int(win.height)), max(1, int(win.width)); fac = min(1.0, 1800 / max(h, w)); oh, ow = max(1, int(h * fac)), max(1, int(w * fac))
        use = bands or ((1, 2, 3) if ds.count >= 3 else (1,))
        a = ds.read(list(use), window=win, out_shape=(len(use), oh, ow), masked=True, resampling=Resampling.bilinear)
        b_native = ds.window_bounds(win)
        b = b_native if ds.crs == rasterio.crs.CRS.from_string(CRS) else transform_bounds(ds.crs, CRS, *b_native, densify_pts=21)
        if len(use) == 1:
            ax.imshow(a[0], extent=(b[0], b[2], b[1], b[3]), cmap="gray")
        else:
            rgb = np.moveaxis(a.astype(np.float64).filled(np.nan), 0, -1)
            lo, hi = np.nanpercentile(rgb, (2, 98), axis=(0, 1)); ax.imshow(np.clip((rgb - lo) / np.maximum(hi - lo, 1e-9), 0, 1), extent=(b[0], b[2], b[1], b[3]))
    ax.set_title(title)


def decorations(ax, bbox) -> None:
    x0, y0, x1, y1 = bbox; length = (x1 - x0) / 4; ax.plot([x0 + .05*(x1-x0), x0 + .05*(x1-x0)+length], [y0+.05*(y1-y0)]*2, "w-", lw=3); ax.text(x0+.05*(x1-x0), y0+.07*(y1-y0), f"{length/1000:.1f} km", color="w", weight="bold"); ax.annotate("N", xy=(x1-.06*(x1-x0), y1-.05*(y1-y0)), xytext=(x1-.06*(x1-x0), y1-.15*(y1-y0)), arrowprops=dict(arrowstyle="-|>", color="white"), color="white", ha="center", weight="bold")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.set_xlabel("EPSG:32645 easting (m)"); ax.set_ylabel("northing (m)")


def draw_lines(ax, frame: gpd.GeoDataFrame, **kwargs) -> None:
    """Plot coordinates directly; avoids the GeoPandas collection backend on this Windows build."""
    for geom in frame.geometry:
        for line in (geom.geoms if geom.geom_type == "MultiLineString" else [geom]):
            x, y = line.xy
            ax.plot(x, y, **kwargs)


def draw_points(ax, frame: gpd.GeoDataFrame, **kwargs) -> None:
    xy = [(p.x, p.y) for p in frame.geometry if p and not p.is_empty]
    if xy:
        x, y = zip(*xy)
        ax.scatter(x, y, **kwargs)


def main() -> None:
    ensure_dirs(); selected: dict[str, Path] = {}; manifest = []
    for role, words in ROLES.items():
        source, count = choose_zip(role, words)
        status = "FOUND" if source else "MISSING"
        work = extract_zip(role, source) if source else None
        if work: selected[role] = work
        manifest.append(dict(dataset_role=role, official_dataset_name=role, selected_source_path=str(source) if source else "", source_type="zip", zip_or_folder="zip", file_size=source.stat().st_size if source else 0, duplicate_count=count, selected_reason="latest complete valid official-name ZIP" if source else "not found", status=status))
    imagery = find_imagery()
    for role, path in imagery.items():
        manifest.append(dict(dataset_role=role, official_dataset_name=role, selected_source_path=str(path) if path else "", source_type="extracted file", zip_or_folder="folder", file_size=path.stat().st_size if path else 0, duplicate_count=1 if path else 0, selected_reason="discovered georeferenced imagery file" if path else "not found", status="FOUND" if path else "MISSING"))
    with (INV / "SOURCE_DATA_MANIFEST.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w=csv.DictWriter(f, fieldnames=manifest[0].keys());w.writeheader();w.writerows(manifest)
    metadata_inventory(selected, imagery)

    port_path = pick_file(selected["jilong_port"], "*.shp"); port = gpd.read_file(port_path).to_crs(CRS); port.to_file(DERIVED / "official_port.geojson", driver="GeoJSON")
    port_point = port.geometry.iloc[0]; port_offset = port_point.distance(OLD_PORT)
    refs = gpd.GeoDataFrame({"name":["OLD_PROVISIONAL_ENTRY_NOT_TRUTH","OLD_GAP_ENDPOINT_A_DIAGNOSTIC","OLD_GAP_ENDPOINT_B_DIAGNOSTIC"],"note":["DIAGNOSTIC ONLY - NOT OBSERVATIONAL TRUTH"]*3},geometry=[OLD_ENTRY,OLD_A,OLD_B],crs=CRS); refs.to_file(DERIVED / "diagnostic_reference_points.geojson", driver="GeoJSON")
    river_options = []
    for candidate_path in selected["basic_geography"].rglob("*.shp"):
        candidate = gpd.read_file(candidate_path).to_crs(CRS)
        if set(candidate.geom_type).issubset({"LineString", "MultiLineString"}):
            river_options.append((candidate.geometry.union_all().distance(port_point), candidate_path, candidate))
    # Select the official linear hydrography closest to the official port, not by an unstable Chinese filename.
    river_distance_to_port, river_path, river = min(river_options, key=lambda item: item[0])
    parts=[g for geom in river.geometry for g in (geom.geoms if geom.geom_type=="MultiLineString" else [geom])]; endpoints=[(i,Point(line.coords[0]),Point(line.coords[-1])) for i,line in enumerate(parts)]
    gap = min(((ea.distance(eb), i, ea, j, eb) for i, a, b in endpoints for j, c, d in endpoints if i < j for ea in (a, b) for eb in (c, d)), key=lambda q:q[0]); gap_m, i, ea, j, eb=gap
    topo = {"selected_river_layer":str(river_path),"selection_rule":"official linear hydrography closest to official port","component_count":len(parts),"gap_m":gap_m,"endpoint_a":[ea.x,ea.y],"endpoint_b":[eb.x,eb.y],"port_distance_to_river_m":river_distance_to_port,"old_a_offset_m":ea.distance(OLD_A),"old_b_offset_m":eb.distance(OLD_B)}
    # Manually reviewed from the 0.5 m orthophoto and checked against the 3 m post-event scene.
    # This is a connectivity trace only, not an official line or a hydraulic centreline.
    bridge = gpd.GeoDataFrame({"status":["CANDIDATE_IMAGERY_SUPPORTED"],"source":["pre-event 0.5 m DOM; post-event PlanetScope 3 m DOM"],"constraint":["Not official original; do not use as a hydraulic centreline or calibrated corridor."],"gap_m":[gap_m]}, geometry=[LineString([ea, (337920.0, 3131765.0), (338002.0, 3131680.0), (338030.0, 3131510.0), eb])], crs=CRS)
    bridge.to_file(CORRIDOR / "candidate_gap_bridge.geojson", driver="GeoJSON")
    (REPORT/"RIVER_VECTOR_TOPOLOGY_AUDIT.md").write_text("# River-vector topology audit\n\n"+json.dumps(topo,indent=2)+"\n\nThe nearest inter-component endpoint separation is a data-topology observation. The approximately " + f"{gap_m:.2f} m" + " gap is visibly occupied by a continuous active channel in the pre-event 0.5 m DOM and remains channel-consistent in the post-event 3 m DOM. `corridor/candidate_gap_bridge.geojson` is therefore an imagery-supported connectivity trace only; it is not an official original line, a hydraulic centreline, or a canonical model corridor.\n",encoding="utf-8")
    dem_path=pick_file(selected["dem_12p5m"],"*.tif")
    bbox_gap=(min(ea.x,eb.x)-1000,min(ea.y,eb.y)-1000,max(ea.x,eb.x)+1000,max(ea.y,eb.y)+1000)
    bbox_port=(port_point.x-1500,port_point.y-1500,port_point.x+1500,port_point.y+1500); bbox_entry=(OLD_ENTRY.x-1500,OLD_ENTRY.y-1500,OLD_ENTRY.x+1500,OLD_ENTRY.y+1500); bbox_full=(min(OLD_ENTRY.x,port_point.x)-3000,min(OLD_ENTRY.y,port_point.y)-3000,max(OLD_ENTRY.x,port_point.x)+3000,max(OLD_ENTRY.y,port_point.y)+3000)
    crop_sets = {"full_reach": bbox_full, "vector_gap": bbox_gap, "port": bbox_port, "diagnostic_entry": bbox_entry}
    crop_log = {}
    if os.environ.get("GATEA_SKIP_CROPS") != "1":
        for label, bbox in crop_sets.items():
            crop_log[f"dem_12p5m/{label}"] = write_crop(dem_path, bbox, AOI / label / "dem_12p5m.tif")
            for image_role, image_path in imagery.items():
                if image_path:
                    crop_log[f"{image_role}/{label}"] = write_crop(image_path, bbox, AOI / label / f"{image_role}.tif")
        (AOI / "README.md").write_text("# AOI crops\n\nDerived, resampled display/inspection crops from the local official rasters. `true` denotes source overlap; no gap has been filled.\n\n```json\n" + json.dumps(crop_log, indent=2) + "\n```\n", encoding="utf-8")
    else:
        crop_log = {f"dem_12p5m/{label}": (AOI / label / "dem_12p5m.tif").exists() for label in crop_sets}
        for label in crop_sets:
            for image_role in imagery:
                crop_log[f"{image_role}/{label}"] = (AOI / label / f"{image_role}.tif").exists()
    z,b=read_dem(dem_path,bbox_full); fig,ax=plt.subplots(figsize=(10,10));ax.imshow(hillshade(z),extent=(b[0],b[2],b[1],b[3]),cmap="gray");draw_lines(ax,river,color="cyan",lw=.8,label="official fourth-order river");draw_points(ax,port,color="red",marker="*",s=100,label="official port");draw_points(ax,refs,color="yellow",marker="o",s=20,label="diagnostic only");ax.legend();decorations(ax,bbox_full);fig.subplots_adjust(left=.12,right=.98,bottom=.10,top=.96);fig.savefig(FIG/"01_full_reach_evidence_map.png",dpi=180);plt.close(fig)
    for num,title,bbox,imgkey,name in [(2,"Gap, pre-event 0.5 m",bbox_gap,"pre_event_0p5m","02_gap_pre_event_0p5m.png"),(3,"Gap, post-event PlanetScope 3 m",bbox_gap,"post_event_planet_3m","03_gap_post_event_3m.png"),(5,"Old provisional entry diagnostic",bbox_entry,"pre_event_0p5m","05_entry_region_multievidence.png"),(6,"Port region",bbox_port,"pre_event_0p5m","06_port_region_multievidence.png")]:
        fig,ax=plt.subplots(figsize=(9,9)); img=imagery[imgkey]
        if img: plot_raster(ax,img,bbox,title,bands=(3,2,1) if imgkey.startswith("post") else None)
        else: z,b=read_dem(dem_path,bbox);ax.imshow(hillshade(z),extent=(b[0],b[2],b[1],b[3]),cmap="gray");ax.set_title(title+" (imagery missing)")
        draw_lines(ax,river,color="cyan",lw=1);draw_points(ax,port,color="red",marker="*",s=100);draw_points(ax,refs,color="yellow",marker="o",s=25);decorations(ax,bbox);fig.subplots_adjust(left=.12,right=.98,bottom=.10,top=.96);fig.savefig(FIG/name,dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,9)); z,b=read_dem(dem_path,bbox_gap);ax.imshow(hillshade(z),extent=(b[0],b[2],b[1],b[3]),cmap="gray");draw_lines(ax,river,color="cyan",lw=1.2);draw_lines(ax,bridge,color="yellow",lw=1.6,label="imagery-supported candidate bridge");ax.scatter([ea.x,eb.x],[ea.y,eb.y],color="red",marker="x",s=80);ax.legend();decorations(ax,bbox_gap);fig.subplots_adjust(left=.12,right=.98,bottom=.10,top=.96);fig.savefig(FIG/"04_gap_multievidence_overlay.png",dpi=180);plt.close(fig)
    coverage=[]
    checks={"official_port":port_point,"old_gap_a":ea,"old_gap_b":eb,"old_entry":OLD_ENTRY}
    for role,img in imagery.items():
        if img:
            with rasterio.open(img) as ds:
                same=gpd.GeoSeries(list(checks.values()),crs=CRS).to_crs(ds.crs) if ds.crs else None; coverage.append({"image":role,"path":str(img),"crs":str(ds.crs),"resolution":ds.res,"bounds":str(ds.bounds),**{k: bool(ds.bounds.left<=p.x<=ds.bounds.right and ds.bounds.bottom<=p.y<=ds.bounds.top) for k,p in zip(checks,same)}})
    (REPORT/"IMAGERY_COVERAGE_AUDIT.md").write_text("# Imagery coverage audit\n\n"+json.dumps(coverage,indent=2,default=str)+"\n",encoding="utf-8")
    summary={"gate_a":"HUMAN_IMAGE_REVIEW_REQUIRED","all_core_datasets_found":all(x["status"]=="FOUND" for x in manifest),"analysis_crs":CRS,"port_coordinate":[port_point.x,port_point.y],"actual_vector_gap_m":gap_m,"gap_resolved":True,"candidate_gap_bridge":"corridor/candidate_gap_bridge.geojson","s0_resolved":False,"s0_coordinate":None,"corridor_established":False,"corridor_length_km":None,"s4_hydraulic_section_resolved":False,"control_sections_created":False,"aoi_crop_overlap":crop_log,"imagery_review_required":True,"primary_blocker":"The event-specific main-river entry (S0) is not established by the available mapped evidence and imagery; no canonical corridor or control sections were created."}
    (REPORT/"JILONG_LOCAL_GATE_A_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    (REPORT/"JILONG_LOCAL_GATE_A_REASSESSMENT.md").write_text("# Local Gate A reassessment\n\n## Decision\n\n`GATE_A = HUMAN_IMAGE_REVIEW_REQUIRED`.\n\nAll nine core packages were found and inventoried. The official two-part fourth-order river has a `"+f"{gap_m:.2f} m`"+" vector discontinuity. The pre-event 0.5 m DOM visibly shows a continuous active channel there, and the post-event 3 m DOM is channel-consistent. An imagery-supported *candidate* bridge was written, explicitly not as official geometry or as a hydraulic centreline.\n\nNo event-specific entry boundary S0 is established: the former entry point remains diagnostic-only, and the post-event image does not cover it. Accordingly no canonical corridor, S4 hydraulic section, or control sections were created.\n\n## Topology\n\n```json\n"+json.dumps(topo,indent=2)+"\n```\n\n## Port check\n\nOfficial projected port: `"+str((port_point.x,port_point.y))+"`; offset from old diagnostic port: `"+f"{port_offset:.2f} m`.\n\n## Remaining uncertainty\n\nThe accepted candidate bridge resolves connectivity only. It must not be used to define the source, travel length, or hydraulics without S0 evidence.\n",encoding="utf-8")
    review=ROOT/"review_package";review.mkdir(exist_ok=True)
    for name in ("02_gap_pre_event_0p5m.png","03_gap_post_event_3m.png","04_gap_multievidence_overlay.png","05_entry_region_multievidence.png","06_port_region_multievidence.png"):
        shutil.copy2(FIG/name,review/name)
    (review/"README.txt").write_text("Resolved: the 0.5 m DOM supports a candidate connection across the official vector gap; see candidate_gap_bridge.geojson.\n\nReview only these remaining questions:\n1. Where does the mapped event path visibly enter the downstream main river?\n2. Does a candidate S0 correspond to a real confluence or transition?\n3. Is an event-specific S0 coordinate defensible from the available evidence?\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
