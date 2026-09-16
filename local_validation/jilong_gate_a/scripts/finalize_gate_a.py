"""Evidence-conservative finalization of the local Jilong spatial Gate A.

Reads only existing local evidence.  It never starts a simulator, changes a DEM,
or invents an event path.  S0 is intentionally withheld unless event-specific
spatial evidence is present.
"""
from __future__ import annotations

import csv
import json
import math
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
from shapely.geometry import LineString, Point, box
from shapely.ops import nearest_points


ROOT = Path(r"E:\Alfred\Jilong_GateA")
RAW = ROOT / "raw_extracted"
CORRIDOR = ROOT / "corridor"
REPORT = ROOT / "reports"
FIG = ROOT / "figures"
REVIEW = ROOT / "review_package"
DOWNLOADS = Path(r"E:\Alfred\Downloads")
CRS = "EPSG:32645"
PORT = Point(340837.89194740437, 3129051.745594877)
OLD_ENTRY = Point(334051.8, 3143392.5)


def bounded_window(ds, bounds):
    inv = ~ds.transform
    coords = [inv * xy for xy in ((bounds[0], bounds[3]), (bounds[2], bounds[3]), (bounds[0], bounds[1]), (bounds[2], bounds[1]))]
    cols, rows = zip(*coords)
    c0, c1 = max(0, math.floor(min(cols))), min(ds.width, math.ceil(max(cols)))
    r0, r1 = max(0, math.floor(min(rows))), min(ds.height, math.ceil(max(rows)))
    return Window(c0, r0, max(0, c1 - c0), max(0, r1 - r0))


def native_bounds(ds, bounds):
    return bounds if ds.crs == rasterio.crs.CRS.from_string(CRS) else transform_bounds(CRS, ds.crs, *bounds, densify_pts=21)


def image_path(pattern):
    return next(DOWNLOADS.rglob(pattern))


def contains_point(ds, point):
    p = gpd.GeoSeries([point], crs=CRS).to_crs(ds.crs).iloc[0]
    return bool(ds.bounds.left <= p.x <= ds.bounds.right and ds.bounds.bottom <= p.y <= ds.bounds.top)


def coverage(ds, point, aoi):
    b = native_bounds(ds, aoi.bounds)
    image_box = box(ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)
    aoi_box = box(*b)
    return {
        "point_covered": contains_point(ds, point),
        "aoi_any_overlap": image_box.intersects(aoi_box),
        "aoi_full_coverage": image_box.covers(aoi_box),
    }


def read_dem(dem_path, bounds, max_dim=1800):
    with rasterio.open(dem_path) as ds:
        win = bounded_window(ds, bounds)
        if win.width <= 0 or win.height <= 0:
            raise ValueError("DEM AOI outside source extent")
        h, w = int(win.height), int(win.width)
        scale = min(1.0, max_dim / max(h, w))
        arr = ds.read(1, window=win, out_shape=(max(1, int(h * scale)), max(1, int(w * scale))), masked=True, resampling=Resampling.bilinear)
        b = ds.window_bounds(win)
        return arr, (b[0], b[1], b[2], b[3])


def hillshade(arr):
    z = arr.astype(float).filled(np.nan)
    dy, dx = np.gradient(z)
    slope = np.pi / 2 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    return np.clip(np.sin(np.deg2rad(45)) * np.sin(slope) + np.cos(np.deg2rad(45)) * np.cos(slope) * np.cos(np.deg2rad(315) - aspect), 0, 1)


def show_image(ax, path, bounds, title, bands=(3, 2, 1)):
    with rasterio.open(path) as ds:
        nb = native_bounds(ds, bounds)
        win = bounded_window(ds, nb)
        if win.width <= 0 or win.height <= 0:
            ax.text(.5, .5, "POST-EVENT IMAGE NOT AVAILABLE AT THIS LOCATION" if "post" in title.lower() else "IMAGE NOT AVAILABLE AT THIS LOCATION", ha="center", va="center", transform=ax.transAxes, wrap=True)
            ax.set_title(title)
            return False
        h, w = int(win.height), int(win.width)
        factor = min(1.0, 1800 / max(h, w))
        use = tuple(b for b in bands if b <= ds.count) or (1,)
        data = ds.read(list(use), window=win, out_shape=(len(use), max(1, int(h * factor)), max(1, int(w * factor))), masked=True, resampling=Resampling.bilinear)
        wb = ds.window_bounds(win)
        ub = wb if ds.crs == rasterio.crs.CRS.from_string(CRS) else transform_bounds(ds.crs, CRS, *wb, densify_pts=21)
        if len(use) == 1:
            ax.imshow(data[0], extent=(ub[0], ub[2], ub[1], ub[3]), cmap="gray")
        else:
            rgb = np.moveaxis(data.astype(float).filled(np.nan), 0, -1)
            lo, hi = np.nanpercentile(rgb, (2, 98), axis=(0, 1))
            ax.imshow(np.clip((rgb - lo) / np.maximum(hi - lo, 1e-9), 0, 1), extent=(ub[0], ub[2], ub[1], ub[3]))
    ax.set_title(title)
    return True


def draw_lines(ax, frame, **kwargs):
    for geom in frame.geometry:
        for line in (geom.geoms if geom.geom_type == "MultiLineString" else [geom]):
            x, y = line.xy
            ax.plot(x, y, **kwargs)


def draw_polygon(ax, frame, **kwargs):
    for geom in frame.geometry:
        geoms = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for poly in geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, **kwargs)


def decorate(ax, bounds):
    x0, y0, x1, y1 = bounds
    scale = (x1 - x0) / 4
    ax.plot([x0 + .05 * (x1 - x0), x0 + .05 * (x1 - x0) + scale], [y0 + .05 * (y1 - y0)] * 2, "w-", lw=3)
    ax.text(x0 + .05 * (x1 - x0), y0 + .07 * (y1 - y0), f"{scale / 1000:.1f} km", color="w", weight="bold")
    ax.annotate("N", xy=(x1 - .06 * (x1 - x0), y1 - .05 * (y1 - y0)), xytext=(x1 - .06 * (x1 - x0), y1 - .15 * (y1 - y0)), arrowprops=dict(arrowstyle="-|>", color="white"), color="white", ha="center", weight="bold")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    ax.set_xlabel("EPSG:32645 easting (m)"); ax.set_ylabel("northing (m)")


def section_at(point, tangent_a, tangent_b, width=180.0):
    dx, dy = tangent_b.x - tangent_a.x, tangent_b.y - tangent_a.y
    angle = math.degrees(math.atan2(dy, dx)) % 180
    nx, ny = -dy / math.hypot(dx, dy), dx / math.hypot(dx, dy)
    half = width / 2
    return LineString([(point.x - nx * half, point.y - ny * half), (point.x + nx * half, point.y + ny * half)]), (angle + 90) % 180


def main():
    for p in (CORRIDOR, REPORT, FIG, REVIEW): p.mkdir(parents=True, exist_ok=True)
    pre, post = image_path("05m.tif"), image_path("*composite.tif")
    dem = next((RAW / "dem_12p5m").rglob("*.tif"))
    core = gpd.read_file(next((RAW / "core_area").rglob("*.shp"))).to_crs(CRS)
    port_gdf = gpd.read_file(next((RAW / "jilong_port").rglob("*.shp"))).to_crs(CRS)
    # The downloaded official point controls all port reporting.
    assert port_gdf.geometry.iloc[0].distance(PORT) < 0.01
    vectors = []
    for p in (RAW / "basic_geography").rglob("*.shp"):
        g = gpd.read_file(p).to_crs(CRS)
        if set(g.geom_type).issubset({"LineString", "MultiLineString"}):
            vectors.append((g.geometry.union_all().distance(PORT), p, g))
    _, river_path, river = min(vectors, key=lambda item: item[0])
    parts = [q for geom in river.geometry for q in (geom.geoms if geom.geom_type == "MultiLineString" else [geom])]
    parts.sort(key=lambda line: line.distance(OLD_ENTRY))
    upstream, downstream = parts[0], parts[1]
    a, b = Point(upstream.coords[-1]), Point(downstream.coords[0])
    endpoint_gap = a.distance(b)
    bridge = gpd.read_file(CORRIDOR / "candidate_gap_bridge.geojson").to_crs(CRS)
    bridge_line = bridge.geometry.iloc[0]
    bridge_length = bridge_line.length

    # S4: nearest point on the official downstream main-channel component to the official port.
    s4 = nearest_points(PORT, downstream)[1]
    coords = list(downstream.coords)
    if s4.equals(Point(coords[-1])):
        tangent_a, tangent_b = Point(coords[-2]), Point(coords[-1])
    else:
        tangent_a, tangent_b = Point(coords[0]), Point(coords[1])
    s4_section, s4_orientation = section_at(s4, tangent_a, tangent_b)
    s4_offset = PORT.distance(s4)
    s4_gdf = gpd.GeoDataFrame({"section_id":["S4"], "official_port_offset_m":[s4_offset], "channel_basis":["nearest point on official fourth-order river; pre/post imagery inspected"], "orientation_deg":[s4_orientation], "confidence":["MODERATE"], "notes":["Hydraulic inspection section, not the port facility point; 180 m span is an inspection extent, not a surveyed channel width."]}, geometry=[s4_section], crs=CRS)
    s4_gdf.to_file(CORRIDOR / "jilong_S4_port_section.geojson", driver="GeoJSON")

    gap_aoi = box(min(a.x, b.x) - 1000, min(a.y, b.y) - 1000, max(a.x, b.x) + 1000, max(a.y, b.y) + 1000)
    port_aoi = PORT.buffer(1500)
    entry_aoi = OLD_ENTRY.buffer(1500)
    full_aoi = box(min(OLD_ENTRY.x, PORT.x) - 3000, min(OLD_ENTRY.y, PORT.y) - 3000, max(OLD_ENTRY.x, PORT.x) + 3000, max(OLD_ENTRY.y, PORT.y) + 3000)
    subjects = {"official_port": (PORT, port_aoi), "vector_gap": (bridge_line.interpolate(.5, normalized=True), gap_aoi), "old_diagnostic_entry": (OLD_ENTRY, entry_aoi), "full_intended_reach": (OLD_ENTRY, full_aoi)}
    coverage_audit = {}
    for name, image in (("pre_event_0p5m", pre), ("post_event_planet_3m", post)):
        with rasterio.open(image) as ds:
            coverage_audit[name] = {key: coverage(ds, p, aoi) for key, (p, aoi) in subjects.items()}
            coverage_audit[name]["image_crs"] = str(ds.crs)
            coverage_audit[name]["image_resolution"] = list(ds.res)

    # The only physically mapped candidate in the intended stage has no public event-path evidence.
    old_entry_channel_distance = (upstream.length - upstream.project(OLD_ENTRY)) + bridge_length + downstream.project(s4)
    s0_row = {
        "candidate_id":"S0_A_OLD_DIAGNOSTIC_MAIN_CHANNEL", "x":OLD_ENTRY.x, "y":OLD_ENTRY.y,
        "feature_type":"official fourth-order main-channel location; not a mapped event confluence",
        "main_channel_supported":True, "tributary_or_event_path_supported":False,
        "pre_event_visible":coverage_audit["pre_event_0p5m"]["old_diagnostic_entry"]["point_covered"],
        "post_event_visible":coverage_audit["post_event_planet_3m"]["old_diagnostic_entry"]["point_covered"],
        "dem_supported":True, "river_vector_supported":True, "hydrology_supported":False,
        "event_specific_evidence":"UNSUPPORTED", "distance_to_official_port_straight_km":OLD_ENTRY.distance(PORT)/1000,
        "estimated_channel_distance_to_port_km":old_entry_channel_distance/1000, "confidence":"LOW",
        "rejection_or_acceptance_reason":"Rejected: core-area containment and a normal mapped channel do not establish where the 2026 event entered the main river; no mapped event path and no post-event coverage at this location."
    }
    with (CORRIDOR / "S0_CANDIDATES.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=s0_row.keys()); w.writeheader(); w.writerow(s0_row)

    # Candidate S0 review figure: no blank panel is allowed to imply post-event evidence.
    entry_bounds = (OLD_ENTRY.x - 1500, OLD_ENTRY.y - 1500, OLD_ENTRY.x + 1500, OLD_ENTRY.y + 1500)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    show_image(axes[0], pre, entry_bounds, "S0 candidate A — pre-event 0.5 m DOM")
    draw_lines(axes[0], river, color="cyan", lw=1)
    draw_polygon(axes[0], core, color="orange", lw=.8)
    axes[0].scatter([OLD_ENTRY.x], [OLD_ENTRY.y], color="yellow", s=55, label="diagnostic candidate only")
    axes[0].legend(loc="upper left"); decorate(axes[0], entry_bounds)
    z, db = read_dem(dem, entry_bounds)
    axes[1].imshow(hillshade(z), extent=(db[0], db[2], db[1], db[3]), cmap="gray")
    draw_lines(axes[1], river, color="cyan", lw=1)
    draw_polygon(axes[1], core, color="orange", lw=.8)
    axes[1].scatter([OLD_ENTRY.x], [OLD_ENTRY.y], color="yellow", s=55)
    axes[1].text(.5, .96, "POST-EVENT IMAGE NOT AVAILABLE AT THIS LOCATION\nNo mapped event path / scour layer available", ha="center", va="top", transform=axes[1].transAxes, color="white", bbox={"facecolor":"black", "alpha":.65})
    axes[1].set_title("S0 candidate A — DEM hillshade"); decorate(axes[1], entry_bounds)
    fig.subplots_adjust(wspace=.14, left=.06, right=.99, bottom=.09, top=.93)
    fig.savefig(FIG / "08_S0_candidate_A.png", dpi=180); plt.close(fig)

    # S4 panel uses post-event imagery, with the official port distinct from the actual channel section.
    port_bounds = (PORT.x - 1500, PORT.y - 1500, PORT.x + 1500, PORT.y + 1500)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    show_image(axes[0], post, port_bounds, "S4 — post-event PlanetScope 3 m DOM")
    draw_lines(axes[0], river, color="cyan", lw=1.1, label="official river")
    draw_polygon(axes[0], core, color="orange", lw=.8, label="official core area")
    axes[0].plot(*s4_section.xy, color="yellow", lw=2.2, label="S4 inspection section")
    axes[0].scatter([PORT.x], [PORT.y], marker="*", s=130, color="red", label="official port point")
    axes[0].scatter([s4.x], [s4.y], s=55, color="yellow", label="S4 channel point")
    axes[0].legend(loc="upper left"); decorate(axes[0], port_bounds)
    z, db = read_dem(dem, port_bounds)
    axes[1].imshow(hillshade(z), extent=(db[0], db[2], db[1], db[3]), cmap="gray")
    draw_lines(axes[1], river, color="cyan", lw=1.1)
    axes[1].plot(*s4_section.xy, color="yellow", lw=2.2)
    axes[1].scatter([PORT.x], [PORT.y], marker="*", s=130, color="red")
    axes[1].scatter([s4.x], [s4.y], s=55, color="yellow")
    axes[1].set_title("S4 — native 12.5 m DEM hillshade"); decorate(axes[1], port_bounds)
    fig.subplots_adjust(wspace=.14, left=.06, right=.99, bottom=.09, top=.93)
    fig.savefig(FIG / "10_S4_port_section.png", dpi=180); plt.close(fig)

    # No S0 => no canonical corridor, profile, or S1--S3.
    final = {
        "gate_a":"HUMAN_IMAGE_REVIEW_REQUIRED", "analysis_crs":CRS,
        "official_port_coordinate":[PORT.x, PORT.y],
        "endpoint_gap_straight_m":endpoint_gap, "candidate_gap_trace_length_m":bridge_length, "gap_resolved":True,
        "s0_resolved":False, "s0_coordinate":None, "s0_event_evidence_class":"UNSUPPORTED",
        "s4_resolved":True, "s4_port_offset_m":s4_offset,
        "canonical_corridor_established":False, "canonical_corridor_length_km":None,
        "control_sections_created":False,
        "pre_event_old_entry_point_covered":coverage_audit["pre_event_0p5m"]["old_diagnostic_entry"]["point_covered"],
        "post_event_old_entry_point_covered":coverage_audit["post_event_planet_3m"]["old_diagnostic_entry"]["point_covered"],
        "human_image_review_required":True,
        "primary_blocker":"NO_EVENT_SPECIFIC_SPATIAL_EVIDENCE: no published or mapped event-path/scour evidence establishes the effective debris-flow entry S0; post-event imagery does not cover the only diagnostic main-channel candidate."
    }
    (REPORT / "JILONG_GATE_A_FINAL.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    (REPORT / "IMAGERY_COVERAGE_AUDIT_FINAL.md").write_text("# Final imagery coverage audit\n\nCoverage semantics are explicit: `point_covered` is point containment; `aoi_any_overlap` means any intersection; `aoi_full_coverage` means the full defined AOI is inside the image.\n\n```json\n" + json.dumps(coverage_audit, indent=2) + "\n```\n", encoding="utf-8")
    topo = {"official_river_layer":str(river_path), "relevant_component_count":len(parts), "endpoint_gap_straight_m":endpoint_gap, "candidate_gap_trace_length_m":bridge_length, "candidate_trace_source":"existing manually digitized candidate_gap_bridge.geojson; reviewed against pre-event DOM, post-event DOM, and 12.5 m DEM", "candidate_trace_confidence":"MODERATE — connectivity only, not surveyed centreline", "artificial_straight_connector_used":False}
    (REPORT / "RIVER_VECTOR_TOPOLOGY_AUDIT_FINAL.md").write_text("# Final river-vector topology audit\n\n```json\n" + json.dumps(topo, indent=2) + "\n```\n\nThe two distances are not interchangeable. The endpoint value is a straight-line separation; the candidate value is the length of the imagery-supported LineString.\n", encoding="utf-8")
    final_md = f"""# Final Jilong/Gyirong spatial Gate A\n\n## Decision\n\n`GATE_A = HUMAN_IMAGE_REVIEW_REQUIRED`.\n\nThe official two-component fourth-order river has a straight endpoint discontinuity of **{endpoint_gap:.2f} m**. The existing candidate channel trace is **{bridge_length:.2f} m** long. It is supported as a connectivity trace by the 0.5 m pre-event DOM, is not contradicted by the 3 m post-event DOM, and is compatible with the 12.5 m DEM valley. It remains explicitly non-official and non-hydraulic.\n\n## S0\n\nOne physically mapped main-channel candidate was assessed: the historical diagnostic location at `({OLD_ENTRY.x:.1f}, {OLD_ENTRY.y:.1f})`. It is rejected as S0. A normal mapped channel, core-area containment, and pre-event visibility do not establish the 26 August event transition into the downstream stage. No mapped event path/scour evidence identifies this location, and the post-event image does not cover it.\n\n## S4\n\nS4 is resolved as a spatially defensible inspection section on the nearest official downstream main-channel geometry, **{s4_offset:.2f} m** from the official port point, rather than at the port facility point itself. See `corridor/jilong_S4_port_section.geojson`.\n\n## Consequence\n\nNo canonical corridor, S0/S1/S2/S3 sections, or longitudinal profile was created. This prevents an unsupported entry assumption from becoming D-Claw geometry.\n\n## Precise review request\n\nThe remaining requirement is an event-specific, georeferenced indication of where the already-formed debris flow entered the modeled main-river stage. The open questions are in `review_package/S0_FINAL_REVIEW_README.txt`.\n"""
    (REPORT / "JILONG_GATE_A_FINAL.md").write_text(final_md, encoding="utf-8")
    (REVIEW / "S0_FINAL_REVIEW_README.txt").write_text("1. Is there a published/mapped event corridor or post-event scour trace that identifies the main-river entry?\n2. Does it intersect the fourth-order river at a distinct physical confluence/transition?\n3. Can one EPSG:32645 S0 control section be defensibly placed there?\n4. Does any already available official contrast/event product cover that exact transition?\n", encoding="utf-8")
    (REVIEW / "README.txt").write_text("Final review figures are stored once under `../figures/` to avoid duplicated binaries:\n- ../figures/08_S0_candidate_A.png\n- ../figures/10_S4_port_section.png\n- ../figures/02_gap_pre_event_0p5m.png\n- ../figures/03_gap_post_event_3m.png\n- ../figures/04_gap_multievidence_overlay.png\n\nRead S0_FINAL_REVIEW_README.txt for the remaining, bounded review questions.\n", encoding="utf-8")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
