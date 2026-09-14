"""Build a checked GeoClaw type-3 terrain raster from the native 12.5-m DEM.

The key convention is explicit: rasterio destination rows run north-to-south,
whereas GeoClaw Topography stores y in increasing (south-to-north) order.
"""
from __future__ import annotations

from pathlib import Path
import os
import json
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling, transform
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
from clawpack.geoclaw import topotools

CASE = Path(__file__).resolve().parent
PROJECT = CASE.parents[2]


def data_root() -> Path:
    """Locate unversioned downloaded data without copying it into Git."""
    requested = os.environ.get("JILONG_DATA_ROOT")
    candidates = [Path(requested) if requested else None,
                  PROJECT / "data/extracted",
                  PROJECT.parent / "Jilong-DebrisFlow/data/extracted"]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    raise FileNotFoundError("Set JILONG_DATA_ROOT to the downloaded Jilong data directory")


DATA = data_root()
DEM12 = DATA / "dem_12_5m/DEM 12.5m.tif"
DEM8 = next((DATA / "dem_8m").rglob("*.tif"))

# A 64-m fixed grid covering the mapped entry-to-port corridor.
XL, YL, DX, MX, MY = 333765.8, 3128840.5, 64.0, 120, 255
XU, YU = XL + MX * DX, YL + MY * DX
X = np.linspace(XL, XU, MX + 1)
Y = np.linspace(YL, YU, MY + 1)  # GeoClaw physical y ordering: south -> north
TOPO = CASE / "jilong_dem_12p5m_64m.tt3"
BUILD_INFO = {}


def bilinear_from_source(src, xy: np.ndarray) -> np.ndarray:
    """Bilinear sample source DEM in its native raster coordinates."""
    sx, sy = xy[:, 0], xy[:, 1]
    if str(src.crs) != "EPSG:32645":
        sx, sy = transform("EPSG:32645", src.crs, sx.tolist(), sy.tolist())
    sx, sy = np.asarray(sx), np.asarray(sy)
    inv = ~src.transform
    cols, rows = inv * (sx, sy)
    # raster coordinate maps pixel corner; shift to pixel-centre index space.
    cols = np.asarray(cols) - 0.5
    rows = np.asarray(rows) - 0.5
    c0, r0 = np.floor(cols).astype(int), np.floor(rows).astype(int)
    dc, dr = cols - c0, rows - r0
    arr = src.read(1).astype(float)
    nodata = src.nodata
    out = np.full(xy.shape[0], np.nan)
    valid = (c0 >= 0) & (r0 >= 0) & (c0 + 1 < src.width) & (r0 + 1 < src.height)
    for n in np.flatnonzero(valid):
        vals = np.array([arr[r0[n], c0[n]], arr[r0[n], c0[n] + 1],
                         arr[r0[n] + 1, c0[n]], arr[r0[n] + 1, c0[n] + 1]])
        if nodata is not None and np.any(vals == nodata):
            continue
        if not np.all(np.isfinite(vals)):
            continue
        out[n] = ((1-dr[n]) * ((1-dc[n]) * vals[0] + dc[n] * vals[1]) +
                  dr[n] * ((1-dc[n]) * vals[2] + dc[n] * vals[3]))
    return out


def build() -> None:
    if not DEM12.exists() or not DEM8.exists():
        raise FileNotFoundError(f"Required observed DEM missing: {DEM8} / {DEM12}")
    # The transform's first output row has physical centre y=YU.  It is
    # deliberately reversed below before handing it to GeoClaw.
    dst_transform = Affine.translation(XL - DX / 2, YU + DX / 2) * Affine.scale(DX, -DX)
    def warp(path):
        out = np.full((MY + 1, MX + 1), np.nan, dtype=float)
        with rasterio.open(path) as src:
            reproject(rasterio.band(src, 1), out, src_transform=src.transform, src_crs=src.crs,
                      src_nodata=src.nodata, dst_transform=dst_transform, dst_crs="EPSG:32645",
                      dst_nodata=np.nan, resampling=Resampling.bilinear)
        return out
    # Native 8-m topography has priority wherever it is valid; 12.5-m fills
    # its coverage gap. Both are independently reprojected to the same
    # explicitly defined physical grid before the one required y reordering.
    z8, z12 = warp(DEM8), warp(DEM12)
    north_to_south = np.where(np.isfinite(z8), z8, z12)
    observed_node_mask = np.isfinite(north_to_south)[::-1, :]  # mask in GeoClaw increasing-y orientation
    BUILD_INFO["observed_node_mask"] = observed_node_mask
    if not np.isfinite(north_to_south).all():
        # The two supplied DEMs have nodata only in off-corridor corners of
        # the rectangular computational box. Fill those cells from the nearest
        # *observed* node, then verify separately that every sampled river
        # corridor/key point is natively observed before allowing a solve.
        observed = np.isfinite(north_to_south)
        _, nearest = distance_transform_edt(~observed, return_indices=True)
        north_to_south[~observed] = north_to_south[tuple(nearest[:, ~observed])]
        BUILD_INFO["nodata_filled_nodes"] = int((~observed).sum())
    else:
        BUILD_INFO["nodata_filled_nodes"] = 0
    topo = topotools.Topography()
    topo.x, topo.y, topo.Z = X, Y, north_to_south[::-1, :]
    topo.write(TOPO, topo_type=3)


def check() -> dict:
    """Compare read-back TT3 elevations with DEM at identical physical points."""
    if not TOPO.exists():
        raise FileNotFoundError(TOPO)
    topo = topotools.Topography()
    topo.read(TOPO, topo_type=3)
    normal = RegularGridInterpolator((topo.y, topo.x), topo.Z, bounds_error=False, fill_value=np.nan)
    mirrored = RegularGridInterpolator((topo.y, topo.x), topo.Z[::-1, :], bounds_error=False, fill_value=np.nan)
    fixed = np.array([
        [334051.8, 3143392.5],  # documented downstream-model entry
        [334450.0, 3141000.0], [335100.0, 3139100.0], [335850.0, 3137000.0],
        [337200.0, 3134700.0],  # mid-corridor
        [340571.8257777202, 3129640.775894154],  # downstream hydraulic proxy
        [340837.9, 3129051.7],  # mapped port point
    ])
    rng = np.random.default_rng(20260915)
    # Random *observed* TT3 nodes avoid treating an interpolation support cell
    # across an external DEM-nodata boundary as an elevation disagreement.
    iy, ix = np.where(BUILD_INFO["observed_node_mask"])
    if len(ix) < 100: raise RuntimeError("fewer than 100 observed TT3 nodes")
    pick = rng.choice(len(ix), 100, replace=False)
    random_xy = np.column_stack((X[ix[pick]], Y[iy[pick]]))
    points = np.vstack((fixed, random_xy))
    with rasterio.open(DEM8) as src8, rasterio.open(DEM12) as src12:
        dem8, dem12 = bilinear_from_source(src8, points), bilinear_from_source(src12, points)
    dem = np.where(np.isfinite(dem8), dem8, dem12)
    generated = normal(points[:, ::-1])
    wrong_y = mirrored(points[:, ::-1])
    err = np.abs(generated - dem)
    mirror_err = np.abs(wrong_y - dem)
    valid = np.isfinite(err)
    stats = {
        "n_total": int(len(points)), "n_valid": int(valid.sum()),
        "median_abs_difference_m": float(np.nanmedian(err)),
        "p95_abs_difference_m": float(np.nanpercentile(err, 95)),
        "max_abs_difference_m": float(np.nanmax(err)),
        "wrong_y_mirror_median_abs_difference_m": float(np.nanmedian(mirror_err)),
        "nodata_filled_nodes_outside_observed_coverage": BUILD_INFO["nodata_filled_nodes"],
        "fixed_points": [
            {"x": float(p[0]), "y": float(p[1]), "dem_m": float(d), "tt3_m": float(g),
             "abs_difference_m": float(e)}
            for p, d, g, e in zip(fixed, dem[:len(fixed)], generated[:len(fixed)], err[:len(fixed)])
        ],
    }
    # 64-m bilinear resampling should be close to the native DEM; a mirror
    # would produce a vastly larger discrepancy in this steep valley.
    stats["passed"] = bool(valid.sum() == len(points) and stats["median_abs_difference_m"] <= 12.5
                           and stats["p95_abs_difference_m"] <= 40.0 and stats["max_abs_difference_m"] <= 50.0
                           and stats["wrong_y_mirror_median_abs_difference_m"] > 100.0)
    (CASE / "terrain_check.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    lines = ["# Terrain consistency check", "", "**Result: {}**".format("PASS" if stats["passed"] else "FAIL"), "",
             "The source raster was reprojected with physical rows north-to-south, then reversed only when assigning GeoClaw's increasing-y `Topography.Z`. The TT3 file was read back before comparison.", "",
             "| Metric | Value |", "|---|---:|",
             f"| Samples / valid | {stats['n_total']} / {stats['n_valid']} |",
             f"| Median |Δz| | {stats['median_abs_difference_m']:.3f} m |",
             f"| 95th percentile |Δz| | {stats['p95_abs_difference_m']:.3f} m |",
             f"| Maximum |Δz| | {stats['max_abs_difference_m']:.3f} m |",
             f"| Counterfactual y-mirror median |Δz| | {stats['wrong_y_mirror_median_abs_difference_m']:.3f} m |", "",
             f"| Nearest-observed nodata fills outside coverage | {stats['nodata_filled_nodes_outside_observed_coverage']} nodes |", "",
             "## Required fixed physical points", "", "| x (m) | y (m) | Native DEM (m) | Generated TT3 (m) | |Δz| (m) |", "|---:|---:|---:|---:|---:|"]
    lines += [f"| {p['x']:.1f} | {p['y']:.1f} | {p['dem_m']:.2f} | {p['tt3_m']:.2f} | {p['abs_difference_m']:.3f} |" for p in stats["fixed_points"]]
    lines += ["", "The 100 seeded random **valid native-DEM** points are included in the aggregate metrics above. Nearest-observed fills are confined to DEM nodata nodes; the required entry, route, mid-corridor and port checks must all be native-observation comparisons. A failed check is a hard stop for the solver."]
    (CASE / "TERRAIN_CHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return stats


if __name__ == "__main__":
    build()
    result = check()
    print(json.dumps(result, indent=2))
