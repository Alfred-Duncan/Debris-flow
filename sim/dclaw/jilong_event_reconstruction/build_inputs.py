"""Build fixed-grid topography and a moving D-Claw qinit source for one candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
from scipy.ndimage import distance_transform_edt
from clawpack.geoclaw import topotools

CASE = Path(__file__).resolve().parent
PROJECT = CASE.parents[2]
DATA = PROJECT / "data" / "extracted"

XL, YL = 331_781.9, 3_128_083.2
DX, MX, MY = 64.0, 176, 261
XU, YU = XL + MX * DX, YL + MY * DX
# Fixed Phase-2B model-entry point: a 0.4 m snap of the original baseline point.
ENTRY_X, ENTRY_Y = 334_051.8, 3_143_392.5
SOURCE_A, SOURCE_B = 192.0, 128.0
# Unit vector along the mapped river immediately downstream of the model entry.
DOWNSTREAM_UX, DOWNSTREAM_UY = 0.7623, -0.6472


def source_path(name: str) -> Path:
    return next(DATA.rglob(name))


def reproject_dem(path: Path, transform: Affine, shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, np.nan, dtype=np.float64)
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1), destination=out,
            src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
            dst_transform=transform, dst_crs="EPSG:32645", dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return out


def write_tt3(name: str, x: np.ndarray, y: np.ndarray, values: np.ndarray) -> None:
    topo = topotools.Topography()
    topo.x, topo.y, topo.Z = x, y, values
    topo.write(CASE / name, topo_type=3)


def main(depth_m: float, speed_ms: float) -> None:
    if depth_m <= 0 or speed_ms < 0:
        raise ValueError("depth must be positive and speed non-negative")
    dem8 = source_path("DongLinZangBu_basin_HMA8mDEM_fillAW3D30.tif")
    dem125 = source_path("DEM 12.5m.tif")
    x, y = np.linspace(XL, XU, MX + 1), np.linspace(YL, YU, MY + 1)
    xx, yy = np.meshgrid(x, y)
    transform = Affine.translation(XL - DX / 2, YU + DX / 2) * Affine.scale(DX, -DX)
    shape = (MY + 1, MX + 1)
    z8, z125 = reproject_dem(dem8, transform, shape), reproject_dem(dem125, transform, shape)
    z = np.where(np.isfinite(z8), z8, z125)
    if not np.isfinite(z).any():
        raise RuntimeError("Neither DEM provided usable elevation in the study domain")
    if not np.isfinite(z).all():
        indices = distance_transform_edt(~np.isfinite(z), return_distances=False, return_indices=True)
        z = z[tuple(indices)]

    source = ((xx - ENTRY_X) / SOURCE_A) ** 2 + ((yy - ENTRY_Y) / SOURCE_B) ** 2 <= 1.0
    h = np.zeros_like(z)
    u = np.zeros_like(z)
    v = np.zeros_like(z)
    h[source] = depth_m
    # D-Claw qinit components 2 and 3 accept u and v, and form hu/hv internally.
    u[source] = speed_ms * DOWNSTREAM_UX
    v[source] = speed_ms * DOWNSTREAM_UY
    write_tt3("jilong_dem_primary_64m.tt3", x, y, z)
    write_tt3("initial_thickness.tt3", x, y, h)
    write_tt3("initial_velocity_u.tt3", x, y, u)
    write_tt3("initial_velocity_v.tt3", x, y, v)
    summary = {
        "crs": "EPSG:32645",
        "domain_utm_m": {"xlower": XL, "xupper": XU, "ylower": YL, "yupper": YU},
        "grid": {"dx_m": DX, "mx": MX, "my": MY},
        "dem": {
            "primary": str(dem8.relative_to(PROJECT)),
            "fallback_for_8m_voids": str(dem125.relative_to(PROJECT)),
            "8m_valid_topo_points_percent": round(float(np.isfinite(z8).mean() * 100), 2),
        },
        "source_inverse_parameters": {
            "representation": "moving elliptical qinit mass at model entry",
            "center_utm_m": [ENTRY_X, ENTRY_Y],
            "semi_axes_m": [SOURCE_A, SOURCE_B],
            "depth_m": depth_m,
            "initial_speed_ms": speed_ms,
            "downstream_unit_vector": [DOWNSTREAM_UX, DOWNSTREAM_UY],
            "grid_volume_m3": float(h.sum() * DX * DX),
        },
    }
    (CASE / "input_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-m", type=float, required=True)
    parser.add_argument("--speed-ms", type=float, required=True)
    args = parser.parse_args()
    main(args.depth_m, args.speed_ms)
