"""Prepare the one real-data topography and initial source for the Jilong baseline."""
from __future__ import annotations

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

# EPSG:32645 (WGS 84 / UTM zone 45N), a single metric CRS for the whole run.
XL, YL = 331_781.9, 3_128_083.2
DX = 64.0
MX, MY = 176, 261
XU, YU = XL + MX * DX, YL + MY * DX
ENTRY_X, ENTRY_Y = 334_051.5, 3_143_392.2
SOURCE_A, SOURCE_B, SOURCE_DEPTH = 192.0, 128.0, 3.0


def source_path(name: str) -> Path:
    return next(DATA.rglob(name))


def reproject_dem(path: Path, transform: Affine, shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, np.nan, dtype=np.float64)
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs="EPSG:32645",
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return out


def main() -> None:
    dem8 = source_path("DongLinZangBu_basin_HMA8mDEM_fillAW3D30.tif")
    dem125 = source_path("DEM 12.5m.tif")
    x = np.linspace(XL, XU, MX + 1)
    y = np.linspace(YL, YU, MY + 1)
    xx, yy = np.meshgrid(x, y)
    # Values are sampled at topo-grid points, hence the half-cell shifted transform.
    transform = Affine.translation(XL - DX / 2, YU + DX / 2) * Affine.scale(DX, -DX)
    shape = (MY + 1, MX + 1)
    z8 = reproject_dem(dem8, transform, shape)
    z125 = reproject_dem(dem125, transform, shape)
    z = np.where(np.isfinite(z8), z8, z125)
    if not np.isfinite(z).any():
        raise RuntimeError("Neither DEM provided usable elevation in the study domain")
    # The 8 m DEM is primary.  Nearest fill is only for residual cells where both DEMs are void.
    if not np.isfinite(z).all():
        indices = distance_transform_edt(~np.isfinite(z), return_distances=False, return_indices=True)
        z = z[tuple(indices)]

    h = np.zeros_like(z)
    source = ((xx - ENTRY_X) / SOURCE_A) ** 2 + ((yy - ENTRY_Y) / SOURCE_B) ** 2 <= 1.0
    h[source] = SOURCE_DEPTH
    for filename, values in (("jilong_dem_primary_64m.tt3", z), ("initial_thickness.tt3", h)):
        topo = topotools.Topography()
        topo.x, topo.y, topo.Z = x, y, values
        topo.write(CASE / filename, topo_type=3)

    summary = {
        "crs": "EPSG:32645",
        "domain_utm_m": {"xlower": XL, "xupper": XU, "ylower": YL, "yupper": YU},
        "grid": {"dx_m": DX, "mx": MX, "my": MY},
        "dem": {
            "primary": str(dem8.relative_to(PROJECT)),
            "fallback_for_8m_voids": str(dem125.relative_to(PROJECT)),
            "8m_valid_topo_points_percent": round(float(np.isfinite(z8).mean() * 100), 2),
        },
        "source": {
            "representation": "D-Claw qinit thickness field; stationary elliptical initial mass at main-river entry",
            "center_utm_m": [ENTRY_X, ENTRY_Y],
            "semi_axes_m": [SOURCE_A, SOURCE_B],
            "depth_m": SOURCE_DEPTH,
            "grid_volume_m3": float(h.sum() * DX * DX),
        },
    }
    (CASE / "input_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
