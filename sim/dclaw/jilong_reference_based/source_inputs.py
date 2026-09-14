"""Create stationary qinit mass rasters for a documented volume scenario.

This is a D-Claw-supported initial-mass construction: h and hm are supplied,
but no initial momentum or custom boundary forcing is imposed.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
from clawpack.geoclaw import topotools
from terrain_preprocess import CASE, X, Y, DX

ENTRY = (334051.8, 3143392.5)
# Downstream tangent of the mapped corridor (not an imposed velocity).
DIRECTION = np.array([0.4284, -0.9036])
M0 = 0.62
SEMIMAJOR_M, SEMIMINOR_M = 360.0, 160.0


def build(volume_m3: float) -> dict:
    xx, yy = np.meshgrid(X, Y, indexing="xy")
    dx, dy = xx - ENTRY[0], yy - ENTRY[1]
    along = dx * DIRECTION[0] + dy * DIRECTION[1]
    cross = -dx * DIRECTION[1] + dy * DIRECTION[0]
    r2 = (along / SEMIMAJOR_M) ** 2 + (cross / SEMIMINOR_M) ** 2
    profile = np.maximum(1.0 - r2, 0.0)
    h = profile * volume_m3 / (profile.sum() * DX * DX)
    for name, z in (("initial_thickness.tt3", h), ("initial_solid_volume.tt3", h * M0)):
        topo = topotools.Topography()
        topo.x, topo.y, topo.Z = X, Y, z
        topo.write(CASE / name, topo_type=3)
    metadata = {
        "representation": "stationary D-Claw qinit initial mass; no imposed initial velocity or boundary inflow",
        "volume_m3": float(volume_m3), "discrete_volume_m3": float(h.sum() * DX * DX),
        "m0": M0, "entry_x_m": ENTRY[0], "entry_y_m": ENTRY[1],
        "semimajor_m": SEMIMAJOR_M, "semiminor_m": SEMIMINOR_M,
        "maximum_initial_thickness_m": float(h.max()),
    }
    (CASE / "source_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume-m3", type=float, default=1.0e6)
    args = parser.parse_args()
    print(json.dumps(build(args.volume_m3), indent=2))
