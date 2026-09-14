"""Compute fixed, reproducible Phase-2B port diagnostics from D-Claw frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from clawpack.pyclaw.solution import Solution

PORT_X, PORT_Y = 340_837.9, 3_129_051.7
WET_THRESHOLD_M = 0.01
# Downstream direction of the mapped-vector terminus to the official port point.
FLOW_DIRECTION = np.array([0.4115, -0.9114])
CROSS_NORMAL = np.array([-FLOW_DIRECTION[1], FLOW_DIRECTION[0]])
CROSS_WIDTH_M, CROSS_DS_M = 384.0, 64.0


def bilinear(values: np.ndarray, x: np.ndarray, y: np.ndarray, xp: float, yp: float) -> float:
    i = int(np.clip(np.searchsorted(x, xp) - 1, 0, len(x) - 2))
    j = int(np.clip(np.searchsorted(y, yp) - 1, 0, len(y) - 2))
    tx, ty = (xp - x[i]) / (x[i + 1] - x[i]), (yp - y[j]) / (y[j + 1] - y[j])
    return float((1-tx)*(1-ty)*values[i,j] + tx*(1-ty)*values[i+1,j] + (1-tx)*ty*values[i,j+1] + tx*ty*values[i+1,j+1])


def read_frame(number: int, output: Path):
    sol = Solution(number, path=output, file_format="ascii")
    q = sol.state.q
    x, y = sol.state.grid.dimensions[0].centers, sol.state.grid.dimensions[1].centers
    return float(sol.t), x, y, q[0], q[1], q[2]


def diagnostic_for_frame(number: int, output: Path) -> dict:
    t, x, y, h, hu, hv = read_frame(number, output)
    hp = bilinear(h, x, y, PORT_X, PORT_Y)
    hup = bilinear(hu, x, y, PORT_X, PORT_Y)
    hvp = bilinear(hv, x, y, PORT_X, PORT_Y)
    speedp = float(np.hypot(hup, hvp) / hp) if hp > WET_THRESHOLD_M else 0.0
    offsets = np.arange(-CROSS_WIDTH_M / 2 + CROSS_DS_M / 2, CROSS_WIDTH_M / 2, CROSS_DS_M)
    qsum = 0.0
    for offset in offsets:
        px, py = np.array([PORT_X, PORT_Y]) + offset * CROSS_NORMAL
        hs = bilinear(h, x, y, px, py)
        if hs > WET_THRESHOLD_M:
            us = bilinear(hu, x, y, px, py) / hs
            vs = bilinear(hv, x, y, px, py) / hs
            qsum += max(0.0, hs * (us * FLOW_DIRECTION[0] + vs * FLOW_DIRECTION[1])) * CROSS_DS_M
    wet = h > WET_THRESHOLD_M
    speed = np.zeros_like(h)
    speed[wet] = np.hypot(hu[wet] / h[wet], hv[wet] / h[wet])
    return {"time_s": t, "port_depth_m": hp, "port_speed_ms": speedp, "port_discharge_m3s": qsum,
            "finite": bool(np.isfinite(q).all()) if (q := np.stack((h, hu, hv))).size else False,
            "domain_max_speed_ms": float(speed.max())}


def main(output: Path, summary: Path) -> None:
    frames = sorted(int(p.name[-4:]) for p in output.glob("fort.q????"))
    if not frames:
        raise RuntimeError(f"no ascii q frames in {output}")
    values = [diagnostic_for_frame(n, output) for n in frames]
    peak = max(values, key=lambda d: d["port_discharge_m3s"])
    arrival = next((d["time_s"] for d in values if d["port_depth_m"] > WET_THRESHOLD_M), None)
    _, x0, y0, h0, _, _ = read_frame(frames[0], output)
    dx, dy = x0[1] - x0[0], y0[1] - y0[0]
    result = {
        "wet_threshold_m": WET_THRESHOLD_M,
        "port_cross_section": {
            "center_utm_m": [PORT_X, PORT_Y], "flow_direction_unit": FLOW_DIRECTION.tolist(),
            "normal_unit": CROSS_NORMAL.tolist(), "width_m": CROSS_WIDTH_M, "strip_width_m": CROSS_DS_M,
            "integration": "sum(max(0, h*(u*dx_downstream + v*dy_downstream))*strip_width) over six 64 m strips",
        },
        "reached_port": arrival is not None, "arrival_time_s": arrival,
        "port_peak_discharge_m3s": peak["port_discharge_m3s"],
        "port_peak_depth_m": max(d["port_depth_m"] for d in values),
        "port_peak_speed_ms": max(d["port_speed_ms"] for d in values),
        "source_volume_m3": float(h0.sum() * dx * dy),
        "stable_fields": bool(all(d["finite"] for d in values)),
        "domain_max_speed_ms": max(d["domain_max_speed_ms"] for d in values),
        "time_series": values,
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "time_series"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    main(args.output, args.summary)
