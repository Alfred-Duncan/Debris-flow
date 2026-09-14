"""Create the minimal visual and numerical checks for the one baseline run."""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from clawpack.pyclaw.solution import Solution
from shapely.geometry import Point

CASE = Path(__file__).resolve().parent
PROJECT = CASE.parents[2]
OUT = CASE / "_output"
PREVIEWS = CASE / "previews"
ENTRY = (334_051.5, 3_143_392.2)
PORT = (340_837.9, 3_129_051.7)


def frame(number: int):
    sol = Solution(number, path=OUT, file_format="ascii")
    q = sol.state.q
    x = sol.state.grid.dimensions[0].centers
    y = sol.state.grid.dimensions[1].centers
    h = q[0]
    speed = np.zeros_like(h)
    wet = h > 1.0e-3
    speed[wet] = np.hypot(q[1][wet] / h[wet], q[2][wet] / h[wet])
    return x, y, h, speed


def centroid(x: np.ndarray, y: np.ndarray, h: np.ndarray) -> Point:
    xx, yy = np.meshgrid(x, y, indexing="ij")
    wet = h > 1.0e-2
    weights = h * wet
    return Point(float((xx * weights).sum() / weights.sum()), float((yy * weights).sum() / weights.sum()))


def draw_river(ax, river) -> None:
    """Overlay the mapped fourth-order river for the one visual corridor check."""
    for part in getattr(river, "geoms", [river]):
        coords = np.asarray(part.coords)
        ax.plot(coords[:, 0], coords[:, 1], color="0.25", lw=0.7, alpha=0.8)


def main() -> None:
    PREVIEWS.mkdir(exist_ok=True)
    river_file = next((PROJECT / "data" / "extracted").rglob("四级河流.shp"))
    river = gpd.read_file(river_file).to_crs("EPSG:32645").geometry.iloc[0]
    frames = [0, 3, 6]
    data = [frame(n) for n in frames]
    x, y, _, _ = data[-1]
    extent = (x.min() - 32, x.max() + 32, y.min() - 32, y.max() + 32)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, number, (_, _, h, _) in zip(axes, frames, data):
        image = ax.imshow(h.T, origin="lower", extent=extent, cmap="Blues", vmin=0, vmax=3)
        draw_river(ax, river)
        ax.scatter(*ENTRY, s=25, c="orange", label="entry" if number == 0 else None)
        ax.scatter(*PORT, s=25, c="red", marker="*", label="port" if number == 0 else None)
        ax.set_title(f"depth, t={number * 300} s")
        ax.set_aspect("equal")
    axes[0].legend(loc="upper right")
    fig.colorbar(image, ax=axes, label="flow depth (m)")
    fig.savefig(PREVIEWS / "depth_snapshots.png", dpi=160)
    plt.close(fig)

    _, _, hfinal, speed = data[-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, values, title, cmap in (
        (axes[0], hfinal, "final flow depth", "Blues"),
        (axes[1], np.ma.masked_where(hfinal <= 1.0e-3, speed), "final speed", "magma"),
    ):
        image = ax.imshow(values.T, origin="lower", extent=extent, cmap=cmap)
        draw_river(ax, river)
        ax.scatter(*ENTRY, s=25, c="cyan")
        ax.scatter(*PORT, s=35, c="lime", marker="*")
        ax.set_title(title + " at t=1800 s")
        ax.set_aspect("equal")
        fig.colorbar(image, ax=ax)
    fig.savefig(PREVIEWS / "final_depth_speed.png", dpi=160)
    plt.close(fig)

    initial = centroid(*data[0][:3])
    final = centroid(*data[-1][:3])
    direction = np.sign(river.project(Point(*PORT)) - river.project(Point(*ENTRY)))
    downstream_progress = (river.project(final) - river.project(initial)) * direction
    route_to_port = abs(river.project(Point(*PORT)) - river.project(Point(*ENTRY)))
    gauge = np.loadtxt(OUT / "gauge00001.txt", comments="#")
    if gauge.ndim == 1:
        gauge = gauge[None, :]
    summary = {
        "completed": True,
        "simulated_time_s": 1800.0,
        "frames": frames,
        "initial_centroid_utm_m": [round(initial.x, 1), round(initial.y, 1)],
        "final_centroid_utm_m": [round(final.x, 1), round(final.y, 1)],
        "centroid_displacement_m": round(initial.distance(final), 1),
        "downstream_progress_along_mapped_river_m": round(float(downstream_progress), 1),
        "mapped_entry_to_port_route_m": round(float(route_to_port), 1),
        "final_wet_cells_depth_gt_0_01m": int((hfinal > 1.0e-2).sum()),
        "final_max_depth_m": round(float(hfinal.max()), 4),
        "final_max_speed_m_s": round(float(speed.max()), 4),
        "port_gauge_max_depth_m": round(float(gauge[:, 2].max()), 6),
        "note": "This is a provisional, uncalibrated baseline; no match to event travel time, velocity, or discharge is implied.",
    }
    (CASE / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
