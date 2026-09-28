"""Local diagnostics for fixed-grid Jilong unit-Froude D-Claw outputs."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
from shapely.geometry import LineString, Point
from clawpack.pyclaw.solution import Solution

CASE = Path(__file__).resolve().parent
HANDOFF = Path("/root/autodl-tmp/Jilong_DClaw_Handoff")
DX, WET = 64.0, 1.0e-3

def geometry():
    corridor = json.loads((HANDOFF/"geometry/jilong_canonical_corridor.geojson").read_text())
    sections = json.loads((HANDOFF/"geometry/jilong_control_sections.geojson").read_text())
    line = LineString(corridor["features"][0]["geometry"]["coordinates"])
    sec = {f["properties"]["section_id"]: LineString(f["geometry"]["coordinates"]) for f in sections["features"]}
    return line, sec

def masks(x, y, line, sections):
    xx, yy = np.meshgrid(x, y, indexing="ij")
    points = [Point(float(a), float(b)) for a, b in zip(xx.ravel(), yy.ravel())]
    dline = np.array([line.distance(p) for p in points]).reshape(xx.shape)
    chain = np.array([line.project(p) for p in points]).reshape(xx.shape)
    section_masks = {name: np.array([s.distance(p) <= DX*.75 for p in points]).reshape(xx.shape)
                     for name, s in sections.items() if name != "S0"}
    return dline <= 256.0, chain, section_masks

def analyze(output: Path) -> dict:
    frame_ids = sorted(int(p.name[-4:]) for p in output.glob("fort.q????"))
    if not frame_ids: raise RuntimeError(f"no frames in {output}")
    line, sections = geometry()
    section_rows = {k: [] for k in ("S1","S2","S3","S4")}
    frames = []
    corridor_mask = chainage = section_masks = None
    for fid in frame_ids:
        sol = Solution(fid, path=output, file_format="ascii")
        q = sol.state.q
        x, y = sol.state.grid.dimensions[0].centers, sol.state.grid.dimensions[1].centers
        h, hu, hv = q[0], q[1], q[2]
        if corridor_mask is None:
            corridor_mask, chainage, section_masks = masks(x, y, line, sections)
        wet = h > WET
        speed = np.zeros_like(h)
        speed[wet] = np.hypot(hu[wet]/h[wet], hv[wet]/h[wet])
        on_corridor = wet & corridor_mask
        for name in section_rows:
            sm = section_masks[name]
            section_rows[name].append({"time_s":float(sol.t),
                "max_depth_m":float(h[sm].max()) if sm.any() else 0.0,
                "max_speed_ms":float(speed[sm].max()) if sm.any() else 0.0})
        frames.append({
            "frame":fid, "time_s":float(sol.t), "finite":bool(np.isfinite(q).all()),
            "max_depth_m":float(h.max()), "max_speed_ms":float(speed[wet].max()) if wet.any() else 0.0,
            "p99_wet_speed_ms":float(np.percentile(speed[wet],99)) if wet.any() else 0.0,
            "wet_cell_count":int(wet.sum()), "wet_volume_m3":float(h.sum()*DX*DX),
            "max_chainage_m":float(chainage[on_corridor].max()) if on_corridor.any() else 0.0,
        })
    arrivals, peaks = {}, {}
    for name, rows in section_rows.items():
        arrivals[name] = {str(th):next((r["time_s"] for r in rows if r["max_depth_m"] > th),None)
                          for th in (.05,.10,.20)}
        peaks[name] = {"max_depth_m":max(r["max_depth_m"] for r in rows),
                       "max_speed_ms":max(r["max_speed_ms"] for r in rows)}
    return {"frames":frames, "section_arrivals_s":arrivals, "section_peaks":peaks,
            "finite":all(r["finite"] for r in frames),
            "max_depth_m":max(r["max_depth_m"] for r in frames),
            "max_speed_ms":max(r["max_speed_ms"] for r in frames),
            "p99_wet_speed_ms":max(r["p99_wet_speed_ms"] for r in frames),
            "max_chainage_m":max(r["max_chainage_m"] for r in frames)}

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True); p.add_argument("--json",type=Path,required=True)
    a=p.parse_args(); a.json.write_text(json.dumps(analyze(a.output),indent=2)+"\n")
