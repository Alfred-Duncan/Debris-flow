"""Produce compact engineering diagnostics for one fixed-grid D-Claw run."""
from __future__ import annotations
import argparse, csv, json, re, os
from pathlib import Path
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import substring
from clawpack.pyclaw.solution import Solution

CASE = Path(__file__).resolve().parent
PROJECT = CASE.parents[2]
ENTRY = (334051.8, 3143392.5)
PROXY = (340571.8257777202, 3129640.775894154)
OFFICIAL_PORT = (340837.9, 3129051.7)
ROUTE_DISTANCE_M, CORRIDOR_TOLERANCE_M, WET = 17624.0, 400.0, 0.01
FLOW = np.array([0.4284, -0.9036])
NORMAL = np.array([-FLOW[1], FLOW[0]])


def route() -> LineString:
    requested = os.environ.get("JILONG_DATA_ROOT")
    candidates = [Path(requested) if requested else None, PROJECT / "data/extracted", PROJECT.parent / "Jilong-DebrisFlow/data/extracted"]
    root = next(p for p in candidates if p and p.exists())
    river = next((root / "basic_geographic_data").rglob("四级河流.shp"))
    multi = gpd.read_file(river).to_crs(32645).geometry.iloc[0]
    a, b = multi.geoms
    return LineString(list(substring(a, a.project(Point(ENTRY)), a.length).coords) + list(b.coords))


def interp(a, x, y, xp, yp):
    i = int(np.clip(np.searchsorted(x, xp) - 1, 0, len(x)-2)); j = int(np.clip(np.searchsorted(y, yp) - 1, 0, len(y)-2))
    tx, ty = (xp-x[i])/(x[i+1]-x[i]), (yp-y[j])/(y[j+1]-y[j])
    return float((1-tx)*(1-ty)*a[i,j] + tx*(1-ty)*a[i+1,j] + (1-tx)*ty*a[i,j+1] + tx*ty*a[i+1,j+1])


def run(output: Path, runtime_s: float, scenario: dict) -> dict:
    frames = sorted(int(p.name[-4:]) for p in output.glob("fort.q????"))
    if not frames:
        raise RuntimeError(f"no output frames in {output}")
    line, scale = route(), None
    scale = ROUTE_DISTANCE_M / line.length
    rows, all_h01 = [], []
    for number in frames:
        sol = Solution(number, path=output, file_format="ascii")
        q, x, y = sol.state.q, sol.state.grid.dimensions[0].centers, sol.state.grid.dimensions[1].centers
        h, hu, hv = q[:3]
        wet = h > WET
        speed = np.zeros_like(h); speed[wet] = np.hypot(hu[wet]/h[wet], hv[wet]/h[wet])
        front = 0.0
        for i, j in zip(*np.where(wet)):
            p = Point(float(x[i]), float(y[j])); d = line.distance(p)
            if d <= CORRIDOR_TOLERANCE_M:
                front = max(front, line.project(p)*scale)
        hp, hup, hvp = interp(h,x,y,*PROXY), interp(hu,x,y,*PROXY), interp(hv,x,y,*PROXY)
        proxy_speed = float(np.hypot(hup,hvp)/hp) if hp > WET else 0.0
        qproxy = 0.0
        for offset in np.arange(-192, 192, 64):
            px, py = np.array(PROXY) + offset*NORMAL
            hs = interp(h,x,y,px,py)
            if hs > WET:
                us, vs = interp(hu,x,y,px,py)/hs, interp(hv,x,y,px,py)/hs
                qproxy += max(0.0, hs*(us*FLOW[0]+vs*FLOW[1]))*64.0
        all_h01.extend(speed[h > .1].tolist())
        rows.append({"time_s":float(sol.t), "front_distance_m":front, "proxy_depth_m":hp,
                     "proxy_speed_ms":proxy_speed, "proxy_discharge_m3s":qproxy,
                     "wet_area_m2":float(wet.sum()*(x[1]-x[0])*(y[1]-y[0])),
                     "raw_max_speed_ms":float(speed[wet].max()) if wet.any() else 0.0,
                     "max_speed_h_gt_01_ms":float(speed[h>.1].max()) if (h>.1).any() else 0.0,
                     "max_speed_h_gt_1_ms":float(speed[h>1].max()) if (h>1).any() else 0.0,
                     "finite":bool(np.isfinite(q).all())})
    log = (output.parent / "run.log").read_text(errors="replace")
    cfl = [float(v.replace("D","E")) for v in re.findall(r"maximum Courant number seen\s*=\s*([0-9.E+\-D]+)", log)]
    arrival = next((r["time_s"] for r in rows if r["proxy_depth_m"] > WET), None)
    result = dict(scenario, solver_completed=True, finite_fields=all(r["finite"] for r in rows),
                  maximum_CFL=max(cfl) if cfl else None, runtime_s=runtime_s,
                  raw_max_speed_ms=max(r["raw_max_speed_ms"] for r in rows),
                  max_speed_h_gt_01_ms=max(r["max_speed_h_gt_01_ms"] for r in rows),
                  max_speed_h_gt_1_ms=max(r["max_speed_h_gt_1_ms"] for r in rows),
                  p99_speed_h_gt_01_ms=float(np.percentile(all_h01,99)) if all_h01 else 0.0,
                  maximum_front_distance_m=max(r["front_distance_m"] for r in rows), final_front_distance_m=rows[-1]["front_distance_m"],
                  proxy_reached=arrival is not None, arrival_time_s=arrival,
                  proxy_peak_discharge_m3s=max(r["proxy_discharge_m3s"] for r in rows),
                  proxy_peak_depth_m=max(r["proxy_depth_m"] for r in rows), proxy_peak_speed_ms=max(r["proxy_speed_ms"] for r in rows),
                  maximum_wet_area_m2=max(r["wet_area_m2"] for r in rows), time_series=rows)
    (output.parent / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    with (output.parent / "front_progress.csv").open("w", newline="") as f:
        w=csv.DictWriter(f,fieldnames=["time_s","front_distance_m","proxy_depth_m","wet_area_m2"]); w.writeheader(); w.writerows([{k:r[k] for k in w.fieldnames} for r in rows])
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True); p.add_argument("--runtime-s",type=float,required=True); p.add_argument("--scenario-json",type=Path,required=True)
    a=p.parse_args(); print(json.dumps(run(a.output,a.runtime_s,json.loads(a.scenario_json.read_text())),indent=2))
