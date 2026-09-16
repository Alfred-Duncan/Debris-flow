"""Compact diagnostics for one 600-s moving-entry D-Claw realization."""
from __future__ import annotations
import json, os, re
from pathlib import Path
import geopandas as gpd
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import LineString, Point
from shapely.ops import substring
from clawpack.geoclaw import topotools
from clawpack.pyclaw.solution import Solution

CASE = Path(__file__).resolve().parent
PROJECT = CASE.parents[2]
ENTRY, PROXY = (334051.8, 3143392.5), (340571.8257777202, 3129640.775894154)
ROUTE_DISTANCE_M, CORRIDOR_TOLERANCE_M, WET = 17624.0, 400.0, 0.01
FLOW = np.array([0.4284, -0.9036]); NORMAL = np.array([-FLOW[1], FLOW[0]])


def data_root() -> Path:
    roots = [Path(os.environ["JILONG_DATA_ROOT"]) if os.environ.get("JILONG_DATA_ROOT") else None,
             PROJECT / "data/extracted", PROJECT.parent / "Jilong-DebrisFlow/data/extracted"]
    return next(p for p in roots if p and p.exists())


def mapped_route() -> LineString:
    river = next((data_root() / "basic_geographic_data").rglob("四级河流.shp"))
    multi = gpd.read_file(river).to_crs(32645).geometry.iloc[0]
    a, b = multi.geoms
    return LineString(list(substring(a, a.project(Point(ENTRY)), a.length).coords) + list(b.coords))


def interp(a, x, y, xp, yp):
    i=int(np.clip(np.searchsorted(x,xp)-1,0,len(x)-2)); j=int(np.clip(np.searchsorted(y,yp)-1,0,len(y)-2))
    tx,ty=(xp-x[i])/(x[i+1]-x[i]),(yp-y[j])/(y[j+1]-y[j])
    return float((1-tx)*(1-ty)*a[i,j]+tx*(1-ty)*a[i+1,j]+(1-tx)*ty*a[i,j+1]+tx*ty*a[i+1,j+1])


def terrain_triplet(s_m: float, line: LineString, scale: float) -> dict:
    topo=topotools.Topography(); topo.read(CASE/"jilong_dem_12p5m_64m.tt3",topo_type=3)
    z=RegularGridInterpolator((topo.y,topo.x),topo.Z,bounds_error=False,fill_value=np.nan)
    out={}
    for label, offset in (("before",-128.0),("at",0.0),("after",128.0)):
        p=line.interpolate(np.clip((s_m+offset)/scale,0,line.length)); out[f"terrain_{label}_m"]=float(z([[p.y,p.x]])[0])
    return out


def postprocess(run_dir: Path, spec: dict, runtime_s: float) -> dict:
    out=run_dir/"_output"; frames=sorted(int(p.name[-4:]) for p in out.glob("fort.q????"))
    if not frames: return dict(spec,solver_completed=False,runtime_s=runtime_s)
    line=mapped_route(); scale=ROUTE_DISTANCE_M/line.length; rows=[]; bulk=[]
    for n in frames:
        sol=Solution(n,path=out,file_format="ascii"); q=sol.state.q; x,y=sol.state.grid.dimensions[0].centers,sol.state.grid.dimensions[1].centers
        h,hu,hv=q[:3]; wet=h>WET; speed=np.zeros_like(h); speed[wet]=np.hypot(hu[wet]/h[wet],hv[wet]/h[wet])
        front=0.0
        for i,j in zip(*np.where(wet)):
            p=Point(float(x[i]),float(y[j])); d=line.distance(p)
            if d<=CORRIDOR_TOLERANCE_M: front=max(front,line.project(p)*scale)
        hp,hup,hvp=interp(h,x,y,*PROXY),interp(hu,x,y,*PROXY),interp(hv,x,y,*PROXY)
        ps=float(np.hypot(hup,hvp)/hp) if hp>WET else 0.0; qp=0.0
        for off in np.arange(-192,192,64):
            px,py=np.array(PROXY)+off*NORMAL; hs=interp(h,x,y,px,py)
            if hs>WET:
                us,vs=interp(hu,x,y,px,py)/hs,interp(hv,x,y,px,py)/hs
                qp+=max(0.0,hs*(us*FLOW[0]+vs*FLOW[1]))*64.0
        bulk.extend(speed[h>.1].tolist())
        rows.append(dict(time_s=float(sol.t),front_distance_m=front,proxy_depth_m=hp,proxy_speed_ms=ps,proxy_discharge_m3s=qp,
                         wet_area_m2=float(wet.sum()*(x[1]-x[0])*(y[1]-y[0])),raw_max_speed_ms=float(speed[wet].max()) if wet.any() else 0.,
                         max_speed_h_gt_01_ms=float(speed[h>.1].max()) if (h>.1).any() else 0.,max_speed_h_gt_1_ms=float(speed[h>1].max()) if (h>1).any() else 0.,finite=bool(np.isfinite(q).all())))
    q0=Solution(frames[0],path=out,file_format="ascii").state.q; x0=Solution(frames[0],path=out,file_format="ascii").state.grid.dimensions[0].centers; y0=Solution(frames[0],path=out,file_format="ascii").state.grid.dimensions[1].centers
    log=(run_dir/"run.log").read_text(errors="replace"); cfl=[float(v.replace("D","E")) for v in re.findall(r"maximum Courant number seen\s*=\s*([0-9.E+\-D]+)",log)]
    arrival=next((r["time_s"] for r in rows if r["proxy_depth_m"]>WET),None)
    at=lambda t:next((r["front_distance_m"] for r in rows if abs(r["time_s"]-t)<1e-6),None)
    result=dict(spec,solver_completed=True,finite_fields=all(r["finite"] for r in rows),maximum_CFL=max(cfl) if cfl else None,runtime_s=runtime_s,
                actual_discrete_initial_volume_m3=float(q0[0].sum()*(x0[1]-x0[0])*(y0[1]-y0[0])),maximum_front_distance_m=max(r["front_distance_m"] for r in rows),final_front_distance_m=rows[-1]["front_distance_m"],
                front_60_s_m=at(60),front_120_s_m=at(120),front_240_s_m=at(240),front_420_s_m=at(420),front_600_s_m=at(600),proxy_reached=arrival is not None,arrival_time_s=arrival,
                proxy_peak_discharge_m3s=max(r["proxy_discharge_m3s"] for r in rows),proxy_peak_depth_m=max(r["proxy_depth_m"] for r in rows),proxy_peak_speed_ms=max(r["proxy_speed_ms"] for r in rows),
                maximum_wet_area_m2=max(r["wet_area_m2"] for r in rows),raw_max_speed_ms=max(r["raw_max_speed_ms"] for r in rows),max_speed_h_gt_01_ms=max(r["max_speed_h_gt_01_ms"] for r in rows),max_speed_h_gt_1_ms=max(r["max_speed_h_gt_1_ms"] for r in rows),p99_speed_h_gt_01_ms=float(np.percentile(bulk,99)) if bulk else 0.,time_series=rows)
    result.update(terrain_triplet(result["final_front_distance_m"],line,scale)); (run_dir/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    return result
