from __future__ import annotations
import csv, json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from clawpack.pyclaw.solution import Solution
CASE=Path(__file__).resolve().parent
OUT=CASE/"runs/C3_entrainment_E4/_output"
DX=64.0; V=2.e6; T=90.0; WET=1.e-3
PROFILE=CASE.parents[2]/"local_validation/jilong_gate_a/terrain_final/jilong_glo30_valley_profile.csv"
SECTIONS={"S1":4881.187672129356,"S2":9762.375344258711,"S3":14643.563016388067,"S4":19524.750688517423}
CHECK={90.,180.,300.,420.,600.,900.}
def project_grid(x,y):
    raw=np.genfromtxt(PROFILE,delimiter=",",names=True,dtype=float,encoding="utf8"); s,px,py=raw["station_m"],raw["x"],raw["y"]
    xx,yy=np.meshgrid(x,y,indexing="ij"); best=np.full(xx.shape,np.inf); ch=np.zeros(xx.shape)
    for k in range(len(s)-1):
        vx,vy=px[k+1]-px[k],py[k+1]-py[k]; vv=vx*vx+vy*vy; u=np.clip(((xx-px[k])*vx+(yy-py[k])*vy)/vv,0,1)
        dx=xx-(px[k]+u*vx);dy=yy-(py[k]+u*vy);d=dx*dx+dy*dy;use=d<best;best[use]=d[use];ch[use]=(s[k]+u*(s[k+1]-s[k]))[use]
    return ch,np.sqrt(best)
def source_volume(t): return 0. if t<=0 else V if t>=T else .5*V*(1-math.cos(math.pi*t/T))
def main():
    fids=sorted(int(p.name[-4:]) for p in OUT.glob("fort.q????"))
    if len(fids)!=31: raise RuntimeError(f"expected 31 frames, got {len(fids)}")
    source=json.loads((CASE/"source_zone_v2.json").read_text());frames=[];arrivals={n:{str(h):None for h in (.001,.05,.10,.20)} for n in SECTIONS};chain=dist=smask=previous=None
    for fid in fids:
        sol=Solution(fid,path=OUT,file_format="ascii");q=sol.state.q;t=float(sol.t);h,hu,hv,bdif=q[0],q[1],q[2],q[6]
        if chain is None:
            x,y=sol.state.grid.dimensions[0].centers,sol.state.grid.dimensions[1].centers;chain,dist=project_grid(x,y);smask=np.zeros(h.shape,bool)
            for c in source["cells"]:smask[c["i"]-1,c["j"]-1]=True
        wet=h>WET;speed=np.zeros_like(h);speed[wet]=np.hypot(hu[wet]/h[wet],hv[wet]/h[wet]);labs,nlab=label(wet,structure=np.ones((3,3),int));labs_at_source=np.unique(labs[smask]);labs_at_source=labs_at_source[labs_at_source>0]
        if labs_at_source.size:comp=np.isin(labs,labs_at_source)
        elif previous is not None and np.any(wet & previous):
            active=np.unique(labs[wet & previous]);comp=np.isin(labs,active[active>0])
        else:comp=wet.copy()
        previous=comp
        for name,s0 in SECTIONS.items():
            band=(np.abs(chain-s0)<=DX)&(dist<=256.);maxh=float(h[band].max()) if np.any(band) else 0.
            for th in (.001,.05,.10,.20):
                if arrivals[name][str(th)] is None and maxh>th:arrivals[name][str(th)]=t
        bd=np.maximum(bdif,0.)
        frames.append({"frame":fid,"time_s":t,"finite":bool(np.isfinite(q).all()),"max_depth_m":float(h.max()),"max_speed_ms":float(speed[wet].max()) if wet.any() else 0.,"p99_wet_speed_ms":float(np.percentile(speed[wet],99)) if wet.any() else 0.,"wet_cell_count":int(wet.sum()),"total_wet_volume_m3":float(h.sum()*DX*DX),"entrained_volume_m3":float(bd.sum()*DX*DX),"max_bdif_m":float(bdif.max()),"unrestricted_max_projected_chainage_m":float(chain[comp].max()) if comp.any() else 0.,"source_ledger_expected_m3":source_volume(t)})
    finite=all(r["finite"] for r in frames);maxh=max(r["max_depth_m"] for r in frames);maxv=max(r["max_speed_ms"] for r in frames);p99=max(r["p99_wet_speed_ms"] for r in frames);maxbd=max(r["max_bdif_m"] for r in frames);grid={int(round(r["time_s"])):r for r in frames}
    if not CHECK.issubset(grid):raise RuntimeError("required times absent")
    sane=finite and maxh<100. and maxv<40.;s1=arrivals["S1"]["0.001"];gate="PASS" if sane and s1 is not None and s1<=900. else "FAIL";classification="E4_MEANINGFUL_DOWNSTREAM_MOBILITY" if gate=="PASS" else ("E4_ENTRAINMENT_INSUFFICIENT" if sane else "E4_NUMERICAL_PATHOLOGY")
    result={"scenario":"C3_entrainment_E4","completed_to_s":frames[-1]["time_s"],"implementation_repair_reruns_used":0,"source_support_cells":len(source["cells"]),"terrain_modified":False,"source_modified":False,"material_baseline_modified":False,"entrainment_rate":.20,"available_layer_thickness_m":2.,"h_e_aux_index_fortran":7,"numerical_gate":{"finite":finite,"max_depth_lt_100":maxh<100.,"max_speed_lt_40":maxv<40.},"max_depth_m":maxh,"max_speed_ms":maxv,"p99_wet_speed_ms":p99,"maximum_bdif_m":maxbd,"E4_GATE":gate,"classification":classification,"arrivals_s":arrivals,"checkpoint_metrics":{str(k):grid[k] for k in sorted(CHECK)},"frames":frames}
    res=CASE/"results";rep=CASE/"reports";fig=CASE/"figures";res.mkdir(exist_ok=True)
    cols=["frame","time_s","finite","max_depth_m","max_speed_ms","p99_wet_speed_ms","wet_cell_count","total_wet_volume_m3","entrained_volume_m3","max_bdif_m","unrestricted_max_projected_chainage_m","source_ledger_expected_m3"]
    with (res/"E4_mass_ledger.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(frames)
    with (res/"E4_front_trajectory.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=["time_s","unrestricted_max_projected_chainage_m","max_depth_m","max_speed_ms","p99_wet_speed_ms","entrained_volume_m3","total_wet_volume_m3"]);w.writeheader();w.writerows([{k:r[k] for k in w.fieldnames} for r in frames])
    with (res/"E4_section_arrivals.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["section","chainage_m","threshold_depth_m","earliest_arrival_s"])
        for name,s0 in SECTIONS.items():
            for th,val in arrivals[name].items():w.writerow([name,s0,th,"" if val is None else val])
    times=np.array([r["time_s"] for r in frames]);front=np.array([r["unrestricted_max_projected_chainage_m"] for r in frames]);ev=np.array([r["entrained_volume_m3"] for r in frames])
    plt.figure(figsize=(7,4));plt.plot(times,front/1000,"o-");plt.xlabel("Time (s)");plt.ylabel("Source-connected front chainage (km)");plt.grid();plt.tight_layout();plt.savefig(fig/"E4_front_trajectory.png",dpi=180);plt.close()
    plt.figure(figsize=(7,4));plt.plot(times,ev/1e6,"o-");plt.xlabel("Time (s)");plt.ylabel("Entrained volume from bdif (10^6 m3)");plt.grid();plt.tight_layout();plt.savefig(fig/"E4_entrained_volume.png",dpi=180);plt.close()
    (rep/"ENTRAINMENT_E4_RESULT.json").write_text(json.dumps(result,indent=2)+"\n")
    lines=["# E4 entrainment result","","E4 reached 900 s in its first formal attempt. bdif is D-Claw state component 7 (Fortran one-based); entrained volume is sum of non-negative bdif times 64 times 64 m2.","",f"- Numerical gate: finite={finite}, max depth={maxh:.6g} m, max speed={maxv:.6g} m/s, P99 wet speed={p99:.6g} m/s.",f"- Maximum bdif: {maxbd:.6g} m.",f"- E2 gate: **{gate}**; classification: **{classification}**.","","| t (s) | front chainage (m) | max depth (m) | max speed (m/s) | P99 wet speed (m/s) | entrained volume (m3) | wet volume (m3) |","|---:|---:|---:|---:|---:|---:|---:|"]
    for k in sorted(CHECK):
        r=grid[k];lines.append(f"| {k} | {r['unrestricted_max_projected_chainage_m']:.3f} | {r['max_depth_m']:.6g} | {r['max_speed_ms']:.6g} | {r['p99_wet_speed_ms']:.6g} | {r['entrained_volume_m3']:.6g} | {r['total_wet_volume_m3']:.6g} |")
    lines+=["","## h > 0.10 m arrivals","","| section | earliest arrival (s) |","|---|---:|"]+[f"| {n} | {'' if arrivals[n]['0.1'] is None else arrivals[n]['0.1']} |" for n in SECTIONS]
    (rep/"ENTRAINMENT_E4_RESULT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k:result[k] for k in ["completed_to_s","max_depth_m","max_speed_ms","p99_wet_speed_ms","maximum_bdif_m","E4_GATE","classification"]},indent=2))
if __name__=="__main__":main()
