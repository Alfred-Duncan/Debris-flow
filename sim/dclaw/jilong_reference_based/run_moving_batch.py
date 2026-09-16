"""One fixed six-case moving-entry batch; no adaptive scientific cases."""
from __future__ import annotations
import csv, json, os, shutil, subprocess, sys, time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from clawpack.pyclaw.solution import Solution
from postprocess_moving import postprocess

CASE=Path(__file__).resolve().parent; PROJECT=CASE.parents[2]
SPECS=[("M1",1e6,8.),("M2",1e6,15.),("M3",1e6,25.),("M4",3e6,8.),("M5",3e6,15.),("M6",3e6,25.)]


def call(cmd,env,log): return subprocess.run(cmd,cwd=CASE,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
def clear_output(): shutil.rmtree(CASE/"_output",ignore_errors=True); (CASE/".output").unlink(missing_ok=True); (CASE/".data").unlink(missing_ok=True)
def spec(sid,vol,speed): return dict(scenario_id=sid,source_volume_m3=vol,prescribed_initial_speed_ms=speed,m0=.62,mcrit=.64,kref=1e-11,pore_pressure_setting="hydrostatic",entrainment_setting="none")


def native_gate(item,base):
    """Zero-time D-Claw initialization only; not one of the six 600-s runs."""
    rd=CASE/"moving_runs"/item["scenario_id"]; rd.mkdir(parents=True,exist_ok=True); env=dict(base,DCLAW_TFINAL="0",DCLAW_NUM_OUTPUT="1")
    with (rd/"initcheck.log").open("w") as log:
        code=call([sys.executable,"source_inputs.py","--volume-m3",str(item["source_volume_m3"]),"--speed-ms",str(item["prescribed_initial_speed_ms"])],env,log)
        if code==0: clear_output(); code=call(["make",".output"],env,log)
    if code: raise RuntimeError(f"native qinit gate failed for {item['scenario_id']}")
    check=rd/"_initcheck"; shutil.rmtree(check,ignore_errors=True); shutil.move(str(CASE/"_output"),str(check))
    sol=Solution(0,path=check,file_format="ascii"); q=sol.state.q; h,hu,hv=q[:3]; mask=h>.1*h.max(); speed=np.hypot(hu[mask]/h[mask],hv[mask]/h[mask])
    mean=float(speed.mean()) if speed.size else 0.; expected=item["prescribed_initial_speed_ms"]; passed=bool(speed.size and abs(mean-expected)<=.02*expected)
    record=dict(scenario_id=item["scenario_id"],requested_speed_ms=expected,mean_t0_speed_ms=mean,n_substantial_cells=int(mask.sum()),passed=passed)
    (rd/"initcheck.json").write_text(json.dumps(record,indent=2)+"\n")
    if not passed: raise RuntimeError(f"native qinit momentum sanity failed for {item['scenario_id']}: {mean} vs {expected}")
    return record


def execute(item,base):
    rd=CASE/"moving_runs"/item["scenario_id"]; shutil.rmtree(rd/"_output",ignore_errors=True); began=time.monotonic()
    with (rd/"run.log").open("w") as log:
        code=call([sys.executable,"source_inputs.py","--volume-m3",str(item["source_volume_m3"]),"--speed-ms",str(item["prescribed_initial_speed_ms"])],base,log)
        if code==0: clear_output(); code=call(["make",".output"],base,log)
    runtime=time.monotonic()-began
    if code: return dict(item,solver_completed=False,finite_fields=False,runtime_s=runtime)
    shutil.move(str(CASE/"_output"),str(rd/"_output")); (rd/"scenario.json").write_text(json.dumps(item,indent=2)+"\n")
    return postprocess(rd,item,runtime)


def write_outputs(results,checks,wall):
    fields=["scenario_id","source_volume_m3","prescribed_initial_speed_ms","solver_completed","finite_fields","maximum_CFL","actual_discrete_initial_volume_m3","maximum_front_distance_m","final_front_distance_m","front_60_s_m","front_120_s_m","front_240_s_m","front_420_s_m","front_600_s_m","proxy_reached","arrival_time_s","proxy_peak_discharge_m3s","proxy_peak_depth_m","proxy_peak_speed_ms","maximum_wet_area_m2","raw_max_speed_ms","max_speed_h_gt_01_ms","max_speed_h_gt_1_ms","p99_speed_h_gt_01_ms","runtime_s"]
    with (CASE/"MOVING_SOURCE_RUNS.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows([{k:r.get(k) for k in fields} for r in results])
    out=PROJECT/"outputs/reference_based"; out.mkdir(parents=True,exist_ok=True); plt.figure(figsize=(8,4.5))
    for r in results:
        ts=r.get("time_series",[]); plt.plot([x["time_s"] for x in ts],[x["front_distance_m"]/1000 for x in ts],label=r["scenario_id"])
    plt.xlabel("Model time (s)"); plt.ylabel("Mapped downstream front distance (km)"); plt.grid(alpha=.3); plt.legend(ncol=3,fontsize=8); plt.tight_layout(); plt.savefig(out/"moving_source_fronts.png",dpi=180); plt.close()
    good=[r for r in results if r.get("solver_completed") and r.get("finite_fields")]; best=max(good,key=lambda r:r["maximum_front_distance_m"]); distances=[r["maximum_front_distance_m"] for r in good]; arrivals=[r["arrival_time_s"] for r in good if r["arrival_time_s"] is not None]
    varied=(max(distances)-min(distances)>=500) or len(set(arrivals))>=2; unblocked=bool(best["maximum_front_distance_m"]>=5000 or best["proxy_reached"]) and varied
    all_under_3=all(r["maximum_front_distance_m"]<3000 for r in good)
    rows="\n".join(f"| {r['scenario_id']} | {r['source_volume_m3']:.0f} | {r['prescribed_initial_speed_ms']:.0f} | {r['maximum_front_distance_m']/1000:.3f} / {r['final_front_distance_m']/1000:.3f} | {r['arrival_time_s']} | {r['max_speed_h_gt_01_ms']:.2f} / {r['max_speed_h_gt_1_ms']:.2f} / {r['p99_speed_h_gt_01_ms']:.2f} |" for r in results)
    diagnostic=""
    if all_under_3:
        diagnostic="\n## Required <3-km diagnostic\n\nAll six moving-entry cases remained below 3 km. The native t=0 gates passed: " + ", ".join(f"{c['scenario_id']}={c['mean_t0_speed_ms']:.2f} m/s" for c in checks) + ".\n\n| Case | final s-distance (km) | terrain 128 m before / at / after final front (m) |\n|---|---:|---:|\n" + "\n".join(f"| {r['scenario_id']} | {r['final_front_distance_m']/1000:.3f} | {r['terrain_before_m']:.1f} / {r['terrain_at_m']:.1f} / {r['terrain_after_m']:.1f} |" for r in good) + "\n\nThe common stop location despite verified 8--25 m/s initial momentum indicates a geometric/corridor-source structural limitation, not missing entry momentum.\n"
    report=f"""# Moving-entry source results

## Native qinit verification

The installed q mapping is `h, hu, hv, hm, pb, hchi, Delta b` = q1--q7. qinit files for q2/q3/q4 supply `u/v/m` and the native implementation multiplies by h. Hydrostatic q5 is applied by unchanged `init_ptype=0`. All t=0 native-qinit gates passed: {', '.join(f"{c['scenario_id']} {c['mean_t0_speed_ms']:.2f}/{c['requested_speed_ms']:.0f} m/s" for c in checks)}.

## Six engineering entry-state scenarios

| Case | Volume (m3) | Prescribed speed (m/s) | max / final front (km) | proxy arrival (s) | max h>.1 / h>1 / p99 speed (m/s) |
|---|---:|---:|---:|---:|---:|
{rows}

Best propagation: **{best['maximum_front_distance_m']/1000:.3f} km** ({best['scenario_id']}). Port proxy reached: **{any(r['proxy_reached'] for r in good)}**. Arrival times: {arrivals if arrivals else 'none'}. Entry-state outcomes meaningfully differ: **{varied}**.\n\n**SIMULATOR_UNBLOCKED = {'YES' if unblocked else 'NO'}**.\n\nTotal wall time (including native-qinit gate): {wall:.1f} s.\n{diagnostic}"""
    (CASE/"MOVING_SOURCE_RESULTS.md").write_text(report)
    (CASE/"moving_batch_metadata.json").write_text(json.dumps(dict(wall_time_s=wall,initchecks=checks,unblocked=unblocked,results=results),indent=2)+"\n")


def main():
    if "**Result: PASS**" not in (CASE/"TERRAIN_CHECK.md").read_text(): raise RuntimeError("accepted terrain artifact is not available")
    if not (CASE/"jilong_dem_12p5m_64m.tt3").exists(): raise RuntimeError("accepted terrain TT3 missing; do not regenerate it in this task")
    began=time.monotonic(); base=dict(os.environ,DCLAW_KREF="1e-11",DCLAW_PINIT_PRATIO="1.0"); subprocess.run(["make"],cwd=CASE,env=base,check=True)
    shutil.rmtree(CASE/"moving_runs",ignore_errors=True); (CASE/"moving_runs").mkdir()
    items=[spec(*s) for s in SPECS]; checks=[native_gate(i,base) for i in items]
    results=[execute(i,base) for i in items]; write_outputs(results,checks,time.monotonic()-began)


if __name__=="__main__": main()
