"""Run exactly the conditional reference-based Jilong scenario plan."""
from __future__ import annotations
import csv, json, os, shutil, subprocess, sys, time
from pathlib import Path
import matplotlib.pyplot as plt

CASE = Path(__file__).resolve().parent
PROJECT = CASE.parents[2]


def invoke(cmd, env, log):
    return subprocess.run(cmd, cwd=CASE, env=env, stdout=log, stderr=subprocess.STDOUT).returncode


def execute(spec: dict, executable_ready: bool) -> dict:
    run_dir = CASE / "runs" / spec["scenario_id"]
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    env = dict(os.environ, DCLAW_KREF=str(spec["kref"]), DCLAW_PINIT_PRATIO=str(spec["p_ratio"]))
    began = time.monotonic()
    with (run_dir / "run.log").open("w") as log:
        code = invoke([sys.executable, "source_inputs.py", "--volume-m3", str(spec["source_volume_m3"])], env, log)
        if code == 0:
            code = invoke([sys.executable, "setrun.py"], env, log)
        if code == 0:
            shutil.rmtree(CASE / "_output", ignore_errors=True)
            (CASE / ".output").unlink(missing_ok=True)
            code = invoke(["make", ".output"], env, log)
    elapsed = time.monotonic() - began
    if code != 0:
        return dict(spec, solver_completed=False, runtime_s=elapsed, solver_exit_code=code)
    shutil.move(str(CASE / "_output"), str(run_dir / "_output"))
    (run_dir / "scenario.json").write_text(json.dumps(spec, indent=2)+"\n")
    code = invoke([sys.executable, "postprocess.py", "--output", str(run_dir / "_output"), "--runtime-s", str(elapsed), "--scenario-json", str(run_dir / "scenario.json")], env, (run_dir / "postprocess.log").open("w"))
    if code != 0:
        return dict(spec, solver_completed=False, runtime_s=elapsed, solver_exit_code=code)
    return json.loads((run_dir / "summary.json").read_text())


def compact_csv(results: list[dict]) -> None:
    fields = ["scenario_id", "source_scale_or_volume", "m0", "mcrit", "kref", "pore_pressure_setting", "entrainment_setting", "runtime_s", "max_front_distance_km", "arrival_time_s", "max_depth_m", "bulk_max_speed_ms", "p99_speed_ms", "affected_area_m2"]
    rows=[]
    for r in results:
        rows.append({"scenario_id":r["scenario_id"], "source_scale_or_volume":r["source_volume_m3"], "m0":r["m0"], "mcrit":r["mcrit"], "kref":r["kref"], "pore_pressure_setting":r["pore_pressure_setting"], "entrainment_setting":r["entrainment_setting"], "runtime_s":r.get("runtime_s"), "max_front_distance_km":r.get("maximum_front_distance_m",0)/1000, "arrival_time_s":r.get("arrival_time_s"), "max_depth_m":r.get("proxy_peak_depth_m"), "bulk_max_speed_ms":r.get("max_speed_h_gt_01_ms"), "p99_speed_ms":r.get("p99_speed_h_gt_01_ms"), "affected_area_m2":r.get("maximum_wet_area_m2")})
    with (CASE / "SCENARIO_RUNS.csv").open("w", newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(rows)


def figure(results: list[dict]) -> None:
    out = PROJECT / "outputs/reference_based"; out.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8,4.5))
    for r in results:
        ts=r.get("time_series",[])
        if ts: plt.plot([v["time_s"] for v in ts],[v["front_distance_m"]/1000 for v in ts],label=r["scenario_id"])
    plt.xlabel("Model time (s)"); plt.ylabel("Mapped-corridor front distance from entry (km)")
    plt.grid(alpha=.3); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(out / "scenario_summary.png",dpi=180); plt.close()


def documents(results: list[dict], mode: str, total_wall_s: float) -> None:
    baseline=next(r for r in results if r["scenario_id"]=="B0")
    ok=[r for r in results if r.get("solver_completed") and r.get("finite_fields")]
    best=max(ok,key=lambda r:r["maximum_front_distance_m"]); worst=min(ok,key=lambda r:r["maximum_front_distance_m"])
    family = mode == "matrix" and len(ok) >= 3
    accepted = family and best["maximum_front_distance_m"] >= 3000 and (best["maximum_front_distance_m"]-worst["maximum_front_distance_m"]) >= 500
    b = f"""# Reference-based baseline results

| Metric | Value |
|---|---:|
| Solver completed / finite fields | {baseline.get('solver_completed')} / {baseline.get('finite_fields')} |
| Maximum CFL | {baseline.get('maximum_CFL')} |
| Raw maximum speed | {baseline.get('raw_max_speed_ms',0):.2f} m/s |
| Maximum speed, h > 0.1 m | {baseline.get('max_speed_h_gt_01_ms',0):.2f} m/s |
| Maximum speed, h > 1.0 m | {baseline.get('max_speed_h_gt_1_ms',0):.2f} m/s |
| P99 speed, h > 0.1 m | {baseline.get('p99_speed_h_gt_01_ms',0):.2f} m/s |
| Maximum / final mapped front | {baseline.get('maximum_front_distance_m',0)/1000:.3f} / {baseline.get('final_front_distance_m',0)/1000:.3f} km |
| Proxy reached / arrival | {baseline.get('proxy_reached')} / {baseline.get('arrival_time_s')} s |
| Proxy Q / depth / speed peak | {baseline.get('proxy_peak_discharge_m3s',0):.2f} m3/s / {baseline.get('proxy_peak_depth_m',0):.3f} m / {baseline.get('proxy_peak_speed_ms',0):.2f} m/s |
| Maximum wet area | {baseline.get('maximum_wet_area_m2',0):.0f} m2 |
| Wall runtime | {baseline.get('runtime_s',0):.1f} s |

The baseline is the 1 Mm3, stationary, hydrostatic qinit mass with no imposed velocity or boundary inflow. It is an engineering source-volume scenario transferred from the smallest published Mount Baker source magnitude, not a reconstruction of unknown Jilong main-river-entry conditions.
"""
    (CASE / "BASELINE_RESULTS.md").write_text(b)
    arrivals=[r["arrival_time_s"] for r in ok if r.get("arrival_time_s") is not None]
    status=f"""# Reference-based Jilong D-Claw status

1. **Primary application:** USGS Mount Baker long-runout lahar scenarios (Gardner et al., 2025; SIR 2024-5133).
2. **Adopted components:** native DEM-to-D-Claw terrain workflow; stationary initial-mass source pattern; m0=0.62, mcrit=0.64, rho_f=1100 kg/m3, rho_s=2700 kg/m3, mu=0.005 Pa s, phi=38 degrees; the published k range.
3. **Old mismatch confirmed?** Yes: the prior pipeline mixed raster row ordering. This case does not reuse it.
4. **Correction:** source rows are north-to-south during raster reprojection, explicitly reversed into GeoClaw increasing-y storage, then TT3-read-back checked at fixed and 100 random points.
5. **Baseline:** 1 Mm3 stationary qinit source, no imposed velocity, kref=1e-11 m2, hydrostatic pressure, no unconstrained entrainment field/rate.
6. **Baseline maximum propagation:** {baseline.get('maximum_front_distance_m',0)/1000:.3f} km.
7. **Downstream proxy reached:** {baseline.get('proxy_reached')}.
8. **Baseline proxy arrival:** {baseline.get('arrival_time_s')} s.
9. **Bulk diagnostic (h>0.1 m):** max {baseline.get('max_speed_h_gt_01_ms',0):.2f} m/s; P99 {baseline.get('p99_speed_h_gt_01_ms',0):.2f} m/s.
10. **Scenario family generated:** {family} ({len(ok)} completed runs; branch `{mode}`).
11. **Range:** front {worst.get('maximum_front_distance_m',0)/1000:.3f}--{best.get('maximum_front_distance_m',0)/1000:.3f} km; arrivals {arrivals if arrivals else 'none'} s; proxy maximum depths {min(r.get('proxy_peak_depth_m',0) for r in ok):.3f}--{max(r.get('proxy_peak_depth_m',0) for r in ok):.3f} m.
12. **Structural comparison:** terrain passed its TT3 read-back check; the baseline matches the primary-reference `m0/mcrit` relation and central kref. The permitted low-kref and 1.5x-hydrostatic corrections test the only documented alternatives without a sweep.
13. **Single remaining blocker:** {'none' if accepted else 'the unobserved Jilong main-river-entry source geometry/saturation; the stationary 1-Mm3 qinit is a transparent placeholder, unlike Mount Baker’s mapped slide geometry.'}
14. **Sufficient for engineering scenario generation:** {'YES' if accepted else 'NO'}. Recommended next step: {'scenario dataset generation' if accepted else 'one source-condition formulation correction before any broader scenario matrix.'}

Total planned-batch wall time: {total_wall_s:.1f} s.
"""
    (PROJECT / "docs/REFERENCE_BASED_JILONG_STATUS.md").write_text(status)
    (CASE / "batch_metadata.json").write_text(json.dumps({"branch":mode,"total_wall_s":total_wall_s,"accepted":accepted,"results":results},indent=2)+"\n")


def main():
    began=time.monotonic(); env=dict(os.environ)
    # Terrain is constructed and verified before the solver executable is built/run.
    subprocess.run([sys.executable,"terrain_preprocess.py"],cwd=CASE,env=env,check=True)
    terrain=json.loads((CASE/"terrain_check.json").read_text())
    if not terrain["passed"]: raise RuntimeError("TERRAIN CHECK FAILED: solver hard-stop")
    subprocess.run(["make"],cwd=CASE,env=env,check=True)
    base=dict(scenario_id="B0",source_volume_m3=1e6,m0=.62,mcrit=.64,kref=1e-11,p_ratio=1.0,pore_pressure_setting="hydrostatic",entrainment_setting="none (no public erodible-thickness constraint)")
    results=[execute(base,True)]
    b=results[0]
    if b.get("solver_completed") and b.get("finite_fields") and b.get("maximum_front_distance_m",0)>2000:
        mode="matrix"
        plan=[("K_LOW",1e6,1e-12,1.0,"hydrostatic"),("K_HIGH",1e6,1e-10,1.0,"hydrostatic"),("V10_KLOW",1e7,1e-12,1.0,"hydrostatic"),("V10_KMID",1e7,1e-11,1.0,"hydrostatic"),("V10_KHIGH",1e7,1e-10,1.0,"hydrostatic")]
    else:
        mode="targeted_correction"
        # At most two corrections, selected from documented D-Claw practice; no sweep.
        plan=[("C_K_LOW",1e6,1e-12,1.0,"hydrostatic"),("C_P15",1e6,1e-11,1.5,"1.5 x hydrostatic (official gully example)")]
    for sid,vol,k,p,ps in plan:
        results.append(execute(dict(scenario_id=sid,source_volume_m3=vol,m0=.62,mcrit=.64,kref=k,p_ratio=p,pore_pressure_setting=ps,entrainment_setting="none (no public erodible-thickness constraint)"),True))
    compact_csv(results); figure(results); documents(results,mode,time.monotonic()-began)


if __name__ == "__main__": main()
