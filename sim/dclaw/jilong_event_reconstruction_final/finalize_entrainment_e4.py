from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
CASE=Path(__file__).resolve().parent
RES=CASE/"results"; REP=CASE/"reports"; FIG=CASE/"figures"
TIMES=[90,180,300,420,600,900]
def rows(p):
    with p.open() as f:return {int(float(r["time_s"])):r for r in csv.DictReader(f)}
def asf(r,k):return float(r[k])
def main():
    e2,e4=rows(RES/"E2_mass_ledger.csv"),rows(RES/"E4_mass_ledger.csv")
    needed=set(TIMES)
    if not needed.issubset(e2) or not needed.issubset(e4):raise RuntimeError("checkpoint mismatch")
    vals=np.array([asf(e4[t],"entrained_volume_m3") for t in sorted(e4)])
    monotonic=bool(np.all(np.diff(vals)>=-1.e-6))
    comp=[]
    for t in TIMES:
        a,b=e2[t],e4[t]
        comp.append(dict(time_s=t, E2_front_m=asf(a,"unrestricted_max_projected_chainage_m"), E4_front_m=asf(b,"unrestricted_max_projected_chainage_m"), front_gain_E4_minus_E2_m=asf(b,"unrestricted_max_projected_chainage_m")-asf(a,"unrestricted_max_projected_chainage_m"), E2_max_depth_m=asf(a,"max_depth_m"), E4_max_depth_m=asf(b,"max_depth_m"), E2_max_speed_ms=asf(a,"max_speed_ms"), E4_max_speed_ms=asf(b,"max_speed_ms"), E2_bdif_integral_m3=asf(a,"entrained_volume_m3"), E4_bdif_integral_m3=asf(b,"entrained_volume_m3")))
    with (RES/"E2_E4_comparison.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(comp[0]));w.writeheader();w.writerows(comp)
    result=json.loads((REP/"ENTRAINMENT_E4_RESULT.json").read_text())
    arrivals=result["arrivals_s"]; sane=all(result["numerical_gate"].values()) and result["completed_to_s"]==900.
    s1,s4=arrivals["S1"]["0.1"],arrivals["S4"]["0.1"]
    if not sane:classification="E4_NUMERICAL_PATHOLOGY";recommend="STOP_DCLAW_PARAMETER_SEARCH_AND_USE_LAST_SANE_CASES";gate="FAIL"
    elif s4 is not None and s4<=900:classification="E4_FULL_DOWNSTREAM_RECONSTRUCTION_CANDIDATE";recommend="FREEZE_DCLAW_NOMINAL_AND_START_FNO";gate="PASS"
    elif s1 is not None and s1<=900:classification="E4_PARTIAL_DOWNSTREAM_MOBILITY";recommend="FREEZE_EVENT_CONSTRAINED_SCENARIO_BRACKET_AND_START_FNO";gate="PASS"
    else:classification="E4_ENTRAINMENT_INSUFFICIENT";recommend="STOP_DCLAW_CALIBRATION_AND_START_FNO_WITH_SCENARIO_FRAMING";gate="FAIL"
    gain=comp[-1]["front_gain_E4_minus_E2_m"];pct=100*gain/comp[-1]["E2_front_m"]
    result.update({"scenario":"C3_entrainment_E4","E4_GATE":gate,"classification":classification,"recommended_next_action":recommend,"bdif_volume_monotonic":monotonic,"bdif_volume_label":"cumulative entrained volume" if monotonic else "instantaneous integrated erosion-state volume","front_gain_900_m":gain,"front_gain_900_percent_relative_to_E2":pct,"E2_mask_reused":True,"E4_h_e_m":4.,"implementation_repair_reruns_used":0,"comparison_rows":comp})
    (REP/"ENTRAINMENT_E4_RESULT.json").write_text(json.dumps(result,indent=2)+"\n")
    labeltxt=result["bdif_volume_label"]
    lines=["# E4 entrainment result","","E4 reached 900 s in its first production attempt. E2 mask membership was reused exactly; only h_e is 4 m.","",f"- Numerical gate: finite={result['numerical_gate']['finite']}, max depth={result['max_depth_m']:.6g} m, max speed={result['max_speed_ms']:.6g} m/s, P99 wet speed={result['p99_wet_speed_ms']:.6g} m/s.",f"- BDIF_VOLUME_MONOTONIC: **{'YES' if monotonic else 'NO'}**. The bdif integral is labelled **{labeltxt}**.",f"- E4 gate: **{gate}**; classification: **{classification}**.",f"- 900 s front gain versus E2: {gain:.6g} m ({pct:.6g}%).","","| t (s) | front chainage (m) | max depth (m) | max speed (m/s) | P99 wet speed (m/s) | bdif integral (m3) | wet volume (m3) |","|---:|---:|---:|---:|---:|---:|---:|"]
    for t in TIMES:
        r=e4[t];lines.append(f"| {t} | {asf(r,'unrestricted_max_projected_chainage_m'):.6f} | {asf(r,'max_depth_m'):.6g} | {asf(r,'max_speed_ms'):.6g} | {asf(r,'p99_wet_speed_ms'):.6g} | {asf(r,'entrained_volume_m3'):.6g} | {asf(r,'total_wet_volume_m3'):.6g} |")
    lines+=["","## h > 0.10 m arrivals","","| section | earliest arrival (s) |","|---|---:|"]+[f"| {s} | {'' if arrivals[s]['0.1'] is None else arrivals[s]['0.1']} |" for s in ("S1","S2","S3","S4")]
    (REP/"ENTRAINMENT_E4_RESULT.md").write_text("\n".join(lines)+"\n")
    te2=np.array(TIMES);f2=np.array([comp[i]["E2_front_m"] for i in range(len(TIMES))]);f4=np.array([comp[i]["E4_front_m"] for i in range(len(TIMES))])
    plt.figure(figsize=(7,4));plt.plot(te2,f2/1000,"o-",label="E2 h_e=2m");plt.plot(te2,f4/1000,"o-",label="E4 h_e=4m");plt.xlabel("Time (s)");plt.ylabel("Front chainage (km)");plt.grid();plt.legend();plt.tight_layout();plt.savefig(FIG/"E2_E4_front_comparison.png",dpi=180);plt.close()
    alltime=np.array(sorted(e4));allbd=np.array([asf(e4[t],"entrained_volume_m3") for t in alltime])
    plt.figure(figsize=(7,4));plt.plot(alltime,allbd/1e6,"o-");plt.xlabel("Time (s)");plt.ylabel("BDIF integrated state (10^6 m3)");plt.grid();plt.tight_layout();plt.savefig(FIG/"E4_erosion_state_timeseries.png",dpi=180);plt.close()
    print(json.dumps({"classification":classification,"gate":gate,"monotonic":monotonic,"gain":gain,"pct":pct},indent=2))
if __name__=="__main__":main()
