"""Summarize frozen conservative-source full cases; raw outputs stay server-local."""
from __future__ import annotations
import csv, json
from pathlib import Path
import matplotlib.pyplot as plt

CASE=Path(__file__).resolve().parent
CASES=["C1","C2","C3","C4","C5","C6"]
TARGET={"C1":1e6,"C2":1e6,"C3":2e6,"C4":2e6,"C5":3e6,"C6":3e6}
VT={"C1":(1e6,90),"C2":(1e6,180),"C3":(2e6,90),"C4":(2e6,180),"C5":(3e6,90),"C6":(3e6,180)}

def main():
    (CASE/"results").mkdir(exist_ok=True); (CASE/"figures").mkdir(exist_ok=True)
    data={n:json.loads((CASE/f"runs/{n}_conservative_full/metrics.json").read_text()) for n in CASES}
    rows=[]; arrivals=[]; peaks=[]
    for n,x in data.items():
        final=x["frames"][-1]; volume=final["wet_volume_m3"]; start=int((CASE/f"runs/{n}_conservative_full/start_epoch").read_text()); end=int((CASE/f"runs/{n}_conservative_full/end_epoch").read_text())
        rows.append({"case":n,"runtime_s":end-start,"target_volume_m3":TARGET[n],"final_model_volume_m3":volume,"relative_injected_volume_error":(volume-TARGET[n])/TARGET[n],"max_chainage_m":x["max_chainage_m"],"s4_reached":x["section_arrivals_s"]["S4"]["0.1"] is not None,"max_depth_m":x["max_depth_m"],"max_speed_ms":x["max_speed_ms"],"p99_wet_speed_ms":x["p99_wet_speed_ms"]})
        for s,a in x["section_arrivals_s"].items(): arrivals.append({"case":n,"section":s,**{f"arrival_h_gt_{k}_s":v for k,v in a.items()}})
        for s,p in x["section_peaks"].items(): peaks.append({"case":n,"section":s,**p})
    def write(path,fields,items):
        with path.open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(items)
    write(CASE/"results/source_only_64m_summary.csv",list(rows[0]),rows); write(CASE/"results/source_mass_budget.csv",["case","target_volume_m3","final_model_volume_m3","relative_injected_volume_error"],[{k:r[k] for k in ["case","target_volume_m3","final_model_volume_m3","relative_injected_volume_error"]} for r in rows]); write(CASE/"results/section_arrivals.csv",list(arrivals[0]),arrivals); write(CASE/"results/section_peaks.csv",list(peaks[0]),peaks)
    summary={"architecture":"CONSERVATIVE_FIXED_S0_SOURCE_ZONE","conservative_source_gate":"PASS","cases":rows,"section_arrivals":arrivals,"section_peaks":peaks,"classification":"SOURCE_ONLY_INSUFFICIENT_FOR_HISTORICAL_RECONSTRUCTION" if not any(r["s4_reached"] for r in rows) else "S4_REACHED_BY_AT_LEAST_ONE_CASE"}
    (CASE/"reports/SOURCE_ONLY_64M_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
    lines=["# Source-only 64 m reconstruction summary","",f"Classification: **{summary['classification']}**.","","| case | runtime s | final mass error | max chainage m | S4 h>0.10 | max depth m | max speed m/s | P99 wet speed m/s |","|---|---:|---:|---:|---|---:|---:|---:|"]
    for r in rows: lines.append(f"| {r['case']} | {r['runtime_s']} | {r['relative_injected_volume_error']:.3e} | {r['max_chainage_m']:.1f} | {r['s4_reached']} | {r['max_depth_m']:.3f} | {r['max_speed_ms']:.3f} | {r['p99_wet_speed_ms']:.3f} |")
    lines += ["", "No retuning was performed after the C3 source gate passed. `S4 reached` uses the required h>0.10 m arrival criterion."]
    (CASE/"reports/SOURCE_ONLY_64M_SUMMARY.md").write_text("\n".join(lines)+"\n")
    names=CASES; plt.figure();
    for n in names:
        v,t=VT[n]; import numpy as np; time=np.linspace(0,t,200); q=np.pi*v/(2*t)*np.sin(np.pi*time/t); plt.plot(time,q,label=n)
    plt.xlabel("time (s)");plt.ylabel("Q (m³/s)");plt.legend();plt.tight_layout();plt.savefig(CASE/"figures/source_hydrographs.png",dpi=160);plt.close()
    plt.figure();plt.bar(names,[r["relative_injected_volume_error"]*100 for r in rows]);plt.ylabel("final volume error (%)");plt.tight_layout();plt.savefig(CASE/"figures/source_mass_budget.png",dpi=160);plt.close()
    plt.figure();plt.bar(names,[r["max_chainage_m"]/1000 for r in rows]);plt.ylabel("max chainage (km)");plt.tight_layout();plt.savefig(CASE/"figures/max_chainage_by_case.png",dpi=160);plt.close()
    plt.figure();
    for th in ("0.05","0.1","0.2"): plt.plot(names,[next(a[f"arrival_h_gt_{th}_s"] for a in arrivals if a["case"]==n and a["section"]=="S4") or float("nan") for n in names],marker="o",label=f"S4 h>{th}")
    plt.ylabel("arrival time (s)");plt.legend();plt.tight_layout();plt.savefig(CASE/"figures/arrival_time_by_case.png",dpi=160);plt.close()
if __name__=="__main__": main()
