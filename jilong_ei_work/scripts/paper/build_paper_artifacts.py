"""Generate publication tables, figures, manifests and release metadata from frozen CSV/JSON outputs."""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]; PAPER = ROOT / "paper_results"
FINAL_SHA = "146184d1fbbf366c0c9c40066a56753b398b10f4"; VAL_SHA = "19b63d10449cefdaeb0ebe0fdd7a84ed31797c4a"


def save(fig, name):
    for ext in ("png", "pdf"): fig.savefig(PAPER / "figures" / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def table(frame, name):
    frame.to_csv(PAPER / "tables" / f"{name}.csv", index=False)
    (PAPER / "tables" / f"{name}.tex").write_text(frame.to_latex(index=False, float_format=lambda x: f"{x:.4g}"), encoding="utf-8")


def existing(path):
    if not path.exists(): raise RuntimeError(f"required historical result missing: {path}")
    return pd.read_csv(path)


def main():
    for name in ("val", "budget_sensitivity", "ablations", "failure_cases", "tables", "figures", "release_metadata"): (PAPER / name).mkdir(parents=True, exist_ok=True)
    test = existing(PAPER / "test" / "test_summary.csv"); test_case = existing(PAPER / "test" / "test_case_metrics.csv")
    final_runs = ROOT / "results" / "engineering_roi_v2" / "runs"; risk = pd.concat([existing(final_runs / f"RiskTemporal_B{x}" / "final_method_summary.csv") for x in ("05", "10", "20")], ignore_index=True)
    risk.to_csv(PAPER / "val" / "risktemporal_budget_sweep.csv", index=False); risk.to_csv(PAPER / "budget_sensitivity" / "risktemporal_b05_b10_b20.csv", index=False)
    v1 = existing(ROOT / "results" / "engineering_roi_v1" / "final_method_summary.csv"); v2 = existing(ROOT / "results" / "engineering_roi_v2" / "final_method_summary.csv")
    v1.to_csv(PAPER / "val" / "engineering_roi_v1_summary.csv", index=False); v2.to_csv(PAPER / "val" / "engineering_roi_v2_summary.csv", index=False)
    ablation_names = ["SupportRisk_Base_B10", "SupportRisk_Temporal_B10", "SupportRisk_DepthGuard_B10"]
    abl = v2[v2.method.isin(ablation_names)].copy(); abl.to_csv(PAPER / "ablations" / "supportrisk_temporal_depthguard.csv", index=False)
    failures = v1[v1.method.isin(["EngineeringROI_B10", "DynamicOnly_B10", "SupportRiskOnly_B10"])].copy(); failures.to_csv(PAPER / "failure_cases" / "failure_mechanism_summary.csv", index=False)
    manifest_rows = []
    def add(name, split, method, budget, deployable, ablation, path, note): manifest_rows.append({"experiment_name": name, "split": split, "method": method, "budget": budget, "checkpoint_global": "best.pt step5000/STAGE_C/K=4", "checkpoint_local": "best.pt update4000", "deployable": deployable, "ablation_type": ablation, "scenario_count": 20 if split in {"VAL", "TEST"} else 1, "result_path": path, "commit": FINAL_SHA if "RiskTemporal" in method or split in {"TEST", "H0"} else "historical frozen result", "notes": note})
    add("RiskTemporal B05", "VAL", "RiskTemporal-v1", .05, True, "budget sensitivity", "results/engineering_roi_v2/runs/RiskTemporal_B05", "frozen final candidate")
    add("RiskTemporal B10", "VAL", "RiskTemporal-v1", .10, True, "primary final configuration", "results/engineering_roi_v2/runs/RiskTemporal_B10", "frozen final candidate")
    add("RiskTemporal B20", "VAL", "RiskTemporal-v1", .20, True, "budget sensitivity", "results/engineering_roi_v2/runs/RiskTemporal_B20", "frozen final candidate")
    for method in ["FrozenGlobal", "SupportRiskOnly_B10", "EngineeringROI_B10", "DynamicOnly_B10", "RandomAll_B10"]: add(method, "VAL", method, .10 if "Frozen" not in method else 0, method not in {"DynamicOnly_B10"}, "failure mechanism" if method in {"EngineeringROI_B10", "DynamicOnly_B10"} else "baseline", "results/engineering_roi_v1", "historical frozen VAL result")
    for method in ablation_names: add(method, "VAL", method, .10, method != "SupportRisk_DepthGuard_B10", "negative ablation" if "Depth" in method else "ablation", "results/engineering_roi_v2", "historical frozen VAL result")
    add("Frozen Global", "TEST", "FrozenGlobal", 0, True, "baseline", "paper_results/test/frozen_global", "untouched TEST")
    add("RiskTemporal B10", "TEST", "RiskTemporal-v1", .10, True, "primary final configuration", "paper_results/test/risktemporal_b10", "untouched TEST")
    add("PARK_V2_H0", "H0", "FrozenGlobal + RiskTemporal-v1 B10", .10, True, "historical application", "paper_results/h0", "event-constrained numerical-reference demonstration")
    pd.DataFrame(manifest_rows).to_csv(PAPER / "all_experiments_manifest.csv", index=False)
    table(risk[["method", "budget_fraction", "trajectory_h_rel_l2", "trajectory_momentum_rel_l2", "mixture_volume_relative_error", "debris_front_mae_km", "mean_wet_iou", "false_positive_wet_fraction", "case_wall_runtime_seconds"]], "budget_sensitivity")
    table(test[["method", "trajectory_h_rel_l2", "trajectory_momentum_rel_l2", "change_region_h_rel_l2", "change_region_momentum_rel_l2", "mean_wet_iou", "false_positive_wet_fraction", "mixture_volume_relative_error", "debris_front_mae_km", "arrival_MAE_s", "case_wall_runtime_seconds"]], "untouched_test_core")
    table(abl, "temporal_depthguard_ablations"); table(existing(PAPER / "h0" / "h0_summary.csv"), "h0_anchor")
    plt.style.use("seaborn-v0_8-whitegrid")
    metrics = ["trajectory_h_rel_l2", "trajectory_momentum_rel_l2", "mixture_volume_relative_error", "debris_front_mae_km", "mean_wet_iou", "false_positive_wet_fraction", "case_wall_runtime_seconds"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6));
    for ax, metric in zip(axes.flat, metrics): ax.plot(risk.budget_fraction, risk[metric], marker="o"); ax.set(title=metric, xlabel="budget fraction")
    axes.flat[-1].axis("off"); save(fig, "figure_A_budget_sensitivity")
    fig, ax = plt.subplots(figsize=(8, 4)); test.set_index("method")[["trajectory_h_rel_l2", "trajectory_momentum_rel_l2", "mixture_volume_relative_error", "debris_front_mae_km"]].T.plot.bar(ax=ax); ax.set_ylabel("error"); save(fig, "figure_B_primary_test_comparison")
    fig, ax = plt.subplots(figsize=(8, 4)); failures.set_index("method")[["trajectory_h_rel_l2", "trajectory_momentum_rel_l2", "mixture_volume_relative_error", "debris_front_mae_km"]].T.plot.bar(ax=ax, logy=True); ax.set_ylabel("error (log scale)"); save(fig, "figure_C_failure_mechanism")
    fig, ax = plt.subplots(figsize=(8, 4)); abl.set_index("method")[["change_region_h_rel_l2", "mixture_volume_relative_error", "debris_front_mae_km"]].T.plot.bar(ax=ax); save(fig, "figure_D_temporal_ablation")
    fig, ax = plt.subplots(figsize=(8, 4)); abl.set_index("method")[["trajectory_h_rel_l2", "trajectory_momentum_rel_l2", "mixture_volume_relative_error"]].T.plot.bar(ax=ax); save(fig, "figure_E_depthguard_negative_ablation")
    fig, ax = plt.subplots(figsize=(6, 5)); ax.scatter(risk.case_wall_runtime_seconds, risk.mixture_volume_relative_error, s=70); [ax.annotate(m, (x, y)) for m,x,y in zip(risk.method,risk.case_wall_runtime_seconds,risk.mixture_volume_relative_error)]; ax.set(xlabel="runtime per case (s)", ylabel="volume relative error"); save(fig, "figure_F_runtime_accuracy")
    g = test_case.pivot(index="scenario_id", columns="method", values="trajectory_h_rel_l2"); fig, ax = plt.subplots(figsize=(5, 5)); ax.scatter(g["FrozenGlobal"], g["RiskTemporal_B10"]); lim=max(g.max()); ax.plot([0,lim],[0,lim],"k--"); ax.set(xlabel="Frozen Global h RelL2", ylabel="RiskTemporal B10 h RelL2"); save(fig, "figure_G_case_robustness")
    h0 = PAPER / "h0"; ref=np.load(h0/"fields"/"reference_t0720s.npz")["h"]; glob=np.load(h0/"fields"/"FrozenGlobal_t0720s.npz")["h"]; riskh=np.load(h0/"fields"/"RiskTemporal_B10_t0720s.npz")["h"]
    fig, axes=plt.subplots(1,4,figsize=(14,4)); vals=[ref,glob,riskh,abs(riskh-ref)]; titles=["reference h","Frozen Global h","RiskTemporal B10 h","|RiskTemporal-reference|"]
    for ax,val,title in zip(axes,vals,titles): im=ax.imshow(np.where(val>.05,val,np.nan),cmap="viridis"); ax.set(title=title,xticks=[],yticks=[]); fig.colorbar(im,ax=ax,shrink=.7)
    save(fig,"figure_H_h0_field_comparison")
    hydro=existing(h0/"stations"/"station_hydrographs.csv"); col="gyirong_cctv_arrival_Q"; fig,ax=plt.subplots(figsize=(8,4));
    for method,frame in hydro.groupby("method"): ax.plot(frame.time_s,frame[col],label=method)
    ax.legend(); ax.set(xlabel="time after onset (s)",ylabel="section discharge proxy",title="H0 station hydrograph"); save(fig,"figure_H_h0_hydrograph")
    volume=existing(h0/"metrics"/"volume_evolution_proxy.csv"); fig,ax=plt.subplots(figsize=(8,4));
    for method,frame in volume.groupby("method"): ax.plot(frame.time_s,frame.wet_volume_proxy_m3/1e6,label=method)
    ax.legend(); ax.set(xlabel="time after onset (s)",ylabel="wet-volume proxy (Mm³)",title="H0 volume evolution"); save(fig,"figure_H_h0_volume_evolution")
    route=np.load(ROOT/"data"/"downloads"/"park_v2"/"inputs"/"upper30h.npz")["route_chainage_m"]; fig,ax=plt.subplots(figsize=(8,4));
    for method in ("reference","FrozenGlobal","RiskTemporal_B10"):
        rows=[]
        for path in sorted((h0/"fields").glob(f"{method}_t*s.npz")):
            value=np.load(path); valid=(value["h"]>.1)&(value["c"]>.05)&np.isfinite(route); front=float(np.max(route[valid])/1e3) if valid.any() else np.nan; rows.append((int(path.stem.split("t")[-1][:-1]),front))
        ax.plot(*zip(*rows),marker="o",label=method)
    ax.legend(); ax.set(xlabel="time after onset (s)",ylabel="debris front (km)",title="H0 propagation-front evolution (sampled)"); save(fig,"figure_H_h0_front_evolution")
    env = {"python":sys.version, "platform":platform.platform(), "torch":torch.__version__, "cuda":torch.version.cuda, "cuda_available":torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    (PAPER/"release_metadata"/"environment.txt").write_text(json.dumps(env,indent=2)+"\n",encoding="utf-8")
    (PAPER/"release_metadata"/"frozen_commits.json").write_text(json.dumps({"final_method_code":FINAL_SHA,"final_val_results":VAL_SHA},indent=2)+"\n",encoding="utf-8")
    shutil.copy2(ROOT/"configs"/"risk_temporal_final_v1.json",PAPER/"release_metadata"/"final_config.json")
    summary = "# Paper results summary\n\nRiskTemporal-v1 B10 is the frozen final method. TEST is evaluation-only and H0 is an event-constrained historical numerical-reference demonstration; neither was used for tuning. See machine-readable audits, tables, figures and manifests in this directory.\n"
    (PAPER/"PAPER_RESULTS_SUMMARY.md").write_text(summary,encoding="utf-8")
    print(json.dumps({"status":"PASS","figures":len(list((PAPER / "figures").glob("*.png"))),"tables":len(list((PAPER / "tables").glob("*.csv")))},sort_keys=True))


if __name__ == "__main__": main()
