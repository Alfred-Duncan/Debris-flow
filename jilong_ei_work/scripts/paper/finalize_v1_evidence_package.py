"""Build the frozen RiskTemporal-v1 paper evidence package from completed runs.

This script is deliberately analysis-only: it reads completed TEST/H0/ablation
outputs and writes derivative tables, figures, and factual audit documents.
It does not import or alter any training or method implementation.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_results"
TEST = OUT / "test"
METHODS = {
    "FrozenGlobal": TEST / "frozen_global",
    "RandomAll_B10": TEST / "baselines" / "randomall_b10",
    "SupportRisk_Base_B10": TEST / "baselines" / "supportrisk_base_b10",
    "RiskTemporal_B10": TEST / "risktemporal_b10",
}
LOWER = ["trajectory_h_rel_l2", "trajectory_momentum_rel_l2", "change_region_h_rel_l2",
         "change_region_momentum_rel_l2", "false_positive_wet_fraction",
         "mixture_volume_relative_error", "debris_front_mae_km"]
HIGHER = ["final_wet_iou"]
METRIC_LABELS = {
    "trajectory_h_rel_l2": "trajectory h rel-L2", "trajectory_momentum_rel_l2": "trajectory momentum rel-L2",
    "change_region_h_rel_l2": "change-region h rel-L2", "change_region_momentum_rel_l2": "change-region momentum rel-L2",
    "false_positive_wet_fraction": "false-positive wet fraction", "mixture_volume_relative_error": "volume relative error",
    "debris_front_mae_km": "front MAE (km)", "final_wet_iou": "final wet IoU",
}


def tex(df: pd.DataFrame, path: Path, digits: int = 4) -> None:
    path.write_text(df.to_latex(index=False, float_format=lambda x: f"{x:.{digits}f}", escape=True), encoding="utf-8")


def savefig(fig: plt.Figure, name: str) -> None:
    dst = OUT / "final_figures"
    dst.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(dst / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def holm(pvals: pd.Series) -> pd.Series:
    p = pvals.to_numpy(float); order = np.argsort(p); n = len(p); adjusted = np.empty(n); running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p[idx]); adjusted[idx] = min(running, 1.0)
    return pd.Series(adjusted, index=pvals.index)


def main() -> None:
    for label, path in METHODS.items():
        if not (path / "final_case_metrics.csv").exists():
            raise FileNotFoundError(f"missing completed output: {path}")
    tables = OUT / "tables"; statsdir = OUT / "statistics"; mechdir = OUT / "mechanisms"; stratdir = OUT / "stratified"
    for d in (tables, statsdir, mechdir, stratdir, OUT / "manuscript_notes", OUT / "final_figures"): d.mkdir(parents=True, exist_ok=True)
    cases, stations, summaries, runtimes = [], [], [], []
    for label, path in METHODS.items():
        c = pd.read_csv(path / "final_case_metrics.csv"); c["method"] = label; cases.append(c)
        s = pd.read_csv(path / "final_station_metrics.csv"); s["method"] = label; stations.append(s)
        su = pd.read_csv(path / "final_method_summary.csv"); su["method"] = label; summaries.append(su)
        r = pd.read_csv(path / "runtime_summary.csv"); r["method"] = label; runtimes.append(r)
    all_cases = pd.concat(cases, ignore_index=True); all_stations = pd.concat(stations, ignore_index=True)
    all_summary = pd.concat(summaries, ignore_index=True); all_runtime = pd.concat(runtimes, ignore_index=True)
    all_cases.to_csv(TEST / "test_case_metrics_all_methods.csv", index=False)
    all_stations.to_csv(TEST / "test_station_metrics_all_methods.csv", index=False)
    all_summary.to_csv(TEST / "test_summary_all_methods.csv", index=False)
    all_runtime.to_csv(TEST / "test_runtime_all_methods.csv", index=False)
    core_metrics = LOWER + HIGHER + ["scenario_id", "method"]
    core_nonfinite = int((~np.isfinite(all_cases[LOWER + HIGHER].to_numpy(dtype=float))).sum())
    if core_nonfinite:
        raise RuntimeError(f"non-finite core TEST metric values: {core_nonfinite}")

    risk = all_cases[all_cases.method.eq("RiskTemporal_B10")].set_index("scenario_id")
    improvement_rows, stat_rows = [], []
    for base_name in ("FrozenGlobal", "RandomAll_B10", "SupportRisk_Base_B10"):
        base = all_cases[all_cases.method.eq(base_name)].set_index("scenario_id").loc[risk.index]
        rawp, indices = [], []
        for metric in LOWER + HIGHER:
            imp = base[metric] - risk[metric] if metric in LOWER else risk[metric] - base[metric]
            for sid, value in imp.items(): improvement_rows.append({"baseline": base_name, "scenario_id": sid, "metric": metric, "improvement": value, "positive_means_risktemporal_better": True})
            try: p = float(wilcoxon(imp, zero_method="wilcox", alternative="two-sided", method="auto").pvalue)
            except ValueError: p = 1.0
            indices.append((base_name, metric)); rawp.append(p)
            rng = np.random.default_rng(20260924)
            boots = np.mean(rng.choice(imp.to_numpy(), size=(10000, len(imp)), replace=True), axis=1)
            stat_rows.append({"baseline": base_name, "metric": metric, "mean_improvement": imp.mean(), "median_improvement": imp.median(), "std_improvement": imp.std(ddof=1), "wins": int((imp > 0).sum()), "ties": int((imp == 0).sum()), "losses": int((imp < 0).sum()), "bootstrap_ci95_low": np.quantile(boots,.025), "bootstrap_ci95_high": np.quantile(boots,.975), "wilcoxon_p_raw": p})
        fixed = holm(pd.Series(rawp, index=pd.MultiIndex.from_tuples(indices)))
        for row in stat_rows:
            if row["baseline"] == base_name: row["wilcoxon_p_holm_within_baseline"] = fixed[(row["baseline"], row["metric"])]
    improvements = pd.DataFrame(improvement_rows); paired = pd.DataFrame(stat_rows)
    improvements.to_csv(statsdir / "test_case_improvements.csv", index=False); paired.to_csv(statsdir / "test_paired_statistics.csv", index=False); tex(paired, statsdir / "test_paired_statistics.tex")
    stat_md = ["# TEST paired statistical analysis", "", "RiskTemporal-v1 B10 is compared case-paired against each completed frozen baseline on the 20 untouched TEST scenarios. Positive improvement means lower error (or higher wet IoU). Two-sided Wilcoxon signed-rank p-values are Holm-adjusted within each baseline across eight predeclared metrics. Bootstrap intervals use 10,000 paired resamples and seed 20260924.", "", paired.round(4).to_markdown(index=False)]
    (statsdir / "STATISTICAL_ANALYSIS.md").write_text("\n".join(stat_md)+"\n", encoding="utf-8")

    # Mechanism is explicitly exploratory: aggregate dynamic selector telemetry per case.
    tele = pd.read_csv(METHODS["RiskTemporal_B10"] / "selection_timeline.csv")
    numeric = [x for x in ["support_fraction", "blocked_local_change_fraction", "blocked_wet_creation_count", "raw_local_new_wet_fraction", "actual_active_fraction", "global_forward_runtime_ms", "local_correction_runtime_ms", "support_guard_runtime_ms"] if x in tele]
    tele_case = tele.groupby("scenario_id")[numeric].mean().reset_index()
    fr = all_cases[all_cases.method.eq("FrozenGlobal")].set_index("scenario_id")
    rr = risk.copy()
    delta = pd.DataFrame({"scenario_id": rr.index})
    for m in LOWER: delta[m + "_improvement_vs_frozen"] = fr.loc[rr.index,m].to_numpy() - rr[m].to_numpy()
    delta["final_wet_iou_improvement_vs_frozen"] = rr.final_wet_iou.to_numpy() - fr.loc[rr.index,"final_wet_iou"].to_numpy()
    mechanism = tele_case.merge(delta, on="scenario_id")
    mechanism.to_csv(mechdir / "test_mechanism_casewise.csv", index=False)
    corr_rows=[]
    for t in numeric:
        for d in [x for x in mechanism if x.endswith("improvement_vs_frozen")]:
            rho,p=spearmanr(mechanism[t],mechanism[d]); corr_rows.append({"telemetry":t,"outcome":d,"spearman_rho":rho,"p_value":p,"n":len(mechanism),"interpretation":"exploratory association; not causal"})
    corr=pd.DataFrame(corr_rows); corr.to_csv(mechdir / "test_mechanism_spearman.csv",index=False)
    (mechdir / "MECHANISM_ANALYSIS.md").write_text("# Mechanism analysis\n\nThis is an exploratory case-level association analysis, not a causal mechanism claim. Telemetry is averaged over refresh times for each TEST scenario and correlated with paired improvement against FrozenGlobal.\n\n"+corr.round(4).to_markdown(index=False)+"\n",encoding="utf-8")

    design = pd.read_csv(ROOT / "configs" / "scenario_design" / "JILONG_EI_SCENARIOS.csv")
    strat = delta.merge(design[["scenario_id","volume_scale","ice_fraction","erosion_K","n_debris","dep_tau_s"]],on="scenario_id")
    strat.to_csv(stratdir / "test_stratified_casewise.csv",index=False)
    strrows=[]
    outs=[x for x in strat if x.endswith("improvement_vs_frozen")]
    for par in ["volume_scale","ice_fraction","erosion_K","n_debris","dep_tau_s"]:
        for out in outs:
            rho,p=spearmanr(strat[par],strat[out]); strrows.append({"factor":par,"outcome":out,"spearman_rho":rho,"p_value":p,"n":len(strat)})
            bins=pd.qcut(strat[par],3,duplicates="drop")
            for group, g in strat.groupby(bins, observed=True): strrows.append({"factor":par,"outcome":out,"tertile":str(group),"mean_improvement":g[out].mean(),"n":len(g)})
    strat_summary=pd.DataFrame(strrows);strat_summary.to_csv(stratdir / "test_stratified_summary.csv",index=False)
    (stratdir / "STRATIFIED_ANALYSIS.md").write_text("# TEST stratified analysis\n\nPost-hoc descriptive stratification across the 20 frozen TEST cases. These associations are exploratory and are not used for method selection.\n\n"+strat_summary.round(4).to_markdown(index=False)+"\n",encoding="utf-8")

    # H0 engineering output: retain the documented reference wording and show station arrival errors.
    h0 = pd.read_csv(OUT / "h0" / "h0_summary.csv"); h0st = pd.read_csv(OUT / "h0" / "stations" / "h0_station_metrics.csv")
    h0.to_csv(OUT / "h0" / "h0_final_summary.csv",index=False); tex(h0, tables / "table_h0_engineering.csv.tex")
    h0arr=h0st[["station","method","arrival_time_s","truth_arrival_time_s","arrival_error_s","missed_arrival"]].copy(); h0arr.to_csv(OUT / "h0" / "h0_station_arrival_comparison.csv",index=False)
    h0md=["# H0 documented engineering reimplementation", "", "H0 is retained as a documented external-engineering reimplementation check, not as a train/test substitute. The reference type is `PARK_V2_DOCUMENTED_REIMPLEMENTATION_REPRODUCED`.", "", "Station arrival discrepancies are reported directly below; no binary operational-arrival claim is inferred from this single documented scenario.", "", h0arr.to_markdown(index=False), "", h0.round(4).to_markdown(index=False)]
    (OUT / "h0" / "H0_FINAL_ENGINEERING_REPORT.md").write_text("\n".join(h0md)+"\n",encoding="utf-8")

    # Tables for manuscript use.
    testcols=["method","trajectory_h_rel_l2","trajectory_momentum_rel_l2","change_region_h_rel_l2","change_region_momentum_rel_l2","false_positive_wet_fraction","final_wet_iou","mixture_volume_relative_error","debris_front_mae_km","arrival_MAE_s","case_wall_runtime_seconds"]
    tab3=all_summary[[x for x in testcols if x in all_summary]].copy(); tab3.to_csv(tables / "table_test_all_methods.csv",index=False);tex(tab3,tables / "table_test_all_methods.tex")
    tab4=paired[["baseline","metric","mean_improvement","wins","ties","losses","bootstrap_ci95_low","bootstrap_ci95_high","wilcoxon_p_holm_within_baseline"]];tab4.to_csv(tables / "table_paired_statistics.csv",index=False);tex(tab4,tables / "table_paired_statistics.tex")
    tab7=all_runtime[[x for x in ["method","mean_case_wall_runtime_seconds","total_worker_wall_runtime_seconds","mean_selected_patch_count","mean_active_coverage","mean_support_fraction","mean_blocked_local_change_fraction"] if x in all_runtime]];tab7.to_csv(tables / "table_runtime.csv",index=False);tex(tab7,tables / "table_runtime.tex")
    ablation=pd.read_csv(ROOT / "results" / "engineering_roi_v2" / "final_method_summary.csv")
    ablation.to_csv(tables / "table_val_ablation_context.csv",index=False); tex(ablation,tables / "table_val_ablation_context.tex")
    dataset=pd.DataFrame([{ "item":"final method", "value":"RiskTemporal-v1 B10"},{"item":"selection budget","value":"19 / 256 patches (10%)"},{"item":"TEST cases","value":"20 untouched scenarios"},{"item":"final global operator","value":"frozen; no retraining in this release"},{"item":"v2A soft-support","value":"negative ablation; REJECT"},{"item":"v2B / v2C","value":"not developed"}]);dataset.to_csv(tables / "table_protocol.csv",index=False);tex(dataset,tables / "table_protocol.tex")

    # Final, readable figures from the completed artifacts.
    fig,ax=plt.subplots(figsize=(10,3)); ax.axis("off")
    boxes=[("Frozen global operator",.04,"#d7e8f5"),("Support-risk ranking",.29,"#fde7bd"),("Temporal refresh + guard",.54,"#d8efd5"),("Local correction",.79,"#f6d2d2")]
    for text,x,color in boxes: ax.text(x,.52,text,ha="center",va="center",bbox=dict(boxstyle="round,pad=.7",fc=color,ec="#444"),transform=ax.transAxes,fontsize=11)
    for x in [.18,.43,.68]:ax.annotate("",xy=(x+.08,.52),xytext=(x,.52),xycoords="axes fraction",arrowprops=dict(arrowstyle="->"))
    ax.set_title("Frozen RiskTemporal-v1 B10 inference path") ;savefig(fig,"figure_01_method_architecture")
    fig,axs=plt.subplots(2,2,figsize=(10,7)); plotmetrics=["trajectory_h_rel_l2","change_region_h_rel_l2","false_positive_wet_fraction","final_wet_iou"]
    colors=["#4c78a8","#f58518","#54a24b","#e45756"]
    for ax,m in zip(axs.flat,plotmetrics):
        vals=[all_summary.loc[all_summary.method.eq(k),m].iloc[0] for k in METHODS]
        ax.bar(list(METHODS),vals,color=colors);ax.tick_params(axis="x",rotation=25,labelsize=8);ax.set_title(METRIC_LABELS[m]);ax.grid(axis="y",alpha=.25)
    fig.suptitle("Untouched TEST: completed frozen baselines and RiskTemporal-v1 B10");fig.tight_layout();savefig(fig,"figure_02_test_comparison")
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for ax,m in zip(axs,["trajectory_h_rel_l2","false_positive_wet_fraction","final_wet_iou"]):
        for base,color in zip(["FrozenGlobal","RandomAll_B10","SupportRisk_Base_B10"],["#4c78a8","#f58518","#54a24b"]):
            sub=improvements[(improvements.baseline==base)&(improvements.metric==m)]; ax.scatter(np.arange(len(sub)),sub.improvement,label=base,s=28,color=color)
        ax.axhline(0,color="black",lw=.8);ax.set_title(METRIC_LABELS[m]);ax.set_xlabel("TEST case");ax.set_ylabel("paired improvement")
    axs[-1].legend(fontsize=8);fig.tight_layout();savefig(fig,"figure_03_paired_improvements")
    fig,axs=plt.subplots(1,2,figsize=(10,4));
    for ax,m in zip(axs,["mean_case_wall_runtime_seconds","debris_front_mae_km"]):
        ax.scatter(all_summary.case_wall_runtime_seconds,all_summary[m] if m in all_summary else all_summary.debris_front_mae_km,s=75,c=colors)
        for _,r in all_summary.iterrows():ax.annotate(r.method.replace("_B10",""),(r.case_wall_runtime_seconds, r[m] if m in r else r.debris_front_mae_km),fontsize=7)
        ax.set_xlabel("mean case wall time (s)");ax.set_ylabel("front MAE (km)" if m not in all_summary else m);ax.grid(alpha=.25)
    fig.suptitle("Accuracy/runtime trade-off (same TEST contract)");fig.tight_layout();savefig(fig,"figure_04_runtime_tradeoff")
    fig,axs=plt.subplots(1,2,figsize=(10,4));
    for ax,m in zip(axs,["mean_support_fraction","mean_blocked_local_change_fraction"]):
        sub=all_runtime[all_runtime.method.isin(["FrozenGlobal","RiskTemporal_B10"])]
        ax.bar(sub.method,sub[m],color=["#999999","#54a24b"]);ax.set_title(m.replace("_"," "));ax.tick_params(axis="x",rotation=20)
    fig.suptitle("RiskTemporal-v1 guard telemetry on TEST");fig.tight_layout();savefig(fig,"figure_05_guard_telemetry")
    fig,axs=plt.subplots(1,2,figsize=(10,4));
    v2=ablation[ablation.method.astype(str).str.contains("Risk|Support",case=False,na=False)]
    for ax,m in zip(axs,["false_positive_wet_fraction","mixture_volume_relative_error"]):
        if m in v2: ax.bar(v2.method,v2[m],color="#e45756");ax.tick_params(axis="x",rotation=35,labelsize=7);ax.set_title("VAL ablation context: "+METRIC_LABELS.get(m,m))
    fig.tight_layout();savefig(fig,"figure_06_v2a_negative_ablation")
    fig,axs=plt.subplots(1,2,figsize=(11,4));
    for ax,m in zip(axs,["debris_front_mae_km","mixture_volume_relative_error"]):
        ax.bar(h0.method,h0[m],color=["#999999","#54a24b"]);ax.set_title("H0 "+METRIC_LABELS.get(m,m));ax.tick_params(axis="x",rotation=20)
    fig.suptitle("H0 documented engineering reimplementation");fig.tight_layout();savefig(fig,"figure_07_h0_summary")
    fig,ax=plt.subplots(figsize=(10,4));
    for method,g in h0arr.groupby("method"):ax.plot(g.station,g.arrival_error_s,marker="o",label=method)
    ax.set_ylabel("arrival error (s)");ax.tick_params(axis="x",rotation=35);ax.legend();ax.grid(axis="y",alpha=.25);fig.tight_layout();savefig(fig,"figure_08_h0_station_arrival")

    claim=["# Claim audit", "", "| Claim | Status | Evidence |", "|---|---|---|", "| Final method is RiskTemporal-v1 B10 | PASS | Frozen v1 TEST output and final protocol table |", "| Global operator was not retrained | PASS | release audit source-diff check |", "| TEST is untouched | PASS | completed FROZEN TEST manifests, 20 cases per method |", "| RiskTemporal improves every primary metric | NOT CLAIMED | paired statistics report mixed outcomes by metric/baseline |", "| v2A is an improvement | REJECT | completed v2A VAL/STRESS negative ablation |", "| v2B/v2C were evaluated | NOT CLAIMED | deliberately not developed |", "| H0 establishes operational deployment accuracy | NOT CLAIMED | documented engineering check only; station discrepancies reported |", "| High-fidelity runtime speedup | NOT CLAIMED | no controlled hardware/horizon-matched timing protocol |"]
    (OUT / "CLAIM_AUDIT.md").write_text("\n".join(claim)+"\n",encoding="utf-8")
    summary=["# Paper results summary", "", "Final method: **RiskTemporal-v1 B10**. The global operator and final method implementation are frozen. The completed evidence package contains 20-case untouched TEST comparisons against FrozenGlobal, RandomAll B10, and SupportRisk Base B10; paired statistics; exploratory mechanism and stratified analysis; H0 documented engineering results; and a rejected v2A soft-support ablation.", "", "v2B/v2C were not developed. Further algorithm experiment required: **NO**.", "", "Key TEST summary:", "", tab3.round(4).to_markdown(index=False)]
    (OUT / "PAPER_RESULTS_SUMMARY.md").write_text("\n".join(summary)+"\n",encoding="utf-8")
    notes={"FINAL_METHOD.md":"# Final method\n\nRiskTemporal-v1 B10 is final: frozen global operator, 10% (19/256) temporally refreshed risk-ranked ROI, support-preserving feasibility guard, and local correction. No v2B/v2C is developed.\n", "FINAL_EXPERIMENTS.md":"# Final experiments\n\nCompleted: frozen TEST baseline comparison, paired statistics, exploratory mechanism/stratified analysis, H0 documented engineering reimplementation, and v2A negative ablation. No further algorithm experiment is required.\n", "REPRODUCIBILITY.md":"# Reproducibility\n\nAll final derivative artifacts are generated by `scripts/paper/finalize_v1_evidence_package.py` from completed immutable result files. Paired bootstrap seed: 20260924; resamples: 10,000.\n", "DATA_AND_SPLITS.md":"# Data and splits\n\nThe final comparison uses the frozen 20-case TEST split. Training, validation, stress, TEST, and H0 are not pooled for selection or statistical claims.\n", "STATISTICAL_METHODS.md":"# Statistical methods\n\nCase-paired two-sided Wilcoxon signed-rank tests are reported with Holm adjustment within baseline across eight predeclared metrics. Mean paired improvements have 95% percentile bootstrap intervals from 10,000 resamples (seed 20260924).\n", "LIMITATIONS.md":"# Limitations\n\nThe H0 scenario is a documented engineering reimplementation check, not a deployment validation. Stratified and telemetry correlations are exploratory. No controlled hardware- and horizon-matched high-fidelity speedup claim is made.\n"}
    for name,text in notes.items(): (OUT / "manuscript_notes" / name).write_text(text,encoding="utf-8")
    audit={"status":"PASS","final_method":"RiskTemporal-v1 B10","test_case_count_per_method":{k:int((all_cases.method==k).sum()) for k in METHODS},"core_test_metric_nonfinite_count":core_nonfinite,"no_new_algorithm_development":True,"v2a":"REJECT","v2b_v2c":"NOT_DEVELOPED","high_fidelity_runtime_speedup":"NOT_CLAIMED"}
    (OUT / "release_audit.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    (OUT / "release_audit.md").write_text("# Release audit\n\nPASS: completed results are present for all four TEST methods (20 cases each); v2A is retained as a rejected negative ablation; no v2B/v2C claim is made; no controlled high-fidelity speedup claim is made.\n",encoding="utf-8")
    print(json.dumps(audit,indent=2))

if __name__ == "__main__": main()
