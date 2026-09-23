"""Read-only V1 persistence and descriptive-association audit; never loads a model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "RandomAll_B10", "RandomSupport_B10", "EngineeringROI_B05", "EngineeringROI_B10",
    "EngineeringROI_B20", "DynamicOnly_B10", "SupportRiskOnly_B10", "EngineeringROI_NoDiversity_B10",
)
CASE_METRICS = ("trajectory_h_rel_l2", "mixture_volume_relative_error", "debris_front_mae_km")
STATION_METRICS = ("peak_hmax_relative_error", "peak_stage_absolute_error", "peak_Q_relative_error", "arrival_error_s")
SIGNALS = ("top10_share", "consecutive_reselection_fraction", "mean_selection_count_per_used_patch")


def persistence(sets):
    """Compute per-scenario patch persistence without touching model artifacts."""
    flat = [patch for selected in sets for patch in selected]
    counts = pd.Series(flat, dtype="int64").value_counts() if flat else pd.Series(dtype="int64")
    gaps, streaks = [], []
    for patch in counts.index:
        hits = [index for index, selected in enumerate(sets) if patch in selected]
        gaps.extend(later - earlier for earlier, later in zip(hits, hits[1:]))
        run, runs = 0, []
        for selected in sets:
            run = run + 1 if patch in selected else 0
            runs.append(run)
        streaks.append(max(runs))
    overlap = sum(len(previous & selected) for previous, selected in zip(sets, sets[1:]))
    total = max(len(flat), 1)
    return {
        "total_selections": len(flat), "unique_selected_patches": len(counts),
        "unique_patch_fraction": len(counts) / total,
        "top1_selection_share": counts.head(1).sum() / total,
        "top5_share": counts.head(5).sum() / total,
        "top10_share": counts.head(10).sum() / total,
        "consecutive_reselection_fraction": overlap / total,
        "revisit_fraction": float((counts > 1).mean()) if len(counts) else 0.0,
        "mean_selection_count_per_used_patch": float(counts.mean()) if len(counts) else 0.0,
        "maximum_selection_count": int(counts.max()) if len(counts) else 0,
        "mean_revisit_gap_steps": float(np.mean(gaps)) if gaps else float("nan"),
        "mean_longest_consecutive_streak": float(np.mean(streaks)) if streaks else 0.0,
    }


def _parse_sets(values):
    return [set(map(int, value.split(";"))) if isinstance(value, str) and value else set() for value in values]


def _station_case_metrics(stations):
    return stations.groupby(["method", "scenario_id"], as_index=False)[list(STATION_METRICS)].mean(numeric_only=True)


def _associations(frame):
    records = []
    for metric in CASE_METRICS + STATION_METRICS:
        if metric not in frame:
            continue
        for signal in SIGNALS:
            pair = frame[[metric, signal]].dropna()
            records.append({
                "metric": metric, "signal": signal, "n_cases": int(len(pair)),
                "pearson": float(pair.corr(method="pearson").iloc[0, 1]) if len(pair) > 1 else float("nan"),
                "spearman": float(pair.corr(method="spearman").iloc[0, 1]) if len(pair) > 1 else float("nan"),
            })
    return records


def _method_observations(per_case):
    observations = {}
    for method, group in per_case.groupby("method"):
        total = float(group.total_selections.sum())
        observations[method] = {
            "cases": int(len(group)), "total_selections": int(total),
            "mean_top10_selection_share": float(group.top10_share.mean()),
            "mean_consecutive_reselection_fraction": float(group.consecutive_reselection_fraction.mean()),
            "mean_revisit_fraction": float(group.revisit_fraction.mean()),
            "mean_longest_consecutive_streak": float(group.mean_longest_consecutive_streak.mean()),
        }
    return observations


def main():
    timeline = pd.read_csv(ROOT / "results/engineering_roi_v1/selection_timeline.csv")
    rows = []
    for (method, scenario), group in timeline[timeline.method.isin(METHODS)].groupby(["method", "scenario_id"], sort=True):
        rows.append({"method": method, "scenario_id": scenario, **persistence(_parse_sets(group.sort_values("time_s").selected_patch_ids))})
    per_selection = pd.DataFrame(rows)
    cases = pd.read_csv(ROOT / "results/engineering_roi_v1/final_case_metrics.csv")
    stations = pd.read_csv(ROOT / "results/engineering_roi_v1/final_station_metrics.csv")
    per_case = per_selection.merge(cases, on=["method", "scenario_id"], how="left").merge(_station_case_metrics(stations), on=["method", "scenario_id"], how="left")
    selected_timeline = timeline[timeline.method.isin(METHODS)].groupby(["method", "scenario_id"], as_index=False).agg(
        mean_selected_dynamic_rank=("mean_selected_dynamic_rank", "mean"),
        mean_selected_support_risk_rank=("mean_selected_support_risk_rank", "mean"),
        mean_selected_base_score=("mean_selected_base_score", "mean"),
        mean_active_coverage=("actual_active_fraction", "mean"),
    )
    per_case = per_case.merge(selected_timeline, on=["method", "scenario_id"], how="left")
    method_summary = per_case.groupby("method", as_index=False).mean(numeric_only=True)
    output = ROOT / "results/engineering_roi_v1_failure_audit"
    output.mkdir(parents=True, exist_ok=True)
    per_selection.to_csv(output / "selection_persistence_summary.csv", index=False)
    per_case.to_csv(output / "per_case_mechanism_summary.csv", index=False)
    method_summary.to_csv(output / "method_mechanism_summary.csv", index=False)
    observations = _method_observations(per_case)
    report = {
        "status": "PASS", "no_new_model_execution": True, "read_only_input_root": "results/engineering_roi_v1",
        "methods": list(METHODS), "selection_persistence_metrics": list(persistence([])),
        "method_observations": observations, "descriptive_associations": _associations(per_case),
        "amplitude_accumulation": "NOT DIRECTLY OBSERVABLE FROM EXISTING V1 ARTIFACTS",
        "interpretation": "Persistence/error correlations are descriptive associations only and do not establish causation.",
    }
    (output / "failure_mechanism_report.json").write_text(json.dumps(report, indent=2, allow_nan=True))
    lines = [
        "# EngineeringROI-v1 failure-mechanism audit", "",
        "Status: PASS. Input artifacts were read only; no model execution occurred.", "",
        "## Observed facts", "",
        "| Method | Mean top-10 share | Mean consecutive reselection | Mean revisit fraction |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        value = observations.get(method, {})
        lines.append(f"| {method} | {value.get('mean_top10_selection_share', float('nan')):.4f} | {value.get('mean_consecutive_reselection_fraction', float('nan')):.4f} | {value.get('mean_revisit_fraction', float('nan')):.4f} |")
    lines.extend([
        "", "## Interpretation boundary", "",
        "The correlations in the CSV/JSON outputs are descriptive associations, not causal proof. "
        "Raw local correction amplitude is **NOT DIRECTLY OBSERVABLE FROM EXISTING V1 ARTIFACTS**; "
        "therefore amplitude accumulation cannot be directly proven by retained V1 telemetry alone.",
    ])
    (ROOT / "reports/ENGINEERING_ROI_V1_FAILURE_MECHANISM.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
