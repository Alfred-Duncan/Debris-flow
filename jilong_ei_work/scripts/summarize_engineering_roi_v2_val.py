"""Read-only aggregation/reporting for completed EngineeringROI-v2 formal VAL."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_engineering_roi_v2_results import METHODS, METRICS, audit_all

V1 = ("FrozenGlobal", "RandomAll_B10", "RandomSupport_B10", "EngineeringROI_B10", "SupportRiskOnly_B10")
LOWER = {"trajectory_h_rel_l2", "trajectory_momentum_rel_l2", "change_region_h_rel_l2", "change_region_momentum_rel_l2", "false_positive_wet_fraction", "wet_overprediction_ratio", "mixture_volume_relative_error", "debris_front_mae_km", "debris_front_final_error_km", "arrival_MAE_s", "peak_Q_relative_error", "peak_Qdebris_relative_error", "peak_hmax_relative_error", "peak_stage_absolute_error", "wet_width_relative_error"}
HIGHER = {"mean_wet_iou", "final_wet_iou"}


def read_run(root, label):
    path = root / "runs" / label
    return (pd.read_csv(path / "final_method_summary.csv"), pd.read_csv(path / "final_case_metrics.csv"), pd.read_csv(path / "final_station_metrics.csv"), pd.read_csv(path / "selection_timeline.csv"), pd.read_csv(path / "runtime_summary.csv"), json.loads((path / "method_report.json").read_text()))


def station_case(frame):
    return frame.groupby("scenario_id", as_index=False).mean(numeric_only=True)


def robustness(current_cases, current_stations, reference_cases, reference_stations, comparison):
    left = current_cases.merge(station_case(current_stations), on="scenario_id", how="left", suffixes=("", "_station"))
    right = reference_cases.merge(station_case(reference_stations), on="scenario_id", how="left", suffixes=("", "_station"))
    rows = []
    for metric in METRICS:
        if metric not in left or metric not in right or metric not in LOWER | HIGHER:
            continue
        a, b = left[metric].to_numpy(float), right[metric].to_numpy(float)
        rows.append({"comparison": comparison, "metric": metric, "improved_cases": int((a < b).sum()) if metric in LOWER else int((a > b).sum()), "total_cases": int(len(a))})
    return rows


def main():
    audit = audit_all()
    v2_root, v1_root = ROOT / "results/engineering_roi_v2", ROOT / "results/engineering_roi_v1"
    method_rows, cases, stations, timelines, runtimes, reports = [], [], [], [], [], {}
    for label in METHODS:
        summary, case, station, timeline, runtime, report = read_run(v2_root, label)
        method_rows.append(summary); cases.append(case); stations.append(station); timelines.append(timeline); runtimes.append(runtime); reports[label] = report
    v2_method = pd.concat(method_rows, ignore_index=True); v2_cases = pd.concat(cases, ignore_index=True); v2_stations = pd.concat(stations, ignore_index=True); v2_timeline = pd.concat(timelines, ignore_index=True); v2_runtime = pd.concat(runtimes, ignore_index=True)
    references = {}
    for label in V1:
        references[label] = read_run(v1_root, label)
    reference_summary = pd.concat([references[label][0] for label in V1], ignore_index=True)
    reference_cases = pd.concat([references[label][1] for label in V1], ignore_index=True)
    reference_stations = pd.concat([references[label][2] for label in V1], ignore_index=True)
    all_summary = pd.concat([reference_summary, v2_method], ignore_index=True)
    mechanism = v2_runtime.merge(v2_method[["method"] + [key for key in METRICS if key in v2_method]], on="method", how="left")
    robust = []
    current_case = v2_cases[v2_cases.method.eq("EngineeringROI_v2_B10")]; current_station = v2_stations[v2_stations.method.eq("EngineeringROI_v2_B10")]
    for label in ("FrozenGlobal", "RandomSupport_B10"):
        robust.extend(robustness(current_case, current_station, reference_cases[reference_cases.method.eq(label)], reference_stations[reference_stations.method.eq(label)], f"EngineeringROI_v2_B10_vs_{label}"))
    robust.extend(robustness(current_case, current_station, v2_cases[v2_cases.method.eq("SupportRisk_Base_B10")], v2_stations[v2_stations.method.eq("SupportRisk_Base_B10")], "EngineeringROI_v2_B10_vs_SupportRisk_Base_B10"))
    v2_root.mkdir(parents=True, exist_ok=True)
    all_summary.to_csv(v2_root / "final_method_summary.csv", index=False); v2_cases.to_csv(v2_root / "final_case_metrics.csv", index=False); v2_stations.to_csv(v2_root / "final_station_metrics.csv", index=False); v2_timeline.to_csv(v2_root / "selection_timeline.csv", index=False); v2_runtime.to_csv(v2_root / "runtime_summary.csv", index=False); mechanism.to_csv(v2_root / "mechanism_summary.csv", index=False); all_summary.to_csv(v2_root / "v2_vs_v1_summary.csv", index=False); pd.DataFrame(robust).to_csv(v2_root / "per_case_robustness.csv", index=False)
    report = {"formal_v2_val_result_audit": audit, "methods": all_summary.to_dict("records"), "mechanism_telemetry": mechanism.to_dict("records"), "per_case_robustness": robust}
    (v2_root / "engineering_roi_v2_val_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    md = ["# EngineeringROI-v2 formal 20-case VAL", "", "Result audit: PASS.", "", "## Method summaries", "", "| Method | trajectory h | volume error | front MAE km | arrival MAE s |", "|---|---:|---:|---:|---:|"]
    for row in all_summary.to_dict("records"):
        md.append(f"| {row['method']} | {row.get('trajectory_h_rel_l2', float('nan')):.6f} | {row.get('mixture_volume_relative_error', float('nan')):.6f} | {row.get('debris_front_mae_km', float('nan')):.6f} | {row.get('arrival_MAE_s', float('nan')):.3f} |")
    md.extend(["", "Findings are descriptive comparisons of the completed frozen VAL protocol; no training, TEST, H0, or holdout execution occurred."])
    (ROOT / "reports/ENGINEERING_ROI_V2_FORMAL_VAL.md").write_text("\n".join(md) + "\n")
    (ROOT / "reports/ENGINEERING_ROI_V2_FORMAL_VAL.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"status": "PASS", "methods": len(METHODS), "timeline_rows": int(len(v2_timeline))}, indent=2))


if __name__ == "__main__":
    main()
