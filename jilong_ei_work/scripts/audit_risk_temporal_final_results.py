"""Read-only audit for the completed RiskTemporal-v1 final VAL sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL_CODE_SHA = "146184d1fbbf366c0c9c40066a56753b398b10f4"
METHODS = ("RiskTemporal_B05", "RiskTemporal_B10", "RiskTemporal_B20")
CASE_METRICS = ("trajectory_h_rel_l2", "trajectory_momentum_rel_l2", "change_region_h_rel_l2", "change_region_momentum_rel_l2", "mean_wet_iou", "final_wet_iou", "false_positive_wet_fraction", "wet_overprediction_ratio", "mixture_volume_relative_error", "debris_front_mae_km", "debris_front_final_error_km")
STATION_METRICS = ("arrival_error_s", "peak_Q_relative_error", "peak_Qdebris_relative_error", "peak_hmax_relative_error", "peak_stage_absolute_error", "wet_width_relative_error")


def _read_run(root: Path, label: str):
    run = root / "runs" / label
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    case, station = pd.read_csv(run / "final_case_metrics.csv"), pd.read_csv(run / "final_station_metrics.csv")
    timeline, runtime = pd.read_csv(run / "selection_timeline.csv"), pd.read_csv(run / "runtime_summary.csv")
    if manifest.get("code_sha") != FINAL_CODE_SHA:
        raise RuntimeError(f"{label}: code SHA is not frozen final candidate")
    if manifest.get("scope") != "VAL_ONLY" or len(case) != 20 or case.scenario_id.nunique() != 20:
        raise RuntimeError(f"{label}: VAL case scope is invalid")
    if len(timeline) != 20 * 144 or not (timeline.groupby("scenario_id").size() == 144).all():
        raise RuntimeError(f"{label}: selection timeline is incomplete")
    if (timeline.selected_patch_consecutive_overlap_count > 0).any() or (timeline.depth_guard_activation_fraction != 0).any():
        raise RuntimeError(f"{label}: temporal/depth semantics are invalid")
    if case.scenario_id.astype(str).str.contains("TEST|H0|HOLDOUT", case=False, regex=True).any():
        raise RuntimeError(f"{label}: non-VAL scenario present")
    if (not np.isfinite(case.select_dtypes(include="number").to_numpy()).all()
            or not np.isfinite(station.select_dtypes(include="number").to_numpy()).all()
            or (runtime[["mean_consecutive_overlap", "mean_depth_guard_activation_fraction"]] != 0).any(axis=None)):
        raise RuntimeError(f"{label}: non-finite or invalid summary telemetry")
    return manifest, case, station, timeline, runtime


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--write-json", action="store_true")
    args = parser.parse_args(); final_root = ROOT / "results" / "risk_temporal_final_v1"
    required = ("budget_curve.csv", "final_method_summary.csv", "final_case_metrics.csv", "final_station_metrics.csv", "selection_timeline.csv", "runtime_summary.csv", "per_case_budget_comparison.csv", "paper_ready_summary.csv", "risk_temporal_final_val_report.json")
    missing = [name for name in required if not (final_root / name).is_file()]
    if missing:
        raise RuntimeError(f"missing required final artifacts: {missing}")
    data = {label: _read_run(final_root, label) for label in METHODS}
    manifests = [item[0] for item in data.values()]
    if len({m["config_sha256"] for m in manifests}) != 1:
        raise RuntimeError("candidate config hashes differ across budgets")
    case = data["RiskTemporal_B10"][1].sort_values("scenario_id").reset_index(drop=True)
    station = data["RiskTemporal_B10"][2].sort_values(["scenario_id", "station"]).reset_index(drop=True)
    timeline = data["RiskTemporal_B10"][3].sort_values(["scenario_id", "time_s"]).reset_index(drop=True)
    ref = ROOT / "results" / "engineering_roi_v2" / "runs" / "SupportRisk_Temporal_B10"
    ref_case = pd.read_csv(ref / "final_case_metrics.csv").sort_values("scenario_id").reset_index(drop=True)
    ref_station = pd.read_csv(ref / "final_station_metrics.csv").sort_values(["scenario_id", "station"]).reset_index(drop=True)
    ref_timeline = pd.read_csv(ref / "selection_timeline.csv").sort_values(["scenario_id", "time_s"]).reset_index(drop=True)
    case_diff = max(float(np.max(np.abs(case[c] - ref_case[c]))) for c in CASE_METRICS)
    station_diff = max(float(np.max(np.abs(station[c] - ref_station[c]))) for c in STATION_METRICS)
    selection_equal = timeline[["scenario_id", "time_s", "selected_patch_ids"]].equals(ref_timeline[["scenario_id", "time_s", "selected_patch_ids"]])
    if max(case_diff, station_diff) > 1e-5 or not selection_equal:
        raise RuntimeError("B10 reproduction against SupportRisk_Temporal_B10 failed")
    evidence = {"FINAL_RISK_TEMPORAL_VAL_AUDIT": "PASS", "final_code_sha": FINAL_CODE_SHA, "methods": list(METHODS), "val_cases_per_budget": 20, "temporal_overlap": 0.0, "depth_guard_activation": 0.0, "b10_reference": "SupportRisk_Temporal_B10", "b10_max_case_metric_abs_diff": case_diff, "b10_max_station_metric_abs_diff": station_diff, "b10_selection_exact_equal": selection_equal, "required_artifacts_present": True}
    if args.write_json:
        (final_root / "final_risk_temporal_result_audit.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
