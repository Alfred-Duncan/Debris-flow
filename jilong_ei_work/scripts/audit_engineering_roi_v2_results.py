"""Read-only formal-result audit for EngineeringROI-v2 VAL outputs."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.provenance import sha256_file

FORMAL_SHA = "8a1b790aeb1421088016f7bbfd796c65018c7176"
METHODS = ("SupportRisk_Base_B10", "SupportRisk_Temporal_B10", "SupportRisk_DepthGuard_B10", "EngineeringROI_v2_B05", "EngineeringROI_v2_B10", "EngineeringROI_v2_B20")
TEMPORAL = {"SupportRisk_Temporal_B10", "EngineeringROI_v2_B05", "EngineeringROI_v2_B10", "EngineeringROI_v2_B20"}
DEPTH = {"SupportRisk_DepthGuard_B10", "EngineeringROI_v2_B05", "EngineeringROI_v2_B10", "EngineeringROI_v2_B20"}
METRICS = ("trajectory_h_rel_l2", "trajectory_momentum_rel_l2", "change_region_h_rel_l2", "change_region_momentum_rel_l2", "mean_wet_iou", "final_wet_iou", "false_positive_wet_fraction", "wet_overprediction_ratio", "mixture_volume_relative_error", "debris_front_mae_km", "debris_front_final_error_km", "arrival_MAE_s", "peak_Q_relative_error", "peak_Qdebris_relative_error", "peak_hmax_relative_error", "peak_stage_absolute_error", "wet_width_relative_error", "missed_arrival_count")


def _load(label):
    path = ROOT / "results/engineering_roi_v2/runs" / label
    required = ("run_manifest.json", "RUN_COMPLETE.json", "final_method_summary.csv", "final_case_metrics.csv", "final_station_metrics.csv", "selection_timeline.csv", "runtime_summary.csv", "method_report.json")
    if any(not (path / name).exists() for name in required):
        raise RuntimeError(f"V2_METHOD_OUTPUT_INCOMPLETE:{label}")
    return path, json.loads((path / "run_manifest.json").read_text()), pd.read_csv(path / "final_method_summary.csv"), pd.read_csv(path / "final_case_metrics.csv"), pd.read_csv(path / "selection_timeline.csv")


def _finite(frame, columns, label):
    missing = [name for name in columns if name not in frame]
    if missing or not np.isfinite(frame[list(columns)].to_numpy(dtype=float)).all():
        raise RuntimeError(f"V2_NONFINITE_METRIC:{label}")


def baseline_check():
    path, manifest, summary, cases, _ = _load("SupportRisk_Base_B10")
    reference = ROOT / "results/engineering_roi_v1/runs/SupportRiskOnly_B10"
    old_manifest = json.loads((reference / "run_manifest.json").read_text())
    for key in ("global_checkpoint_sha256", "local_checkpoint_sha256", "local_normalization_sha256", "support_guard_version", "local_checkpoint_update"):
        if manifest.get(key) != old_manifest.get(key):
            raise RuntimeError("SUPPORTRISK_BASELINE_REPRODUCTION_FAIL")
    old_cases = pd.read_csv(reference / "final_case_metrics.csv").sort_values("scenario_id").reset_index(drop=True)
    cases = cases.sort_values("scenario_id").reset_index(drop=True)
    if list(cases.scenario_id.astype(str)) != list(old_cases.scenario_id.astype(str)):
        raise RuntimeError("SUPPORTRISK_BASELINE_REPRODUCTION_FAIL")
    common = [key for key in METRICS if key in cases and key in old_cases]
    differences = {key: float(np.max(np.abs(cases[key].to_numpy(float) - old_cases[key].to_numpy(float)))) for key in common}
    if any(not math.isfinite(value) or value > 1e-5 for value in differences.values()):
        raise RuntimeError("SUPPORTRISK_BASELINE_REPRODUCTION_FAIL")
    return {"status": "PASS", "max_absolute_differences": differences, "v1_reference": "SupportRiskOnly_B10", "v2_method": "SupportRisk_Base_B10"}


def audit_all():
    expected_config = sha256_file(ROOT / "configs/engineering_roi_v2.json")
    provenance = None; totals = {}; loaded = {}
    for label in METHODS:
        path, manifest, summary, cases, timeline = _load(label); loaded[label] = (path, manifest, summary, cases, timeline)
        if manifest.get("code_sha") != FORMAL_SHA or manifest.get("config_sha256") != expected_config:
            raise RuntimeError("V2_FORMAL_PROVENANCE_MISMATCH")
        if len(cases) != 20 or cases.scenario_id.astype(str).nunique() != 20 or len(timeline) != 2880 or any(len(group) != 144 for _, group in timeline.groupby("scenario_id")):
            raise RuntimeError(f"V2_FORMAL_COMPLETENESS_FAIL:{label}")
        if timeline.scenario_id.astype(str).str.contains("TEST|H0|HOLDOUT", case=False, regex=True).any():
            raise RuntimeError("V2_NONVAL_ID_PRESENT")
        _finite(summary, METRICS, label)
        _finite(timeline, ("blocked_wet_creation_count", "raw_local_new_wet_fraction", "depth_guard_activation_fraction", "depth_guard_mean_abs_clip_m", "depth_guard_max_abs_clip_m"), label)
        if (timeline.blocked_wet_creation_count < 0).any() or ((timeline.raw_local_new_wet_fraction < 0) | (timeline.raw_local_new_wet_fraction > 1)).any() or ((timeline.depth_guard_activation_fraction < 0) | (timeline.depth_guard_activation_fraction > 1)).any() or (timeline[["depth_guard_mean_abs_clip_m", "depth_guard_max_abs_clip_m"]] < 0).any().any():
            raise RuntimeError("V2_TELEMETRY_RANGE_INVALID")
        if label in TEMPORAL and (timeline.selected_patch_consecutive_overlap_count > 0).any():
            raise RuntimeError("TEMPORAL_REFRESH_VIOLATION")
        if label not in DEPTH and (timeline.depth_guard_activation_fraction != 0).any():
            raise RuntimeError("V2_DEPTH_ABLATION_SEMANTICS_INVALID")
        identity = {key: manifest.get(key) for key in ("config_sha256", "global_checkpoint_sha256", "local_checkpoint_sha256", "local_normalization_sha256", "local_checkpoint_update", "support_guard_version", "full_scenario_ids")}
        if provenance is None: provenance = identity
        elif provenance != identity: raise RuntimeError("V2_FORMAL_VAL_INVALID")
        totals[label] = {"cases": int(len(cases)), "timeline_rows": int(len(timeline))}
    return {"status": "PASS", "formal_code_sha": FORMAL_SHA, "methods": totals, "baseline_reproduction": baseline_check()}


def main(args):
    report = baseline_check() if args.baseline_only else audit_all()
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--baseline-only", action="store_true"); main(parser.parse_args())
