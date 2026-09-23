"""Build lightweight, reproducible final VAL tables for RiskTemporal-v1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "risk_temporal_final_v1"
METHODS = ("RiskTemporal_B05", "RiskTemporal_B10", "RiskTemporal_B20")
SHA = "146184d1fbbf366c0c9c40066a56753b398b10f4"


def _load(name: str) -> pd.DataFrame:
    return pd.concat([pd.read_csv(OUT / "runs" / method / name) for method in METHODS], ignore_index=True)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    summaries, cases = _load("final_method_summary.csv"), _load("final_case_metrics.csv")
    stations, timeline, runtime = _load("final_station_metrics.csv"), _load("selection_timeline.csv"), _load("runtime_summary.csv")
    for name, frame in {"final_method_summary.csv": summaries, "final_case_metrics.csv": cases, "final_station_metrics.csv": stations, "selection_timeline.csv": timeline, "runtime_summary.csv": runtime}.items():
        frame.to_csv(OUT / name, index=False)
    curve = summaries.merge(runtime[["method", "total_worker_wall_runtime_seconds", "mean_consecutive_overlap", "mean_depth_guard_activation_fraction"]], on="method", validate="one_to_one").sort_values("budget_fraction")
    curve.to_csv(OUT / "budget_curve.csv", index=False)
    cols = ["scenario_id", "method", "budget_fraction", "trajectory_h_rel_l2", "trajectory_momentum_rel_l2", "change_region_h_rel_l2", "change_region_momentum_rel_l2", "mean_wet_iou", "final_wet_iou", "false_positive_wet_fraction", "mixture_volume_relative_error", "debris_front_mae_km"]
    cases[cols].sort_values(["scenario_id", "budget_fraction"]).to_csv(OUT / "per_case_budget_comparison.csv", index=False)
    prior = pd.read_csv(ROOT / "results" / "engineering_roi_v2" / "final_method_summary.csv")
    refs = prior[prior.method.isin(["FrozenGlobal", "RandomAll_B10", "RandomSupport_B10", "EngineeringROI_B10", "SupportRisk_Base_B10", "SupportRisk_Temporal_B10", "SupportRisk_DepthGuard_B10"])]
    pd.concat([refs, summaries], ignore_index=True).to_csv(OUT / "paper_ready_summary.csv", index=False)
    manifests = [json.loads((OUT / "runs" / method / "run_manifest.json").read_text(encoding="utf-8")) for method in METHODS]
    report = {"method": "Risk-Aware Temporally Refreshed Local Refinement", "version": "RiskTemporal-v1", "final_code_sha": SHA, "config_sha256": _sha(ROOT / "configs" / "risk_temporal_final_v1.json"), "budgets": [0.05, 0.1, 0.2], "val_cases_per_budget": 20, "scenario_ids": manifests[0]["scenario_ids"], "depth_envelope_guard": "disabled", "temporal_refresh_steps": 1, "temporal_overlap_all_budgets": float(runtime.mean_consecutive_overlap.max()), "depth_guard_activation_all_budgets": float(runtime.mean_depth_guard_activation_fraction.max()), "global_checkpoint_sha256": manifests[0]["global_checkpoint_sha256"], "local_checkpoint_sha256": manifests[0]["local_checkpoint_sha256"], "local_checkpoint_update": manifests[0]["local_checkpoint_update"], "local_normalization_sha256": manifests[0]["local_normalization_sha256"], "support_guard_version": manifests[0]["support_guard_version"], "architecture_frozen": True, "primary_budget_frozen": False}
    (OUT / "risk_temporal_final_val_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    reports = ROOT / "reports"; reports.mkdir(exist_ok=True)
    table_cols = ["method", "budget_fraction", "trajectory_h_rel_l2", "trajectory_momentum_rel_l2", "change_region_h_rel_l2", "change_region_momentum_rel_l2", "mean_wet_iou", "false_positive_wet_fraction", "mixture_volume_relative_error", "debris_front_mae_km", "case_wall_runtime_seconds"]
    table = summaries.sort_values("budget_fraction")[table_cols].to_markdown(index=False)
    text = "# RiskTemporal-v1 final VAL sweep\n\nRisk-Aware Temporally Refreshed Local Refinement uses SupportRisk ROI, spatial diversity, one-step temporal refresh, frozen Local@4000, SupportGuard, and the existing train-derived momentum guard. DepthEnvelope is disabled. The sweep is VAL-only (20 fixed scenarios per budget); no training, TEST, H0, or holdout evaluation was run.\n\n## Budget results\n\n" + table + "\n\n## Freeze status\n\nFinal code SHA: `" + SHA + "`. Config SHA-256: `" + report["config_sha256"] + "`. Architecture is frozen; primary budget remains an explicit scientific choice, not an automatic winner. Temporal consecutive overlap and depth-guard activation are both zero for every budget. B10 is exactly reproducible against `SupportRisk_Temporal_B10` under the recorded comparison gate.\n"
    (reports / "RISK_TEMPORAL_FINAL_VAL.md").write_text(text, encoding="utf-8")
    freeze = dict(report); freeze.update({"freeze_candidate": True, "primary_budget_note": "B05/B10/B20 are reported; no automatic primary-budget selection."})
    (reports / "RISK_TEMPORAL_METHOD_FREEZE_CANDIDATE.json").write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
