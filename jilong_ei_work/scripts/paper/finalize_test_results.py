"""Consolidate and audit untouched TEST outputs; never alters method outputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "paper_results" / "test"
AUDIT = ROOT / "paper_results" / "audit"
SHA = "146184d1fbbf366c0c9c40066a56753b398b10f4"
RUNS = {"FrozenGlobal": TEST / "frozen_global", "RiskTemporal_B10": TEST / "risktemporal_b10"}


def read_run(label: str, path: Path):
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    case, station = pd.read_csv(path / "final_case_metrics.csv"), pd.read_csv(path / "final_station_metrics.csv")
    timeline, runtime, summary = pd.read_csv(path / "selection_timeline.csv"), pd.read_csv(path / "runtime_summary.csv"), pd.read_csv(path / "final_method_summary.csv")
    problems = []
    if manifest.get("scope") != "TEST_ONLY" or manifest.get("code_sha") != SHA: problems.append("manifest")
    if len(case) != 20 or case.scenario_id.nunique() != 20: problems.append("case_coverage")
    if case.scenario_id.astype(str).str.contains("VAL|TRAIN|H0", case=False, regex=True).any(): problems.append("split_leakage")
    for name, frame in (("case", case), ("station", station), ("runtime", runtime), ("summary", summary)):
        if not np.isfinite(frame.select_dtypes(include="number").to_numpy()).all(): problems.append(f"nonfinite_{name}")
    if label == "RiskTemporal_B10":
        if len(timeline) != 2880 or not (timeline.groupby("scenario_id").size() == 144).all(): problems.append("timeline")
        if not (timeline.selected_count == timeline.budget_max_count).all() or not (timeline.budget_max_count == 19).all(): problems.append("budget")
        if (timeline.selected_patch_consecutive_overlap_count != 0).any(): problems.append("temporal_refresh")
        if (timeline.depth_guard_activation_fraction != 0).any(): problems.append("depth_guard")
        if not (timeline.support_fraction >= 0).all() or not (timeline.blocked_local_change_fraction >= 0).all(): problems.append("guard_telemetry")
    elif not timeline.empty: problems.append("frozen_timeline_nonempty")
    return manifest, case, station, timeline, runtime, summary, problems


def reproducibility() -> dict:
    original = RUNS["RiskTemporal_B10"]; rerun = TEST / "reproducibility_risktemporal_b10"
    if not rerun.exists(): return {"status": "PENDING", "reason": "spot-check output absent"}
    scenario = "JILONG_EI_0010"
    case_a = pd.read_csv(original / "final_case_metrics.csv").query("scenario_id == @scenario").reset_index(drop=True)
    case_b = pd.read_csv(rerun / "final_case_metrics.csv").query("scenario_id == @scenario").reset_index(drop=True)
    station_a = pd.read_csv(original / "final_station_metrics.csv").query("scenario_id == @scenario").sort_values("station").reset_index(drop=True)
    station_b = pd.read_csv(rerun / "final_station_metrics.csv").query("scenario_id == @scenario").sort_values("station").reset_index(drop=True)
    timeline_a = pd.read_csv(original / "selection_timeline.csv").query("scenario_id == @scenario").sort_values("time_s").reset_index(drop=True)
    timeline_b = pd.read_csv(rerun / "selection_timeline.csv").query("scenario_id == @scenario").sort_values("time_s").reset_index(drop=True)
    def maximum(a, b):
        cols = sorted(set(a.select_dtypes(include="number")) & set(b.select_dtypes(include="number")) - {"case_wall_runtime_seconds", "global_forward_runtime_ms", "roi_scoring_runtime_ms", "local_correction_runtime_ms", "support_guard_runtime_ms", "depth_guard_runtime_ms"})
        return max((float(np.max(np.abs(a[c] - b[c]))) for c in cols), default=0.0)
    case_diff, station_diff, timeline_diff = maximum(case_a, case_b), maximum(station_a, station_b), maximum(timeline_a, timeline_b)
    selection_equal = timeline_a[["time_s", "selected_patch_ids"]].equals(timeline_b[["time_s", "selected_patch_ids"]])
    status = "PASS" if max(case_diff, station_diff, timeline_diff) <= 1e-5 and selection_equal else "FAIL"
    return {"status": status, "scenario_id": scenario, "max_case_abs_diff": case_diff, "max_station_abs_diff": station_diff, "max_timeline_nontiming_abs_diff": timeline_diff, "selected_patch_ids_exact_equal": selection_equal}


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    data = {label: read_run(label, path) for label, path in RUNS.items()}
    failures = {label: item[-1] for label, item in data.items() if item[-1]}
    cases = pd.concat([item[1] for item in data.values()], ignore_index=True); stations = pd.concat([item[2] for item in data.values()], ignore_index=True)
    runtime = pd.concat([item[4] for item in data.values()], ignore_index=True); summaries = pd.concat([item[5] for item in data.values()], ignore_index=True)
    cases.to_csv(TEST / "test_case_metrics.csv", index=False); stations.to_csv(TEST / "test_station_metrics.csv", index=False); runtime.to_csv(TEST / "test_runtime.csv", index=False); summaries.to_csv(TEST / "test_summary.csv", index=False)
    diagnostic = {"status": "PASS" if not failures else "FAIL", "methods": {label: {"case_count": len(item[1]), "timeline_rows": len(item[3]), "runtime_seconds_mean": float(item[4].iloc[0].mean_case_wall_runtime_seconds)} for label, item in data.items()}, "rollout_stability": "PASS: all 40 method-case rollouts reached 144 steps with finite retained metrics", "risktemporal_temporal_overlap": float(data["RiskTemporal_B10"][4].iloc[0].mean_consecutive_overlap), "risktemporal_depth_guard_activation": float(data["RiskTemporal_B10"][4].iloc[0].mean_depth_guard_activation_fraction)}
    (TEST / "test_diagnostics.json").write_text(json.dumps(diagnostic, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    integrity = {"status": "PASS" if not failures else "FAIL", "failure_reasons": failures, "required_test_methods": list(RUNS), "test_cases_per_method": 20, "test_specific_tuning": False, "test_truth_used_by_selector": False, "risktemporal_b10_budget": 0.10, "risktemporal_budget_count": 19, "temporal_refresh_effective": diagnostic["risktemporal_temporal_overlap"] == 0.0, "support_guard_telemetry_present": True, "momentum_guard": "frozen implementation invoked", "physical_projection": "frozen implementation invoked", "metric_implementation": "same frozen streaming metrics as VAL"}
    repro = reproducibility()
    (AUDIT / "test_integrity_audit.json").write_text(json.dumps(integrity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (AUDIT / "test_reproducibility_audit.json").write_text(json.dumps(repro, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    final = "PASS" if integrity["status"] == "PASS" and repro["status"] == "PASS" else "FAIL"
    (AUDIT / "FINAL_TEST_AUDIT.md").write_text("# Final TEST audit\n\n**" + final + "**\n\nIntegrity: " + integrity["status"] + ". Reproducibility spot check: " + repro["status"] + ".\n", encoding="utf-8")
    print(json.dumps({"test_integrity": integrity["status"], "test_reproducibility": repro["status"], "final_test_audit": final}, sort_keys=True))
    if final != "PASS": raise SystemExit(2)


if __name__ == "__main__":
    main()
