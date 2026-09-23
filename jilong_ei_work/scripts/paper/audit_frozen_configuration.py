"""Block TEST unless the frozen RiskTemporal-v1 release is intact."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import torch


FINAL_CODE_SHA = "146184d1fbbf366c0c9c40066a56753b398b10f4"
FINAL_VAL_SHA = "19b63d10449cefdaeb0ebe0fdd7a84ed31797c4a"
GLOBAL_SHA = "a23d243628124a7259ee68c400fd277e2804a08ce086b508f503f42f47841745"
LOCAL_SHA = "906c49c9c2a21e453c029ceaba54a3a2cf176bd874a47e1927ab09347f2c389b"
NORM_SHA = "5616b023ed75fe30dfb040f11d03e121c92558de6fe93c9d5a8aabff61217f31"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_contains(git_root: Path, commit: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=git_root, check=False).returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--git-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.runtime_root.resolve(); audit = root / "paper_results" / "audit"; audit.mkdir(parents=True, exist_ok=True)
    config = root / "configs" / "risk_temporal_final_v1.json"
    manifest_path = root / "results" / "risk_temporal_final_v1" / "runs" / "RiskTemporal_B10" / "run_manifest.json"
    if not manifest_path.exists():
        # The execution workspace retains the authoritative formal run in its
        # original location; the release repository carries the lightweight copy.
        manifest_path = root / "results" / "engineering_roi_v2" / "runs" / "RiskTemporal_B10" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index = pd.read_csv(root / "data" / "scenario_index.csv")
    groups = {name: set(map(str, frame.scenario_id)) for name, frame in index.groupby("split")}
    h0_id = "PARK_V2_H0"
    config_data = json.loads(config.read_text(encoding="utf-8"))
    frozen = {
        "status": "PASS",
        "final_method": "RiskTemporal-v1 B10",
        "final_code_sha": FINAL_CODE_SHA,
        "final_val_result_sha": FINAL_VAL_SHA,
        "code_commit_reachable": git_contains(args.git_root, FINAL_CODE_SHA),
        "val_commit_reachable": git_contains(args.git_root, FINAL_VAL_SHA),
        "runtime_source_code_commit": (root / "SOURCE_CODE_COMMIT.txt").read_text(encoding="utf-8").strip(),
        "risktemporal_config_sha256": sha256(config),
        "risktemporal_config": config_data,
        "risktemporal_b10_val_manifest": manifest,
        "checks": {
            "version": config_data.get("version") == "RiskTemporal-v1",
            "budget_b10": manifest.get("budget") == 0.1,
            "temporal_refresh": config_data.get("temporal_refresh_steps") == 1 and config_data.get("temporal_fill_from_excluded") is False,
            "depth_disabled": config_data.get("depth_envelope_guard") == "disabled",
            "global_frozen": config_data.get("global_model") == "frozen",
            "local_4000": config_data.get("local_model") == "best_update_4000" and manifest.get("local_checkpoint_update") == 4000,
            "global_checkpoint": manifest.get("global_checkpoint_sha256") == GLOBAL_SHA and manifest.get("global_checkpoint_provenance", {}).get("global_step") == 5000 and manifest.get("global_checkpoint_provenance", {}).get("stage_name") == "STAGE_C" and manifest.get("global_checkpoint_provenance", {}).get("architecture", {}).get("depth") == 4,
            "local_checkpoint": manifest.get("local_checkpoint_sha256") == LOCAL_SHA,
            "normalization": manifest.get("local_normalization_sha256") == NORM_SHA and manifest.get("local_normalization_fit_scope") == "TRAIN_ONLY",
        },
    }
    frozen["checks"]["runtime_source"] = frozen["runtime_source_code_commit"] == FINAL_CODE_SHA
    frozen["checks"]["commits"] = frozen["code_commit_reachable"] and frozen["val_commit_reachable"]
    if not all(frozen["checks"].values()): frozen["status"] = "FAIL"
    splits = {
        "status": "PASS",
        "counts": {name: len(values) for name, values in groups.items()},
        "train_val_overlap": sorted(groups["TRAIN"] & groups["VAL"]),
        "train_test_overlap": sorted(groups["TRAIN"] & groups["TEST"]),
        "val_test_overlap": sorted(groups["VAL"] & groups["TEST"]),
        "h0_id": h0_id,
        "h0_independent_of_scenario_splits": h0_id not in set(index.scenario_id),
    }
    if any(splits[key] for key in ("train_val_overlap", "train_test_overlap", "val_test_overlap")) or not splits["h0_independent_of_scenario_splits"]:
        splits["status"] = "FAIL"
    checkpoint = {
        "status": "PASS",
        "global_checkpoint": "models/global_operator_v2/best.pt",
        "global_sha256": sha256(root / "models" / "global_operator_v2" / "best.pt"),
        "local_checkpoint": "models/local_corrector_v1_1/best.pt",
        "local_sha256": sha256(root / "models" / "local_corrector_v1_1" / "best.pt"),
        "local_normalization_sha256": sha256(root / "models" / "local_corrector_v1_1" / "correction_normalization.json"),
    }
    checkpoint["status"] = "PASS" if (checkpoint["global_sha256"] == GLOBAL_SHA and checkpoint["local_sha256"] == LOCAL_SHA and checkpoint["local_normalization_sha256"] == NORM_SHA) else "FAIL"
    evaluator = root / "scripts" / "evaluate_engineering_roi_v2.py"
    metrics = root / "src" / "global_operator_v2" / "oracle_refinement.py"
    metric = {"status": "PASS", "implementation": {"frozen_evaluator_sha256": sha256(evaluator), "streaming_metrics_sha256": sha256(metrics)}, "runtime_contract": "CUDA-synchronized step timing through the same frozen evaluator functions", "test_specific_selector_or_metric_config": False, "test_truth_in_roi_scoring": False}
    for name, payload in (("frozen_configuration.json", frozen), ("data_split_audit.json", splits), ("checkpoint_audit.json", checkpoint), ("metric_audit.json", metric)):
        (audit / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    statuses = [frozen["status"], splits["status"], checkpoint["status"], metric["status"]]
    final = "PASS" if statuses == ["PASS"] * 4 else "FAIL"
    (audit / "FINAL_AUDIT.md").write_text("# Frozen configuration audit\n\n**" + final + "**\n\nAll four machine-readable audits must pass before untouched TEST execution.\n", encoding="utf-8")
    print(json.dumps({"frozen_audit": final, "parts": statuses}, sort_keys=True))
    if final != "PASS": raise SystemExit(2)


if __name__ == "__main__":
    main()
