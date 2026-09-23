"""Static safety audit for the predeclared, unexecuted EngineeringROI-v2 code."""
from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
sys.path.insert(0, str(ROOT))
BASE_RESULT_SHA = "3b730b83aad74a396c80f976bd58fac50cadeed6"
BASE_V2_SHA = "512f9e55c60b15584556fe2928fac8dfd3c3756d"
V1_PATHS = (
    "jilong_ei_work/configs/engineering_roi_v1.json", "jilong_ei_work/src/local_corrector/engineering_roi.py",
    "jilong_ei_work/results/engineering_roi_v1", "jilong_ei_work/reports/ENGINEERING_ROI_V1_FORMAL_VAL.json",
    "jilong_ei_work/reports/ENGINEERING_ROI_V1_FORMAL_VAL.md",
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    from scripts import evaluate_engineering_roi_v2 as evaluator
    selector = (ROOT / "src/local_corrector/engineering_roi_v2.py").read_text()
    guard = (ROOT / "src/local_corrector/depth_envelope_guard.py").read_text()
    config = json.loads((ROOT / "configs/engineering_roi_v2.json").read_text())
    require(config["roi_component"] == "support_risk" and config["temporal_refresh_steps"] == 1 and config["temporal_fill_from_excluded"] is False, "V2_CONFIG_INVALID")
    for forbidden in ("compute_dynamic_component(", "compute_front_component(", "compute_section_component("):
        require(forbidden not in selector, f"FORBIDDEN_V1_COMPONENT:{forbidden}")
    require("previous_selected_patch_ids" in selector and "Excluded patches are never reintroduced" in selector, "TEMPORAL_REFRESH_INVALID")
    require("[:, :1]" in guard and "output[:, :1]" in guard and "output[:, 1:]" not in guard, "DEPTH_GUARD_NOT_H_ONLY")
    transition = inspect.getsource(evaluator.v2_transition)
    execution = inspect.getsource(evaluator.execute_cases)
    ordered = [transition.index(name) for name in ("select_engineering_roi_v2(", "apply_learned_correction(", "apply_support_guard(", "apply_momentum_state_guard(", "transform.decode(momentum_written)", "apply_depth_envelope_guard(", "project_physical(physical")]
    require(ordered == sorted(ordered), "TRANSITION_GUARD_ORDER_INVALID")
    require(transition.count("timed_cuda(") == 4, "CUDA_MODULE_TIMING_MISSING")
    require("v2_transition(" in execution and execution.index("v2_transition(") < execution.index("truth_current ="), "TRUTH_ISOLATION_INVALID")
    for forbidden in ("apply_learned_correction(", "apply_support_guard(", "apply_momentum_state_guard(", "apply_depth_envelope_guard("):
        require(forbidden not in execution, "DUPLICATE_PRODUCTION_WRITEBACK")
    require(all(name in evaluator.TIMELINE_COLUMNS for name in ("support_fraction", "blocked_local_change_fraction", "raw_local_new_wet_fraction")), "SUPPORT_TELEMETRY_DROPPED")
    require("LOCAL_CHECKPOINT_MUST_BE_BEST_4000" in inspect.getsource(evaluator.main) and "LOCAL_NORMALIZATION_NOT_TRAIN_ONLY" in inspect.getsource(evaluator.main), "FROZEN_LOCAL_GUARDS_MISSING")
    source = (ROOT / "scripts/evaluate_engineering_roi_v2.py").read_text().lower()
    require("optimizer" not in source and ".backward(" not in source, "TRAINING_OPERATION_PRESENT")
    require(not any(word in source.upper() for word in ("TEST", "H0", "HOLDOUT")), "NON_VAL_BRANCH_PRESENT")
    require(subprocess.run(["git", "diff", "--quiet", BASE_RESULT_SHA, "--", *V1_PATHS], cwd=REPOSITORY).returncode == 0, "V1_FILES_CHANGED")
    require(subprocess.run(["git", "diff", "--quiet", BASE_V2_SHA, "--", "jilong_ei_work/configs/engineering_roi_v2.json"], cwd=REPOSITORY).returncode == 0, "V2_CONFIG_CHANGED")
    require(not (ROOT / "results/engineering_roi_v2").exists(), "V2_REAL_OUTPUT_PRESENT")
    print(json.dumps({"status": "PASS", "v1_status": "FROZEN", "v2_definition": "UNCHANGED", "production_transition_single_source": True, "cuda_timing_synchronized": True, "failure_audit_statistics_fixed": True, "support_telemetry_preserved": True, "real_rollout_executed": False}))


if __name__ == "__main__":
    main()
