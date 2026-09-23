"""Static safety audit for the predeclared, unexecuted EngineeringROI-v2 code."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
BASE_RESULT_SHA = "3b730b83aad74a396c80f976bd58fac50cadeed6"
V1_PATHS = (
    "jilong_ei_work/configs/engineering_roi_v1.json",
    "jilong_ei_work/src/local_corrector/engineering_roi.py",
    "jilong_ei_work/results/engineering_roi_v1",
    "jilong_ei_work/reports/ENGINEERING_ROI_V1_FORMAL_VAL.json",
    "jilong_ei_work/reports/ENGINEERING_ROI_V1_FORMAL_VAL.md",
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    selector = (ROOT / "src/local_corrector/engineering_roi_v2.py").read_text()
    guard = (ROOT / "src/local_corrector/depth_envelope_guard.py").read_text()
    evaluator = (ROOT / "scripts/evaluate_engineering_roi_v2.py").read_text()
    config = json.loads((ROOT / "configs/engineering_roi_v2.json").read_text())
    require(config["roi_component"] == "support_risk" and config["temporal_refresh_steps"] == 1, "V2_CONFIG_INVALID")
    require(config["temporal_fill_from_excluded"] is False and config["local_model"] == "best_update_4000", "V2_CONFIG_INVALID")
    for forbidden in ("compute_dynamic_component(", "compute_front_component(", "compute_section_component("):
        require(forbidden not in selector, f"FORBIDDEN_V1_COMPONENT:{forbidden}")
    require("previous_selected_patch_ids" in selector and "temporal_refresh: bool" in selector, "TEMPORAL_REFRESH_MISSING")
    require("Excluded patches are never reintroduced" in selector, "TEMPORAL_FALLBACK_NOT_EXPLICITLY_PROHIBITED")
    require("[:, :1]" in guard and "output[:, :1]" in guard and "output[:, 1:]" not in guard, "DEPTH_GUARD_NOT_H_ONLY")
    start = selector.index("def select_engineering_roi_v2")
    signature = selector[start:selector.index("):\n", start) + 2]
    require(not any(word in signature for word in ("truth", "target", "future", "oracle")), "SELECTION_TRUTH_INTERFACE")
    require("LOCAL_CHECKPOINT_MUST_BE_BEST_4000" in evaluator and "LOCAL_NORMALIZATION_NOT_TRAIN_ONLY" in evaluator, "FROZEN_LOCAL_GUARDS_MISSING")
    require("apply_support_guard(" in evaluator and "apply_momentum_state_guard(" in evaluator, "EXISTING_GUARDS_MISSING")
    support_at, momentum_at, decode_at = evaluator.index("apply_support_guard("), evaluator.index("apply_momentum_state_guard("), evaluator.index("transform.decode(momentum_written)")
    require(support_at < momentum_at < decode_at < evaluator.index("apply_depth_envelope_guard(", decode_at), "GUARD_ORDER_INVALID")
    require(evaluator.index("corrected = project_physical") < evaluator.index("truth_current ="), "TRUTH_ISOLATION_ORDER_INVALID")
    lowered = evaluator.lower()
    require("optimizer" not in lowered and ".backward(" not in lowered, "TRAINING_OPERATION_PRESENT")
    require(not any(word in evaluator for word in ("TEST", "H0", "HOLDOUT")), "NON_VAL_BRANCH_PRESENT")
    changed = subprocess.run(["git", "diff", "--quiet", BASE_RESULT_SHA, "--", *V1_PATHS], cwd=REPOSITORY).returncode
    require(changed == 0, "V1_FILES_CHANGED")
    require(not (ROOT / "results/engineering_roi_v2").exists(), "V2_REAL_OUTPUT_PRESENT")
    print(json.dumps({"status": "PASS", "v1_status": "FROZEN", "v2_definition": "PREDECLARED", "new_training": False, "real_rollout_executed": False}))


if __name__ == "__main__":
    main()
