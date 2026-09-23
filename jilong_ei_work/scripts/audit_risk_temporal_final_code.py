"""Static audit for the frozen RiskTemporal-v1 candidate before VAL."""
from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

EXPECTED = {
    "version": "RiskTemporal-v1", "scientific_name": "Risk-Aware Temporally Refreshed Local Refinement",
    "wet_threshold_m": .05, "shallow_margin_upper_h_m": .15, "roi_component": "support_risk",
    "support_risk_definition": "new_wet_or_shallow_margin", "normalization": "positive_percentile_rank",
    "spatial_diversity": "nonadjacent_first_then_fill", "temporal_refresh": "exclude_previous_step_selected_patches",
    "temporal_refresh_steps": 1, "temporal_fill_from_excluded": False, "depth_envelope_guard": "disabled",
    "budgets": [.05, .10, .20], "global_model": "frozen", "local_model": "best_update_4000",
    "support_guard": "current_or_global_provisional_wet", "momentum_guard": "existing_train_derived_absolute_guard",
    "truth_in_roi_scoring": False,
}


def require(value, message):
    if not value: raise AssertionError(message)


def main():
    from scripts import evaluate_engineering_roi_v2 as evaluator
    config = json.loads((ROOT / "configs/risk_temporal_final_v1.json").read_text())
    selector = (ROOT / "src/local_corrector/engineering_roi_v2.py").read_text()
    source = (ROOT / "scripts/evaluate_engineering_roi_v2.py").read_text()
    require(config == EXPECTED, "RISK_TEMPORAL_FINAL_CONFIG_INVALID")
    require(evaluator.METHODS.get("RiskTemporal") == (None, True, False), "RISK_TEMPORAL_METHOD_MAPPING_INVALID")
    for forbidden in ("compute_dynamic_component(", "compute_front_component(", "compute_section_component("):
        require(forbidden not in selector, "FINAL_SELECTOR_NOT_SUPPORTRISK_ONLY")
    transition = inspect.getsource(evaluator.v2_transition)
    require("apply_support_guard(" in transition and "apply_momentum_state_guard(" in transition, "FINAL_GUARDS_MISSING")
    require("if use_depth:" in transition and "RiskTemporal\": (None, True, False)" in source, "FINAL_DEPTH_SEMANTICS_INVALID")
    require("LOCAL_CHECKPOINT_MUST_BE_BEST_4000" in source and "LOCAL_NORMALIZATION_NOT_TRAIN_ONLY" in source, "FINAL_FROZEN_LOCAL_INVALID")
    lowered = source.lower(); require("optimizer" not in lowered and ".backward(" not in lowered, "TRAINING_OPERATION_PRESENT")
    require(not any(word in source.upper() for word in ("TEST", "H0", "HOLDOUT")), "NON_VAL_BRANCH_PRESENT")
    require(subprocess.run(["git", "diff", "--quiet", "a82a8a76667b563362cca3a2930c815e2474f8de", "--", "configs/engineering_roi_v2.json", "src/local_corrector/engineering_roi_v2.py", "src/local_corrector/depth_envelope_guard.py", "src/local_corrector/support_guard.py"], cwd=ROOT).returncode == 0, "EXISTING_METHOD_CODE_CHANGED")
    print(json.dumps({"FINAL_CANDIDATE_CODE_AUDIT": "PASS", "risk_temporal_mapping": "temporal_on_depth_off", "new_training": False}))


if __name__ == "__main__": main()
