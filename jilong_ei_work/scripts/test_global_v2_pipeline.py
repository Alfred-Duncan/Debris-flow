"""Assertion tests for the serializable V2 training curriculum."""
from __future__ import annotations

import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.global_operator_v2.pipeline import (
    PipelineState,
    apply_capacity_result,
    capacity_schedule,
    checkpoint_agreement,
    effective_stage_names,
    load_state,
    next_effective_stage,
    save_state,
    stage_config,
)


PROBE_AB = [
    {"K": 1, "status": "SAFE", "safe": True},
    {"K": 2, "status": "SAFE", "safe": True},
    {"K": 4, "status": "OOM", "safe": False},
]
PROBE_ABCD = [
    {"K": k, "status": "SAFE", "safe": True} for k in (1, 2, 4, 6)
]


def test_persistence():
    state = PipelineState(
        "v2", "abc", effective_curriculum=capacity_schedule(PROBE_AB),
        max_safe_k=2, total_planned_updates=4000, stage_best_score=0.75,
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "run_state.json"
        save_state(path, state)
        restored = load_state(path, PipelineState("default", "default"))
    assert restored.effective_curriculum == state.effective_curriculum
    assert restored.max_safe_k == state.max_safe_k
    assert restored.total_planned_updates == state.total_planned_updates
    assert restored.stage_best_score == state.stage_best_score


def test_capacity_mapping_and_helpers():
    expected = [
        {"stage": "STAGE_A", "k": 1, "updates": 2000},
        {"stage": "STAGE_B", "k": 2, "updates": 2000},
    ]
    assert capacity_schedule(PROBE_AB) == expected
    state = apply_capacity_result(PipelineState("v2", "abc"), PROBE_AB)
    assert state.max_safe_k == 2
    assert state.total_planned_updates == 4000
    assert effective_stage_names(state) == ["STAGE_A", "STAGE_B"]
    assert stage_config(state, "STAGE_B") == expected[1]
    assert stage_config(state, "STAGE_C") is None
    assert next_effective_stage(state, "ARCHITECTURE") == "STAGE_A"
    assert next_effective_stage(state, "STAGE_A") == "STAGE_B"
    assert next_effective_stage(state, "STAGE_B") == "VAL_CONFIRM"
    assert next_effective_stage(state, "VAL_CONFIRM") == "FREEZE"
    assert next_effective_stage(state, "STAGE_C") is None


def test_full_capacity_transitions():
    state = apply_capacity_result(PipelineState("v2", "abc"), PROBE_ABCD)
    assert state.total_planned_updates == 14000
    assert next_effective_stage(state, "ARCHITECTURE") == "STAGE_A"
    assert next_effective_stage(state, "STAGE_A") == "STAGE_B"
    assert next_effective_stage(state, "STAGE_B") == "STAGE_C"
    assert next_effective_stage(state, "STAGE_C") == "STAGE_D"
    assert next_effective_stage(state, "STAGE_D") == "VAL_CONFIRM"
    assert next_effective_stage(state, "VAL_CONFIRM") == "FREEZE"


def test_no_safe_capacity():
    try:
        capacity_schedule([{"K": 1, "status": "OOM", "safe": False}])
    except RuntimeError as error:
        assert str(error) == "NO_SAFE_CAPACITY"
    else:
        raise AssertionError("capacity_schedule accepted an unsafe probe")


def test_checkpoint_agreement():
    state = PipelineState(
        "v2", "abc", architecture={"width": 32}, current_stage="STAGE_B",
        stage_step=500, global_step=2500, stage_updates_total=2000,
    )
    checkpoint = {
        "stage_name": "STAGE_B", "stage_step": 500, "global_step": 2500,
        "stage_updates_total": 2000, "config_hash": "abc",
        "architecture": {"width": 32},
    }
    checkpoint_agreement(state, checkpoint)
    checkpoint["stage_updates_total"] = 4000
    try:
        checkpoint_agreement(state, checkpoint)
    except RuntimeError as error:
        assert "RUN_STATE_CHECKPOINT_MISMATCH" in str(error)
    else:
        raise AssertionError("checkpoint mismatch was accepted")


if __name__ == "__main__":
    test_persistence()
    test_capacity_mapping_and_helpers()
    test_full_capacity_transitions()
    test_no_safe_capacity()
    test_checkpoint_agreement()
