"""Synthetic, no-model tests for the predeclared EngineeringROI-v2 path."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.global_operator_v2.losses import project_physical
from src.global_operator_v2.oracle_refinement import PatchLayout
from src.local_corrector.depth_envelope_guard import apply_depth_envelope_guard
from src.local_corrector.engineering_roi import build_roi_static_metadata, compute_support_risk_component
from src.local_corrector.engineering_roi_v2 import initial_temporal_state, load_engineering_roi_v2_config, select_engineering_roi_v2
from scripts.audit_engineering_roi_v1_failure_mechanism import persistence


def metadata_for(active_np, rows=4, cols=4):
    active = torch.as_tensor(active_np, dtype=torch.bool)[None, None]
    layout = PatchLayout(active_np, rows, cols)
    return build_roi_static_metadata(layout, active, np.zeros(active_np.shape, np.float32), torch.zeros(active_np.shape, dtype=torch.bool))


def selected_ids(selected):
    return {patch.patch_id for patch in selected}


def test_config_and_signature(config):
    assert config["temporal_refresh_steps"] == 1 and config["temporal_fill_from_excluded"] is False
    names = set(inspect.signature(select_engineering_roi_v2).parameters)
    assert not names.intersection({"truth", "target", "future", "local_prediction", "oracle_error"})


def test_support_risk_boundaries(config):
    active = np.ones((4, 4), bool); metadata = metadata_for(active)
    current = torch.zeros(1, 6, 4, 4); provisional = current.clone()
    provisional[:, 0, 0, 0] = .05; provisional[:, 0, 0, 1] = .149
    provisional[:, 0, 0, 2] = .15; current[:, 0, 0, 2] = .15
    provisional[:, 0, 0, 3] = .20; current[:, 0, 0, 3] = .20
    raw = compute_support_risk_component(current, provisional, metadata, config)
    assert raw[0].item() == 1.0 and raw[1].item() == 1.0 and raw[2].item() == 0.0 and raw[3].item() == 0.0


def test_selection_temporal_and_diversity(config):
    active = np.ones((8, 8), bool); metadata = metadata_for(active)
    current = torch.zeros(1, 6, 8, 8); provisional = current.clone(); provisional[:, 0] = .10
    first, state1, telemetry1 = select_engineering_roi_v2(current, provisional, metadata, config, .10, initial_temporal_state())
    again, _, _ = select_engineering_roi_v2(current, provisional, metadata, config, .10, initial_temporal_state())
    second, state2, telemetry2 = select_engineering_roi_v2(current, provisional, metadata, config, .10, state1)
    third, _, _ = select_engineering_roi_v2(current, provisional, metadata, config, .10, state2)
    assert selected_ids(first) == selected_ids(again) and selected_ids(first).isdisjoint(selected_ids(second))
    assert telemetry2["selected_patch_consecutive_overlap_count"] == 0
    assert selected_ids(first).intersection(selected_ids(third)), "one-step exclusion must expire at t+2"
    assert telemetry1["selected_count"] <= telemetry1["budget_max_count"]
    for left in first:
        for right in first:
            if left.patch_id < right.patch_id:
                assert max(abs(left.row_id-right.row_id), abs(left.col_id-right.col_id)) >= 2
    reset, _, _ = select_engineering_roi_v2(current, provisional, metadata, config, .10, initial_temporal_state())
    assert selected_ids(reset) == selected_ids(first)
    one_active = np.zeros((4, 4), bool); one_active[0, 0] = True; one = metadata_for(one_active)
    c1 = torch.zeros(1, 6, 4, 4); p1 = c1.clone(); p1[:, 0, 0, 0] = .10
    chosen, state, _ = select_engineering_roi_v2(c1, p1, one, config, .10, initial_temporal_state())
    empty, _, empty_telemetry = select_engineering_roi_v2(c1, p1, one, config, .10, state)
    assert len(chosen) == 1 and not empty and empty_telemetry["candidate_count"] == 0
    zeros = torch.zeros_like(provisional)
    none, _, none_telemetry = select_engineering_roi_v2(zeros, zeros, metadata, config, .10, initial_temporal_state())
    assert not none and none_telemetry["selected_count"] == 0


def test_depth_envelope():
    active = torch.tensor([[[[True, True], [True, False]]]])
    current = torch.full((1, 6, 2, 2), .10); provisional = torch.full((1, 6, 2, 2), .20)
    raw = torch.full((1, 6, 2, 2), .30); raw[:, 1:3] = .7; raw[:, 3] = .8; raw[:, 4] = .6; raw[:, 5] = -.4; raw[:, 0, 1, 1] = .50
    upper, telemetry = apply_depth_envelope_guard(current, provisional, raw, active)
    assert np.isclose(upper[:, 0, 0, 0].item(), .20) and np.isclose(upper[:, 0, 1, 1].item(), .50)
    assert torch.equal(upper[:, 1:], raw[:, 1:]) and telemetry["depth_guard_activation_fraction"] == 1.0
    assert np.isclose(telemetry["depth_guard_mean_abs_clip_m"], .10) and np.isclose(telemetry["depth_guard_max_abs_clip_m"], .10)
    down = torch.full((1, 6, 2, 2), .05); lower, _ = apply_depth_envelope_guard(provisional, current, down, active)
    assert np.isclose(lower[:, 0, 0, 0].item(), .10)
    inside = torch.full((1, 6, 2, 2), .15); unchanged, inside_telemetry = apply_depth_envelope_guard(current, provisional, inside, active)
    assert torch.equal(unchanged, inside) and inside_telemetry["depth_guard_activation_fraction"] == 0.0
    equal, _ = apply_depth_envelope_guard(current, current, raw, active)
    assert np.isclose(equal[:, 0, 0, 0].item(), .10)
    projected = project_physical(upper, active)
    assert projected[:, 0].min().item() >= 0 and torch.all(projected[:, 4] <= projected[:, 3])


def test_persistence_fixture():
    result = persistence([{1, 2, 3}, {2, 3, 4}, {3, 5}])
    assert result["total_selections"] == 8 and result["consecutive_reselection_fraction"] == .375
    assert result["top1_selection_share"] == .375 and result["maximum_selection_count"] == 3
    assert result["mean_revisit_gap_steps"] > 0 and result["mean_longest_consecutive_streak"] > 1


def main():
    config, _ = load_engineering_roi_v2_config(ROOT / "configs/engineering_roi_v2.json")
    test_config_and_signature(config); test_support_risk_boundaries(config)
    test_selection_temporal_and_diversity(config); test_depth_envelope(); test_persistence_fixture()
    print("PASS EngineeringROI-v2 synthetic selector, refresh, envelope, and audit tests")


if __name__ == "__main__":
    main()
