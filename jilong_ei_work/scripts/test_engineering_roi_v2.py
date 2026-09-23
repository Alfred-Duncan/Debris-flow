"""Synthetic, no-model tests for the predeclared EngineeringROI-v2 path."""
from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.global_operator_v2.losses import project_physical
from src.global_operator_v2.oracle_refinement import PatchLayout
from src.local_corrector.depth_envelope_guard import apply_depth_envelope_guard
from src.local_corrector.engineering_roi import build_roi_static_metadata, compute_support_risk_component
from src.local_corrector.engineering_roi_v2 import initial_temporal_state, load_engineering_roi_v2_config, select_engineering_roi_v2
from src.local_corrector.support_guard import apply_support_guard
from scripts import evaluate_engineering_roi_v2 as evaluator
from scripts.audit_engineering_roi_v1_failure_mechanism import _associations, persistence


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
    base_first, base_state, _ = select_engineering_roi_v2(current, provisional, metadata, config, .10, initial_temporal_state(), temporal_refresh=False)
    base_second, _, _ = select_engineering_roi_v2(current, provisional, metadata, config, .10, base_state, temporal_refresh=False)
    assert selected_ids(base_first).intersection(selected_ids(base_second)), "refresh-off ablations may repeat patches"
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
    result = persistence([{1, 2, 3}, {2, 3, 4}, {3, 5}], eligible_patch_count=10)
    assert result["total_selections"] == 8 and result["unique_patch_fraction"] == .5
    assert result["consecutive_overlap_count"] == 3 and result["consecutive_denominator_count"] == 5 and result["consecutive_reselection_fraction"] == .6
    assert result["top1_selection_share"] == .375 and result["maximum_selection_count"] == 3
    assert result["mean_revisit_gap_steps"] > 0 and result["mean_longest_consecutive_streak"] > 1
    coverage = persistence([set(range(4)) for _ in range(25)], eligible_patch_count=10)
    assert coverage["total_selections"] == 100 and coverage["unique_patch_fraction"] == .4


def test_support_guard_physical_telemetry():
    active = torch.ones(1, 1, 1, 1, dtype=torch.bool); current = torch.zeros(1, 6, 1, 1); provisional = current.clone()
    encoded_global = torch.zeros_like(current); encoded_local = encoded_global.clone(); encoded_local[:, :1] = .01
    physical_local = current.clone(); physical_local[:, :1] = .10
    _, telemetry = apply_support_guard(encoded_global, encoded_local, current, provisional, active, local_corrected_physical=physical_local)
    assert telemetry["blocked_wet_creation_count"] == 1 and telemetry["raw_local_new_wet_fraction"] == 1.0


def test_support_guard_encoded_physical_disagreement_and_invariance():
    active = torch.ones(1, 1, 1, 1, dtype=torch.bool); current = torch.zeros(1, 6, 1, 1); provisional = current.clone()
    encoded_global = torch.zeros_like(current); encoded_local = encoded_global.clone(); encoded_local[:, :1] = .20
    physical_local = current.clone(); physical_local[:, :1] = .01
    old_written, old_telemetry = apply_support_guard(encoded_global, encoded_local, current, provisional, active)
    new_written, new_telemetry = apply_support_guard(encoded_global, encoded_local, current, provisional, active, local_corrected_physical=physical_local)
    assert torch.equal(old_written, new_written)
    assert old_telemetry["blocked_wet_creation_count"] == 1 and new_telemetry["blocked_wet_creation_count"] == 0 and new_telemetry["raw_local_new_wet_fraction"] == 0.0


def test_timing_contract_and_telemetry_schema():
    timing = {"method": "EngineeringROI_v2_B10", "scenario_id": "case", "case_wall_runtime_seconds": 1.0, **{key: [1.0] * 144 for key in evaluator.TIMING_STEP_KEYS}}
    assert evaluator.validate_case_timing_record(timing) == timing
    for malformed, expected in (({"parts": []}, "V2_TIMING_CACHE_SCHEMA_INVALID"), ({**timing, "scenario_id": ""}, "V2_CASE_TIMING_INVALID"), ({**timing, "roi_steps_ms": [1.0]}, "V2_CASE_TIMING_INVALID")):
        try: evaluator.validate_case_timing_record(malformed)
        except RuntimeError as error: assert str(error) == expected
        else: raise AssertionError("malformed timing accepted")
    row = {column: 0.0 for column in evaluator.TIMELINE_COLUMNS}; row.update({"method": "x", "scenario_id": "y", "selected_patch_ids": "", "support_fraction": .7, "blocked_local_change_fraction": .2})
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "timeline.csv"; pd.DataFrame([row], columns=evaluator.TIMELINE_COLUMNS).to_csv(path, index=False)
        restored = pd.read_csv(path); assert np.isclose(restored.support_fraction.iloc[0], .7) and np.isclose(restored.blocked_local_change_fraction.iloc[0], .2)


def test_association_and_single_source_contract():
    frame = pd.DataFrame({"method": ["A", "A", "B", "B"], "trajectory_h_rel_l2": [1., 2., 3., 4.], "top10_share": [.1, .2, .3, .4], "consecutive_reselection_fraction": [.1, .2, .3, .4], "mean_selection_count_per_used_patch": [1., 2., 3., 4.], "mean_longest_consecutive_streak": [1., 2., 3., 4.]})
    pooled = _associations(frame); within = [record for method, group in frame.groupby("method") for record in _associations(group, method)]
    assert pooled and within and all("method" in record for record in within)
    source = inspect.getsource(evaluator.execute_cases)
    assert "v2_transition(" in source
    assert not any(name in source for name in ("apply_learned_correction(", "apply_support_guard(", "apply_momentum_state_guard(", "apply_depth_envelope_guard("))


def test_cuda_timing_and_transition_order():
    calls = []; original_sync = evaluator.torch.cuda.synchronize
    evaluator.torch.cuda.synchronize = lambda device: calls.append("sync")
    try: evaluator.timed_cuda(lambda: calls.append("fn"), torch.device("cuda"))
    finally: evaluator.torch.cuda.synchronize = original_sync
    assert calls == ["sync", "fn", "sync"]
    order = []; originals = {name: getattr(evaluator, name) for name in ("select_engineering_roi_v2", "build_features", "apply_learned_correction", "apply_support_guard", "apply_momentum_state_guard", "apply_depth_envelope_guard", "project_physical")}
    class Transform:
        def decode(self, value): order.append("decode"); return value
    metadata = type("Metadata", (), {"active": torch.ones(1, 1, 1, 1, dtype=torch.bool)})()
    try:
        evaluator.select_engineering_roi_v2 = lambda *args, **kwargs: (("patch",), "next", {"selected_count": 1}) if not order.append("select") else None
        evaluator.build_features = lambda *args, **kwargs: "features"
        evaluator.apply_learned_correction = lambda *args, **kwargs: (order.append("local") or args[2], None, None)
        evaluator.apply_support_guard = lambda *args, **kwargs: (order.append("support") or args[0], {"support_fraction": 0., "support_cell_count": 0, "blocked_local_change_fraction": 0., "blocked_local_change_cell_count": 0, "blocked_wet_creation_count": 0, "raw_local_new_wet_fraction": 0.})
        evaluator.apply_momentum_state_guard = lambda value, *args: order.append("momentum") or value
        evaluator.apply_depth_envelope_guard = lambda *args: (order.append("depth") or args[2], {"depth_guard_activation_fraction": 0., "depth_guard_mean_abs_clip_m": 0., "depth_guard_max_abs_clip_m": 0.})
        evaluator.project_physical = lambda value, *args: order.append("project") or value
        value = torch.zeros(1, 6, 1, 1); delta = type("Delta", (), {"scales": [1.] * 6})()
        evaluator.v2_transition(value, value, value, value, value, metadata, {}, .1, None, "EngineeringROI_v2", None, None, delta, {"scales": [1.] * 6, "bounds": [1.] * 6}, None, None, 0., Transform(), None, None, torch.device("cpu"))
    finally:
        for name, value in originals.items(): setattr(evaluator, name, value)
    assert order == ["select", "local", "decode", "project", "support", "momentum", "decode", "depth", "project"]
    assert "local_corrected_physical=raw_local_physical" in inspect.getsource(evaluator.v2_transition)


def main():
    config, _ = load_engineering_roi_v2_config(ROOT / "configs/engineering_roi_v2.json")
    test_config_and_signature(config); test_support_risk_boundaries(config)
    test_selection_temporal_and_diversity(config); test_depth_envelope(); test_persistence_fixture()
    test_support_guard_physical_telemetry(); test_support_guard_encoded_physical_disagreement_and_invariance()
    test_timing_contract_and_telemetry_schema(); test_association_and_single_source_contract(); test_cuda_timing_and_transition_order()
    print("PASS EngineeringROI-v2 synthetic selector, refresh, envelope, and audit tests")


if __name__ == "__main__":
    main()
