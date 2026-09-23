"""Future VAL-only executor for the predeclared EngineeringROI-v2 design.

This file is deliberately an execution implementation, rather than a design
stub.  It is not invoked by the V2 code-review workflow: calling it is a
separate, explicitly authorized future rollout action.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.provenance import resolve_code_sha, sha256_file
from src.global_operator_v2.dataset import FrameStore, INPUT, build_features, scenario_rows
from src.global_operator_v2.frame_adapter import static_and_exogenous
from src.global_operator_v2.losses import project_physical
from src.global_operator_v2.metrics import load_transects
from src.global_operator_v2.oracle_refinement import PatchLayout, StreamingMetrics
from src.local_corrector.depth_envelope_guard import apply_depth_envelope_guard
from src.local_corrector.engineering_roi import build_roi_static_metadata, build_section_mask
from src.local_corrector.engineering_roi_execution import (
    cached_engineering_method, create_or_validate_manifest, partition_rows,
    run_output_path, validate_shard_coverage, validate_shard_manifests,
    write_run_complete,
)
from src.local_corrector.engineering_roi_v2 import (
    initial_temporal_state, load_engineering_roi_v2_config, select_engineering_roi_v2,
)
from src.local_corrector.model import JilongLocalCorrector
from src.local_corrector.support_guard import apply_support_guard
from src.local_corrector.trainer import apply_learned_correction, apply_momentum_state_guard
from scripts.run_global_v2_oracle_refinement import (
    append_station, load_model, predict, station_row, station_summary, summarize,
)

SEED = 20260920
METHODS = {
    "SupportRisk_Base": (0.10, False, False),
    "SupportRisk_Temporal": (0.10, True, False),
    "SupportRisk_DepthGuard": (0.10, False, True),
    "EngineeringROI_v2": (None, True, True),
}
TIMELINE_COLUMNS = (
    "method", "scenario_id", "time_s", "budget", "selected_patch_ids",
    "budget_fraction", "budget_max_count", "pre_refresh_candidate_count",
    "candidate_count", "selected_count", "actual_active_fraction",
    "mean_selected_support_risk_rank", "first_pass_count", "second_pass_count",
    "selected_patch_consecutive_overlap_count", "selected_patch_consecutive_overlap_fraction",
    "support_cell_count", "support_fraction", "blocked_local_change_cell_count",
    "blocked_local_change_fraction", "blocked_wet_creation_count", "raw_local_new_wet_fraction",
    "depth_guard_activation_fraction", "depth_guard_mean_abs_clip_m",
    "depth_guard_max_abs_clip_m", "global_forward_runtime_ms", "roi_scoring_runtime_ms",
    "local_correction_runtime_ms", "support_guard_runtime_ms", "depth_guard_runtime_ms",
)
TIMING_STEP_KEYS = ("global_steps_ms", "roi_steps_ms", "local_steps_ms", "support_steps_ms", "depth_steps_ms")


def validate_args(method: str, budget: float) -> None:
    if method not in METHODS or float(budget) not in (0.05, 0.10, 0.20):
        raise ValueError("ENGINEERING_ROI_V2_METHOD_INVALID")
    fixed_budget, _, _ = METHODS[method]
    if fixed_budget is not None and float(budget) != fixed_budget:
        raise ValueError("ENGINEERING_ROI_V2_ABLATION_B10_ONLY")


def method_label(method: str, budget: float) -> str:
    validate_args(method, budget)
    return f"{method}_B{int(round(budget * 100)):02d}"


def tensor(value, device):
    return torch.from_numpy(np.asarray(value, np.float32)).unsqueeze(0).to(device)


def timed_cuda(fn, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, time.perf_counter() - started


def _zero_depth_telemetry() -> dict[str, float]:
    return {
        "depth_guard_activation_fraction": 0.0,
        "depth_guard_mean_abs_clip_m": 0.0,
        "depth_guard_max_abs_clip_m": 0.0,
    }


def validate_case_timing_record(record):
    """Reject ambiguous or incomplete timing cache payloads before aggregation."""
    if not isinstance(record, dict) or "parts" in record:
        raise RuntimeError("V2_TIMING_CACHE_SCHEMA_INVALID")
    if not record.get("scenario_id"):
        raise RuntimeError("V2_CASE_TIMING_INVALID")
    try:
        wall = float(record["case_wall_runtime_seconds"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("V2_CASE_TIMING_INVALID") from error
    if not np.isfinite(wall):
        raise RuntimeError("V2_CASE_TIMING_INVALID")
    for key in TIMING_STEP_KEYS:
        values = record.get(key)
        if not isinstance(values, list) or len(values) != 144:
            raise RuntimeError("V2_CASE_TIMING_INVALID")
        try:
            if not np.isfinite(np.asarray(values, dtype=float)).all():
                raise RuntimeError("V2_CASE_TIMING_INVALID")
        except (TypeError, ValueError) as error:
            raise RuntimeError("V2_CASE_TIMING_INVALID") from error
    return record


def v2_transition(
    previous, current, provisional, encoded_current, encoded_provisional, metadata, config,
    budget, state, method, local, layout, delta, normalization, static, params, time_value,
    transform, normalizer, global_model, device,
):
    """Run the complete deterministic V2 writeback chain before teacher access."""
    _, use_temporal, use_depth = METHODS[method]
    (selected, next_state, selection), roi_seconds = timed_cuda(
        lambda: select_engineering_roi_v2(current, provisional, metadata, config, budget, state, temporal_refresh=use_temporal), device
    )
    def local_writeback():
        if not selected:
            return encoded_provisional
        features = build_features(previous, current, static, params, torch.tensor([time_value], device=current.device), transform, normalizer)
        raw_transformed, _, _ = apply_learned_correction(
            local, features, encoded_provisional, encoded_current, selected, layout, delta.scales,
            normalization["scales"], normalization["bounds"], metadata.active, global_model,
            apply_momentum_guard=False,
        )
        return raw_transformed
    raw_transformed, local_seconds = timed_cuda(local_writeback, device)
    def support_and_momentum():
        # Physical proposal is telemetry-only; transformed writeback stays unchanged.
        raw_local_physical = project_physical(transform.decode(raw_transformed), metadata.active)
        support_written, support_telemetry = apply_support_guard(
            encoded_provisional, raw_transformed, current, provisional, metadata.active,
            local_corrected_physical=raw_local_physical,
        )
        return apply_momentum_state_guard(support_written, global_model), support_telemetry
    (momentum_written, support_telemetry), support_seconds = timed_cuda(support_and_momentum, device)
    def depth_and_projection():
        physical = transform.decode(momentum_written)
        if use_depth:
            physical, depth_telemetry = apply_depth_envelope_guard(current, provisional, physical, metadata.active)
        else:
            depth_telemetry = _zero_depth_telemetry()
        return project_physical(physical, metadata.active), depth_telemetry
    (corrected, depth_telemetry), depth_seconds = timed_cuda(depth_and_projection, device)
    timing = {
        "roi_scoring_runtime_ms": 1000 * roi_seconds,
        "local_correction_runtime_ms": 1000 * local_seconds,
        "support_guard_runtime_ms": 1000 * support_seconds,
        "depth_guard_runtime_ms": 1000 * depth_seconds,
    }
    return corrected, selected, next_state, selection | support_telemetry | depth_telemetry, timing


def _checkpoint_provenance(path, checkpoint):
    return {
        "path": str(path), "global_step": checkpoint.get("global_step"),
        "stage_name": checkpoint.get("stage_name"), "architecture": checkpoint.get("architecture"),
        "source_code_sha": checkpoint.get("source_code_sha", checkpoint.get("code_sha", "UNKNOWN")),
    }


def manifest_for(label, method, budget, full_rows, rows, config_sha, code_sha, global_info, local_info, normalization, layout, shard_index, shard_count):
    global_path = ROOT / "models/global_operator_v2/best.pt"
    local_path = ROOT / "models/local_corrector_v1_1/best.pt"
    norm_path = ROOT / "models/local_corrector_v1_1/correction_normalization.json"
    return {
        "schema_version": 1, "method_label": label, "strategy": method, "budget": budget,
        "scope": "VAL_ONLY", "scenario_ids": list(map(str, rows.scenario_id)),
        "full_scenario_ids": list(map(str, full_rows.scenario_id)),
        "assigned_scenario_ids": list(map(str, rows.scenario_id)), "shard_count": shard_count,
        "shard_index": shard_index, "partition_rule": "global_index_mod_shard_count",
        "config_path": "configs/engineering_roi_v2.json", "config_sha256": config_sha,
        "engineering_roi_config_sha256": config_sha, "code_sha": code_sha,
        "global_checkpoint_path": str(global_path.relative_to(ROOT)),
        "global_checkpoint_sha256": sha256_file(global_path), "global_checkpoint_provenance": global_info,
        "local_checkpoint_path": str(local_path.relative_to(ROOT)),
        "local_checkpoint_sha256": sha256_file(local_path), "local_checkpoint_provenance": local_info,
        "local_checkpoint_update": 4000, "local_normalization_path": str(norm_path.relative_to(ROOT)),
        "local_normalization_sha256": sha256_file(norm_path),
        "local_normalization_fit_scope": normalization.get("fit_scope"),
        "support_guard_rule": "current_or_global_provisional_wet", "support_guard_version": "SupportGuard-v1",
        "seed_base": SEED, "patch_layout_rows": layout.rows, "patch_layout_cols": layout.cols,
        "eligible_patch_count": len(layout.eligible), "created_by": "evaluate_engineering_roi_v2.py",
    }


@torch.no_grad()
def execute_cases(label, method, budget, rows, layout, metadata, config, global_model, local, transform, normalizer, delta, normalization, static, route, z0, transects, device):
    if len(rows) != 1:
        raise RuntimeError("V2_CASE_RUNNER_REQUIRES_EXACTLY_ONE_SCENARIO")
    store = FrameStore(rows)
    _, row = next(rows.iterrows())
    store_index = int(np.where(store.rows.scenario_id.eq(row.scenario_id))[0][0])
    previous, current, _, params, time0, _ = store.sample(store_index, 0)
    previous, current, params = tensor(previous, device), tensor(current, device), tensor(params, device)
    metric = StreamingMetrics(metadata.active, route, delta.change_threshold); predicted, teacher = [], []
    append_station(predicted, 0, station_row(current, z0, transects)); append_station(teacher, 0, station_row(current, z0, transects))
    temporal_state = initial_temporal_state(); timings = {key: [] for key in TIMING_STEP_KEYS}; counts, coverage, timeline = [], [], []
    started = time.perf_counter()
    for step in range(144):
        provisional, global_seconds = timed_cuda(lambda: predict(global_model, previous, current, static, params, time0 + step / 144.0, transform, normalizer, metadata.active), device)
        encoded_current, encoded_provisional = transform.encode(current), transform.encode(provisional)
        corrected, selected, temporal_state, telemetry, transition_timing = v2_transition(
            previous, current, provisional, encoded_current, encoded_provisional, metadata, config, budget,
            temporal_state, method, local, layout, delta, normalization, static, params, time0 + step / 144.0,
            transform, normalizer, global_model, device,
        )
        # The transition is entirely complete before teacher frames are materialized.
        truth_current = tensor(store.frame(row, step * 10), device); truth_next = tensor(store.frame(row, (step + 1) * 10), device)
        metric.add(corrected, truth_current, truth_next, transform)
        append_station(predicted, (step + 1) * 10, station_row(corrected, z0, transects)); append_station(teacher, (step + 1) * 10, station_row(truth_next, z0, transects))
        timing = {"global_forward_runtime_ms": 1000 * global_seconds, **transition_timing}
        timeline.append(telemetry | timing | {"method": label, "scenario_id": row.scenario_id, "time_s": (step + 1) * 10, "budget": f"B{int(budget * 100):02d}", "selected_patch_ids": ";".join(str(p.patch_id) for p in selected)})
        timings["global_steps_ms"].append(timing["global_forward_runtime_ms"])
        timings["roi_steps_ms"].append(timing["roi_scoring_runtime_ms"])
        timings["local_steps_ms"].append(timing["local_correction_runtime_ms"])
        timings["support_steps_ms"].append(timing["support_guard_runtime_ms"])
        timings["depth_steps_ms"].append(timing["depth_guard_runtime_ms"])
        counts.append(len(selected)); coverage.append(telemetry["actual_active_fraction"]); previous, current = current, corrected
    wall = time.perf_counter() - started
    case = metric.result() | {"method": label, "scenario_id": row.scenario_id, "budget_fraction": budget, "budget_max_count": layout.count_for_budget(budget), "selected_patch_count": float(np.mean(counts)), "active_cell_coverage_fraction": float(np.mean(coverage)), "case_wall_runtime_seconds": wall}
    stations = [item | {"method": label, "scenario_id": row.scenario_id} for item in station_summary(predicted, teacher, transects)]
    timing_record = validate_case_timing_record({"method": label, "scenario_id": str(row.scenario_id), "case_wall_runtime_seconds": wall, **timings})
    return [case], stations, timeline, timing_record


def write_method_outputs(run_out, label, cases, stations, timeline, timing_parts, manifest):
    if not isinstance(timing_parts, list):
        raise RuntimeError("V2_TIMING_CACHE_SCHEMA_INVALID")
    parts = [validate_case_timing_record(part) for part in timing_parts]
    if len(parts) != len(cases):
        raise RuntimeError("V2_TIMING_CACHE_SCHEMA_INVALID")
    cleaned = [{key: value for key, value in row.items() if key != "_global_case_index"} for row in cases]
    frame = pd.DataFrame(timeline, columns=TIMELINE_COLUMNS).sort_values(["scenario_id", "time_s"])
    summary = summarize(cleaned, stations)
    all_steps = {key: [value for part in parts for value in part[key]] for key in TIMING_STEP_KEYS}
    mean_p95 = lambda values: (float(np.mean(values)), float(np.quantile(values, .95)))
    runtime = {
        "total_worker_wall_runtime_seconds": float(sum(p["case_wall_runtime_seconds"] for p in parts)),
        "mean_case_wall_runtime_seconds": float(np.mean([p["case_wall_runtime_seconds"] for p in parts])),
        "mean_selected_patch_count": float(frame.selected_count.mean()), "mean_active_coverage": float(frame.actual_active_fraction.mean()),
        "mean_consecutive_overlap": float(frame.selected_patch_consecutive_overlap_fraction.mean()),
        "mean_support_fraction": float(frame.support_fraction.mean()), "mean_blocked_local_change_fraction": float(frame.blocked_local_change_fraction.mean()),
        "mean_blocked_wet_creation_count": float(frame.blocked_wet_creation_count.mean()), "mean_raw_local_new_wet_fraction": float(frame.raw_local_new_wet_fraction.mean()),
        "mean_depth_guard_activation_fraction": float(frame.depth_guard_activation_fraction.mean()),
        "mean_depth_guard_abs_clip_m": float(frame.depth_guard_mean_abs_clip_m.mean()), "max_depth_guard_abs_clip_m": float(frame.depth_guard_max_abs_clip_m.max()),
    }
    for prefix, key in (("global_forward", "global_steps_ms"), ("roi_scoring", "roi_steps_ms"), ("local_correction", "local_steps_ms"), ("support_guard", "support_steps_ms"), ("depth_guard", "depth_steps_ms")):
        runtime[f"{prefix}_mean_ms_step"], runtime[f"{prefix}_p95_ms_step"] = mean_p95(all_steps[key])
    run_out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{ "method": label, **summary }]).to_csv(run_out / "final_method_summary.csv", index=False)
    pd.DataFrame(cleaned).sort_values("scenario_id").to_csv(run_out / "final_case_metrics.csv", index=False)
    pd.DataFrame(stations).sort_values(["scenario_id", "station"]).to_csv(run_out / "final_station_metrics.csv", index=False)
    frame.to_csv(run_out / "selection_timeline.csv", index=False)
    pd.DataFrame([{ "method": label, **runtime }]).to_csv(run_out / "runtime_summary.csv", index=False)
    report = {"method_label": label, "manifest": manifest, "summary": summary, "runtime": runtime, "timing_parts": parts}
    (run_out / "method_report.json").write_text(json.dumps(report, indent=2, allow_nan=True))


def merge_shards(args):
    """Deterministically assemble already-complete VAL shards without inference."""
    validate_args(args.method, args.budget)
    if args.shard_count <= 0:
        raise RuntimeError("V2_MERGE_REQUIRES_SHARDS")
    label = method_label(args.method, args.budget)
    run_root = ROOT / "results" / (args.output_dir or f"engineering_roi_v2/runs/{label}")
    manifests = validate_shard_manifests(run_root, args.shard_count)
    shard_paths = [run_root / "shards" / f"shard_{index}" for index in range(args.shard_count)]
    cases = pd.concat([pd.read_csv(path / "final_case_metrics.csv") for path in shard_paths], ignore_index=True)
    stations = pd.concat([pd.read_csv(path / "final_station_metrics.csv") for path in shard_paths], ignore_index=True)
    timeline = pd.concat([pd.read_csv(path / "selection_timeline.csv") for path in shard_paths], ignore_index=True)
    full = scenario_rows("VAL")
    validate_shard_coverage(cases.to_dict("records"), full.scenario_id)
    if any(len(group) != 144 for _, group in timeline.groupby("scenario_id")):
        raise RuntimeError("SELECTION_TIMELINE_INCOMPLETE")
    merged = dict(manifests[0])
    merged.update({"scenario_ids": list(map(str, full.scenario_id)), "assigned_scenario_ids": list(map(str, full.scenario_id)), "shard_count": 0, "shard_index": 0})
    create_or_validate_manifest(run_root, merged, True)
    timing_parts = []
    for path in shard_paths:
        timing_parts.extend(json.loads((path / "method_report.json").read_text()).get("timing_parts", []))
    write_method_outputs(run_root, label, cases.to_dict("records"), stations.to_dict("records"), timeline.to_dict("records"), timing_parts, merged)
    write_run_complete(run_root, merged, cases.to_dict("records"), timeline.to_dict("records"), True)


def main(args):
    validate_args(args.method, args.budget)
    if args.merge_shards:
        return merge_shards(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    config, config_sha = load_engineering_roi_v2_config(ROOT / "configs/engineering_roi_v2.json")
    label = method_label(args.method, args.budget)
    full = scenario_rows("VAL")
    if len(full) != 20 or not set(full["split"]).issubset({"VAL"}):
        raise RuntimeError("VAL_SCOPE_REQUIRED")
    rows = partition_rows(full, args.shard_index, args.shard_count)
    run_root = ROOT / "results" / (args.output_dir or f"engineering_roi_v2/runs/{label}")
    run_out = run_output_path(run_root, args.shard_index, args.shard_count)
    device = torch.device("cuda")
    global_model, transform, normalizer, delta, global_checkpoint = load_model(device)
    static_np, _ = static_and_exogenous(INPUT); static = tensor(static_np, device); active = static[:, 1:2]
    layout = PatchLayout(static_np[1]); model_dir = ROOT / "models/local_corrector_v1_1"
    normalization = json.loads((model_dir / "correction_normalization.json").read_text())
    if normalization.get("fit_scope") != "TRAIN_ONLY":
        raise RuntimeError("LOCAL_NORMALIZATION_NOT_TRAIN_ONLY")
    local_checkpoint = torch.load(model_dir / "best.pt", map_location=device, weights_only=False)
    if local_checkpoint.get("update") != 4000:
        raise RuntimeError("LOCAL_CHECKPOINT_MUST_BE_BEST_4000")
    local = JilongLocalCorrector(47).to(device); local.load_state_dict(local_checkpoint["model"]); local.eval()
    route = np.asarray(np.load(INPUT)["route_chainage_m"], np.float32)
    transects = load_transects(ROOT / "data/downloads/park_v2/inputs/upper30h_transects.json")
    metadata = build_roi_static_metadata(layout, active, route, build_section_mask(transects, static_np.shape[-2:]), device)
    code_sha = resolve_code_sha(args.source_code_sha, ROOT, require=True)
    manifest = manifest_for(label, args.method, args.budget, full, rows, config_sha, code_sha, _checkpoint_provenance(ROOT / "models/global_operator_v2/best.pt", global_checkpoint), _checkpoint_provenance(model_dir / "best.pt", local_checkpoint), normalization, layout, args.shard_index, args.shard_count)
    runner = lambda one: execute_cases(label, args.method, args.budget, one, layout, metadata, config, global_model, local, transform, normalizer, delta, normalization, static, route, static_np[0], transects, device)
    cases, stations, timeline, timing = cached_engineering_method(run_out, manifest, rows, runner, True, args.resume, expected_station_count=len(transects))
    write_method_outputs(run_out, label, cases, stations, timeline, timing, manifest)
    write_run_complete(run_out, manifest, cases, timeline, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--budget", choices=(0.05, 0.10, 0.20), type=float, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=0)
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--source-code-sha")
    main(parser.parse_args())
