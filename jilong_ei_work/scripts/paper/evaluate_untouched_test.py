"""Untouched TEST evaluator using the frozen RiskTemporal-v1 transition verbatim.

This is deliberately a paper-stage wrapper: it does not modify or reimplement
the frozen selector, guards, projection, metrics, or CUDA timing functions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import evaluate_engineering_roi_v2 as frozen
from scripts.run_global_v2_oracle_refinement import append_station, station_row, station_summary, summarize
from src.common.provenance import resolve_code_sha, sha256_file
from src.global_operator_v2.dataset import FrameStore, INPUT, scenario_rows
from src.global_operator_v2.frame_adapter import static_and_exogenous
from src.global_operator_v2.metrics import load_transects
from src.global_operator_v2.oracle_refinement import PatchLayout, StreamingMetrics
from src.local_corrector.engineering_roi import build_roi_static_metadata, build_section_mask
from src.local_corrector.engineering_roi_execution import cached_engineering_method, partition_rows, run_output_path, write_run_complete
from src.local_corrector.engineering_roi_v2 import load_engineering_roi_v2_config
from src.local_corrector.model import JilongLocalCorrector

FINAL_CODE_SHA = "146184d1fbbf366c0c9c40066a56753b398b10f4"


def mean_p95(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.quantile(values, .95))


@torch.no_grad()
def frozen_global_case(label, rows, metadata, global_model, transform, normalizer, delta, static, route, z0, transects, device):
    if len(rows) != 1:
        raise RuntimeError("TEST_CASE_RUNNER_REQUIRES_EXACTLY_ONE_SCENARIO")
    store = FrameStore(rows); _, row = next(rows.iterrows())
    previous, current, _, params, time0, _ = store.sample(0, 0)
    previous, current, params = frozen.tensor(previous, device), frozen.tensor(current, device), frozen.tensor(params, device)
    metric = StreamingMetrics(metadata.active, route, delta.change_threshold)
    predicted, teacher = [], []
    truth0 = frozen.tensor(store.frame(row, 0), device)
    append_station(predicted, 0, station_row(truth0, z0, transects)); append_station(teacher, 0, station_row(truth0, z0, transects))
    timings = {key: [] for key in frozen.TIMING_STEP_KEYS}; started = time.perf_counter()
    for step in range(144):
        provisional, seconds = frozen.timed_cuda(lambda: frozen.predict(global_model, previous, current, static, params, time0 + step / 144.0, transform, normalizer, metadata.active), device)
        truth_current = frozen.tensor(store.frame(row, step * 10), device); truth_next = frozen.tensor(store.frame(row, (step + 1) * 10), device)
        metric.add(provisional, truth_current, truth_next, transform)
        append_station(predicted, (step + 1) * 10, station_row(provisional, z0, transects)); append_station(teacher, (step + 1) * 10, station_row(truth_next, z0, transects))
        timings["global_steps_ms"].append(1000 * seconds)
        for key in frozen.TIMING_STEP_KEYS[1:]: timings[key].append(0.0)
        previous, current = current, provisional
    wall = time.perf_counter() - started
    case = metric.result() | {"method": label, "scenario_id": row.scenario_id, "budget_fraction": 0.0, "budget_max_count": 0, "selected_patch_count": 0.0, "active_cell_coverage_fraction": 0.0, "case_wall_runtime_seconds": wall}
    stations = [item | {"method": label, "scenario_id": row.scenario_id} for item in station_summary(predicted, teacher, transects)]
    return [case], stations, [], frozen.validate_case_timing_record({"method": label, "scenario_id": str(row.scenario_id), "case_wall_runtime_seconds": wall, **timings})


def write_frozen_outputs(run_out: Path, label: str, cases, stations, timing_parts, manifest) -> None:
    summary = summarize(cases, stations)
    all_steps = {key: [value for part in timing_parts for value in part[key]] for key in frozen.TIMING_STEP_KEYS}
    runtime = {"total_worker_wall_runtime_seconds": float(sum(item["case_wall_runtime_seconds"] for item in timing_parts)), "mean_case_wall_runtime_seconds": float(np.mean([item["case_wall_runtime_seconds"] for item in timing_parts])), "mean_selected_patch_count": 0.0, "mean_active_coverage": 0.0, "mean_consecutive_overlap": 0.0, "mean_support_fraction": 0.0, "mean_blocked_local_change_fraction": 0.0, "mean_blocked_wet_creation_count": 0.0, "mean_raw_local_new_wet_fraction": 0.0, "mean_depth_guard_activation_fraction": 0.0, "mean_depth_guard_abs_clip_m": 0.0, "max_depth_guard_abs_clip_m": 0.0}
    for prefix, key in (("global_forward", "global_steps_ms"), ("roi_scoring", "roi_steps_ms"), ("local_correction", "local_steps_ms"), ("support_guard", "support_steps_ms"), ("depth_guard", "depth_steps_ms")):
        runtime[f"{prefix}_mean_ms_step"], runtime[f"{prefix}_p95_ms_step"] = mean_p95(all_steps[key])
    run_out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{ "method": label, **summary }]).to_csv(run_out / "final_method_summary.csv", index=False)
    pd.DataFrame(cases).sort_values("scenario_id").to_csv(run_out / "final_case_metrics.csv", index=False)
    pd.DataFrame(stations).sort_values(["scenario_id", "station"]).to_csv(run_out / "final_station_metrics.csv", index=False)
    pd.DataFrame(columns=frozen.TIMELINE_COLUMNS).to_csv(run_out / "selection_timeline.csv", index=False)
    pd.DataFrame([{ "method": label, **runtime }]).to_csv(run_out / "runtime_summary.csv", index=False)
    (run_out / "method_report.json").write_text(json.dumps({"method_label": label, "manifest": manifest, "summary": summary, "runtime": runtime, "timing_parts": timing_parts}, indent=2, allow_nan=True), encoding="utf-8")


def setup():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA_REQUIRED")
    device = torch.device("cuda"); config_path = ROOT / "configs" / "risk_temporal_final_v1.json"; raw = config_path.read_bytes(); config = json.loads(raw)
    if config.get("version") != "RiskTemporal-v1" or config.get("depth_envelope_guard") != "disabled" or config.get("temporal_refresh_steps") != 1 or config.get("temporal_fill_from_excluded") is not False: raise RuntimeError("RISK_TEMPORAL_CONFIG_INVALID")
    global_model, transform, normalizer, delta, global_checkpoint = frozen.load_model(device)
    static_np, _ = static_and_exogenous(INPUT); static = frozen.tensor(static_np, device); active = static[:, 1:2]
    layout = PatchLayout(static_np[1]); model_dir = ROOT / "models" / "local_corrector_v1_1"; normalization = json.loads((model_dir / "correction_normalization.json").read_text())
    if normalization.get("fit_scope") != "TRAIN_ONLY": raise RuntimeError("LOCAL_NORMALIZATION_NOT_TRAIN_ONLY")
    local_checkpoint = torch.load(model_dir / "best.pt", map_location=device, weights_only=False)
    if local_checkpoint.get("update") != 4000: raise RuntimeError("LOCAL_CHECKPOINT_MUST_BE_BEST_4000")
    local = JilongLocalCorrector(47).to(device); local.load_state_dict(local_checkpoint["model"]); local.eval()
    route = np.asarray(np.load(INPUT)["route_chainage_m"], np.float32); transects = load_transects(ROOT / "data" / "downloads" / "park_v2" / "inputs" / "upper30h_transects.json")
    metadata = build_roi_static_metadata(layout, active, route, build_section_mask(transects, static_np.shape[-2:]), device)
    return device, config, hashlib.sha256(raw).hexdigest(), global_model, transform, normalizer, delta, global_checkpoint, static_np, static, layout, normalization, local_checkpoint, local, route, transects, metadata


def run_method(method: str, args) -> None:
    setup_values = setup(); device, config, config_sha, global_model, transform, normalizer, delta, global_checkpoint, static_np, static, layout, normalization, local_checkpoint, local, route, transects, metadata = setup_values
    full = scenario_rows("TEST")
    if len(full) != 20 or not set(full["split"]).issubset({"TEST"}) or full.scenario_id.astype(str).str.contains("H0|VAL|TRAIN", case=False, regex=True).any(): raise RuntimeError("TEST_SCOPE_REQUIRED")
    if args.case_id:
        full = full.loc[full.scenario_id.eq(args.case_id)].reset_index(drop=True)
        if len(full) != 1: raise RuntimeError("UNKNOWN_TEST_CASE")
    rows = partition_rows(full, args.shard_index, args.shard_count)
    code_sha = resolve_code_sha(FINAL_CODE_SHA, ROOT, require=True)
    if code_sha != FINAL_CODE_SHA: raise RuntimeError("FINAL_CODE_SHA_REQUIRED")
    if method == "risktemporal_b10":
        label, strategy, budget, folder = "RiskTemporal_B10", "RiskTemporal", 0.10, "risktemporal_b10"
    else:
        label, strategy, budget, folder = "FrozenGlobal", "FrozenGlobal", 0.0, "frozen_global"
    run_root = ROOT / (args.output_dir or f"paper_results/test/{folder}"); run_out = run_output_path(run_root, args.shard_index, args.shard_count)
    manifest = frozen.manifest_for(label, strategy, budget, full, rows, config_sha, code_sha, frozen._checkpoint_provenance(ROOT / "models" / "global_operator_v2" / "best.pt", global_checkpoint), frozen._checkpoint_provenance(ROOT / "models" / "local_corrector_v1_1" / "best.pt", local_checkpoint), normalization, layout, args.shard_index, args.shard_count, "configs/risk_temporal_final_v1.json")
    manifest.update({"scope": "TEST_ONLY", "paper_stage": "untouched_test", "selector_truth_access": False, "metric_implementation": "frozen_engineering_roi_v2"})
    if method == "risktemporal_b10":
        runner = lambda one: frozen.execute_cases(label, "RiskTemporal", budget, one, layout, metadata, config, global_model, local, transform, normalizer, delta, normalization, static, route, static_np[0], transects, device)
        cases, stations, timeline, timing = cached_engineering_method(run_out, manifest, rows, runner, True, args.resume, expected_station_count=len(transects))
        frozen.write_method_outputs(run_out, label, cases, stations, timeline, timing, manifest); write_run_complete(run_out, manifest, cases, timeline, True)
    else:
        runner = lambda one: frozen_global_case(label, one, metadata, global_model, transform, normalizer, delta, static, route, static_np[0], transects, device)
        cases, stations, timeline, timing = cached_engineering_method(run_out, manifest, rows, runner, False, args.resume, expected_station_count=len(transects))
        write_frozen_outputs(run_out, label, cases, stations, timing, manifest); write_run_complete(run_out, manifest, cases, timeline, False)
    print(json.dumps({"status": "PASS", "method": label, "scope": "TEST_ONLY", "output": str(run_out), "cases": len(cases)}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--method", choices=("frozen_global", "risktemporal_b10"), required=True); parser.add_argument("--case-id"); parser.add_argument("--output-dir"); parser.add_argument("--shard-index", type=int, default=0); parser.add_argument("--shard-count", type=int, default=0); parser.add_argument("--resume", action="store_true")
    parsed = parser.parse_args(); run_method(parsed.method, parsed)
