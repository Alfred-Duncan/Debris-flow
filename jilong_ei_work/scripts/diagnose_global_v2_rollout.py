"""Read-only diagnosis of a failed Global Operator V2 rollout run.

The script deliberately accepts a runtime root.  This keeps source code and the
large, locally generated Park artifacts separate and guarantees that it never
updates a checkpoint or a training-state file.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.global_operator_v2 import dataset as dataset_module
from src.global_operator_v2.dataset import FrameStore, build_features, feature_names
from src.global_operator_v2.frame_adapter import static_and_exogenous
from src.global_operator_v2.losses import (
    STORAGE_WET_THRESHOLD_M,
    project_physical,
    teacher_weight,
)
from src.global_operator_v2.metrics import debris_front
from src.global_operator_v2.model import JilongGlobalOperatorV2
from src.global_operator_v2.transforms import FeatureNormalizer, PhysicalTransform
from src.global_operator_v2.validation import select_fixed_val_subset

HORIZONS = (1, 2, 4, 8, 16, 32, 64, 144)
CELL_AREA_M2 = 900.0


def _tensor(value, device):
    return torch.from_numpy(np.asarray(value)).unsqueeze(0).to(device)

def _jsonable(value):
    if isinstance(value,dict):return {str(key):_jsonable(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [_jsonable(item) for item in value]
    if isinstance(value,(np.floating,float)):return float(value) if np.isfinite(value) else None
    if isinstance(value,(np.integer,)):return int(value)
    return value


def _runtime_rows(runtime_root: Path, split: str) -> pd.DataFrame:
    index = pd.read_csv(runtime_root / "data/scenario_index.csv")
    design = pd.read_csv(runtime_root / "configs/scenario_design/JILONG_EI_SCENARIOS.csv")
    columns = ["scenario_id", *dataset_module.PARAMS, "design_index"]
    rows = index.merge(design[columns], on="scenario_id", validate="one_to_one")
    return rows.loc[rows.split.eq(split)].reset_index(drop=True)


def _configure_runtime_data(runtime_root: Path) -> None:
    """Point the reusable FrameStore at the explicit runtime artifacts."""
    runtime_rvpi = runtime_root / "external/RVPI-PDE"
    if runtime_rvpi.exists():
        sys.path.insert(0, str(runtime_rvpi))
    dataset_module.ROOT = runtime_root
    dataset_module.INPUT = runtime_root / "data/downloads/park_v2/inputs/upper30h.npz"


def _load_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    semantics=checkpoint.get('model_semantics',{});bounded=bool(semantics.get('bounded_residual',False));bounds=semantics.get('delta_bounds') if bounded else None
    model = JilongGlobalOperatorV2(len(feature_names()), **architecture,delta_bounds=bounds).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if unexpected or set(missing) - {"delta_bounds"}:
        raise RuntimeError(f"incompatible checkpoint state: missing={missing}, unexpected={unexpected}")
    if bounded and (not model.bounded_residual or not torch.equal(model.delta_bounds.detach().cpu(),torch.as_tensor(bounds))):raise RuntimeError('DELTA_MODEL_SEMANTICS_MISMATCH')
    model.eval()
    transform = PhysicalTransform.from_dict(checkpoint["transform"])
    normalizer = FeatureNormalizer.from_dict(checkpoint["feature_normalizer"])
    return model, transform, normalizer, checkpoint


def _front_error(predicted: float, truth: float, route_length_km: float) -> float | None:
    if np.isfinite(predicted) and np.isfinite(truth):
        return abs(predicted - truth)
    if not np.isfinite(predicted) and not np.isfinite(truth):
        return None
    return route_length_km


def _metric_snapshot(acc: dict, horizon: int) -> dict:
    def rel(name):
        return math.sqrt(acc[name + "_num"] / max(acc[name + "_den"], 1e-12))

    fronts = acc["front_errors"]
    return {
        "horizon": horizon,
        "trajectory_h_rel_l2": rel("h"),
        "trajectory_momentum_rel_l2": rel("momentum"),
        "trajectory_c_rel_l2": rel("c"),
        "trajectory_ice_rel_l2": rel("ice"),
        "trajectory_dz_rel_l2": rel("dz"),
        "mean_wet_iou": acc["wet_sum"] / max(acc["frames"], 1),
        "mixture_volume_relative_error": acc["volume_sum"] / max(acc["frames"], 1),
        "debris_front_mae_km": float(np.mean(fronts)) if fronts else acc["route_length_km"],
    }


@torch.no_grad()
def _ladder_for_model(label, model, transform, normalizer, store, rows, static, route, device, leakage_rows):
    active = static[:, 1:2]
    active_bool = active[:, 0].bool()
    route_length_km = float(np.nanmax(route) / 1000.0)
    per_horizon = {h: [] for h in HORIZONS}
    finite_by_horizon = {h: [] for h in HORIZONS}
    first_nonfinite = []
    for _, row in rows.iterrows():
        index = int(np.where(store.rows.scenario_id.eq(row.scenario_id))[0][0])
        previous, current, _, params, time_fraction, _ = store.sample(index, 0)
        previous, current, params = (_tensor(x, device) for x in (previous, current, params))
        acc = {name + suffix: 0.0 for name in ("h", "momentum", "c", "ice", "dz") for suffix in ("_num", "_den")}
        acc.update(wet_sum=0.0, volume_sum=0.0, frames=0, front_errors=[], route_length_km=route_length_km)
        bad_step = None
        snapshots = {}
        for step in range(1, max(HORIZONS) + 1):
            features = build_features(previous, current, static, params, torch.tensor([time_fraction + (step - 1) / 144.0], device=device), transform, normalizer)
            predicted = project_physical(transform.decode(model(features, transform.encode(current))))
            if not torch.isfinite(predicted).all():
                bad_step = step
                break
            target = _tensor(store.frame(row, step * 10), device)
            outside = ~active_bool
            record = {"model": label, "scenario_id": row.scenario_id, "step": step}
            for channel, name in enumerate(("h", "hu", "hv", "c", "ice", "dz")):
                values = predicted[:, channel][outside]
                record[name + "_outside_rms"] = float(values.square().mean().sqrt().cpu()) if values.numel() else 0.0
                record[name + "_outside_max"] = float(values.abs().max().cpu()) if values.numel() else 0.0
            leakage_rows.append(record)
            masks = {"h": active_bool, "momentum": active, "c": active_bool, "ice": active_bool, "dz": active_bool}
            pairs = {
                "h": (predicted[:, 0], target[:, 0]),
                "momentum": (predicted[:, 1:3], target[:, 1:3]),
                "c": (predicted[:, 3], target[:, 3]),
                "ice": (predicted[:, 4], target[:, 4]),
                "dz": (predicted[:, 5], target[:, 5]),
            }
            for name, (prediction, truth) in pairs.items():
                mask = masks[name]
                acc[name + "_num"] += float(((prediction - truth).square() * mask).sum().cpu())
                acc[name + "_den"] += float((truth.square() * mask).sum().cpu())
            p_wet = predicted[:, 0] >= STORAGE_WET_THRESHOLD_M
            t_wet = target[:, 0] >= STORAGE_WET_THRESHOLD_M
            acc["wet_sum"] += float(((p_wet & t_wet & active_bool).sum().float() / ((p_wet | t_wet) & active_bool).sum().clamp_min(1)).cpu())
            predicted_volume = float((predicted[:, 0:1] * active).sum().cpu()) * CELL_AREA_M2
            truth_volume = float((target[:, 0:1] * active).sum().cpu()) * CELL_AREA_M2
            acc["volume_sum"] += abs(predicted_volume - truth_volume) / max(abs(truth_volume), 1.0)
            front = _front_error(debris_front(predicted[0].cpu().numpy(), route), debris_front(target[0].cpu().numpy(), route), route_length_km)
            if front is not None:
                acc["front_errors"].append(front)
            acc["frames"] += 1
            previous, current = current, predicted
            if step in per_horizon:
                snapshots[step] = _metric_snapshot(acc, step)
        first_nonfinite.append(bad_step)
        for horizon in HORIZONS:
            finite = bad_step is None or bad_step > horizon
            finite_by_horizon[horizon].append(finite)
            if finite:
                snapshots[horizon]["scenario_id"] = row.scenario_id
                per_horizon[horizon].append(snapshots[horizon])
    rows_out = []
    for horizon in HORIZONS:
        valid = per_horizon[horizon]
        result = {"model": label, "horizon": horizon, "finite_fraction": float(np.mean(finite_by_horizon[horizon])), "finite_rollout": bool(all(finite_by_horizon[horizon])), "first_nonfinite_step": min((x for x in first_nonfinite if x is not None), default=None)}
        for key in ("trajectory_h_rel_l2", "trajectory_momentum_rel_l2", "trajectory_c_rel_l2", "trajectory_ice_rel_l2", "trajectory_dz_rel_l2", "mean_wet_iou", "mixture_volume_relative_error", "debris_front_mae_km"):
            result[key] = float(np.mean([item[key] for item in valid])) if valid else float("nan")
        rows_out.append(result)
    return rows_out


@torch.no_grad()
def _ladder_persistence(store, rows, static, route, device):
    active = static[:, 1:2]
    active_bool = active[:, 0].bool()
    route_length_km = float(np.nanmax(route) / 1000.0)
    accumulators = {h: [] for h in HORIZONS}
    for _, row in rows.iterrows():
        index = int(np.where(store.rows.scenario_id.eq(row.scenario_id))[0][0])
        _, current, _, _, _, _ = store.sample(index, 0)
        current = _tensor(current, device)
        acc = {name + suffix: 0.0 for name in ("h", "momentum", "c", "ice", "dz") for suffix in ("_num", "_den")}
        acc.update(wet_sum=0.0, volume_sum=0.0, frames=0, front_errors=[], route_length_km=route_length_km)
        for step in range(1, max(HORIZONS) + 1):
            target = _tensor(store.frame(row, step * 10), device)
            pairs = {"h": (current[:, 0], target[:, 0]), "momentum": (current[:, 1:3], target[:, 1:3]), "c": (current[:, 3], target[:, 3]), "ice": (current[:, 4], target[:, 4]), "dz": (current[:, 5], target[:, 5])}
            for name, (prediction, truth) in pairs.items():
                mask = active if name == "momentum" else active_bool
                acc[name + "_num"] += float(((prediction - truth).square() * mask).sum().cpu())
                acc[name + "_den"] += float((truth.square() * mask).sum().cpu())
            p_wet = current[:, 0] >= STORAGE_WET_THRESHOLD_M
            t_wet = target[:, 0] >= STORAGE_WET_THRESHOLD_M
            acc["wet_sum"] += float(((p_wet & t_wet & active_bool).sum().float() / ((p_wet | t_wet) & active_bool).sum().clamp_min(1)).cpu())
            pv = float((current[:, 0:1] * active).sum().cpu()) * CELL_AREA_M2
            tv = float((target[:, 0:1] * active).sum().cpu()) * CELL_AREA_M2
            acc["volume_sum"] += abs(pv - tv) / max(abs(tv), 1.0)
            error = _front_error(debris_front(current[0].cpu().numpy(), route), debris_front(target[0].cpu().numpy(), route), route_length_km)
            if error is not None:
                acc["front_errors"].append(error)
            acc["frames"] += 1
            if step in accumulators:
                snapshot = _metric_snapshot(acc, step)
                snapshot["scenario_id"] = row.scenario_id
                accumulators[step].append(snapshot)
    output = []
    for horizon, values in accumulators.items():
        result = {"model": "persistence", "horizon": horizon, "finite_fraction": 1.0, "finite_rollout": True, "first_nonfinite_step": None}
        for key in ("trajectory_h_rel_l2", "trajectory_momentum_rel_l2", "trajectory_c_rel_l2", "trajectory_ice_rel_l2", "trajectory_dz_rel_l2", "mean_wet_iou", "mixture_volume_relative_error", "debris_front_mae_km"):
            result[key] = float(np.mean([item[key] for item in values]))
        output.append(result)
    return output


@torch.no_grad()
def _source_release(model, transform, normalizer, store, rows, static, device):
    active, source = static[:, 1:2], static[:, 2:3].bool()
    output = []
    for _, row in rows.iterrows():
        index = int(np.where(store.rows.scenario_id.eq(row.scenario_id))[0][0])
        previous, current, _, params, time_fraction, _ = store.sample(index, 0)
        previous, current, params = (_tensor(x, device) for x in (previous, current, params))
        predicted = current.clone()
        volumes = {"truth": {}, "prediction": {}, "persistence": {}}
        errors = {}
        for step in range(0, 13):
            time_s = step * 10
            truth = _tensor(store.frame(row, time_s), device)
            for label, state in (("truth", truth), ("prediction", predicted), ("persistence", current)):
                volumes[label][time_s] = float((state[:, 0:1] * active).sum().cpu()) * CELL_AREA_M2
            if time_s in (10, 20, 30):
                for channel, name in ((0, "h"), (3, "c"), (4, "ice")):
                    errors[f"source_{name}_mae_{time_s}s"] = float((predicted[:, channel:channel + 1].sub(truth[:, channel:channel + 1]).abs() * source).sum().cpu() / source.sum().clamp_min(1).cpu())
            if step < 12:
                features = build_features(previous, predicted, static, params, torch.tensor([time_fraction + step / 144.0], device=device), transform, normalizer)
                predicted_next = project_physical(transform.decode(model(features, transform.encode(predicted))))
                previous, predicted = predicted, predicted_next
        for start, end in ((0, 10), (10, 20), (20, 30), (30, 40), (40, 60), (60, 120)):
            for label in volumes:
                output.append({"scenario_id": row.scenario_id, "interval": f"{start}-{end}", "kind": label, "mixture_volume": volumes[label][end], "volume_increment": volumes[label][end] - volumes[label][start], **errors})
        truth_injection = volumes["truth"][30] - volumes["truth"][0]
        pred_injection = volumes["prediction"][30] - volumes["prediction"][0]
        output.append({"scenario_id": row.scenario_id, "interval": "0-30_summary", "kind": "comparison", "source_0_30_volume_injection_truth": truth_injection, "source_0_30_volume_injection_pred": pred_injection, "source_0_30_relative_error": abs(pred_injection - truth_injection) / max(abs(truth_injection), 1.0), **errors})
    return output


def _sampler_audit(n_scenarios: int):
    # This is the exact pre-stabilization sampler used by the failed run, kept
    # local so future sampler changes cannot rewrite the historical audit.
    def legacy_sample(global_step, k):
        rng = np.random.default_rng(np.random.SeedSequence([20260920, global_step]))
        rng.integers(n_scenarios)
        return int(rng.integers(0, 145 - k)) * 10
    stages = ((0, 2000, 1), (2000, 4000, 2), (4000, 8000, 4), (8000, 14000, 6))
    starts = []
    for low, high, k in stages:
        for global_step in range(low, high):
            starts.append(legacy_sample(global_step, k))
    starts = np.asarray(starts)
    bins = {"0-30": (0, 30), "0-120": (0, 120), "120-300": (120, 300), "300-600": (300, 600), "600-900": (600, 900), "900-1200": (900, 1200), "1200-1440": (1200, 1441)}
    return {"total_updates": int(starts.size), "starts_0s": int((starts == 0).sum()), "starts_10s": int((starts == 10).sum()), "starts_20s": int((starts == 20).sum()), "source_active_fraction": float((starts < 30).mean()), "ranges": {name: int(((starts >= low) & (starts < high)).sum()) for name, (low, high) in bins.items()}}


@torch.no_grad()
def _loss_audit(model, transform, normalizer, store, static, device):
    active = static[:, 1:2]
    totals = {name: 0.0 for name in ("dry_active", "wet", "strong_change", "source", "transition_halo")}
    cells = {name: 0 for name in totals}
    for step in range(32):
        rng=np.random.default_rng(np.random.SeedSequence([20260920,step]));sample={"scenario_index":int(rng.integers(len(store.rows))),"time_s":int(rng.integers(0,144))*10}
        previous, current, target, params, time_fraction, _ = store.sample(sample["scenario_index"], sample["time_s"])
        previous, current, target, params = (_tensor(x, device) for x in (previous, current, target, params))
        predicted_t = model(build_features(previous, current, static, params, torch.tensor([time_fraction], device=device), transform, normalizer), transform.encode(current))
        target_t = transform.encode(target)
        base = torch.nn.functional.smooth_l1_loss(predicted_t, target_t, reduction="none").mean(1, keepdim=True)
        weights, wet = teacher_weight(current, target, active)
        weighted = base * weights
        change = (target_t - transform.encode(current)).abs().mean(1, keepdim=True)
        threshold = torch.quantile(change[active.bool()], 0.9)
        masks = {
            "dry_active": active.bool() & ~wet.bool(),
            "wet": wet.bool(),
            "strong_change": active.bool() & (change >= threshold),
            "source": static[:, 2:3].bool(),
            "transition_halo": active.bool() & (current[:, 0:1] >= 0.02) & (current[:, 0:1] < 0.08),
        }
        for name, mask in masks.items():
            totals[name] += float((weighted * mask).sum().cpu())
            cells[name] += int(mask.sum().cpu())
    all_loss = max(sum(totals.values()), 1e-12)
    return {name: {"cells": cells[name], "weighted_loss": totals[name], "fraction_of_reported_components": totals[name] / all_loss} for name in totals}


def main(args):
    runtime_root = Path(args.runtime_root).resolve()
    artifacts = runtime_root / "models/global_operator_v2"
    if not artifacts.exists():
        raise FileNotFoundError(artifacts)
    _configure_runtime_data(runtime_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows = _runtime_rows(runtime_root, "TRAIN")
    val_rows = _runtime_rows(runtime_root, "VAL")
    subset = select_fixed_val_subset(val_rows, 8, 20260920, train_rows)
    store = FrameStore(pd.concat((train_rows, subset), ignore_index=True))
    # FrameStore's rows must include the exact VAL subset used by the old run.
    val_store = FrameStore(subset)
    input_path = runtime_root / "data/downloads/park_v2/inputs/upper30h.npz"
    static_np, _ = static_and_exogenous(input_path)
    static = torch.from_numpy(static_np).unsqueeze(0).to(device)
    with np.load(input_path) as data:
        route = np.asarray(data["route_chainage_m"], np.float32)
    leakage_rows, comparisons = [], []
    checkpoints = {"best_candidate": artifacts / "best_candidate.pt", "last": artifacts / "last.pt"}
    loaded = {}
    for label, checkpoint_path in checkpoints.items():
        if not checkpoint_path.exists():
            continue
        model, transform, normalizer, checkpoint = _load_model(checkpoint_path, device)
        loaded[label] = (model, transform, normalizer, checkpoint)
        comparisons.extend(_ladder_for_model(label, model, transform, normalizer, val_store, subset, static, route, device, leakage_rows))
    persistence = _ladder_persistence(val_store, subset, static, route, device)
    comparisons.extend(persistence)
    if "best_candidate" not in loaded:
        raise RuntimeError("best_candidate.pt is required for source and loss diagnosis")
    best_model, best_transform, best_normalizer, best_checkpoint = loaded["best_candidate"]
    source_rows = _source_release(best_model, best_transform, best_normalizer, val_store, subset, static, device)
    sampler = _sampler_audit(len(train_rows))
    loss = _loss_audit(best_model, best_transform, best_normalizer, store, static, device)
    comparison_frame = pd.DataFrame(comparisons)
    diagnostic_dir = runtime_root / "results/global_operator_v2_diagnostic"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    comparison_frame.to_csv(diagnostic_dir / "horizon_comparison.csv", index=False)
    pd.DataFrame(source_rows).to_csv(diagnostic_dir / "source_release_diagnostic.csv", index=False)
    pd.DataFrame(leakage_rows).to_csv(diagnostic_dir / "outside_active_leakage.csv", index=False)
    persistence_144 = comparison_frame.query("model == 'persistence' and horizon == 144").iloc[0].to_dict()
    best_144 = comparison_frame.query("model == 'best_candidate' and horizon == 144").iloc[0].to_dict()
    primary = ("trajectory_h_rel_l2", "trajectory_momentum_rel_l2", "mean_wet_iou", "mixture_volume_relative_error", "debris_front_mae_km")
    worse_or_close = 0
    ratios = {}
    for key in primary:
        if key == "mean_wet_iou":
            ratio = best_144[key] / max(persistence_144[key], 1e-12)
            worse = ratio <= 1.05
        else:
            ratio = best_144[key] / max(persistence_144[key], 1e-12)
            worse = ratio >= 0.95
        ratios[key] = ratio
        worse_or_close += int(worse)
    source_summary = pd.DataFrame(source_rows).query("interval == '0-30_summary'")
    summary = {
        "runtime_root": str(runtime_root),
        "device": str(device),
        "val_subset_scenarios": subset.scenario_id.tolist(),
        "checkpoint_steps": {label: int(value[3].get("global_step", value[3].get("step", -1))) for label, value in loaded.items()},
        "persistence_collapse": "YES" if worse_or_close >= 3 else "NO",
        "model_to_persistence_ratio_at_144": ratios,
        "first_nonfinite_step": {row["model"]: row["first_nonfinite_step"] for row in comparisons if row["horizon"] == 144 and row["model"] != "persistence"},
        "source_0_30_relative_error_mean": float(source_summary.source_0_30_relative_error.mean()),
        "outside_active_max_abs": float(pd.DataFrame(leakage_rows).filter(regex="_outside_max$").max().max()),
        "old_sampler": sampler,
        "loss_contributions": loss,
        "best_candidate_144": best_144,
        "persistence_144": persistence_144,
    }
    report = runtime_root / "reports/GLOBAL_V2_FAILURE_DIAGNOSTIC.json"
    summary=_jsonable(summary)
    report.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True)
    main(parser.parse_args())
