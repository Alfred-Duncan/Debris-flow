"""Frozen deterministic EngineeringROI-v1 patch allocation primitives.

The scoring API deliberately contains no teacher, target, oracle, or future
state.  It is a deployable function of the current prediction, one frozen
Global provisional step, and fixed engineering metadata only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from src.global_operator_v2.oracle_refinement import PatchLayout, select_random
from .support_guard import support_mask

COMPONENTS = ("dynamic", "predicted_front", "support_risk", "engineering_section")
STRATEGIES = ("random_all", "random_support", "dynamic_only", "front_only", "support_risk_only",
              "section_only", "engineering_roi", "engineering_roi_no_diversity")


def load_engineering_roi_config(path: str | Path) -> tuple[dict, str]:
    raw = Path(path).read_bytes(); config = json.loads(raw.decode("utf-8"))
    if config.get("version") != "EngineeringROI-v1" or config.get("components") != list(COMPONENTS):
        raise RuntimeError("ENGINEERING_ROI_CONFIG_INVALID")
    weights = config.get("weights", {})
    if any(float(weights.get(name, float("nan"))) != .25 for name in COMPONENTS) or sum(map(float, weights.values())) != 1.0:
        raise RuntimeError("ENGINEERING_ROI_WEIGHTS_INVALID")
    if config.get("truth_in_roi_scoring") is not False:
        raise RuntimeError("ENGINEERING_ROI_TRUTH_CONFIG_INVALID")
    return config, hashlib.sha256(raw).hexdigest()


def build_section_mask(transects: Mapping, shape: tuple[int, int]) -> torch.Tensor:
    """Build a strict static mask from the production transect row/col points."""
    height, width = map(int, shape); mask = torch.zeros((height, width), dtype=torch.bool); points = 0
    if not isinstance(transects, Mapping) or not transects:
        raise RuntimeError("ENGINEERING_SECTION_METADATA_INVALID")
    for name, transect in transects.items():
        if not isinstance(transect, Mapping):
            raise RuntimeError(f"ENGINEERING_SECTION_METADATA_INVALID:{name}")
        rows, cols = transect.get("rows"), transect.get("cols")
        if rows is None or cols is None or len(rows) != len(cols):
            raise RuntimeError(f"ENGINEERING_SECTION_METADATA_INVALID:{name}")
        for row, col in zip(rows, cols):
            row, col = int(row), int(col)
            if not (0 <= row < height and 0 <= col < width):
                raise RuntimeError(f"ENGINEERING_SECTION_METADATA_INVALID:{name}")
            mask[row, col] = True; points += 1
    if points == 0 or not bool(mask.any()):
        raise RuntimeError("ENGINEERING_SECTION_METADATA_INVALID")
    return mask


def positive_percentile_rank(values) -> np.ndarray:
    """Rank strictly positive finite values by their inclusive empirical CDF."""
    raw = np.asarray(values, dtype=np.float64); ranked = np.zeros_like(raw)
    positive = np.isfinite(raw) & (raw > 0)
    pool = raw[positive]
    if pool.size:
        ranked[positive] = np.array([(pool <= value).sum() / pool.size for value in pool], dtype=np.float64)
    return ranked


def _field(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim != 4 or value.shape[0] != 1:
        raise ValueError(f"{name} must have shape [1,C,H,W]")
    return value


def compute_roi_components(current_physical: torch.Tensor, provisional_physical: torch.Tensor,
                           encoded_current: torch.Tensor, encoded_provisional: torch.Tensor,
                           active: torch.Tensor, route_chainage_m, section_mask: torch.Tensor,
                           global_delta_scales, layout: PatchLayout, config: Mapping) -> dict:
    """Compute all four immutable EngineeringROI-v1 components once per step."""
    current_physical = _field(current_physical, "current_physical"); provisional_physical = _field(provisional_physical, "provisional_physical")
    encoded_current = _field(encoded_current, "encoded_current"); encoded_provisional = _field(encoded_provisional, "encoded_provisional")
    if current_physical.shape != provisional_physical.shape or encoded_current.shape != encoded_provisional.shape or current_physical.shape[-2:] != encoded_current.shape[-2:]:
        raise ValueError("ROI state shapes must match")
    if encoded_current.shape[1] != 6:
        raise ValueError("EngineeringROI requires six encoded state channels")
    active_mask = (active if active.ndim == 4 else active[:, None]).to(current_physical.device).bool()
    if active_mask.shape != current_physical[:, :1].shape:
        raise ValueError("active mask shape mismatch")
    route = np.asarray(route_chainage_m, dtype=np.float64)
    if route.shape != tuple(current_physical.shape[-2:]):
        raise ValueError("route_chainage_m shape mismatch")
    section = torch.as_tensor(section_mask, device=current_physical.device, dtype=torch.bool)
    if tuple(section.shape) != tuple(current_physical.shape[-2:]):
        raise ValueError("section_mask shape mismatch")
    wet = float(config["wet_threshold_m"]); shallow = float(config["shallow_margin_upper_h_m"])
    support = support_mask(current_physical, provisional_physical, active_mask, wet)
    scales = torch.as_tensor(global_delta_scales, device=encoded_current.device, dtype=encoded_current.dtype)[None, :, None, None]
    if scales.shape[1] != 6:
        raise ValueError("global_delta_scales must contain six channels")
    dynamic_cell = (((encoded_provisional - encoded_current) / scales).square().mean(1, keepdim=True)).sqrt()
    h, c = provisional_physical[:, :1], provisional_physical[:, 3:4]
    route_tensor = torch.as_tensor(route, device=h.device)
    front_cells = (h[:, 0] > float(config["debris_front_h_threshold_m"])) & (c[:, 0] > float(config["debris_front_c_threshold"])) & torch.isfinite(route_tensor)[None]
    front_available = bool(front_cells.any().item())
    front_chainage_m = float(route_tensor[front_cells[0]].max().item()) if front_available else float("nan")
    front_zone = torch.zeros_like(support)
    if front_available:
        front_zone = (torch.abs(route_tensor[None, None] - front_chainage_m) <= float(config["front_half_width_m"])) & active_mask & support
    new_wet = (current_physical[:, :1] < wet) & (h >= wet)
    shallow_margin = (h >= wet) & (h < shallow)
    risk_mask = active_mask & (new_wet | shallow_margin)
    raw = {name: [] for name in COMPONENTS}; support_overlap = []; patch_ids = []
    for patch in layout.eligible:
        core = (slice(patch.r0, patch.r1), slice(patch.c0, patch.c1)); support_core = support[0, 0][core]; active_core = active_mask[0, 0][core]
        active_count = int(active_core.sum().item()); overlap = int(support_core.sum().item()); section_core = section[core]
        patch_ids.append(patch.patch_id); support_overlap.append(overlap)
        raw["dynamic"].append(float(dynamic_cell[0, 0][core][support_core].mean().item()) if overlap else 0.0)
        raw["predicted_front"].append(float(front_zone[0, 0][core].sum().item() / max(active_count, 1)))
        raw["support_risk"].append(float(risk_mask[0, 0][core].sum().item() / max(active_count, 1)))
        section_count = int(section_core.sum().item())
        raw["engineering_section"].append(float((section_core & support_core).sum().item() / section_count) if section_count else 0.0)
    raw = {name: np.asarray(values, dtype=np.float64) for name, values in raw.items()}
    ranks = {name: positive_percentile_rank(raw[name]) for name in COMPONENTS}
    base = sum(float(config["weights"][name]) * ranks[name] for name in COMPONENTS)
    overlap = np.asarray(support_overlap, dtype=np.int64); candidates = (overlap > 0) & (base > 0)
    return {"patches": tuple(layout.eligible), "patch_ids": np.asarray(patch_ids, dtype=np.int64), "support_overlap": overlap,
            "raw": raw, "ranks": ranks, "base_score": base, "candidates": candidates,
            "predicted_front_chainage_m": front_chainage_m, "front_available": front_available}


def _ordered(indices: np.ndarray, scores: np.ndarray, patches) -> list[int]:
    return sorted(map(int, indices), key=lambda index: (-float(scores[index]), int(patches[index].patch_id)))


def _diverse(indices: np.ndarray, scores: np.ndarray, patches, maximum: int) -> tuple[list[int], int, int]:
    ranked = _ordered(indices, scores, patches); chosen = []
    for index in ranked:
        patch = patches[index]
        if all(max(abs(patch.row_id - patches[other].row_id), abs(patch.col_id - patches[other].col_id)) >= 2 for other in chosen):
            chosen.append(index)
            if len(chosen) == maximum: return chosen, len(chosen), 0
    first = len(chosen)
    for index in ranked:
        if index not in chosen:
            chosen.append(index)
            if len(chosen) == maximum: break
    return chosen, first, len(chosen) - first


def select_engineering_roi(current_physical: torch.Tensor, provisional_physical: torch.Tensor,
                           encoded_current: torch.Tensor, encoded_provisional: torch.Tensor,
                           active: torch.Tensor, route_chainage_m, section_mask: torch.Tensor,
                           global_delta_scales, layout: PatchLayout, config: Mapping,
                           budget_fraction: float, strategy: str = "engineering_roi", seed: int | None = None) -> tuple[tuple, dict]:
    """Select up-to-budget patches under the immutable deterministic V1 rule."""
    if strategy not in STRATEGIES: raise ValueError("ENGINEERING_ROI_STRATEGY_INVALID")
    if float(budget_fraction) not in tuple(map(float, config["budgets"])): raise ValueError("ENGINEERING_ROI_BUDGET_INVALID")
    info = compute_roi_components(current_physical, provisional_physical, encoded_current, encoded_provisional, active, route_chainage_m, section_mask, global_delta_scales, layout, config)
    patches, maximum = info["patches"], layout.count_for_budget(budget_fraction)
    if strategy == "random_all":
        selected = tuple(select_random(layout, maximum, int(seed)))
        selected_indices = [next(i for i, patch in enumerate(patches) if patch.patch_id == value.patch_id) for value in selected]; first, second = len(selected), 0
    elif strategy == "random_support":
        candidates = np.flatnonzero(info["support_overlap"] > 0); rng = np.random.default_rng(int(seed)); picks = rng.choice(candidates, size=min(maximum, len(candidates)), replace=False) if len(candidates) else np.empty(0, dtype=int)
        selected_indices = sorted(map(int, picks), key=lambda index: patches[index].patch_id); selected = tuple(patches[index] for index in selected_indices); first, second = len(selected), 0
    else:
        key = {"dynamic_only":"dynamic", "front_only":"predicted_front", "support_risk_only":"support_risk", "section_only":"engineering_section"}.get(strategy)
        scores = info["ranks"][key] if key else info["base_score"]
        candidates = np.flatnonzero((info["support_overlap"] > 0) & (scores > 0))
        if strategy == "engineering_roi_no_diversity": selected_indices = _ordered(candidates, scores, patches)[:maximum]; first, second = len(selected_indices), 0
        else: selected_indices, first, second = _diverse(candidates, scores, patches, maximum)
        selected = tuple(patches[index] for index in selected_indices)
    def average(name): return float(np.mean(info["ranks"][name][selected_indices])) if selected_indices else 0.0
    candidate_count = len(layout.eligible) if strategy == "random_all" else (int((info["support_overlap"] > 0).sum()) if strategy == "random_support" else int(((info["support_overlap"] > 0) & (scores > 0)).sum()))
    telemetry = {"budget_fraction":float(budget_fraction), "budget_max_count":maximum, "candidate_count":candidate_count,
        "selected_count":len(selected), "actual_active_fraction":layout.selected_active_fraction(selected), "predicted_front_chainage_m":info["predicted_front_chainage_m"], "front_available":info["front_available"],
        "mean_selected_dynamic_rank":average("dynamic"), "mean_selected_front_rank":average("predicted_front"), "mean_selected_support_risk_rank":average("support_risk"), "mean_selected_section_rank":average("engineering_section"),
        "mean_selected_base_score":float(np.mean(info["base_score"][selected_indices])) if selected_indices else 0.0, "selected_section_patch_count":int(sum(info["ranks"]["engineering_section"][index] > 0 for index in selected_indices)), "first_pass_count":first, "second_pass_count":second}
    return selected, telemetry
