"""Predeclared EngineeringROI-v2 SupportRisk selector.

V2 is a mechanism-driven simplification of V1, not a retuning of its four
component weights. It intentionally scores only support risk: V1 formal
ablations identified that signal as the stable propagation-oriented input.
The one-step state prevents immediate repeated autoregressive writes to the
same logical patch; it is reset by the evaluator for every scenario.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .engineering_roi import (
    ROIStaticMetadata,
    _diverse,
    compute_support_candidates,
    compute_support_risk_component,
    positive_percentile_rank,
)

_EXPECTED_CONFIG = {
    "version": "EngineeringROI-v2", "wet_threshold_m": 0.05,
    "shallow_margin_upper_h_m": 0.15, "roi_component": "support_risk",
    "support_risk_definition": "new_wet_or_shallow_margin",
    "candidate_rule": "support_overlap_and_positive_support_risk",
    "normalization": "positive_percentile_rank",
    "spatial_diversity": "nonadjacent_first_then_fill",
    "temporal_refresh": "exclude_previous_step_selected_patches",
    "temporal_refresh_steps": 1, "temporal_fill_from_excluded": False,
    "depth_envelope_guard": "current_global_convex_hull",
    "depth_envelope_channels": ["h"], "budgets": [0.05, 0.10, 0.20],
    "global_model": "frozen", "local_model": "best_update_4000",
    "support_guard": "current_or_global_provisional_wet",
    "truth_in_roi_scoring": False,
}


@dataclass(frozen=True)
class TemporalRefreshState:
    """Only immediately preceding logical-patch IDs are retained."""
    previous_selected_patch_ids: frozenset[int] = frozenset()


def initial_temporal_state() -> TemporalRefreshState:
    return TemporalRefreshState()


def load_engineering_roi_v2_config(path: str | Path) -> tuple[dict, str]:
    """Load the fixed predeclared design and its byte-level SHA256."""
    raw = Path(path).read_bytes()
    config = json.loads(raw.decode("utf-8"))
    if config != _EXPECTED_CONFIG:
        raise RuntimeError("ENGINEERING_ROI_V2_CONFIG_INVALID")
    return config, hashlib.sha256(raw).hexdigest()


def select_engineering_roi_v2(
    current_physical,
    provisional_physical,
    metadata: ROIStaticMetadata,
    config: Mapping,
    budget_fraction: float,
    temporal_state: TemporalRefreshState,
    *,
    temporal_refresh: bool = True,
):
    """Allocate a maximum patch budget from current/global states only.

    The reused risk primitive is precisely V1's newly-wet OR shallow-margin
    definition, active-core normalized and positive-percentile ranked. This
    API intentionally has no target, future state, Local proposal, or teacher
    parameter. Excluded patches are never reintroduced to fill a budget.
    """
    if float(budget_fraction) not in tuple(map(float, config["budgets"])):
        raise ValueError("ENGINEERING_ROI_V2_BUDGET_INVALID")
    if config.get("temporal_refresh_steps") != 1 or config.get("temporal_fill_from_excluded") is not False:
        raise RuntimeError("ENGINEERING_ROI_V2_TEMPORAL_POLICY_INVALID")
    _, support_overlap = compute_support_candidates(
        current_physical, provisional_physical, metadata, config["wet_threshold_m"]
    )
    raw_risk = compute_support_risk_component(
        current_physical, provisional_physical, metadata, config
    ).detach().cpu().numpy()
    risk_rank = positive_percentile_rank(raw_risk)
    patches = tuple(metadata.layout.eligible)
    candidates = np.flatnonzero((support_overlap.detach().cpu().numpy() > 0) & (risk_rank > 0))
    prior = temporal_state.previous_selected_patch_ids
    usable = np.asarray(
        [index for index in candidates if not temporal_refresh or patches[int(index)].patch_id not in prior], dtype=int
    )
    maximum = metadata.layout.count_for_budget(float(budget_fraction))
    indices, first_pass, second_pass = _diverse(usable, risk_rank, patches, maximum)
    selected = tuple(patches[index] for index in indices)
    selected_ids = frozenset(patch.patch_id for patch in selected)
    consecutive = len(selected_ids & prior)
    return selected, TemporalRefreshState(selected_ids), {
        "budget_fraction": float(budget_fraction), "budget_max_count": int(maximum),
        "pre_refresh_candidate_count": int(len(candidates)), "candidate_count": int(len(usable)),
        "selected_count": int(len(selected)),
        "actual_active_fraction": float(metadata.layout.selected_active_fraction(selected)) if selected else 0.0,
        "mean_selected_support_risk_rank": float(np.mean(risk_rank[indices])) if indices else 0.0,
        "first_pass_count": int(first_pass), "second_pass_count": int(second_pass),
        "selected_patch_consecutive_overlap_count": int(consecutive),
        "selected_patch_consecutive_overlap_fraction": float(consecutive / max(len(selected), 1)),
    }
