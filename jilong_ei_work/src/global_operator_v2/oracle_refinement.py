"""VAL-only, ground-truth oracle patch-refinement feasibility primitives.

This module deliberately contains no selector or trainable local model.
``OraclePerfect`` ranks cores with the target state and replaces those cores
with the target; it is therefore an explicitly non-deployable upper bound.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import torch

from .losses import STORAGE_WET_THRESHOLD_M
from .metrics import debris_front


STATE_NAMES = ("h", "hu", "hv", "c", "ice", "dz")


@dataclass(frozen=True)
class Patch:
    patch_id: int
    row_id: int
    col_id: int
    r0: int
    r1: int
    c0: int
    c1: int
    active_cell_count: int


class PatchLayout:
    """A deterministic non-overlapping 16x16 core layout, clipped to fields."""
    def __init__(self, active: np.ndarray, rows: int = 16, cols: int = 16):
        active = np.asarray(active, dtype=bool)
        if active.ndim != 2:
            raise ValueError("active must have shape H,W")
        self.height, self.width, self.rows, self.cols = *active.shape, int(rows), int(cols)
        self.pad_height = int(math.ceil(self.height / self.rows) * self.rows)
        self.pad_width = int(math.ceil(self.width / self.cols) * self.cols)
        self.core_height = self.pad_height // self.rows
        self.core_width = self.pad_width // self.cols
        patches = []
        for row_id in range(self.rows):
            for col_id in range(self.cols):
                r0, c0 = row_id * self.core_height, col_id * self.core_width
                r1, c1 = min(r0 + self.core_height, self.height), min(c0 + self.core_width, self.width)
                count = int(active[r0:r1, c0:c1].sum())
                patches.append(Patch(len(patches), row_id, col_id, r0, r1, c0, c1, count))
        self.patches = tuple(patches)
        self.eligible = tuple(p for p in self.patches if p.active_cell_count > 0)
        self.active_cell_count = int(active.sum())

    def count_for_budget(self, fraction: float) -> int:
        if not 0 <= float(fraction) <= 1:
            raise ValueError("patch budget must be in [0, 1]")
        return min(len(self.eligible), int(math.ceil(float(fraction) * len(self.eligible))))

    def selected_active_fraction(self, selected: list[Patch] | tuple[Patch, ...]) -> float:
        return float(sum(p.active_cell_count for p in selected) / max(self.active_cell_count, 1))

    def masks(self, selected: list[Patch] | tuple[Patch, ...], device=None) -> torch.Tensor:
        mask = torch.zeros((self.height, self.width), dtype=torch.bool, device=device)
        for patch in selected:
            mask[patch.r0:patch.r1, patch.c0:patch.c1] = True
        return mask


def oracle_patch_scores(pred_t: torch.Tensor, truth_t: torch.Tensor, truth_current: torch.Tensor,
                        truth_next: torch.Tensor, active: torch.Tensor, layout: PatchLayout,
                        delta_scales: list[float] | torch.Tensor) -> tuple[np.ndarray, float]:
    """Immediate teacher-normalized error mass in each eligible core.

    Scores use only active, teacher-relevant cells.  Padding is never
    represented by a patch core and cannot affect either rankings or totals.
    """
    scale = torch.as_tensor(delta_scales, dtype=pred_t.dtype, device=pred_t.device)[None, :, None, None]
    teacher_relevant = ((truth_current[:, 0:1] >= .03) | (truth_next[:, 0:1] >= .03) |
                        ((truth_next[:, 5:6] - truth_current[:, 5:6]).abs() > 1e-8))
    relevant = teacher_relevant & active.bool()
    cell_error = ((pred_t - truth_t) / scale).square().mean(1)[0] * relevant[:, 0].to(pred_t.dtype)
    scores = np.zeros(len(layout.eligible), dtype=np.float64)
    for index, patch in enumerate(layout.eligible):
        scores[index] = float(cell_error[patch.r0:patch.r1, patch.c0:patch.c1].sum().detach().cpu())
    return scores, float(cell_error.sum().detach().cpu())


def select_oracle(layout: PatchLayout, scores: np.ndarray, count: int) -> tuple[Patch, ...]:
    """Stable patch-id tie breaking makes the ground-truth upper bound exact."""
    if len(scores) != len(layout.eligible):
        raise ValueError("score/layout length mismatch")
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), layout.eligible[i].patch_id))
    return tuple(layout.eligible[i] for i in order[:int(count)])


def select_random(layout: PatchLayout, count: int, seed: int) -> tuple[Patch, ...]:
    rng = np.random.default_rng(int(seed))
    picks = rng.choice(len(layout.eligible), size=int(count), replace=False) if count else np.empty(0, int)
    return tuple(layout.eligible[int(i)] for i in sorted(picks))


def correct_cores(provisional: torch.Tensor, truth_next: torch.Tensor, selected: tuple[Patch, ...],
                  layout: PatchLayout, active: torch.Tensor) -> torch.Tensor:
    """Perfect CORE-only correction; inactive cells retain physical projection mask."""
    mask = layout.masks(selected, provisional.device)[None, None] & active.bool()
    return torch.where(mask, truth_next, provisional)


def concentration_fractions(scores: np.ndarray, fractions=(.01, .05, .10, .20, .30, .50)) -> dict[str, float]:
    total = float(np.sum(scores))
    answer = {}
    for fraction in fractions:
        count = min(len(scores), int(math.ceil(float(fraction) * len(scores))))
        answer[f"top{int(round(fraction * 100))}_fraction"] = (float(np.sort(scores)[-count:].sum()) / total) if total > 0 and count else 0.
    return answer


class StreamingMetrics:
    """Metric accumulation with global numerators/denominators, never trajectories."""
    def __init__(self, active: torch.Tensor, route_chainage_m: np.ndarray, change_threshold: float, cell_area: float = 900.):
        self.active = active.bool()
        self.mask = self.active[:, 0]
        self.route = np.asarray(route_chainage_m, np.float32)
        self.route_length_km = float(np.nanmax(self.route) / 1000.)
        self.change_threshold = float(change_threshold)
        self.cell_area = float(cell_area)
        self.num = {name: 0. for name in STATE_NAMES}; self.den = {name: 0. for name in STATE_NAMES}
        self.change_num = {name: 0. for name in STATE_NAMES}; self.change_den = {name: 0. for name in STATE_NAMES}
        self.front_num = {name: 0. for name in ("h", "momentum")}; self.front_den = {name: 0. for name in ("h", "momentum")}
        self.wet_sum = 0.; self.wet_count = 0; self.final_wet_iou = float("nan")
        self.front_wet_sum = 0.; self.front_wet_count = 0
        self.volume_errors = []; self.front_errors = []; self.final_front_error = float("nan")
        self.new_intersection = 0; self.new_union = 0; self.new_pred = 0; self.new_truth = 0

    @staticmethod
    def _rel(num, den): return math.sqrt(num / max(den, 1e-12))
    @staticmethod
    def _iou(pred, truth, mask):
        union = ((pred | truth) & mask).sum().item()
        return float(((pred & truth) & mask).sum().item() / union) if union else float("nan")

    def add(self, pred: torch.Tensor, truth_current: torch.Tensor, truth_next: torch.Tensor, transform) -> None:
        p, cur, target = (item[0] for item in (pred, truth_current, truth_next))
        active = self.mask
        for index, name in enumerate(STATE_NAMES):
            self.num[name] += float((((p[index] - target[index]).square()) * active).sum().detach().cpu())
            self.den[name] += float(((target[index].square()) * active).sum().detach().cpu())
        current_t, next_t = transform.encode(truth_current), transform.encode(truth_next)
        changed = ((next_t - current_t).abs().mean(1)[0] >= self.change_threshold) & active
        for index, name in enumerate(STATE_NAMES):
            self.change_num[name] += float((((p[index] - target[index]).square()) * changed).sum().detach().cpu())
            self.change_den[name] += float(((target[index].square()) * changed).sum().detach().cpu())
        p_wet, t_wet = p[0] >= STORAGE_WET_THRESHOLD_M, target[0] >= STORAGE_WET_THRESHOLD_M
        iou = self._iou(p_wet, t_wet, active)
        if math.isfinite(iou): self.wet_sum += iou; self.wet_count += 1; self.final_wet_iou = iou
        teacher_dry = cur[0] < .05; truth_new = teacher_dry & (target[0] >= .05) & active; pred_new = teacher_dry & (p[0] >= .05) & active
        self.new_intersection += int((truth_new & pred_new).sum().item()); self.new_union += int((truth_new | pred_new).sum().item())
        self.new_pred += int(pred_new.sum().item()); self.new_truth += int(truth_new.sum().item())
        volume_p = float((p[0] * active).sum().detach().cpu()) * self.cell_area; volume_t = float((target[0] * active).sum().detach().cpu()) * self.cell_area
        self.volume_errors.append(abs(volume_p - volume_t) / max(abs(volume_t), 1.))
        p_front = debris_front(p.detach().cpu().numpy(), self.route); t_front = debris_front(target.detach().cpu().numpy(), self.route)
        if math.isfinite(t_front):
            if math.isfinite(p_front): error = abs(p_front - t_front)
            else: error = self.route_length_km
            self.front_errors.append(error); self.final_front_error = error
            zone = active & torch.from_numpy(np.isfinite(self.route) & (np.abs(self.route - t_front * 1000.) <= 1000.)).to(active.device)
            for key, indices in (("h", (0,)), ("momentum", (1, 2))):
                self.front_num[key] += float(((p[list(indices)] - target[list(indices)]).square() * zone).sum().detach().cpu())
                self.front_den[key] += float((target[list(indices)].square() * zone).sum().detach().cpu())
            front_iou = self._iou(p_wet, t_wet, zone)
            if math.isfinite(front_iou): self.front_wet_sum += front_iou; self.front_wet_count += 1

    def result(self) -> dict[str, float]:
        out = {f"trajectory_{name}_rel_l2": self._rel(self.num[name], self.den[name]) for name in STATE_NAMES}
        out["trajectory_momentum_rel_l2"] = self._rel(self.num["hu"] + self.num["hv"], self.den["hu"] + self.den["hv"])
        out.update({f"change_region_{name}_rel_l2": self._rel(self.change_num[name], self.change_den[name]) for name in STATE_NAMES})
        out["change_region_momentum_rel_l2"] = self._rel(self.change_num["hu"] + self.change_num["hv"], self.change_den["hu"] + self.change_den["hv"])
        out.update({"mean_wet_iou": self.wet_sum / max(self.wet_count, 1), "final_wet_iou": self.final_wet_iou,
                    "mixture_volume_relative_error": float(np.mean(self.volume_errors)) if self.volume_errors else float("nan"),
                    "debris_front_mae_km": float(np.mean(self.front_errors)) if self.front_errors else float("nan"),
                    "debris_front_final_error_km": self.final_front_error,
                    "newly_wet_iou": self.new_intersection / self.new_union if self.new_union else float("nan"),
                    "newly_wet_precision": self.new_intersection / self.new_pred if self.new_pred else float("nan"),
                    "newly_wet_recall": self.new_intersection / self.new_truth if self.new_truth else float("nan"),
                    "front_zone_h_rel_l2": self._rel(self.front_num["h"], self.front_den["h"]),
                    "front_zone_momentum_rel_l2": self._rel(self.front_num["momentum"], self.front_den["momentum"]),
                    "front_zone_wet_iou": self.front_wet_sum / max(self.front_wet_count, 1)})
        return out
