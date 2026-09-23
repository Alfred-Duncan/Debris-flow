"""Truth-free, h-only macro-transition envelope for EngineeringROI-v2."""
from __future__ import annotations

import torch


def apply_depth_envelope_guard(current, provisional, corrected, active):
    """Clamp active-cell depth into the current-to-Global interval only.

    It runs after SupportGuard and MomentumGuard writeback and before final
    physical projection. Channels other than h are bitwise preserved here.
    """
    if current.shape != provisional.shape or current.shape != corrected.shape:
        raise ValueError("DEPTH_ENVELOPE_STATE_SHAPE_INVALID")
    if current.ndim != 4 or current.shape[1] < 1:
        raise ValueError("DEPTH_ENVELOPE_CHANNELS_INVALID")
    mask = (active if active.ndim == 4 else active[:, None]).to(corrected.device).bool()
    if mask.shape[1] != 1 or mask.shape[0] != corrected.shape[0] or mask.shape[-2:] != corrected.shape[-2:]:
        raise ValueError("DEPTH_ENVELOPE_ACTIVE_SHAPE_INVALID")
    low, high = torch.minimum(current[:, :1], provisional[:, :1]), torch.maximum(current[:, :1], provisional[:, :1])
    before = corrected[:, :1]
    after = torch.where(mask, torch.clamp(before, low, high), before)
    output = corrected.clone()
    output[:, :1] = after
    clip = (before - after).abs()
    active_clip = clip.masked_select(mask)
    return output, {
        "depth_guard_activation_fraction": float((((clip > 0) & mask).sum().item()) / max(mask.sum().item(), 1)),
        "depth_guard_mean_abs_clip_m": float(active_clip.mean().item()) if active_clip.numel() else 0.0,
        "depth_guard_max_abs_clip_m": float(active_clip.max().item()) if active_clip.numel() else 0.0,
    }
