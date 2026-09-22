"""Deterministic wet-support-preserving Local Corrector writeback."""
from __future__ import annotations

import torch

WET_THRESHOLD_M = .05


def _active(active: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    mask = active if active.ndim == 4 else active[:, None]
    return mask.to(device=reference.device).bool()


def support_mask(current_physical: torch.Tensor, provisional_physical: torch.Tensor,
                 active: torch.Tensor, wet_threshold: float = WET_THRESHOLD_M) -> torch.Tensor:
    """Deployable support: active and wet in current or Global provisional h."""
    if current_physical.shape != provisional_physical.shape or current_physical.shape[1] < 1:
        raise ValueError("current/provisional physical states must have matching state channels")
    return _active(active, current_physical) & ((current_physical[:, :1] >= wet_threshold) |
                                                (provisional_physical[:, :1] >= wet_threshold))


def apply_support_guard(encoded_provisional: torch.Tensor, encoded_local_corrected: torch.Tensor,
                        current_physical: torch.Tensor, provisional_physical: torch.Tensor,
                        active: torch.Tensor, wet_threshold: float = WET_THRESHOLD_M,
                        local_corrected_physical: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    """Allow the complete six-channel local proposal only inside flow support.

    ``local_corrected_physical`` is telemetry-only and is supplied by inference
    after decoding/projecting the raw proposal.  It is never used to decide
    writeback, which depends exclusively on current/provisional/active fields.
    """
    if encoded_provisional.shape != encoded_local_corrected.shape:
        raise ValueError("encoded provisional and local proposal shapes must match")
    support = support_mask(current_physical, provisional_physical, active, wet_threshold)
    active_mask = _active(active, encoded_provisional)
    changed = (encoded_local_corrected != encoded_provisional).any(dim=1, keepdim=True) & active_mask
    blocked = changed & ~support
    if local_corrected_physical is None:
        # Intended for small tensor unit tests. Production passes a decoded,
        # physically projected proposal for this threshold-based telemetry.
        raw_h = encoded_local_corrected[:, :1]
    else:
        if local_corrected_physical.shape[:2] != encoded_local_corrected.shape[:2]:
            raise ValueError("telemetry physical proposal shape mismatch")
        raw_h = local_corrected_physical[:, :1]
    raw_new_wet = active_mask & ~support & (raw_h >= wet_threshold)
    active_count = int(active_mask.sum().item())
    changed_count = int(changed.sum().item())
    telemetry = {
        "support_cell_count": int(support.sum().item()),
        "support_fraction": float(support.sum().item() / max(active_count, 1)),
        "blocked_local_change_cell_count": int(blocked.sum().item()),
        "blocked_local_change_fraction": float(blocked.sum().item() / max(changed_count, 1)),
        "blocked_wet_creation_count": int(raw_new_wet.sum().item()),
        "raw_local_new_wet_fraction": float(raw_new_wet.sum().item() / max(active_count, 1)),
    }
    return torch.where(support, encoded_local_corrected, encoded_provisional), telemetry
