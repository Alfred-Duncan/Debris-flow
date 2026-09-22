"""Fixed-shape logical patch extraction and core-only writeback."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.global_operator_v2.oracle_refinement import Patch, PatchLayout

HALO = 8


def _logical_shape(layout: PatchLayout) -> tuple[int, int]:
    return layout.core_height, layout.core_width


def _pad_logical(value: torch.Tensor, layout: PatchLayout) -> torch.Tensor:
    """Zero-pad fields to the logical 16x16 canvas."""
    ph, pw = layout.pad_height - value.shape[-2], layout.pad_width - value.shape[-1]
    if ph < 0 or pw < 0:
        raise ValueError("field is larger than the patch layout")
    return F.pad(value, (0, pw, 0, ph))


def extract(value: torch.Tensor, patch: Patch, layout: PatchLayout, halo: int = HALO) -> torch.Tensor:
    """Return the fixed logical footprint (core + halo) for one patch."""
    padded = _pad_logical(value, layout)
    # Torch's replication padding has no Bool kernel; masks retain their
    # semantics after a temporary float representation.
    context = F.pad(padded.float() if padded.dtype == torch.bool else padded,
                    (halo, halo, halo, halo), mode="replicate")
    if padded.dtype == torch.bool:
        context = context.bool()
    ch, cw = _logical_shape(layout)
    r0, c0 = patch.row_id * ch, patch.col_id * cw
    return context[:, :, r0:r0 + ch + 2 * halo, c0:c0 + cw + 2 * halo]


def core(value: torch.Tensor, layout: PatchLayout, halo: int = HALO) -> torch.Tensor:
    ch, cw = _logical_shape(layout)
    return value[:, :, halo:halo + ch, halo:halo + cw]


def local_features(global_features: torch.Tensor, encoded_provisional: torch.Tensor,
                   encoded_current: torch.Tensor, scales, patches, layout: PatchLayout) -> torch.Tensor:
    scale = torch.as_tensor(scales, device=encoded_provisional.device,
                            dtype=encoded_provisional.dtype)[None, :, None, None]
    transition = (encoded_provisional - encoded_current) / scale
    full = torch.cat((global_features, encoded_provisional, transition), dim=1)
    if full.shape[1] != global_features.shape[1] + 12:
        raise RuntimeError("local feature channel construction failed")
    return torch.cat([extract(full, patch, layout) for patch in patches], dim=0)


def core_active(active: torch.Tensor, patches, layout: PatchLayout) -> torch.Tensor:
    return torch.cat([core(extract(active, patch, layout), layout) for patch in patches], dim=0).bool()


def apply_core(provisional_t: torch.Tensor, correction_t: torch.Tensor, patches, layout: PatchLayout,
               active: torch.Tensor) -> torch.Tensor:
    """Apply correction cores only; logical padding is never written back."""
    out = provisional_t.clone()
    for index, patch in enumerate(patches):
        h, w = patch.r1 - patch.r0, patch.c1 - patch.c0
        if h <= 0 or w <= 0:
            continue
        mask = active[:, :, patch.r0:patch.r1, patch.c0:patch.c1].bool()
        candidate = provisional_t[:, :, patch.r0:patch.r1, patch.c0:patch.c1] + correction_t[index:index + 1, :, :h, :w]
        out[:, :, patch.r0:patch.r1, patch.c0:patch.c1] = torch.where(mask, candidate, out[:, :, patch.r0:patch.r1, patch.c0:patch.c1])
    return out
