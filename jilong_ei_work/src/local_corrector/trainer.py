"""Loss and inference primitives for the learned local corrector."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .normalization import bounded
from .patches import apply_core, core, core_active, extract, local_features


def bounded_correction(raw: torch.Tensor, scales, bounds) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = bounded(raw, scales, bounds)
    scale = torch.as_tensor(scales, device=raw.device, dtype=raw.dtype)[None, :, None, None]
    return encoded, encoded / scale


def apply_learned_correction(model, features, encoded_provisional, encoded_current, patches, layout,
                             global_delta_scales, local_correction_scales, local_correction_bounds, active, global_model=None):
    """Infer fixed-shape patches, then make a core-only transformed update."""
    inputs = local_features(features, encoded_provisional, encoded_current, global_delta_scales, patches, layout)
    raw = model(inputs)
    correction, normalized = bounded_correction(core(raw, layout), local_correction_scales, local_correction_bounds)
    corrected = apply_core(encoded_provisional, correction, patches, layout, active)
    if global_model is not None and bool(getattr(global_model, "momentum_state_guard", False)):
        guard = global_model.momentum_state_bounds[None, :, None, None].to(corrected)
        corrected = torch.cat((corrected[:, :1], corrected[:, 1:3].clamp(-guard, guard), corrected[:, 3:]), dim=1)
    nb = torch.as_tensor(local_correction_bounds, device=raw.device, dtype=raw.dtype)[None, :, None, None] / torch.as_tensor(local_correction_scales, device=raw.device, dtype=raw.dtype)[None, :, None, None]
    saturation = (raw.abs() >= nb).float().mean((0, 2, 3))
    return corrected, normalized, saturation


def correction_loss(model, features, provisional_t, truth_t, current_t, provisional_p, truth_p,
                    patches, layout, global_delta_scales, local_correction_scales, local_correction_bounds,
                    active, change_threshold, global_model=None):
    """Core/physical-only normalized Huber objective with prescribed weights."""
    corrected, prediction, saturation = apply_learned_correction(
        model, features, provisional_t, current_t, patches, layout, global_delta_scales, local_correction_scales, local_correction_bounds, active, global_model)
    targets = torch.cat([core(extract(truth_t - provisional_t, patch, layout), layout) for patch in patches], dim=0)
    target = targets / torch.as_tensor(local_correction_scales, device=targets.device, dtype=targets.dtype)[None, :, None, None]
    activity = core_active(active, patches, layout).to(target.dtype)
    prov = torch.cat([core(extract(provisional_p, patch, layout), layout) for patch in patches], dim=0)
    truth = torch.cat([core(extract(truth_p, patch, layout), layout) for patch in patches], dim=0)
    # Change-region thresholds are fitted in transformed space, so both
    # operands must remain transformed.  Physical truth is deliberately not
    # consulted for this mask.
    truth_transformed = torch.cat([core(extract(truth_t, patch, layout), layout) for patch in patches], dim=0)
    current_transformed = torch.cat([core(extract(current_t, patch, layout), layout) for patch in patches], dim=0)
    wet = (truth[:, :1] >= .05) | (prov[:, :1] >= .05)
    fp = (prov[:, :1] >= .05) & (truth[:, :1] < .05)
    changed = (truth_transformed - current_transformed).abs().mean(1, keepdim=True) >= float(change_threshold)
    weight = activity * (1. + wet.to(target.dtype) + fp.to(target.dtype) + changed.to(target.dtype))
    weight = weight / weight.sum().clamp_min(1.) * weight.numel()
    huber = (F.smooth_l1_loss(prediction, target, reduction="none") * weight).mean()
    small = (target.abs().mean(1, keepdim=True) < .05).to(target.dtype) * activity
    magnitude = (prediction.square() * small).sum() / small.expand_as(prediction).sum().clamp_min(1.)
    fp_mask = fp.to(target.dtype) * activity
    fp_direction = (torch.relu(prediction[:, :1]).square() * fp_mask).sum() / fp_mask.sum().clamp_min(1.)
    dry_dry = (prov[:, :1] < .05) & (truth[:, :1] < .05) & (target[:, :1].abs() < .05)
    dry_mask = dry_dry.to(target.dtype) * activity
    dry_identity = (prediction[:, :1].square() * dry_mask).sum() / dry_mask.sum().clamp_min(1.)
    total = huber + .01 * magnitude + .05 * fp_direction + .01 * dry_identity
    return total, {"huber": float(huber.detach()), "mag": float(magnitude.detach()),
                   "fp_direction": float(fp_direction.detach()), "dry_identity": float(dry_identity.detach()),
                   "saturation": saturation.detach().cpu().tolist(), "corrected_t": corrected}
