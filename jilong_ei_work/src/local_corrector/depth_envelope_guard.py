"""Truth-free h-only envelope for V2 macro-transition refinement."""
from __future__ import annotations
import torch
def apply_depth_envelope_guard(current, provisional, corrected, active):
    low=torch.minimum(current[:,:1],provisional[:,:1]);high=torch.maximum(current[:,:1],provisional[:,:1]);before=corrected[:,:1];after=torch.where(active,torch.clamp(before,low,high),before);out=corrected.clone();out[:,:1]=after;clip=(before-after).abs();den=active.sum().clamp_min(1);return out,{'depth_guard_activation_fraction':float(((clip>0)&active).sum().item()/den.item()),'depth_guard_mean_abs_clip_m':float(clip[active].mean().item()) if active.any() else 0.,'depth_guard_max_abs_clip_m':float(clip.max().item())}
