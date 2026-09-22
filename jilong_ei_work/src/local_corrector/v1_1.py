"""V1.1 protocol helpers: route-front sampling and exact RNG persistence."""
from __future__ import annotations
import copy
import numpy as np
import torch
from src.global_operator_v2.oracle_refinement import PatchLayout


def true_front_patch(layout: PatchLayout, truth_physical: torch.Tensor, active: torch.Tensor,
                     route_chainage_m: np.ndarray):
    """Return the eligible core with most cells in the true ±1 km debris front."""
    route=np.asarray(route_chainage_m); valid=np.isfinite(route)
    debris=((truth_physical[0,0]>.1)&(truth_physical[0,3]>.05)&active[0,0].bool()).detach().cpu().numpy()&valid
    if not debris.any():return None
    front=float(route[debris].max());zone=(np.abs(route-front)<=1000.)&valid
    ranked=sorted(layout.eligible,key=lambda p:(-int(zone[p.r0:p.r1,p.c0:p.c1].sum()),p.patch_id))
    return ranked[0] if ranked and int(zone[ranked[0].r0:ranked[0].r1,ranked[0].c0:ranked[0].c1].sum()) else None


def audit_normalization(data: dict, train_ids) -> None:
    if data.get('fit_scope')!='TRAIN_ONLY':raise RuntimeError('LOCAL_NORMALIZATION_NOT_TRAIN_ONLY')
    sources=set(data.get('source_cases',[]));allowed=set(train_ids)
    if not sources or not sources.issubset(allowed) or any(x.upper().find('TEST')>=0 or x.upper().find('H0')>=0 for x in sources):raise RuntimeError('LOCAL_NORMALIZATION_SCOPE_INVALID')
    channels=data.get('per_channel',{})
    if len(channels)!=6 or any(float(value.get('coverage',0.))<.999 for value in channels.values()):raise RuntimeError('LOCAL_NORMALIZATION_COVERAGE_INVALID')


def snapshot_local_rng(rng: np.random.Generator):return copy.deepcopy(rng.bit_generator.state)
def restore_local_rng(state):
    rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(state);return rng
