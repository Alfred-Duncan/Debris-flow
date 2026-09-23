"""RiskTemporal-v2A soft, front-connected support writeback; v1 guard is untouched."""
from __future__ import annotations
import torch
import torch.nn.functional as F

WET_THRESHOLD_M = .05

def apply_soft_support_guard(encoded_provisional, encoded_local, current_physical, provisional_physical, active, transform, *, radius_cells=1, alpha_front=.5):
    if radius_cells not in (1, 2) or not 0. <= alpha_front <= 1.: raise ValueError("SOFT_SUPPORT_CONFIG_INVALID")
    active_mask=(active if active.ndim==4 else active[:,None]).bool(); a=active_mask & ((current_physical[:,:1]>=WET_THRESHOLD_M)|(provisional_physical[:,:1]>=WET_THRESHOLD_M))
    dilated=F.max_pool2d(a.float(),2*radius_cells+1,1,radius_cells).bool(); b=active_mask & ~a & dilated; c=active_mask & ~a & ~b
    proposal=transform.decode(encoded_local); soft=provisional_physical+float(alpha_front)*(proposal-provisional_physical); soft_encoded=transform.encode(soft)
    written=torch.where(a,encoded_local,torch.where(b,soft_encoded,encoded_provisional)); delta=(encoded_local-encoded_provisional); changed=delta.abs().any(1,keepdim=True)&active_mask
    def frac(mask): return float(mask.sum().item()/max(int(active_mask.sum().item()),1))
    def norm(mask): return float((delta.square()*mask).sum().sqrt().item())
    raw_h=proposal[:,:1]; soft_h=soft[:,:1]; newly=(provisional_physical[:,:1]<WET_THRESHOLD_M)
    return written,{"support_cell_count":int(a.sum().item()),"support_fraction":frac(a),"blocked_local_change_cell_count":int((changed&c).sum().item()),"blocked_local_change_fraction":float((changed&c).sum().item()/max(int(changed.sum().item()),1)),"blocked_wet_creation_count":int((newly&c&(raw_h>=WET_THRESHOLD_M)).sum().item()),"raw_local_new_wet_fraction":frac(newly&~a&(raw_h>=WET_THRESHOLD_M)),"fraction_zone_A":frac(a),"fraction_zone_B":frac(b),"fraction_zone_C":frac(c),"local_delta_norm_A":norm(a),"local_delta_norm_B":norm(b),"new_wet_created_A":int((newly&a&(raw_h>=WET_THRESHOLD_M)).sum().item()),"new_wet_created_B":int((newly&b&(soft_h>=WET_THRESHOLD_M)).sum().item()),"softened_local_change_fraction":float((changed&b).sum().item()/max(int(changed.sum().item()),1)),"newly_wet_fraction":frac(newly&(torch.where(a,raw_h,torch.where(b,soft_h,provisional_physical[:,:1]))>=WET_THRESHOLD_M))}
