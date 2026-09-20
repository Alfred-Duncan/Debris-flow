from __future__ import annotations
import torch
import torch.nn.functional as F
def masked_huber(pred,target,active,dry=.03,wet_weight=2.):
    base=F.smooth_l1_loss(pred,target,reduction='none'); a=active[:,None].float(); wet=((pred[:,0]>dry)|(target[:,0]>dry))[:,None].float(); w=a*(1+wet_weight*wet)
    each=(base*w).sum((0,2,3))/w.sum((0,2,3)).clamp_min(1); return each.mean(),each
