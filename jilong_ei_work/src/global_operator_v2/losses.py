"""Physical-consistent objectives.  Weight masks always derive from teacher fields."""
from __future__ import annotations
import torch
import torch.nn.functional as F

STORAGE_WET_THRESHOLD_M=.05
MOMENTUM_DRY_THRESHOLD_M=.03

def project_physical(x,dry:float=MOMENTUM_DRY_THRESHOLD_M):
    """Project only the physical constraints represented by the Park-v2 state.

    ``c`` is the total solid/debris column concentration and ``ice`` is its ice
    component.  Therefore the valid simplex is ``0 <= ice <= c <= 1``; there
    is deliberately no separate-total constraint.  Concentrations remain
    meaningful in shallow retained ML cells and are never erased by this pass.
    """
    y=x.clone();y[:,0].clamp_(min=0);y[:,3:5].clamp_(0,1)
    y[:,4]=torch.minimum(y[:,4],y[:,3])
    drymask=y[:,0:1]<dry;y[:,1:3]*=(~drymask).to(y.dtype)
    return y
def teacher_weight(current,target,active,dry:float=STORAGE_WET_THRESHOLD_M):
    a=active if active.ndim==4 else active[:,None];wet=((current[:,0]>=dry)|(target[:,0]>=dry))[:,None].float();return a.float()*(.1+.9*wet),wet
def state_loss(pred_t,target_t,current_p,target_p,active):
    w,wet=teacher_weight(current_p,target_p,active);base=F.smooth_l1_loss(pred_t,target_t,reduction='none');weights=w.expand_as(base).clone();a=active if active.ndim==4 else active[:,None];weights[:,3:5]=wet.expand(-1,2,-1,-1)*a
    each=(base*weights).sum((0,2,3))/weights.sum((0,2,3)).clamp_min(1);return each.mean(),each
def wet_loss(pred_p,target_p,active,tau=.02,wet_threshold:float=STORAGE_WET_THRESHOLD_M):
    p=torch.sigmoid((pred_p[:,0]-wet_threshold)/tau)*active;q=torch.sigmoid((target_p[:,0]-wet_threshold)/tau)*active
    dice=1-(2*(p*q).sum((1,2)) +1e-6)/(p.square().sum((1,2))+q.square().sum((1,2))+1e-6);return dice.mean()
def integral_loss(pred_p,target_p,active,cell_area:float,eps=1.):
    a=(active if active.ndim==4 else active[:,None]).to(pred_p.dtype); quantities=lambda x:torch.stack(((x[:,0:1]*a).sum((1,2,3))*cell_area,(x[:,0:1]*x[:,3:4]*a).sum((1,2,3))*cell_area,(x[:,0:1]*x[:,4:5]*a).sum((1,2,3))*cell_area),1)
    p,t=quantities(pred_p),quantities(target_p); rel=(p-t).abs()/t.abs().clamp_min(eps);return rel.mean(),{'mixture':rel[:,0].mean(),'solid':rel[:,1].mean(),'ice':rel[:,2].mean()}
def amplitude_guard(pred_t,target_t,active,eps=1e-6):
    a=(active if active.ndim==4 else active[:,None]).to(pred_t.dtype); pr=((pred_t.square()*a).sum((0,2,3))/a.sum((0,2,3)).clamp_min(1)).sqrt();tr=((target_t.square()*a).sum((0,2,3))/a.sum((0,2,3)).clamp_min(1)).sqrt();valid=tr>eps
    return (torch.relu(pr[valid]/tr[valid].clamp_min(eps)-2).square().mean() if valid.any() else pred_t.new_zeros(()))
def step_loss(pred_t,target_t,teacher_current,pred_p,target_p,active,cell_area):
    s,ch=state_loss(pred_t,target_t,teacher_current,target_p,active);w=wet_loss(pred_p,target_p,active);i,parts=integral_loss(pred_p,target_p,active,cell_area);return s+.10*w+.05*i,{'state':s,'wet':w,'integral':i,'channels':ch,**parts}
def rollout_objective(step_terms,amp_terms):
    multi=torch.stack(step_terms).mean();one=step_terms[0];amp=torch.stack(amp_terms).mean() if amp_terms else multi.new_zeros(());return multi+.5*one+.01*amp,{'multi':multi.detach(),'one_step_anchor':one.detach(),'amplitude_guard':amp.detach()}
