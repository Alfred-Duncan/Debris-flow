"""Physical-consistent objectives.  Weight masks always derive from teacher fields."""
from __future__ import annotations
import torch
import torch.nn.functional as F

STORAGE_WET_THRESHOLD_M=.05
MOMENTUM_DRY_THRESHOLD_M=.03

def project_physical(x,active=None,dry:float=MOMENTUM_DRY_THRESHOLD_M):
    """Project only the physical constraints represented by the Park-v2 state.

    ``c`` is the total solid/debris column concentration and ``ice`` is its ice
    component.  Therefore the valid simplex is ``0 <= ice <= c <= 1``; there
    is deliberately no separate-total constraint.  Concentrations remain
    meaningful in shallow retained ML cells and are never erased by this pass.
    """
    # Construct a fresh tensor: indexed in-place writes here invalidate the
    # decoded rollout graph during capacity/backward probes on CUDA.
    h=x[:,0:1].clamp(min=0);c=x[:,3:4].clamp(0,1);ice=torch.minimum(x[:,4:5].clamp(0,1),c)
    momentum=x[:,1:3]*(h>=dry).to(x.dtype)
    output=torch.cat((h,momentum,c,ice,x[:,5:6]),dim=1)
    if active is None:return output
    mask=(active if active.ndim==4 else active[:,None]).to(dtype=output.dtype)
    return output*mask
def teacher_weight(current,target,active,teacher_delta=None,static=None,source_active=False,weighting=None,dry:float=STORAGE_WET_THRESHOLD_M):
    """Bounded teacher-only weighting; no mask ever depends on prediction."""
    weighting=weighting or {};a=(active if active.ndim==4 else active[:,None]).float();wet=((current[:,0]>=dry)|(target[:,0]>=dry))[:,None].float()
    dry_weight=float(weighting.get('dry_active_weight',.02));wet_weight=float(weighting.get('wet_weight',1.0));weights=a*(dry_weight+(wet_weight-dry_weight)*wet)
    if teacher_delta is not None:
        threshold=float(weighting.get('change_threshold',float('inf')));strong=(teacher_delta.abs().mean(1,keepdim=True)>=threshold).float();weights=weights*(1+(float(weighting.get('strong_change_weight',1.0))-1)*strong)
    if static is not None and source_active:
        source=static[:,2:3].to(weights.dtype);weights=weights*(1+(float(weighting.get('source_weight',1.0))-1)*source)
    return weights,wet
def state_loss(pred_t,target_t,current_p,target_p,active,teacher_delta=None,static=None,source_active=False,weighting=None):
    w,_=teacher_weight(current_p,target_p,active,teacher_delta,static,source_active,weighting);base=F.smooth_l1_loss(pred_t,target_t,reduction='none');weights=w.expand_as(base)
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
def normalized_delta_huber_components(pred_t,encoded_current,target_t,teacher_current_t,active,delta_scales):
    """Directly penalize a zero residual whenever the teacher changes."""
    scales=torch.as_tensor(delta_scales,device=pred_t.device,dtype=pred_t.dtype)[None,:,None,None]
    predicted_delta=(pred_t-encoded_current)/scales;teacher_delta=(target_t-teacher_current_t)/scales;a=(active if active.ndim==4 else active[:,None]).to(pred_t.dtype)
    per=F.smooth_l1_loss(predicted_delta,teacher_delta,reduction='none');channels=(per*a).sum((0,2,3))/a.sum((0,2,3)).clamp_min(1);return channels.mean(),channels
def normalized_delta_huber(pred_t,encoded_current,target_t,teacher_current_t,active,delta_scales):
    return normalized_delta_huber_components(pred_t,encoded_current,target_t,teacher_current_t,active,delta_scales)[0]
def step_loss(pred_t,target_t,teacher_current,pred_p,target_p,active,cell_area,encoded_current=None,teacher_current_t=None,delta_scales=None,static=None,source_active=False,weighting=None):
    encoded_current=target_t if encoded_current is None else encoded_current;teacher_current_t=target_t if teacher_current_t is None else teacher_current_t
    teacher_delta=target_t-teacher_current_t;s,ch=state_loss(pred_t,target_t,teacher_current,target_p,active,teacher_delta,static,source_active,weighting);w=wet_loss(pred_p,target_p,active);i,parts=integral_loss(pred_p,target_p,active,cell_area)
    if delta_scales is None:delta=pred_t.new_zeros(());delta_channels=pred_t.new_zeros(6)
    else:delta,delta_channels=normalized_delta_huber_components(pred_t,encoded_current,target_t,teacher_current_t,active,delta_scales)
    names=('h','hu','hv','c','ice','dz');return s+.10*w+.05*i+.50*delta,{'state':s,'wet':w,'integral':i,'delta':delta,**{f'delta_{name}':value for name,value in zip(names,delta_channels)},'channels':ch,**parts}
def rollout_objective(step_terms,amp_terms):
    multi=torch.stack(step_terms).mean();one=step_terms[0];amp=torch.stack(amp_terms).mean() if amp_terms else multi.new_zeros(());return multi+.5*one+.01*amp,{'multi':multi.detach(),'one_step_anchor':one.detach(),'amplitude_guard':amp.detach()}
