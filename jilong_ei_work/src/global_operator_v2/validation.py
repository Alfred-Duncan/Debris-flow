"""Formal validation utilities.  They never access FINAL_HOLDOUT paths."""
from __future__ import annotations
import numpy as np,torch
from .dataset import build_features
from .losses import project_physical
def fixed_val_subset(rows,n=8):
    """Deterministic parameter-space coverage: sorted rank interleaving, no hand-picking."""
    q=rows.sort_values(['volume_scale','ice_fraction','erosion_K','n_debris','dep_tau_s']).reset_index(drop=True);return q.iloc[np.linspace(0,len(q)-1,min(n,len(q)),dtype=int)].copy()
@torch.no_grad()
def rollout(model,transform,normalizer,previous,current,static,params,steps,time,active):
    states=[]
    for k in range(steps):
        inp=build_features(previous,current,static,params,time+k/144.,transform,normalizer);current_next=project_physical(transform.decode(model(inp,transform.encode(current))));previous,current=current,current_next;states.append(current)
    return states
def field_metrics(pred,target,active):
    a=active.bool();hrel=torch.linalg.vector_norm(((pred[:,0]-target[:,0])*a[:,0]).reshape(pred.shape[0],-1),dim=1)/torch.linalg.vector_norm((target[:,0]*a[:,0]).reshape(pred.shape[0],-1),dim=1).clamp_min(1e-6);mom=(pred[:,1:3]-target[:,1:3]).abs().mean();wetp=pred[:,0]>=.03;wett=target[:,0]>=.03;iou=((wetp&wett&a[:,0]).sum((1,2)).float()/((wetp|wett)&a[:,0]).sum((1,2)).clamp_min(1)).mean();return {'h_rel_l2':float(hrel.mean()),'momentum_mae':float(mom),'wet_iou':float(iou)}
def composite(metrics):return metrics['h_rel_l2']+metrics['momentum_mae']+(1-metrics['wet_iou'])
def dimensionless_score(m):
    g=lambda k,d:float(m.get(k,0.) if np.isfinite(m.get(k,0.)) else 0.)/d
    return .40*g('trajectory_h_rel_l2',1)+.20*g('trajectory_momentum_rel_l2',1)+.25*g('wet_error',1)+.10*g('front_mae_km',.5)+.05*g('volume_relative_error',.10)
@torch.no_grad()
def validate_cases(model,store,transform,normalizer,device,rows=None):
    """Real held-out one-step and 0→1440 closed-loop evaluation entry; caller supplies VAL only."""
    if rows is None: rows=store.rows
    if not set(rows['split']).issubset({'VAL'}):raise ValueError('validation accepts VAL only')
    static=torch.from_numpy(store.static).unsqueeze(0).to(device);active=static[:,1:2];records=[]
    for i in range(len(rows)):
      prev,cur,tgt,p,t,row=store.sample(i,0);a=lambda x:torch.from_numpy(x).unsqueeze(0).to(device);pred=project_physical(transform.decode(model(build_features(a(prev),a(cur),static,torch.from_numpy(p).unsqueeze(0).to(device),torch.tensor([t],device=device),transform,normalizer),transform.encode(a(cur)))));m=field_metrics(pred,a(tgt),active);records.append({'scenario_id':row.scenario_id,**m})
    return records
def validate_one_step(*args,**kwargs):return validate_cases(*args,**kwargs)
def validate_short_rollout(model,store,transform,normalizer,device,steps=12):return validate_full_rollout(model,store,transform,normalizer,device,steps)
@torch.no_grad()
def validate_full_rollout(model,store,transform,normalizer,device,steps=144):
    """True autoregressive S0→1440 rollout: after S0 no teacher field enters input."""
    static=torch.from_numpy(store.static).unsqueeze(0).to(device);active=static[:,1:2];out=[]
    for i,row in store.rows.iterrows():
      prev,cur,targets,p,time,_=store.window(i,0,steps);a=lambda x:torch.from_numpy(x).unsqueeze(0).to(device);prev,cur=a(prev),a(cur);pa=torch.from_numpy(p).unsqueeze(0).to(device);truth=[a(x) for x in targets];preds=[]
      for k in range(len(truth)):
       pred=project_physical(transform.decode(model(build_features(prev,cur,static,pa,torch.tensor([time+k/144.],device=device),transform,normalizer),transform.encode(cur))));prev,cur=cur,pred;preds.append(pred)
      fm=[field_metrics(p,t,active) for p,t in zip(preds,truth)];out.append({'scenario_id':row.scenario_id,'trajectory_h_rel_l2':float(np.mean([x['h_rel_l2'] for x in fm])),'final_h_rel_l2':fm[-1]['h_rel_l2'],'trajectory_momentum_rel_l2':float(np.mean([x['momentum_mae'] for x in fm])),'mean_wet_iou':float(np.mean([x['wet_iou'] for x in fm])),'final_wet_iou':fm[-1]['wet_iou']})
    return out
def validate_all_val(*args,**kwargs):return validate_full_rollout(*args,**kwargs)
