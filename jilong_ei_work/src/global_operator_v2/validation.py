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
@torch.no_grad()
def validate_cases(model,store,transform,normalizer,device,rows=None):
    """Real held-out one-step and 0→1440 closed-loop evaluation entry; caller supplies VAL only."""
    if rows is None: rows=store.rows
    if not set(rows['split']).issubset({'VAL'}):raise ValueError('validation accepts VAL only')
    static=torch.from_numpy(store.static).unsqueeze(0).to(device);active=static[:,1:2];records=[]
    for i in range(len(rows)):
      prev,cur,tgt,p,t,row=store.sample(i,0);a=lambda x:torch.from_numpy(x).unsqueeze(0).to(device);pred=project_physical(transform.decode(model(build_features(a(prev),a(cur),static,torch.from_numpy(p).unsqueeze(0).to(device),torch.tensor([t],device=device),transform,normalizer),transform.encode(a(cur)))));m=field_metrics(pred,a(tgt),active);records.append({'scenario_id':row.scenario_id,**m})
    return records
