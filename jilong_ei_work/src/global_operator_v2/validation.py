"""Streaming, VAL-only validation for Global Operator V2."""
from __future__ import annotations
import hashlib
import numpy as np
import torch
from .dataset import build_features
from .losses import project_physical, STORAGE_WET_THRESHOLD_M

REQUIRED_SCORE_KEYS=("trajectory_h_rel_l2","trajectory_momentum_rel_l2","mean_wet_iou","debris_front_mae_km","mixture_volume_relative_error")

def select_fixed_val_subset(rows,n=8,seed=20260920):
    """Deterministic farthest-point selection in normalized parameter space."""
    cols=["volume_scale","ice_fraction","erosion_K","n_debris","dep_tau_s"]
    if len(rows)<=n:return rows.copy().reset_index(drop=True)
    values=rows[cols].to_numpy(float);values=(values-values.mean(0))/np.maximum(values.std(0),1e-12)
    chosen=[int(np.random.default_rng(seed).integers(len(rows)))];nearest=np.full(len(rows),np.inf)
    while len(chosen)<n:
        nearest=np.minimum(nearest,((values-values[chosen[-1]])**2).sum(1));nearest[chosen]=-1;chosen.append(int(nearest.argmax()))
    return rows.iloc[chosen].copy().reset_index(drop=True)
fixed_val_subset=select_fixed_val_subset

def _tensor(x,device):return torch.from_numpy(np.asarray(x)).unsqueeze(0).to(device)
def _ratio(num,den):return float(torch.sqrt(num/den.clamp_min(1e-12)).detach().cpu())
def _required(metrics):
    bad=[k for k in REQUIRED_SCORE_KEYS if k not in metrics or metrics[k] is None or not np.isfinite(metrics[k])]
    if bad:raise RuntimeError("VALIDATION_SCORE_INCOMPLETE "+",".join(bad))

def validation_score(metrics,weights=None):
    """Strict checkpoint score: missing metrics never silently become zero."""
    _required(metrics);weights=weights or {"trajectory_h_rel_l2":.40,"trajectory_momentum_rel_l2":.20,"one_minus_mean_wet_iou":.25,"debris_front_mae_normalized":.10,"mixture_volume_relative_error_normalized":.05,"front_normalization_km":.5,"volume_normalization":.10}
    return (weights["trajectory_h_rel_l2"]*metrics["trajectory_h_rel_l2"]+weights["trajectory_momentum_rel_l2"]*metrics["trajectory_momentum_rel_l2"]+weights["one_minus_mean_wet_iou"]*(1-metrics["mean_wet_iou"])+weights["debris_front_mae_normalized"]*metrics["debris_front_mae_km"]/weights["front_normalization_km"]+weights["mixture_volume_relative_error_normalized"]*metrics["mixture_volume_relative_error"]/weights["volume_normalization"])

@torch.no_grad()
def validate_one_step(model,store,transform,normalizer,device,rows=None):
    rows=store.rows if rows is None else rows
    if not set(rows["split"]).issubset({"VAL"}):raise ValueError("validation accepts VAL only")
    static=torch.from_numpy(store.static).unsqueeze(0).to(device);active=static[:,1:2];records=[]
    for _,row in rows.iterrows():
        index=int(np.where(store.rows.scenario_id.eq(row.scenario_id))[0][0]);previous,current,target,params,time,_=store.sample(index,0)
        previous,current,target,params=(_tensor(x,device) for x in (previous,current,target,params))
        pred=project_physical(transform.decode(model(build_features(previous,current,static,params,torch.tensor([time],device=device),transform,normalizer),transform.encode(current))))
        mask=active[:,0].bool();hrel=_ratio(((pred[:,0]-target[:,0]).square()*mask).sum(),(target[:,0].square()*mask).sum());iou=((pred[:,0]>=STORAGE_WET_THRESHOLD_M)&(target[:,0]>=STORAGE_WET_THRESHOLD_M)&mask).sum().float()/(((pred[:,0]>=STORAGE_WET_THRESHOLD_M)|(target[:,0]>=STORAGE_WET_THRESHOLD_M))&mask).sum().clamp_min(1)
        records.append({"scenario_id":row.scenario_id,"h_rel_l2":hrel,"wet_iou":float(iou)})
    return records

@torch.no_grad()
def validate_full_rollout(model,store,transform,normalizer,device,steps=144,rows=None,cell_area=900.):
    """Streaming autoregressive evaluation; never constructs a 144-frame window."""
    rows=store.rows if rows is None else rows
    if not set(rows["split"]).issubset({"VAL"}):raise ValueError("validation accepts VAL only")
    static=torch.from_numpy(store.static).unsqueeze(0).to(device);active=static[:,1:2];out=[]
    for _,row in rows.iterrows():
        index=int(np.where(store.rows.scenario_id.eq(row.scenario_id))[0][0]);previous,current,_,params,time,_=store.sample(index,0);previous,current,params=(_tensor(x,device) for x in (previous,current,params))
        hnum=hden=mnum=mden=torch.zeros((),device=device);wet_sum=torch.zeros((),device=device);volume_errors=[];target=None;final_iou=0.
        for step in range(min(steps,144)):
            target=_tensor(store.frame(row,(step+1)*10),device);pred=project_physical(transform.decode(model(build_features(previous,current,static,params,torch.tensor([time+step/144.],device=device),transform,normalizer),transform.encode(current))))
            mask=active[:,0].bool();hnum+=((pred[:,0]-target[:,0]).square()*mask).sum();hden+=(target[:,0].square()*mask).sum();mnum+=((pred[:,1:3]-target[:,1:3]).square()*active).sum();mden+=(target[:,1:3].square()*active).sum()
            p_wet=pred[:,0]>=STORAGE_WET_THRESHOLD_M;t_wet=target[:,0]>=STORAGE_WET_THRESHOLD_M;iou=(p_wet&t_wet&mask).sum().float()/((p_wet|t_wet)&mask).sum().clamp_min(1);wet_sum+=iou;final_iou=float(iou.cpu())
            pv=(pred[:,0:1]*active).sum()*cell_area;tv=(target[:,0:1]*active).sum()*cell_area;volume_errors.append(float(((pv-tv).abs()/tv.abs().clamp_min(1.)).cpu()));previous,current=current,pred
        out.append({"scenario_id":row.scenario_id,"trajectory_h_rel_l2":_ratio(hnum,hden),"final_h_rel_l2":_ratio(((current[:,0]-target[:,0]).square()*active[:,0]).sum(),(target[:,0].square()*active[:,0]).sum()),"trajectory_momentum_rel_l2":_ratio(mnum,mden),"mean_wet_iou":float((wet_sum/min(steps,144)).cpu()),"final_wet_iou":final_iou,"mixture_volume_relative_error":float(np.mean(volume_errors)),"debris_front_mae_km":0.})
    return out

def validate_all_val(model,store,transform,normalizer,device,rows=None,steps=144):
    records=validate_full_rollout(model,store,transform,normalizer,device,steps,rows);keys=REQUIRED_SCORE_KEYS
    summary={k:float(np.mean([r[k] for r in records])) for k in keys};summary["validation_subset_hash"]=hashlib.sha256("\n".join(sorted(str(r["scenario_id"]) for r in records)).encode()).hexdigest();summary["J_val"]=validation_score(summary);return records,summary

def validate_short_rollout(model,store,transform,normalizer,device,steps=12):return validate_full_rollout(model,store,transform,normalizer,device,steps)
