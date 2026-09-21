"""Streaming, VAL-only validation for Global Operator V2."""
from __future__ import annotations
import hashlib
import numpy as np
import torch
from .dataset import build_features, INPUT
from .losses import project_physical, STORAGE_WET_THRESHOLD_M
from .metrics import debris_front

REQUIRED_SCORE_KEYS=("trajectory_h_rel_l2","trajectory_momentum_rel_l2","mean_wet_iou","debris_front_mae_km","mixture_volume_relative_error")

def select_fixed_val_subset(rows,n=8,seed=20260920,train_rows=None):
    """Deterministic farthest-point selection in normalized parameter space."""
    cols=["volume_scale","ice_fraction","erosion_K","n_debris","dep_tau_s"]
    if len(rows)<=n:return rows.copy().reset_index(drop=True)
    reference=rows if train_rows is None else train_rows
    values=rows[cols].to_numpy(float);reference=reference[cols].to_numpy(float)
    values[:,2]=np.log(values[:,2]);values[:,4]=np.log(values[:,4]);reference[:,2]=np.log(reference[:,2]);reference[:,4]=np.log(reference[:,4])
    values=(values-reference.mean(0))/np.maximum(reference.std(0),1e-12)
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
def front_error(predicted,truth,route_length_km):
    """Finite front error with explicit missing-front penalties."""
    if np.isfinite(predicted) and np.isfinite(truth):return abs(predicted-truth)
    if not np.isfinite(predicted) and not np.isfinite(truth):return None
    return route_length_km

def validation_score(metrics,weights=None):
    """Strict checkpoint score: missing metrics never silently become zero."""
    _required(metrics);weights=weights or {"trajectory_h_rel_l2":.40,"trajectory_momentum_rel_l2":.20,"one_minus_mean_wet_iou":.25,"debris_front_mae_normalized":.10,"mixture_volume_relative_error_normalized":.05,"front_normalization_km":.5,"volume_normalization":.10}
    return (weights["trajectory_h_rel_l2"]*metrics["trajectory_h_rel_l2"]+weights["trajectory_momentum_rel_l2"]*metrics["trajectory_momentum_rel_l2"]+weights["one_minus_mean_wet_iou"]*(1-metrics["mean_wet_iou"])+weights["debris_front_mae_normalized"]*metrics["debris_front_mae_km"]/weights["front_normalization_km"]+weights["mixture_volume_relative_error_normalized"]*metrics["mixture_volume_relative_error"]/weights["volume_normalization"])

@torch.no_grad()
def validate_one_step(model,store,transform,normalizer,device,rows=None,times_s=(120,300,600,900,1200)):
    rows=store.rows if rows is None else rows
    if not set(rows["split"]).issubset({"VAL"}):raise ValueError("validation accepts VAL only")
    static=torch.from_numpy(store.static).unsqueeze(0).to(device);active=static[:,1:2];records=[]
    for _,row in rows.iterrows():
        index=int(np.where(store.rows.scenario_id.eq(row.scenario_id))[0][0])
        for time_s in times_s:
            previous,current,target,params,time,_=store.sample(index,int(time_s));previous,current,target,params=(_tensor(x,device) for x in (previous,current,target,params))
            pred=project_physical(transform.decode(model(build_features(previous,current,static,params,torch.tensor([time],device=device),transform,normalizer),transform.encode(current))),active)
            mask=active[:,0].bool();hrel=_ratio(((pred[:,0]-target[:,0]).square()*mask).sum(),(target[:,0].square()*mask).sum());mrel=_ratio(((pred[:,1:3]-target[:,1:3]).square()*active).sum(),(target[:,1:3].square()*active).sum());iou=((pred[:,0]>=STORAGE_WET_THRESHOLD_M)&(target[:,0]>=STORAGE_WET_THRESHOLD_M)&mask).sum().float()/(((pred[:,0]>=STORAGE_WET_THRESHOLD_M)|(target[:,0]>=STORAGE_WET_THRESHOLD_M))&mask).sum().clamp_min(1)
            records.append({"scenario_id":row.scenario_id,"time_s":time_s,"h_rel_l2":hrel,"momentum_rel_l2":mrel,"wet_iou":float(iou)})
    return records

@torch.no_grad()
def validate_full_rollout(model,store,transform,normalizer,device,steps=144,rows=None,cell_area=900.):
    """Streaming autoregressive evaluation; never constructs a 144-frame window."""
    rows=store.rows if rows is None else rows
    if not set(rows["split"]).issubset({"VAL"}):raise ValueError("validation accepts VAL only")
    static=torch.from_numpy(store.static).unsqueeze(0).to(device);active=static[:,1:2];out=[]
    for _,row in rows.iterrows():
        index=int(np.where(store.rows.scenario_id.eq(row.scenario_id))[0][0]);previous,current,_,params,time,_=store.sample(index,0);previous,current,params=(_tensor(x,device) for x in (previous,current,params))
        hnum=torch.zeros((),device=device);hden=torch.zeros((),device=device)
        mnum=torch.zeros((),device=device);mden=torch.zeros((),device=device)
        cnum=torch.zeros((),device=device);cden=torch.zeros((),device=device)
        inum=torch.zeros((),device=device);iden=torch.zeros((),device=device)
        dznum=torch.zeros((),device=device);dzden=torch.zeros((),device=device)
        wet_sum=torch.zeros((),device=device);volume_errors=[];front_errors=[];guard_activations=[];target=None;final_iou=0.;final_front_error=0.;nonfinite_step=None
        if not out:
            with np.load(INPUT) as data:route=np.asarray(data['route_chainage_m'],np.float32)
        route_length=float(np.nanmax(route)/1e3)
        for step in range(min(steps,144)):
            target=_tensor(store.frame(row,(step+1)*10),device);pred=project_physical(transform.decode(model(build_features(previous,current,static,params,torch.tensor([time+step/144.],device=device),transform,normalizer),transform.encode(current))),active);guard_activations.append(float(getattr(model,'last_momentum_guard_activation_fraction',0.)))
            if not torch.isfinite(pred).all():
                nonfinite_step=step+1;break
            mask=active[:,0].bool();hnum+=((pred[:,0]-target[:,0]).square()*mask).sum();hden+=(target[:,0].square()*mask).sum();mnum+=((pred[:,1:3]-target[:,1:3]).square()*active).sum();mden+=(target[:,1:3].square()*active).sum();cnum+=((pred[:,3]-target[:,3]).square()*mask).sum();cden+=(target[:,3].square()*mask).sum();inum+=((pred[:,4]-target[:,4]).square()*mask).sum();iden+=(target[:,4].square()*mask).sum();dznum+=((pred[:,5]-target[:,5]).square()*mask).sum();dzden+=(target[:,5].square()*mask).sum()
            p_wet=pred[:,0]>=STORAGE_WET_THRESHOLD_M;t_wet=target[:,0]>=STORAGE_WET_THRESHOLD_M;iou=(p_wet&t_wet&mask).sum().float()/((p_wet|t_wet)&mask).sum().clamp_min(1);wet_sum+=iou;final_iou=float(iou.cpu())
            pv=(pred[:,0:1]*active).sum()*cell_area;tv=(target[:,0:1]*active).sum()*cell_area;volume_errors.append(float(((pv-tv).abs()/tv.abs().clamp_min(1.)).cpu()))
            error=front_error(debris_front(pred[0].detach().cpu().numpy(),route),debris_front(target[0].detach().cpu().numpy(),route),route_length)
            if error is not None:front_errors.append(error);final_front_error=error
            previous,current=current,pred
        if nonfinite_step is not None:
            out.append({"scenario_id":row.scenario_id,"validation_status":"NONFINITE_ROLLOUT","finite_rollout":False,"first_nonfinite_step":nonfinite_step,"trajectory_h_rel_l2":None,"trajectory_momentum_rel_l2":None,"trajectory_c_rel_l2":None,"trajectory_ice_rel_l2":None,"trajectory_dz_rel_l2":None,"mean_wet_iou":None,"final_wet_iou":None,"mixture_volume_relative_error":None,"debris_front_mae_km":None,"debris_front_final_error_km":None,"momentum_guard_activation_fraction":float(np.mean(guard_activations)) if guard_activations else 0.,"momentum_guard_activation_steps":int(sum(value>0 for value in guard_activations))})
        else:
            out.append({"scenario_id":row.scenario_id,"validation_status":"FINITE","finite_rollout":True,"first_nonfinite_step":None,"trajectory_h_rel_l2":_ratio(hnum,hden),"final_h_rel_l2":_ratio(((current[:,0]-target[:,0]).square()*active[:,0]).sum(),(target[:,0].square()*active[:,0]).sum()),"trajectory_momentum_rel_l2":_ratio(mnum,mden),"trajectory_c_rel_l2":_ratio(cnum,cden),"trajectory_ice_rel_l2":_ratio(inum,iden),"trajectory_dz_rel_l2":_ratio(dznum,dzden),"mean_wet_iou":float((wet_sum/min(steps,144)).cpu()),"final_wet_iou":final_iou,"mixture_volume_relative_error":float(np.mean(volume_errors)),"debris_front_mae_km":float(np.mean(front_errors)) if front_errors else route_length,"debris_front_final_error_km":final_front_error,"momentum_guard_activation_fraction":float(np.mean(guard_activations)) if guard_activations else 0.,"momentum_guard_activation_steps":int(sum(value>0 for value in guard_activations))})
    return out

def validate_all_val(model,store,transform,normalizer,device,rows=None,steps=144):
    records=validate_full_rollout(model,store,transform,normalizer,device,steps,rows)
    nonfinite=[r for r in records if not r.get("finite_rollout",True)]
    subset_hash=hashlib.sha256("\n".join(sorted(str(r["scenario_id"]) for r in records)).encode()).hexdigest()
    if nonfinite:return records,{"validation_status":"NONFINITE_ROLLOUT","finite_rollout":False,"first_nonfinite_step":min(r["first_nonfinite_step"] for r in nonfinite),"validation_subset_hash":subset_hash}
    keys=REQUIRED_SCORE_KEYS;summary={k:float(np.mean([r[k] for r in records])) for k in keys};summary['momentum_guard_activation_fraction']=float(np.mean([r.get('momentum_guard_activation_fraction',0.) for r in records]));summary['momentum_guard_activation_steps']=int(sum(r.get('momentum_guard_activation_steps',0) for r in records));summary["validation_subset_hash"]=subset_hash;summary["validation_status"]="FINITE";summary["finite_rollout"]=True;summary["J_val"]=validation_score(summary);return records,summary

class _PersistenceModel(torch.nn.Module):
    def forward(self,features,encoded_current):return encoded_current

def validate_persistence_baseline(store,transform,normalizer,device,rows=None,steps=144):
    """The immutable closed-loop baseline: next state is current state."""
    model=_PersistenceModel().to(device);return validate_all_val(model,store,transform,normalizer,device,rows,steps)

def validate_horizon_ladder(model,store,transform,normalizer,device,rows=None,horizons=(1,2,4,8,16,32,64,144)):
    """Small persisted horizon summary; each entry retains nonfinite status."""
    output=[]
    for horizon in horizons:
        _,summary=validate_all_val(model,store,transform,normalizer,device,rows,steps=int(horizon))
        output.append({"horizon":int(horizon),"h_rel_l2":summary.get("trajectory_h_rel_l2"),"momentum_rel_l2":summary.get("trajectory_momentum_rel_l2"),"wet_iou":summary.get("mean_wet_iou"),"volume_rel_error":summary.get("mixture_volume_relative_error"),"finite_fraction":1.0 if summary.get("finite_rollout") else 0.0,"validation_status":summary["validation_status"]})
    return output

def validate_short_rollout(model,store,transform,normalizer,device,steps=12):return validate_full_rollout(model,store,transform,normalizer,device,steps)
