"""Train-only fitted, zero-preserving state and feature transforms."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch

STATE_NAMES=("h","hu","hv","c","ice","dz")
PARAMETER_NAMES=("volume_scale","ice_fraction","erosion_K","n_debris","dep_tau_s")
def _q(values,q,floor):
    xs=[np.asarray(v).reshape(-1) for v in values if np.asarray(v).size]
    return max(float(np.quantile(np.abs(np.concatenate(xs)),q)),floor) if xs else floor

@dataclass
class PhysicalTransform:
    h_scale:float=1.;hu_scale:float=1.;hv_scale:float=1.;dz_scale:float=.05
    def encode(self,x):
        y=x.clone();y[:,0]=torch.log1p(x[:,0].clamp_min(0)/self.h_scale);y[:,1]=torch.asinh(x[:,1]/self.hu_scale);y[:,2]=torch.asinh(x[:,2]/self.hv_scale);y[:,5]=torch.asinh(x[:,5]/self.dz_scale);return y
    def decode(self,y):
        x=y.clone();x[:,0]=self.h_scale*torch.expm1(y[:,0]).clamp_min(0);x[:,1]=self.hu_scale*torch.sinh(y[:,1]);x[:,2]=self.hv_scale*torch.sinh(y[:,2]);x[:,5]=self.dz_scale*torch.sinh(y[:,5]);return x
    def to_dict(self):return {"h":{"kind":"log1p","scale":self.h_scale},"hu":{"kind":"asinh","scale":self.hu_scale},"hv":{"kind":"asinh","scale":self.hv_scale},"c":{"kind":"identity"},"ice":{"kind":"identity"},"dz":{"kind":"asinh","scale":self.dz_scale},"fit_scope":"TRAIN_ONLY"}
    @classmethod
    def from_dict(cls,d):return cls(d['h']['scale'],d['hu']['scale'],d['hv']['scale'],d['dz']['scale'])

@dataclass
class FeatureNormalizer:
    terrain_mean:float;terrain_std:float;lateral_scale:float;external_scale:float;param_mean:list;param_std:list
    def params(self,p):
        q=p.clone();q[:,2]=torch.log(q[:,2]);q[:,4]=torch.log(q[:,4]);m=torch.tensor(self.param_mean,device=p.device);s=torch.tensor(self.param_std,device=p.device);return(q-m)/s
    def terrain(self,z):return(z-self.terrain_mean)/self.terrain_std
    def forcing(self,q,kind):return torch.asinh(q/(self.lateral_scale if kind=='lateral' else self.external_scale))
    def to_dict(self):return {"terrain":{"mean_active":self.terrain_mean,"std_active":self.terrain_std},"lateral_q":{"kind":"asinh","scale":self.lateral_scale},"external_inflow_q":{"kind":"asinh","scale":self.external_scale},"parameters":{"names":list(PARAMETER_NAMES),"log_transformed":["erosion_K","dep_tau_s"],"mean":self.param_mean,"std":self.param_std},"fit_scope":"TRAIN_ONLY"}
    @classmethod
    def from_dict(cls,d):return cls(d['terrain']['mean_active'],d['terrain']['std_active'],d['lateral_q']['scale'],d['external_inflow_q']['scale'],d['parameters']['mean'],d['parameters']['std'])

@dataclass
class DeltaNormalization:
    """TRAIN-only robust scales, bounds, and change threshold for 10-s deltas."""
    scales:list
    bounds:list
    change_threshold:float
    coverage:float
    quantile:float
    bound_quantile:float
    safety_factor:float
    sample_count:int
    per_channel:dict|None=None
    change_nonzero_fraction:float=0.
    def scale_tensor(self,device,dtype=torch.float32):return torch.tensor(self.scales,device=device,dtype=dtype)[None,:,None,None]
    def bound_tensor(self,device,dtype=torch.float32):return torch.tensor(self.bounds,device=device,dtype=dtype)[None,:,None,None]
    def to_dict(self):return {"state_names":list(STATE_NAMES),"scales":self.scales,"bounds":self.bounds,"change_threshold":self.change_threshold,"coverage":self.coverage,"coverage_per_channel":{name:float((self.per_channel or {}).get(name,{}).get("coverage_all_samples",self.coverage)) for name in STATE_NAMES},"quantile":self.quantile,"bound_quantile":self.bound_quantile,"safety_factor":self.safety_factor,"sample_count":self.sample_count,"per_channel":self.per_channel or {},"change_nonzero_fraction":self.change_nonzero_fraction,"fit_scope":"TRAIN_ONLY"}
    @classmethod
    def from_dict(cls,d):return cls(d["scales"],d["bounds"],d["change_threshold"],d["coverage"],d["quantile"],d["bound_quantile"],d["safety_factor"],d["sample_count"],d.get("per_channel"),d.get("change_nonzero_fraction",0.))

def fit_training_transforms(rows,store,static,seed=20260920,times=(0,120,300,600,900,1200,1440),max_cells=2048):
    """Streaming deterministic fit over all and only TRAIN scenarios."""
    if not set(rows['split']).issubset({'TRAIN'}):raise ValueError('TRAIN-only fit required')
    rng=np.random.default_rng(seed); h=[];hu=[];hv=[];dz=[]
    for _,row in rows.sort_values('scenario_id').iterrows():
      for t in times:
        s=store.frame(row,t);wet=np.flatnonzero(s[0].ravel()>=.03);nz=np.flatnonzero(np.abs(s[5]).ravel()>0)
        pick=lambda v:v[rng.choice(v.size,min(v.size,max_cells),replace=False)] if v.size else v
        a,b=pick(wet),pick(nz);h.append(s[0].ravel()[a]);hu.append(s[1].ravel()[a]);hv.append(s[2].ravel()[a]);dz.append(s[5].ravel()[b])
    tr=PhysicalTransform(_q(h,.90,1e-6),_q(hu,.95,1e-6),_q(hv,.95,1e-6),_q(dz,.95,.05)); active=static[1].astype(bool);z=static[0][active];lat=static[4][static[4]!=0];ext=static[5][static[5]!=0]
    p=rows.loc[:,PARAMETER_NAMES].to_numpy(float);p[:,2]=np.log(p[:,2]);p[:,4]=np.log(p[:,4]);norm=FeatureNormalizer(float(z.mean()),float(z.std() or 1),_q([lat],.95,1e-6),_q([ext],.95,1e-6),p.mean(0).tolist(),np.maximum(p.std(0),1e-6).tolist())
    return tr,norm

def _encode_numpy(state,transform):
    """Numpy counterpart used only while fitting persisted TRAIN statistics."""
    out=np.asarray(state,dtype=np.float32).copy()
    out[0]=np.log1p(np.maximum(out[0],0)/transform.h_scale)
    out[1]=np.arcsinh(out[1]/transform.hu_scale)
    out[2]=np.arcsinh(out[2]/transform.hv_scale)
    out[5]=np.arcsinh(out[5]/transform.dz_scale)
    return out

def fit_delta_normalization(rows,store,static,transform,seed=20260920,samples_per_case=12,max_cells=512,quantile=.995,bound_quantile=.999,safety_factor=1.2,floor=1e-6,change_quantile=.90,epsilon=1e-8):
    """Fit delta scales/bounds from deterministic samples of TRAIN pairs only.

    The same sampled distribution is retained for the reported coverage, so a
    resume or another machine derives byte-for-byte equivalent configuration.
    """
    if not set(rows['split']).issubset({'TRAIN'}):raise ValueError('TRAIN-only delta fit required')
    rng=np.random.default_rng(seed);active=np.asarray(static[1],bool);samples=[[] for _ in STATE_NAMES];cell_change=[];epsilon=max(float(epsilon),1e-8)
    available=np.arange(144,dtype=int)
    for _,row in rows.sort_values('scenario_id').iterrows():
        forced=np.asarray([0,1,2],int);count=max(int(samples_per_case)-len(forced),0)
        picked=np.unique(np.concatenate((forced,rng.choice(available,size=count,replace=False))))
        for index in picked:
            before=_encode_numpy(store.frame(row,int(index*10)),transform);after=_encode_numpy(store.frame(row,int((index+1)*10)),transform);delta=after-before
            cells=np.flatnonzero(active.ravel());take=rng.choice(cells,size=min(len(cells),int(max_cells)),replace=False)
            for channel in range(len(STATE_NAMES)):samples[channel].append(delta[channel].ravel()[take])
            cell_change.append(np.mean(np.abs(delta[:,active]),axis=0))
    flattened=[np.concatenate(parts) for parts in samples];scales=[];bounds=[];per_channel={}
    for name,values in zip(STATE_NAMES,flattened):
        absolute=np.abs(values);changed=absolute[absolute>epsilon];nonzero_count=int(changed.size);zero_fraction=float(1-nonzero_count/max(int(absolute.size),1))
        scale=max(float(np.quantile(changed,quantile)),floor) if nonzero_count else float(floor)
        bound=(max(float(np.quantile(changed,bound_quantile)),floor) if nonzero_count else float(floor))*float(safety_factor)
        coverage=float(np.mean(absolute<=bound))
        scales.append(scale);bounds.append(bound)
        per_channel[name]={"nonzero_sample_count":nonzero_count,"zero_fraction":zero_fraction,"scale":scale,"bound":bound,"coverage_all_samples":coverage,"q50_nonzero":float(np.quantile(changed,.50)) if nonzero_count else None,"q90_nonzero":float(np.quantile(changed,.90)) if nonzero_count else None,"q99_nonzero":float(np.quantile(changed,.99)) if nonzero_count else None,"q995_nonzero":float(np.quantile(changed,.995)) if nonzero_count else None,"q999_nonzero":float(np.quantile(changed,.999)) if nonzero_count else None,"max_abs_sampled":float(absolute.max()) if absolute.size else 0.}
    all_change=np.concatenate(cell_change);positive_change=all_change[all_change>epsilon];threshold=max(float(np.quantile(positive_change,change_quantile)),floor) if positive_change.size else float(floor)
    coverage=float(np.mean(np.concatenate([np.abs(values)<=bound for values,bound in zip(flattened,bounds)])))
    return DeltaNormalization(scales,bounds,threshold,coverage,float(quantile),float(bound_quantile),float(safety_factor),int(sum(len(x) for x in flattened)),per_channel,float(positive_change.size/max(all_change.size,1)))
