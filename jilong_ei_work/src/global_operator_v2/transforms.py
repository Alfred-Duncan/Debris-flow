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
