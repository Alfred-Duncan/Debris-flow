from __future__ import annotations
import json,numpy as np,torch
from pathlib import Path
STATE=('h','hu','hv','c','ice','dz')
def fit(values,source_cases,source_times,seed=20260920,floor=1e-6):
 """Train-only robust correction coordinate statistics."""
 out={'fit_scope':'TRAIN_ONLY','seed':seed,'source_cases':list(source_cases),'source_times':list(source_times),'state_names':list(STATE),'per_channel':{}}
 scales=[];bounds=[]
 for name,x in zip(STATE,values):
  x=np.asarray(x).ravel();a=np.abs(x);nz=a[a>1e-8]
  q=lambda p:float(np.quantile(nz,p)) if nz.size else floor
  scale=max(q(.995),floor);bound=max(1.2*q(.999),1.05*q(.995),floor);coverage=float(np.mean(a<=bound))
  if coverage<.999:raise RuntimeError('LOCAL_CORRECTION_COVERAGE_INSUFFICIENT '+name)
  out['per_channel'][name]={'nonzero_count':int(nz.size),'zero_fraction':float(1-nz.size/max(a.size,1)),'q50':q(.5),'q90':q(.9),'q99':q(.99),'q995':q(.995),'q999':q(.999),'q9999':q(.9999),'max':float(a.max(initial=0)),'scale':scale,'bound':bound,'coverage':coverage};scales.append(scale);bounds.append(bound)
 out['scales']=scales;out['bounds']=bounds;return out
def save(path,data):Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(data,indent=2))
def bounded(raw,scales,bounds):
 s=torch.as_tensor(scales,device=raw.device,dtype=raw.dtype)[None,:,None,None];b=torch.as_tensor(bounds,device=raw.device,dtype=raw.dtype)[None,:,None,None];nb=b/s;return s*nb*torch.tanh(raw/nb)
