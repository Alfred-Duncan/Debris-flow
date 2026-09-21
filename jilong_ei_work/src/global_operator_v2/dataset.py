"""Scenario sampler and V2 feature construction (no predictor-side route data)."""
from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from .frame_adapter import dynamic_state, static_and_exogenous
from .transforms import PARAMETER_NAMES

ROOT=Path(__file__).resolve().parents[2]
INPUT=ROOT/"data/downloads/park_v2/inputs/upper30h.npz"
PARAMS=PARAMETER_NAMES

def scenario_rows(split: str | None=None) -> pd.DataFrame:
    idx=pd.read_csv(ROOT/"data/scenario_index.csv")
    design=pd.read_csv(ROOT/"configs/scenario_design/JILONG_EI_SCENARIOS.csv")
    out=idx.merge(design[["scenario_id",*PARAMS,"design_index"]],on="scenario_id",validate="one_to_one")
    return out if split is None else out.loc[out["split"].eq(split)].reset_index(drop=True)

class FrameStore:
    def __init__(self, rows: pd.DataFrame, cache_size: int=12):
        self.rows=rows.reset_index(drop=True); self.static,self.meta=static_and_exogenous(INPUT); self.shape=tuple(self.meta["shape"]); self.cache=OrderedDict(); self.cache_size=cache_size
    def frame(self,row: pd.Series,t: int) -> np.ndarray:
        path=Path(row.frames_dir)/f"state_{t:04d}s.npz"; key=str(path)
        if key not in self.cache:
            self.cache[key]=dynamic_state(path,self.shape)[0]
            while len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        self.cache.move_to_end(key); return self.cache[key]
    def sample(self, i: int, t: int):
        row=self.rows.iloc[i]; prev=max(0,t-10); current=self.frame(row,t); previous=self.frame(row,prev); target=self.frame(row,t+10)
        param=np.asarray([row[p] for p in PARAMS],np.float32)
        return previous,current,target,param,float(t)/1440.,row

    def window(self,i:int,t:int,k:int):
        """True future target window, clipped only at the 1440-s dataset end."""
        row=self.rows.iloc[i]; previous=self.frame(row,max(0,t-10));current=self.frame(row,t)
        targets=[self.frame(row,tt) for tt in range(t+10,min(1440,t+10*k)+1,10)]
        return previous,current,targets,np.asarray([row[p] for p in PARAMS],np.float32),float(t)/1440.,row

def feature_names():
    return [f'previous_{x}' for x in ('h','hu','hv','c','ice','dz')]+[f'current_{x}' for x in ('h','hu','hv','c','ice','dz')]+[f'delta_{x}' for x in ('h','hu','hv','c','ice','dz')]+['z_initial_normalized','current_bed_normalized','active','source_mask','channel','lateral_q_normalized','external_inflow_q_normalized']+[f'parameter_{x}_normalized' for x in PARAMS]+['time_norm','release_remaining','source_forcing_map','x_coord','y_coord']

def build_features(previous: torch.Tensor, current: torch.Tensor, static: torch.Tensor, params: torch.Tensor, time_fraction: torch.Tensor, transform, normalizer) -> torch.Tensor:
    """Named 35-channel V2 input; only history is state-transformed, never route data."""
    b,_,h,w=current.shape
    if static.shape[0] == 1 and b != 1: static=static.expand(b,-1,-1,-1)
    pp,pc=transform.encode(previous),transform.encode(current);delta=pc-pp
    z0,active,source,channel,lateral,external=[static[:,i:i+1] for i in range(6)]
    bed=normalizer.terrain(z0+current[:,5:6]); terrain=normalizer.terrain(z0)
    maps=[normalizer.params(params)[:,i:i+1,None,None].expand(-1,-1,h,w) for i in range(params.shape[1])]
    t=time_fraction[:,None,None,None].expand(-1,1,h,w); remaining=(1-(time_fraction*1440)/30).clamp(0,1)[:,None,None,None].expand(-1,1,h,w)
    # Solver injection has a constant rate while t<release_duration (30 s),
    # not a decaying rate. ``remaining`` is separately supplied as a feature.
    release_active=(time_fraction*1440<30).to(current.dtype)[:,None,None,None].expand(-1,1,h,w)
    source_force=source*(params[:,0:1,None,None].expand(-1,-1,h,w))*release_active
    xx=torch.linspace(-1,1,w,device=current.device)[None,None,None,:].expand(b,1,h,w); yy=torch.linspace(1,-1,h,device=current.device)[None,None,:,None].expand(b,1,h,w)
    out=torch.cat((pp,pc,delta,terrain,bed,active,source,channel,normalizer.forcing(lateral,'lateral'),normalizer.forcing(external,'external'),*maps,t,remaining,source_force,xx,yy),dim=1)
    if out.shape[1]!=len(feature_names()):raise RuntimeError(f'feature mismatch {out.shape[1]} != {len(feature_names())}')
    return out

def static_tensor(device: torch.device) -> torch.Tensor:
    static,_=static_and_exogenous(INPUT); return torch.from_numpy(static).unsqueeze(0).to(device)

def design_json() -> dict:
    return {"state_channels":["h","hu","hv","c","ice","dz"],"history":"transformed previous,current,current-previous; t0 previous=current", "predictor_excludes":"route_chainage_m and transects are evaluation-only", "feature_channels":len(feature_names()),"feature_names":feature_names(),"parameters":list(PARAMS),"source_schedule":"Park-v2 progressive release: 30-s active duration; forcing is source_mask * volume_scale * remaining-release activity"}
