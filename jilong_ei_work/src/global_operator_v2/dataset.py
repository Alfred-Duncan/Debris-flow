"""Scenario sampler and V2 feature construction (no predictor-side route data)."""
from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from .frame_adapter import dynamic_state, static_and_exogenous

ROOT=Path(__file__).resolve().parents[2]
INPUT=ROOT/"data/downloads/park_v2/inputs/upper30h.npz"
PARAMS=("volume_scale","ice_fraction","erosion_K","n_debris","dep_tau_s")

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

def build_features(previous: torch.Tensor, current: torch.Tensor, static: torch.Tensor, params: torch.Tensor, time_fraction: torch.Tensor) -> torch.Tensor:
    """[prev,current,delta,z_current,static maps,param maps,time]; 6+6+6+1+5+5+1=30."""
    b,_,h,w=current.shape
    if static.shape[0] == 1 and b != 1: static=static.expand(b,-1,-1,-1)
    delta=current-previous; bed=static[:,0:1]+current[:,5:6]
    maps=[params[:,i:i+1,None,None].expand(-1,-1,h,w) for i in range(params.shape[1])]
    t=time_fraction[:,None,None,None].expand(-1,1,h,w)
    return torch.cat((previous,current,delta,bed,static[:,1:],*maps,t),dim=1)

def static_tensor(device: torch.device) -> torch.Tensor:
    static,_=static_and_exogenous(INPUT); return torch.from_numpy(static).unsqueeze(0).to(device)

def design_json() -> dict:
    return {"state_channels":["h","hu","hv","c","ice","dz"],"history":"previous,current,current-previous; t0 previous=current", "predictor_excludes":"route_chainage_m and transects are evaluation-only", "feature_channels":30, "parameters":list(PARAMS)}
