from __future__ import annotations
import json
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
from src.physics_frame_adapter import dynamic_tensor, static_tensor

ROOT=Path(__file__).resolve().parents[2]
STATE=('h','hu','hv','c','ice'); SHAPE=(921,882); DRY=.03

def frame_path(row, t:int): return Path(row.frames_dir)/f'state_{t:04d}s.npz'

class ScenarioFrames:
    def __init__(self,index:pd.DataFrame,cache_size=8): self.index=index.reset_index(drop=True); self.cache=OrderedDict(); self.cache_size=cache_size; self.static,self.meta=static_tensor(ROOT/'data/downloads/park_v2/inputs/upper30h.npz'); self.static=self.static[[0,3]]
    def state(self,path):
        k=str(path)
        if k not in self.cache:
            x,_=dynamic_tensor(path,SHAPE); self.cache[k]=x
            if len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        self.cache.move_to_end(k); return self.cache[k]
    def sample(self,i:int,t:int):
        row=self.index.iloc[i]; x=self.state(frame_path(row,t)); y=self.state(frame_path(row,t+10));
        p=np.array([row.volume_scale,row.ice_fraction,np.log(row.erosion_K),row.n_debris,np.log(row.dep_tau_s)],np.float32)
        return x,y,p,float(t)/1440.,max(0.,1.-float(t)/30.),row

def post_project(x:np.ndarray,dry=DRY):
    x=x.copy(); x[0]=np.maximum(x[0],0); x[3:5]=np.clip(x[3:5],0,1); s=x[3]+x[4]; bad=s>1
    x[3,bad]/=s[bad];x[4,bad]/=s[bad]; x[1:3,x[0]<dry]=0; return x
