"""CPU-only preparation: train-only normalization and fixed validation sampling."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator.dataset import ScenarioFrames
rng=np.random.default_rng(20260921); idx=pd.read_csv(ROOT/'data/scenario_index.csv'); design=pd.read_csv(ROOT/'configs/scenario_design/JILONG_EI_SCENARIOS.csv'); idx=idx.merge(design[['scenario_id','volume_scale','ice_fraction','erosion_K','n_debris','dep_tau_s']],on='scenario_id',validate='one_to_one'); train=idx[idx.split=='TRAIN'].reset_index(drop=True); val=idx[idx.split=='VAL'].reset_index(drop=True); ds=ScenarioFrames(train,cache_size=2)
# One deterministic transition per each of all 160 train scenarios; random cells
# retain broad state coverage without materializing a dense training dataset.
state=[]
for i in range(len(train)):
    x,_,_,_,_,_=ds.sample(i,int(rng.integers(0,144))*10); flat=x.reshape(5,-1)[:,rng.choice(x.shape[1]*x.shape[2],8192,replace=False)]; state.append(flat)
a=np.concatenate(state,axis=1); bed=ds.static[0].reshape(-1); norms={'state':{'mean':a.mean(1).tolist(),'std':np.maximum(a.std(1),1e-6).tolist(),'min':a.min(1).tolist(),'max':a.max(1).tolist(),'q01':np.quantile(a,.01,axis=1).tolist(),'q99':np.quantile(a,.99,axis=1).tolist()},'bed':{'mean':float(bed.mean()),'std':float(max(bed.std(),1e-6))},'params':{}}
p=train[['volume_scale','ice_fraction','erosion_K','n_debris','dep_tau_s']].copy();p.erosion_K=np.log(p.erosion_K);p.dep_tau_s=np.log(p.dep_tau_s)
norms['params']={'mean':p.mean().tolist(),'std':p.std().clip(lower=1e-6).tolist(),'names':['volume_scale','ice_fraction','log_erosion_K','n_debris','log_dep_tau_s'],'source_split':'TRAIN_ONLY'}
out=ROOT/'models/global_operator';out.mkdir(parents=True,exist_ok=True);(out/'normalization.json').write_text(json.dumps(norms,indent=2));
subset=[]
for r in val.itertuples():
    for t in rng.choice(np.arange(0,1440,10),8,replace=False):subset.append({'scenario_id':r.scenario_id,'time_s':int(t)})
pd.DataFrame(subset).sort_values(['scenario_id','time_s']).to_csv(out/'validation_subset.csv',index=False)
print('prepared',len(train),'train scenarios and',len(subset),'fixed validation pairs')
