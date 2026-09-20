from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np,torch,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator.dataset import ScenarioFrames,post_project
from src.global_operator.model import JilongGlobalFNO
from scripts.train_global_operator import compose
idx=pd.read_csv(ROOT/'data/scenario_index.csv');d=pd.read_csv(ROOT/'configs/scenario_design/JILONG_EI_SCENARIOS.csv');row=idx[idx['split']=='TEST'].merge(d.drop(columns=['split']),on='scenario_id').reset_index(drop=True).iloc[[0]];ds=ScenarioFrames(row);n=json.loads((ROOT/'models/global_operator/normalization.json').read_text());c=json.loads((ROOT/'models/global_operator/config.json').read_text());s=c['selected'];m=JilongGlobalFNO(s['width'],s['modes'],s['depth']).cuda().eval();m.load_state_dict(torch.load(ROOT/'models/global_operator/last.pt',map_location='cuda',weights_only=False)['model']);x,_,p,_,_,_=ds.sample(0,0)
def roll():
 global x
 q=x.copy();torch.cuda.synchronize();a=time.perf_counter()
 for t in range(0,1440,10):
  inp,_=compose(ds,q,p,t/1440,max(0,1-t/30),n,'cuda')
  with torch.no_grad():z=m(inp).float().cpu().numpy()[0]
  q=post_project(z*np.array(n['state']['std'],np.float32)[:,None,None]+np.array(n['state']['mean'],np.float32)[:,None,None])
 torch.cuda.synchronize();return time.perf_counter()-a
for _ in range(5):roll()
r=[roll() for _ in range(20)];o=ROOT/'results/global_operator';o.mkdir(parents=True,exist_ok=True);pd.DataFrame({'repeat':range(1,21),'wall_s':r}).to_csv(o/'runtime_benchmark.csv',index=False);print(json.dumps({'mean_s':float(np.mean(r)),'median_s':float(np.median(r)),'p95_s':float(np.quantile(r,.95))},indent=2))
