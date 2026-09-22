from __future__ import annotations
import sys,numpy as np,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.dataset import FrameStore,scenario_rows,INPUT
from src.global_operator_v2.frame_adapter import static_and_exogenous
from src.local_corrector.normalization import fit,save
from scripts.run_global_v2_oracle_refinement import load_model,predict
def main():
 dev=torch.device('cuda');rows=scenario_rows('TRAIN');rng=np.random.default_rng(20260920);chosen=rows.iloc[rng.choice(len(rows),32,replace=False)].sort_values('scenario_id');store=FrameStore(chosen);model,tr,norm,delta,_=load_model(dev);static_np,_=static_and_exogenous(INPUT);static=torch.from_numpy(static_np).unsqueeze(0).to(dev);active=static[:,1:2];values=[[] for _ in range(6)];times=[0,10,20,30,60,120,240,480,720,960,1200,1430]
 for i,(_,row) in enumerate(chosen.reset_index(drop=True).iterrows()):
  for t in times:
   prev,current,_,params,time,_=store.sample(i,t);to=lambda x:torch.from_numpy(np.asarray(x)).unsqueeze(0).to(dev);prev,current,params=to(prev),to(current),to(params);truth=to(store.frame(row,t+10));prov=predict(model,prev,current,static,params,time,tr,norm,active);corr=(tr.encode(truth)-tr.encode(prov))[0].detach().cpu().numpy();cells=rng.choice(np.flatnonzero(static_np[1].ravel()),min(4096,int(static_np[1].sum())),replace=False)
   for c in range(6):values[c].append(corr[c].ravel()[cells])
 data=fit([np.concatenate(x) for x in values],chosen.scenario_id.tolist(),times);save(ROOT/'models/local_corrector_v1/correction_normalization.json',data);print('PASS',data['scales'])
if __name__=='__main__':main()
