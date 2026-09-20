"""Frozen-checkpoint autoregressive evaluation; never writes training parameters."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch,matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator.dataset import ScenarioFrames,post_project,SHAPE
from src.global_operator.model import JilongGlobalFNO
from scripts.train_global_operator import compose
TIMES={60,120,300,600,900,1200,1440}
def metric(p,y,active):
    out={}
    for i,k in enumerate(['h','hu','hv','c','ice']):
        e=p[i]-y[i]; out[f'{k}_rmse']=float(np.sqrt(np.mean(e[active]**2)));out[f'{k}_mae']=float(np.mean(np.abs(e[active])));out[f'{k}_rell2']=float(np.linalg.norm(e[active])/max(np.linalg.norm(y[i][active]),1e-8))
    wetp=p[0]>.03;wety=y[0]>.03;out['wet_area_error_cells']=int(wetp.sum()-wety.sum());out['wet_iou']=float((wetp&wety).sum()/max((wetp|wety).sum(),1));out['front_position_error']='NOT_FIELD_DERIVABLE: no published route-coordinate postprocessor in retained frame schema';return out
def main():
    out=ROOT/'results/global_operator';fig=ROOT/'figures/global_operator';out.mkdir(parents=True,exist_ok=True);fig.mkdir(parents=True,exist_ok=True)
    norm=json.loads((ROOT/'models/global_operator/normalization.json').read_text());cfg=json.loads((ROOT/'models/global_operator/config.json').read_text());sel=cfg['selected'];m=JilongGlobalFNO(sel['width'],sel['modes'],sel['depth']).cuda().eval();ck=torch.load(ROOT/'models/global_operator/last.pt',map_location='cuda',weights_only=False);m.load_state_dict(ck['model']);idx=pd.read_csv(ROOT/'data/scenario_index.csv');design=pd.read_csv(ROOT/'configs/scenario_design/JILONG_EI_SCENARIOS.csv');idx=idx.merge(design[['scenario_id','volume_scale','ice_fraction','erosion_K','n_debris','dep_tau_s']],on='scenario_id');allrows=[]
    for split in ['VAL','TEST']:
      sub=idx[idx.split==split].reset_index(drop=True);ds=ScenarioFrames(sub,cache_size=2);active=ds.static[1]>0
      for i,row in sub.iterrows():
        x,_,p,_,_,_=ds.sample(i,0); pred=x.copy()
        for t in range(0,1440,10):
          inp,_=compose(ds,pred,p,t/1440,max(0,1-t/30),norm,'cuda')
          with torch.no_grad(): z=m(inp).float().cpu().numpy()[0]
          pred=post_project(z*np.array(norm['state']['std'],np.float32)[:,None,None]+np.array(norm['state']['mean'],np.float32)[:,None,None])
          true=ds.state(Path(row.frames_dir)/f'state_{t+10:04d}s.npz')
          if t+10 in TIMES: allrows.append({'split':split,'scenario_id':row.scenario_id,'time_s':t+10,**metric(pred,true,active)})
    metrics=pd.DataFrame(allrows);metrics[metrics.split=='VAL'].to_csv(out/'val_scenario_metrics.csv',index=False);metrics[metrics.split=='TEST'].to_csv(out/'test_scenario_metrics.csv',index=False)
    # compact formal diagnostic figure: Test h error trajectory
    g=metrics[metrics.split=='TEST'].groupby('time_s').h_rell2.agg(['mean','median']);plt.style.use('seaborn-v0_8-whitegrid');ax=g.plot(marker='o');ax.set(xlabel='rollout time (s)',ylabel='h RelL2',title='Frozen Global-FNO test rollout error');ax.figure.savefig(fig/'test_rollout_error_vs_time.png',dpi=180,bbox_inches='tight');plt.close(ax.figure)
    summary={'checkpoint':'models/global_operator/last.pt','checkpoint_selection':'NOT_AVAILABLE: pre-validation training launcher; frozen last checkpoint evaluated without Test-guided tuning','val_rows':int((metrics.split=='VAL').shape[0]),'test_rows':int((metrics.split=='TEST').shape[0]),'test_mean':metrics[metrics.split=='TEST'].mean(numeric_only=True).to_dict(),'front_metric':'NOT_FIELD_DERIVABLE'};(out/'frozen_checkpoint_evaluation.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
