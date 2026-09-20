from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch,matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator.dataset import ScenarioFrames,post_project
from src.global_operator.model import JilongGlobalFNO
from scripts.train_global_operator import compose
def main():
 out=ROOT/'results/global_operator/error_maps';fig=ROOT/'figures/global_operator/error_maps';out.mkdir(parents=True,exist_ok=True);fig.mkdir(parents=True,exist_ok=True); norm=json.loads((ROOT/'models/global_operator/normalization.json').read_text());cfg=json.loads((ROOT/'models/global_operator/config.json').read_text());s=cfg['selected'];m=JilongGlobalFNO(s['width'],s['modes'],s['depth']).cuda().eval();m.load_state_dict(torch.load(ROOT/'models/global_operator/last.pt',map_location='cuda',weights_only=False)['model']);idx=pd.read_csv(ROOT/'data/scenario_index.csv');d=pd.read_csv(ROOT/'configs/scenario_design/JILONG_EI_SCENARIOS.csv');r=pd.read_csv(ROOT/'data/scenario_response_summary.csv');z=idx.merge(d[['scenario_id','volume_scale','ice_fraction','erosion_K','n_debris','dep_tau_s']],on='scenario_id').query("split=='TEST'"); rr=r.set_index('scenario_id'); med=z.iloc[(z.scenario_id.map(rr.peak_gyirong_Q_m3_s).sub(r[r.split=='TEST'].peak_gyirong_Q_m3_s.median()).abs()).argmin()]; chosen=pd.DataFrame([med,z.loc[z.volume_scale.idxmax()],z.loc[z.volume_scale.idxmin()],z.loc[z.n_debris.idxmax()],z.loc[z.dep_tau_s.idxmax()]]).drop_duplicates('scenario_id').reset_index(drop=True);ds=ScenarioFrames(chosen,cache_size=10);states=[];params=[]
 for i in range(len(chosen)): x,_,p,_,_,_=ds.sample(i,0);states.append(x);params.append(p)
 for t in range(0,1440,10):
  ins=[compose(ds,states[i],params[i],t/1440,max(0,1-t/30),norm,'cuda')[0] for i in range(len(chosen))]
  with torch.no_grad(): zz=m(torch.cat(ins)).float().cpu().numpy()
  states=[post_project(zz[i]*np.array(norm['state']['std'],np.float32)[:,None,None]+np.array(norm['state']['mean'],np.float32)[:,None,None]) for i in range(len(chosen))]
  if t+10 in {300,600,900,1200,1440}:
   for i,row in chosen.iterrows():
    truth=ds.state(Path(row.frames_dir)/f'state_{t+10:04d}s.npz');err=np.abs(states[i][0]-truth[0]);np.savez_compressed(out/f'{row.scenario_id}_{t+10:04d}s_h_error.npz',h_abs_error=err,wet_difference=(states[i][0]>.03)!=(truth[0]>.03));fig1,ax=plt.subplots(1,3,figsize=(10,3));
    for a,v,title in zip(ax,[states[i][0],truth[0],err],['prediction h','physics h','absolute h error']): im=a.imshow(v,cmap='turbo',vmin=0,vmax=np.percentile(v,99) if np.any(v) else 1);a.set_title(title);a.axis('off');fig1.colorbar(im,ax=a,shrink=.7)
    fig1.suptitle(f'{row.scenario_id} t={t+10}s');fig1.savefig(fig/f'{row.scenario_id}_{t+10:04d}s.png',dpi=150,bbox_inches='tight');plt.close(fig1)
 pd.DataFrame({'scenario_id':chosen.scenario_id,'selection':['median_response','high_volume','low_volume','high_roughness','long_deposition'][:len(chosen)]}).to_csv(out/'representative_cases.csv',index=False)
if __name__=='__main__':main()
