from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch,matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator.dataset import post_project
from src.physics_frame_adapter import dynamic_tensor,static_tensor
from src.global_operator.model import JilongGlobalFNO
from scripts.train_global_operator import compose
def main():
 d=ROOT/'outputs/park_v2_h0/h0_global_eval_reference/frames'; frames=[]
 for p in d.glob('frame_*.npz'):
  with np.load(p) as z:frames.append((float(z['time_s']),p))
 ref=lambda t:min(frames,key=lambda q:abs(q[0]-t))[1]; norm=json.loads((ROOT/'models/global_operator/normalization.json').read_text());cfg=json.loads((ROOT/'models/global_operator/config.json').read_text());s=cfg['selected'];m=JilongGlobalFNO(s['width'],s['modes'],s['depth']).cuda().eval();m.load_state_dict(torch.load(ROOT/'models/global_operator/last.pt',map_location='cuda',weights_only=False)['model']); st,_=static_tensor(ROOT/'data/downloads/park_v2/inputs/upper30h.npz');
 class DS: pass
 ds=DS();ds.static=st[[0,3]]; x,_=dynamic_tensor(ref(0),(921,882));p=np.array([1,.2,np.log(.0074),.0145,np.log(337)],np.float32);rows=[]
 for t in range(0,1440,10):
  inp,_=compose(ds,x,p,t/1440,max(0,1-t/30),norm,'cuda')
  with torch.no_grad():z=m(inp).float().cpu().numpy()[0]
  x=post_project(z*np.array(norm['state']['std'],np.float32)[:,None,None]+np.array(norm['state']['mean'],np.float32)[:,None,None]);y,_=dynamic_tensor(ref(t+10),(921,882)); e=x[0]-y[0];rows.append({'time_s':t+10,'h_rmse':float(np.sqrt(np.mean(e**2))),'h_rell2':float(np.linalg.norm(e)/max(np.linalg.norm(y[0]),1e-8)),'wet_iou':float(((x[0]>.03)&(y[0]>.03)).sum()/max(((x[0]>.03)|(y[0]>.03)).sum(),1))})
 out=ROOT/'results/global_operator';fig=ROOT/'figures/global_operator';out.mkdir(parents=True,exist_ok=True);fig.mkdir(parents=True,exist_ok=True);q=pd.DataFrame(rows);q.to_csv(out/'h0_rollout_metrics.csv',index=False);ax=q.plot(x='time_s',y=['h_rell2','wet_iou'],secondary_y='wet_iou',title='H0 historical-anchor rollout');ax.figure.savefig(fig/'h0_rollout_comparison.png',dpi=180,bbox_inches='tight');plt.close(ax.figure)
if __name__=='__main__':main()
