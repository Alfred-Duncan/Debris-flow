"""Formal Global-FNO launcher.  It refuses to train unless the full audit passed."""
from __future__ import annotations
import argparse,json,random,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator.dataset import ScenarioFrames,post_project
from src.global_operator.model import JilongGlobalFNO
from src.global_operator.losses import masked_huber

def rows():
    idx=pd.read_csv(ROOT/'data/scenario_index.csv'); d=pd.read_csv(ROOT/'configs/scenario_design/JILONG_EI_SCENARIOS.csv'); return idx.merge(d[['scenario_id','volume_scale','ice_fraction','erosion_K','n_debris','dep_tau_s']],on='scenario_id',validate='one_to_one')
def compose(ds,x,p,t,rel,n,device):
    xn=(x-np.array(n['state']['mean'],np.float32)[:,None,None])/np.array(n['state']['std'],np.float32)[:,None,None]
    bed=(ds.static[0]-n['bed']['mean'])/n['bed']['std']; active=ds.static[1]; pn=(p-np.array(n['params']['mean'],np.float32))/np.array(n['params']['std'],np.float32)
    cond=np.concatenate([np.broadcast_to(v,x.shape[1:])[None] for v in pn]+[np.full((1,*x.shape[1:]),t,np.float32),np.full((1,*x.shape[1:]),rel,np.float32)])
    return torch.from_numpy(np.concatenate([xn,bed[None],active[None],cond])[None]).to(device),torch.from_numpy(active[None]).to(device)
def main(a):
    audit=json.loads((ROOT/'reports/GLOBAL_OPERATOR_DATA_AUDIT.json').read_text())
    if audit['status']!='PASS': raise RuntimeError('STOP: GLOBAL_OPERATOR_DATA_AUDIT is not PASS')
    cfg=json.loads((ROOT/'configs/GLOBAL_OPERATOR_BASELINE.json').read_text()); n=json.loads((ROOT/'models/global_operator/normalization.json').read_text()); device='cuda'; torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])
    train=rows().query("split=='TRAIN'").reset_index(drop=True); ds=ScenarioFrames(train,cache_size=8)
    candidates=cfg['model_candidates']; probe=[]
    for c in candidates:
        m=JilongGlobalFNO(**c).to(device); opt=torch.optim.AdamW(m.parameters(),lr=1e-3); torch.cuda.reset_peak_memory_stats(); tic=time.perf_counter()
        for k in range(20):
            x,y,p,t,r,_=ds.sample(k%len(ds.index),(k%144)*10); inp,active=compose(ds,x,p,t,r,n,device); target=torch.from_numpy((y-np.array(n['state']['mean'],np.float32)[:,None,None])/np.array(n['state']['std'],np.float32)[:,None,None])[None].to(device); opt.zero_grad(); pred=m(inp); loss,_=masked_huber(pred,target,active); loss.backward();opt.step()
        torch.cuda.synchronize(); probe.append({**c,'parameters':m.parameter_count,'peak_vram_mib':torch.cuda.max_memory_allocated()/2**20,'step_s':(time.perf_counter()-tic)/20}); del m,opt;torch.cuda.empty_cache()
    (ROOT/'reports/GLOBAL_OPERATOR_MODEL_PROBE.json').write_text(json.dumps(probe,indent=2));
    if a.probe_only: print(json.dumps(probe,indent=2));return
    good=[z for z in probe if z['peak_vram_mib']<.7*8151]; chosen=max(good,key=lambda z:(z['width'],z['modes'])) if good else min(probe,key=lambda z:z['peak_vram_mib'])
    out=ROOT/'models/global_operator';out.mkdir(parents=True,exist_ok=True); logdir=ROOT/'results/global_operator';logdir.mkdir(parents=True,exist_ok=True); (out/'config.json').write_text(json.dumps({**cfg,'selected':chosen},indent=2)); model=JilongGlobalFNO(**{k:chosen[k] for k in ('width','modes','depth')}).to(device); opt=torch.optim.AdamW(model.parameters(),lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay']); log=[]
    # Deliberately step-based stochastic sampling; test split is never constructed here.
    for step in range(1,cfg['training']['max_steps']+1):
        i=np.random.randint(len(train)); t=np.random.randint(144)*10; x,y,p,tn,r,_=ds.sample(i,t); inp,active=compose(ds,x,p,tn,r,n,device); target=torch.from_numpy((y-np.array(n['state']['mean'],np.float32)[:,None,None])/np.array(n['state']['std'],np.float32)[:,None,None])[None].to(device); opt.zero_grad(); pred=model(inp); loss,each=masked_huber(pred,target,active,wet_weight=cfg['training']['wet_weight']); loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%50==0: log.append({'step':step,'train_total_loss':float(loss),'h_loss':float(each[0]),'hu_loss':float(each[1]),'hv_loss':float(each[2]),'c_loss':float(each[3]),'ice_loss':float(each[4]),'gpu_vram_mib':torch.cuda.max_memory_allocated()/2**20});pd.DataFrame(log).to_csv(logdir/'training_log.csv',index=False); print(f'JiLong Global FNO step {step}/12000 loss {float(loss):.5g} VRAM {torch.cuda.max_memory_allocated()/2**20:.0f} MiB',flush=True)
        if step%500==0: torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'normalization':n,'config':cfg},out/'last.pt')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--probe-only',action='store_true');p.add_argument('--resume',action='store_true');main(p.parse_args())
