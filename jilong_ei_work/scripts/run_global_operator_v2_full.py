"""Single V2 launcher with an explicit code-only safety gate.

Tomorrow's approved command is ``python scripts/run_global_operator_v2_full.py --formal``.
Today only ``--dry-run`` and ``--smoke`` are accepted; neither can touch formal
model/result directories or generate final-holdout physics.
"""
from __future__ import annotations
import argparse,json,random,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.dataset import FrameStore,build_features,design_json,scenario_rows,static_tensor
from src.global_operator_v2.frame_adapter import static_and_exogenous
from src.global_operator_v2.losses import project_physical
from src.global_operator_v2.model import JilongGlobalOperatorV2
from src.global_operator_v2.trainer import CURRICULUM,choose_amp,restore_checkpoint,rollout_loss,save_checkpoint
from src.global_operator_v2.transforms import PhysicalTransform

CONFIG=json.loads((ROOT/'configs/GLOBAL_OPERATOR_V2.json').read_text(encoding='utf-8'))
SMOKE_MODELS=ROOT/'models/global_operator_v2_smoke'; SMOKE_RESULTS=ROOT/'results/global_operator_v2_smoke'
FORMAL_MODELS=ROOT/'models/global_operator_v2'; FORMAL_RESULTS=ROOT/'results/global_operator_v2'

def verify_layout():
    static,meta=static_and_exogenous(ROOT/'data/downloads/park_v2/inputs/upper30h.npz')
    rows=scenario_rows(); required={'z_initial','active','source_mask','channel','lateral_q_active_m3_s','external_inflow_q_m3_s'}
    assert required==set(meta['static_channels']), meta['static_channels']
    assert static.shape==(6,921,882),static.shape
    assert len(rows)==200 and set(rows['split'])=={'TRAIN','VAL','TEST'}
    assert design_json()['feature_channels']==30
    return {'input_shape':list(static.shape),'scenario_count':int(len(rows)),'splits':rows['split'].value_counts().to_dict(),'feature_channels':30,'formal_model_dir':str(FORMAL_MODELS),'formal_result_dir':str(FORMAL_RESULTS),'policy':CONFIG['today_policy']}

def make_batch(store,device,crop=None):
    # A nonzero frame has momentum and bed-change support, so transform and
    # projection tests exercise all six retained state components.
    previous,current,target,p,t,row=store.sample(0,120)
    a=lambda x:torch.from_numpy(x).unsqueeze(0).to(device)
    prev,cur,tgt=a(previous),a(current),a(target); params=torch.from_numpy(p).unsqueeze(0).to(device); times=torch.tensor([t],device=device)
    static=static_tensor(device); active=static[:,1:2]
    if crop:
        # Fixed active-corridor crop: a developer smoke only, never a formal capacity claim.
        y0,x0,size=240,250,crop; prev=prev[:,:,y0:y0+size,x0:x0+size];cur=cur[:,:,y0:y0+size,x0:x0+size];tgt=tgt[:,:,y0:y0+size,x0:x0+size];static=static[:,:,y0:y0+size,x0:x0+size];active=active[:,:,y0:y0+size,x0:x0+size]
    return prev,cur,tgt,params,times,static,active

def smoke(updates:int=64,crop:int=256):
    if updates<1 or updates>64: raise ValueError('--smoke-updates must be 1..64')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.manual_seed(CONFIG['seed']);np.random.seed(CONFIG['seed']);random.seed(CONFIG['seed'])
    rows=scenario_rows('TRAIN');store=FrameStore(rows); batch=make_batch(store,device,crop)
    prev,cur,tgt,p,t,static,active=batch; transform=PhysicalTransform.fit(torch.cat((prev,cur,tgt)))
    amp,amp_dtype=choose_amp(device); model=JilongGlobalOperatorV2(30,width=16,modes=8,depth=2).to(device);opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4);scaler=torch.amp.GradScaler('cuda',enabled=amp and amp_dtype==torch.float16)
    SMOKE_MODELS.mkdir(parents=True,exist_ok=True);SMOKE_RESULTS.mkdir(parents=True,exist_ok=True); logs=[]
    for step in range(1,updates+1):
        opt.zero_grad(set_to_none=True); loss,detail,pred=rollout_loss(model,transform,prev,cur,[tgt]*2,static,p,t,active,2,amp_dtype)
        if scaler.is_enabled(): scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(opt);scaler.update()
        else: loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step in {1,updates}: logs.append({'step':step,'loss':float(loss.detach().cpu()),'pred_h_max_m':float(pred[:,0].max().detach().cpu()),'vram_mib':float(torch.cuda.max_memory_allocated()/2**20) if device.type=='cuda' else 0.})
    # BPTT development probes: bounded, same V2 path; crop declaration prevents a false full-grid claim.
    probes=[]
    for k in (2,4,6):
        opt.zero_grad(set_to_none=True)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        loss,_,_=rollout_loss(model,transform,prev,cur,[tgt]*k,static,p,t,active,k,amp_dtype);loss.backward();opt.zero_grad(set_to_none=True)
        probes.append({'rollout_steps':k,'developer_crop_px':crop,'peak_vram_mib':float(torch.cuda.max_memory_allocated()/2**20) if device.type=='cuda' else 0.,'status':'DEVELOPER_SMOKE_NOT_FORMAL_CAPACITY_CERTIFICATION'})
    ck=SMOKE_MODELS/'smoke_last.pt'; save_checkpoint(ck,model,opt,None,updates,transform,CONFIG); restored=JilongGlobalOperatorV2(30,width=16,modes=8,depth=2).to(device); restore_checkpoint(ck,restored)
    result={'status':'PASS','mode':'CODE_ONLY_SMOKE','updates':updates,'crop_px':crop,'device':str(device),'amp_dtype':str(amp_dtype),'transform':transform.to_dict(),'logs':logs,'bptt_probes':probes,'checkpoint_roundtrip':'PASS','formal_directories_untouched':True}
    (SMOKE_RESULTS/'SMOKE_REPORT.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

def formal():
    """Approved-day entry point; stage execution is deliberately segregated from code-only actions."""
    raise RuntimeError('Formal runner scaffold is present but intentionally locked during code-only mode. Run only after explicit approval, then implement/resume full A/B/C/D execution in this entry point.')

def main(a):
    layout=verify_layout()
    if a.dry_run:
        print(json.dumps({'status':'PASS','mode':'DRY_RUN',**layout},indent=2));return
    if a.smoke: smoke(a.smoke_updates,a.smoke_crop);return
    if a.formal: formal();return
    raise SystemExit('Choose exactly one of --dry-run, --smoke, or --formal.')
if __name__=='__main__':
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--dry-run',action='store_true');g.add_argument('--smoke',action='store_true');g.add_argument('--formal',action='store_true');p.add_argument('--smoke-updates',type=int,default=64);p.add_argument('--smoke-crop',type=int,default=256);main(p.parse_args())
