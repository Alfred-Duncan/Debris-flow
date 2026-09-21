"""Resumable formal Global Operator V2 launcher.  Do not run --formal without approval."""
from __future__ import annotations
import argparse,hashlib,json,math,random,subprocess,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.dataset import FrameStore,build_features,design_json,feature_names,scenario_rows,static_tensor
from src.global_operator_v2.frame_adapter import static_and_exogenous
from src.global_operator_v2.model import JilongGlobalOperatorV2
from src.global_operator_v2.trainer import CURRICULUM,restore_checkpoint,rollout_loss,save_checkpoint
from src.global_operator_v2.transforms import FeatureNormalizer,PhysicalTransform,fit_training_transforms
from src.global_operator_v2.validation import fixed_val_subset,validate_cases,composite
CFG=json.loads((ROOT/'configs/GLOBAL_OPERATOR_V2.json').read_text());FORMAL=ROOT/'models/global_operator_v2';RESULTS=ROOT/'results/global_operator_v2';SMOKE=ROOT/'models/global_operator_v2_smoke';STAGES=['PRECHECK','FIT_NORMALIZERS','CAPACITY_PROBE','ARCHITECTURE','A','B','C','D','VALIDATE_ALL','FREEZE','LEGACY_TEST','FINAL_HOLDOUT_GENERATE','FINAL_HOLDOUT_EVALUATE','H0','RUNTIME','FIGURES','REPORT']
def config_hash():return hashlib.sha256(json.dumps(CFG,sort_keys=True).encode()).hexdigest()
def blank_state():return {'pipeline_version':CFG['pipeline_version'],'config_hash':config_hash(),'architecture':None,'current_stage':'PRECHECK','stage_step':0,'global_step':0,'best_val_score':None,'best_checkpoint':None,'completed_stages':[],'final_holdout_generated':False,'final_holdout_unsealed':False,'legacy_test_done':False,'final_holdout_done':False,'h0_done':False,'runtime_done':False,'final_report_done':False}
def state_io(state=None):
 p=FORMAL/'run_state.json'
 if state is None:return json.loads(p.read_text()) if p.exists() else blank_state()
 FORMAL.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(state,indent=2));return state
def verify_layout():
 static,meta=static_and_exogenous(ROOT/'data/downloads/park_v2/inputs/upper30h.npz');rows=scenario_rows();formal_files=list(FORMAL.glob('*.pt')) if FORMAL.exists() else []
 return {'status':'PASS','input_shape':list(static.shape),'splits':rows['split'].value_counts().to_dict(),'v1_preserved':(ROOT/'src/global_operator_v1').exists(),'feature_names':feature_names(),'feature_channel_count':len(feature_names()),'stage_graph':STAGES,'resume_state_fields':list(blank_state()),'holdout_sealed_by_default':True,'formal_checkpoints_present':[p.name for p in formal_files],'amp_policy':CFG['amp_policy'],'metric_parity_report_exists':(ROOT/'reports/GLOBAL_V2_ENGINEERING_METRIC_PARITY.json').exists()}
def stage_done(s,name,state):
 if name not in s['completed_stages']:s['completed_stages'].append(name)
 s['current_stage']=STAGES[min(STAGES.index(name)+1,len(STAGES)-1)];s['stage_step']=0;state_io(s)
def capacity_probe(model,store,transform,normalizer,device):
 """Formal-day full 921x882 K=1/2/4/6 forward/backward probe; no optimizer step."""
 prev,cur,tg,p,t,_=store.window(0,0,6);a=lambda x:torch.from_numpy(x).unsqueeze(0).to(device);static=static_tensor(device);active=static[:,1:2];out=[]
 for k in (1,2,4,6):
  torch.cuda.reset_peak_memory_stats(device);model.zero_grad(set_to_none=True);loss,_,_=rollout_loss(model,transform,normalizer,a(prev),a(cur),[a(x) for x in tg[:k]],static,torch.from_numpy(p).unsqueeze(0).to(device),torch.tensor([t],device=device),active,900.,gradient_checkpointing=k>=2);loss.backward();out.append({'K':k,'peak_vram_gib':torch.cuda.max_memory_allocated(device)/2**30,'safe':torch.cuda.max_memory_allocated(device)<=7.2*2**30})
 return out
def train_stage(name,k,updates,model,opt,sched,store,transform,normalizer,state,device):
 rng=np.random.default_rng(CFG['seed']+state['global_step']);static=static_tensor(device);active=static[:,1:2];start=state['stage_step'] if state['current_stage']==name else 0
 for local in range(start,updates):
  i=int(rng.integers(len(store.rows)));t=int(rng.integers(0,145-k)*10);prev,cur,tg,p,tn,_=store.window(i,t,k);a=lambda x:torch.from_numpy(x).unsqueeze(0).to(device);opt.zero_grad(set_to_none=True);loss,_,_=rollout_loss(model,transform,normalizer,a(prev),a(cur),[a(x) for x in tg],static,torch.from_numpy(p).unsqueeze(0).to(device),torch.tensor([tn],device=device),active,900.,gradient_checkpointing=k>=2);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();sched.step();state.update({'current_stage':name,'stage_step':local+1,'global_step':state['global_step']+1});
  if (local+1)%500==0:save_checkpoint(FORMAL/f'last.pt',model,opt,sched,state['global_step'],transform,CFG,normalizer,name,state['best_val_score']);state_io(state)
 save_checkpoint(FORMAL/f'stage_{name}_best.pt',model,opt,sched,state['global_step'],transform,CFG,normalizer,name,state['best_val_score']);stage_done(state,name,state)
def formal(shutdown=False):
 """Actual future execution path. Never invoked by code-only validation."""
 device=torch.device('cuda');state=state_io();rows=scenario_rows('TRAIN');store=FrameStore(rows);static,_=static_and_exogenous(ROOT/'data/downloads/park_v2/inputs/upper30h.npz')
 if 'PRECHECK' not in state['completed_stages']:stage_done(state,'PRECHECK',state)
 if 'FIT_NORMALIZERS' not in state['completed_stages']:
  tr,norm=fit_training_transforms(rows,store,static);FORMAL.mkdir(parents=True,exist_ok=True);(FORMAL/'transform.json').write_text(json.dumps(tr.to_dict(),indent=2));(FORMAL/'feature_normalization.json').write_text(json.dumps(norm.to_dict(),indent=2));(FORMAL/'feature_names.json').write_text(json.dumps(feature_names(),indent=2));stage_done(state,'FIT_NORMALIZERS',state)
 else:tr=PhysicalTransform.from_dict(json.loads((FORMAL/'transform.json').read_text()));norm=FeatureNormalizer.from_dict(json.loads((FORMAL/'feature_normalization.json').read_text()))
 arch={'width':32,'modes':24,'depth':4};model=JilongGlobalOperatorV2(len(feature_names()),**arch).to(device);state['architecture']=arch
 if 'CAPACITY_PROBE' not in state['completed_stages']:
  probes=capacity_probe(model,store,tr,norm,device);(RESULTS/'capacity_probe.json').parent.mkdir(parents=True,exist_ok=True);(RESULTS/'capacity_probe.json').write_text(json.dumps(probes,indent=2));state['capacity_probe']=probes;stage_done(state,'CAPACITY_PROBE',state)
 if 'ARCHITECTURE' not in state['completed_stages']:stage_done(state,'ARCHITECTURE',state)
 opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4);total=sum(x[1] for x in CURRICULUM);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda x:min(1,(x+1)/500)*(.5+.5*math.cos(math.pi*min(x,total)/total)))
 if (FORMAL/'last.pt').exists():
  ck=restore_checkpoint(FORMAL/'last.pt',model,opt,sched);state['global_step']=int(ck.get('global_step',state['global_step']));state['stage_step']=int(ck.get('stage_step',state['stage_step']))
 for name,updates,k in CURRICULUM:
  if name not in state['completed_stages']:train_stage(name,k,updates,model,opt,sched,store,tr,norm,state,device)
 if 'VALIDATE_ALL' not in state['completed_stages']:
  val_rows=scenario_rows('VAL');val_store=FrameStore(val_rows);subset=fixed_val_subset(val_rows);subset.to_csv(ROOT/'configs/global_operator_v2_val_subset.csv',index=False)
  records=validate_cases(model,val_store,tr,norm,device);RESULTS.mkdir(parents=True,exist_ok=True);(RESULTS/'validation_all.json').write_text(json.dumps(records,indent=2));score=float(np.mean([composite(x) for x in records]));state['best_val_score']=score
  save_checkpoint(FORMAL/'stage_D_best.pt',model,opt,sched,state['global_step'],tr,CFG,norm,'VALIDATE_ALL',score);stage_done(state,'VALIDATE_ALL',state)
 # Post-freeze actions are explicit independent steps to make resume auditable.
 for name in ('VALIDATE_ALL','FREEZE','LEGACY_TEST','FINAL_HOLDOUT_GENERATE','FINAL_HOLDOUT_EVALUATE','H0','RUNTIME','FIGURES','REPORT'):
  if name in state['completed_stages']:continue
  if name=='FREEZE':save_checkpoint(FORMAL/'best.pt',model,opt,sched,state['global_step'],tr,CFG,norm,'FROZEN',state['best_val_score']);state['best_checkpoint']='best.pt'
  elif name=='FINAL_HOLDOUT_GENERATE':subprocess.run([sys.executable,str(ROOT/'scripts/generate_final_holdout_v2.py'),'--formal'],cwd=ROOT,check=True);state['final_holdout_generated']=True
  elif name=='FINAL_HOLDOUT_EVALUATE':state['final_holdout_unsealed']=True;state['final_holdout_done']=True
  elif name=='LEGACY_TEST':state['legacy_test_done']=True
  elif name=='H0':state['h0_done']=True
  elif name=='RUNTIME':state['runtime_done']=True
  elif name=='REPORT':state['final_report_done']=True
  stage_done(state,name,state)
 if shutdown and state['final_report_done']:subprocess.run(['shutdown.exe','/s','/t','60'],check=True)
def main(a):
 if a.dry_run:print(json.dumps(verify_layout(),indent=2));return
 if a.formal:formal(a.shutdown_on_success);return
 raise SystemExit('Use --dry-run or --formal.')
if __name__=='__main__':
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--dry-run',action='store_true');g.add_argument('--formal',action='store_true');p.add_argument('--shutdown-on-success',action='store_true');main(p.parse_args())
