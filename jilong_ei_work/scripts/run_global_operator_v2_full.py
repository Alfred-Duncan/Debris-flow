"""Single-state-machine V2 training-core launcher; --formal is deliberately not invoked in code review."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,random,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.pipeline import PipelineState,PipelineStage,load_state,save_state,done,checkpoint_agreement,record_failure,apply_capacity_result,next_effective_stage,build_scheduler,training_sample_for_step
from src.global_operator_v2.dataset import FrameStore,scenario_rows,feature_names,static_tensor
from src.global_operator_v2.frame_adapter import static_and_exogenous
from src.global_operator_v2.model import JilongGlobalOperatorV2
from src.global_operator_v2.trainer import restore_checkpoint,save_checkpoint,rollout_loss,train_stage,is_improvement
from src.global_operator_v2.validation import select_fixed_val_subset,validate_one_step,validate_all_val
from src.global_operator_v2.evaluation import freeze_best_candidate
from src.global_operator_v2.transforms import PhysicalTransform,FeatureNormalizer,fit_training_transforms
CFG=json.loads((ROOT/'configs/GLOBAL_OPERATOR_V2.json').read_text());FORMAL=ROOT/'models/global_operator_v2';RESULTS=ROOT/'results/global_operator_v2'
model_ref=[None]
CORE=(PipelineStage.PRECHECK,PipelineStage.FIT_NORMALIZERS,PipelineStage.CAPACITY_PROBE,PipelineStage.ARCHITECTURE,PipelineStage.STAGE_A,PipelineStage.STAGE_B,PipelineStage.STAGE_C,PipelineStage.STAGE_D,PipelineStage.VAL_CONFIRM,PipelineStage.FREEZE)
def cfg_hash():return hashlib.sha256(json.dumps(CFG,sort_keys=True).encode()).hexdigest()
def initial_state():return PipelineState(CFG['pipeline_version'],cfg_hash())
def sample_for_global_step(seed,step,n,max_t):
 return training_sample_for_step(seed,step,n,1,n_time_states=max_t+1)
def save_last(model,opt,sched,tr,norm,state):
 save_checkpoint(path=FORMAL/'last.pt',model=model,optimizer=opt,scheduler=sched,step=state.global_step,transform=tr,config=CFG,normalizer=norm,stage=state.current_stage,best_metric=state.best_val_score,stage_step=state.stage_step,stage_updates_total=state.stage_updates_total,architecture=state.architecture,best_checkpoint_path=state.best_checkpoint_path,stage_best_score=state.stage_best_score)
def save_resume_pair(model,opt,sched,tr,norm,state):
 save_last(model,opt,sched,tr,norm,state);save_state(FORMAL/'run_state.json',state)
def enter_stage(model,opt,sched,tr,norm,state,stage):
 state.current_stage=stage;state.stage_step=0;state.stage_best_score=None
 config=next((item for item in state.effective_curriculum if item['stage']==stage),None)
 if config:state.stage_updates_total=config['updates']
 save_resume_pair(model,opt,sched,tr,norm,state)
def probe_one_k(model,k,loss_fn=None):
 """One formal-grid forward/backward capacity measurement, without an update."""
 if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED_FOR_CAPACITY_PROBE')
 torch.cuda.reset_peak_memory_stats();model.zero_grad(set_to_none=True)
 try:
  if loss_fn is None: raise RuntimeError('CAPACITY_PROBE_INPUT_REQUIRED')
  loss=loss_fn(k);loss.backward();torch.cuda.synchronize()
  peak=torch.cuda.max_memory_allocated()/1024**3;model.zero_grad(set_to_none=True)
  return {'K':k,'status':'SAFE' if peak<=7.2 else 'UNSAFE','peak_vram_gib':peak,'safe':peak<=7.2}
 except torch.cuda.OutOfMemoryError:
  model.zero_grad(set_to_none=True);torch.cuda.empty_cache();return {'K':k,'status':'OOM','peak_vram_gib':None,'safe':False}
def capacity_probe(model,loss_fn=None,probe_fn=probe_one_k):
 out=[]
 for k in (1,2,4,6):
  result=probe_fn(model,k,loss_fn);out.append(result)
  if not result['safe']:break
 return out
def stage_entry_checkpoint(model,opt,sched,tr,norm,state):save_last(model,opt,sched,tr,norm,state)
def _advance(state):
 state.current_stage=next_effective_stage(state,state.current_stage) or PipelineStage.COMPLETE.value;state.stage_step=0
def core_stage_path(state):
 """Pure effective-curriculum transition trace used by the mock state-machine test."""
 trace=[];current=state.current_stage
 while current!=PipelineStage.FREEZE.value:
  trace.append(current);current=next_effective_stage(state,current) or PipelineStage.FREEZE.value
 return trace+[PipelineStage.FREEZE.value]
def _capacity_loss(store,static,tr,norm,device):
 row=store.rows.iloc[0];previous,current,_,params,times,_=store.sample(0,0);tensor=lambda x:torch.from_numpy(np.asarray(x)).unsqueeze(0).to(device);previous,current,params=tensor(previous),tensor(current),tensor(params);active=static[:,1:2]
 def closure(k):
  targets=[tensor(store.frame(row,10*(i+1))) for i in range(k)]
  return rollout_loss(model_ref[0],tr,norm,previous,current,targets,static,params,torch.tensor([times],device=device),active,900.,gradient_checkpointing=True)[0]
 return closure
def _save_subset(rows,train_rows):
 subset=select_fixed_val_subset(rows,8,20260920,train_rows);path=ROOT/'configs/global_operator_v2_val_subset.csv';subset.to_csv(path,index=False);return subset
def _stage_training(model,opt,sched,state,store,val_subset_store,val_subset_rows,static,tr,norm,device,stage):
 assert set(val_subset_rows.scenario_id)==set(val_subset_store.rows.scenario_id)
 config=next(item for item in state.effective_curriculum if item['stage']==stage);state.stage_updates_total=config['updates'];save_resume_pair(model,opt,sched,tr,norm,state)
 active=static[:,1:2]
 def batch(step):
  sample=training_sample_for_step(CFG['seed'],step,len(store.rows),config['k']);row=store.rows.iloc[sample['scenario_index']];previous,current,_,params,times,_=store.sample(sample['scenario_index'],sample['time_s']);tensor=lambda x:torch.from_numpy(np.asarray(x)).unsqueeze(0).to(device)
  return tensor(previous),tensor(current),[tensor(store.frame(row,sample['time_s']+10*(j+1))) for j in range(config['k'])],tensor(params),torch.tensor([times],device=device)
 def loss(batch):return rollout_loss(model,tr,norm,*batch[:2],batch[2],static,batch[3],batch[4],active,900.,gradient_checkpointing=True)[:2]
 def updated(current,details,cadence):
  if current.global_step%250==0:save_resume_pair(model,opt,sched,tr,norm,current)
  if current.global_step%50==0:
   RESULTS.mkdir(parents=True,exist_ok=True);path=RESULTS/'training_log.csv';new=not path.exists();row={'global_step':current.global_step,'stage':stage,'stage_step':current.stage_step,'K':config['k'],'total_loss':float(details['multi']),'state_loss':float(details['state']),'wet_loss':float(details['wet']),'integral_loss':float(details['integral']),'amplitude_guard':float(details['amplitude_guard']),'learning_rate':opt.param_groups[0]['lr'],'step_seconds':float(details.get('step_seconds',float('nan'))),'peak_vram_gib':float(torch.cuda.max_memory_allocated()/1024**3)}
   with path.open('a',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=row.keys());
    if new:writer.writeheader()
    writer.writerow(row)
  if current.global_step%500==0:validate_one_step(model,val_subset_store,tr,norm,device,val_subset_rows,times_s=(120,300,600,900,1200))
  if current.global_step%1000==0:
   records,summary=validate_all_val(model,val_subset_store,tr,norm,device,rows=val_subset_rows,steps=144);score=summary['J_val'];global_best=current.best_val_score
   if is_improvement(score,global_best):
    current.best_val_score=score;current.best_checkpoint_path=str(FORMAL/'best_candidate.pt');save_checkpoint(path=FORMAL/'best_candidate.pt',model=model,optimizer=opt,scheduler=sched,step=current.global_step,transform=tr,config=CFG,normalizer=norm,stage=current.current_stage,best_metric=score,stage_step=current.stage_step,stage_updates_total=current.stage_updates_total,architecture=current.architecture,best_checkpoint_path=current.best_checkpoint_path,stage_best_score=current.stage_best_score);save_state(FORMAL/'run_state.json',current)
   if is_improvement(score,current.stage_best_score):
    current.stage_best_score=score;save_checkpoint(path=FORMAL/f'{stage.lower()}_best.pt',model=model,optimizer=opt,scheduler=sched,step=current.global_step,transform=tr,config=CFG,normalizer=norm,stage=current.current_stage,best_metric=score,stage_step=current.stage_step,stage_updates_total=current.stage_updates_total,architecture=current.architecture,best_checkpoint_path=current.best_checkpoint_path,stage_best_score=current.stage_best_score);save_state(FORMAL/'run_state.json',current)
 train_stage(model,opt,sched,state,config['updates'],batch,loss,on_update=updated)
 save_resume_pair(model,opt,sched,tr,norm,state)
def formal(stop_after_freeze=False):
 state=load_state(FORMAL/'run_state.json',initial_state());model=None
 try:
  rows=scenario_rows('TRAIN');all_val_rows=scenario_rows('VAL');val_subset_rows=_save_subset(all_val_rows,rows);store=FrameStore(rows);val_subset_store=FrameStore(val_subset_rows);all_val_store=FrameStore(all_val_rows);static_np,_=static_and_exogenous(ROOT/'data/downloads/park_v2/inputs/upper30h.npz');static=torch.from_numpy(static_np).unsqueeze(0).to('cuda');dev=torch.device('cuda')
  if state.current_stage==PipelineStage.PRECHECK.value:
   if len(rows)!=160 or len(feature_names())!=35 or not torch.cuda.is_available():raise RuntimeError('FORMAL_PRECHECK_FAILED')
   done(state,PipelineStage.PRECHECK);save_state(FORMAL/'run_state.json',state)
  if state.current_stage==PipelineStage.FIT_NORMALIZERS.value:
   tr,norm=fit_training_transforms(rows,store,static_np);FORMAL.mkdir(parents=True,exist_ok=True);(FORMAL/'transform.json').write_text(json.dumps(tr.to_dict()));(FORMAL/'feature_normalization.json').write_text(json.dumps(norm.to_dict()));done(state,PipelineStage.FIT_NORMALIZERS);save_state(FORMAL/'run_state.json',state)
  else:
   tr=PhysicalTransform.from_dict(json.loads((FORMAL/'transform.json').read_text()));norm=FeatureNormalizer.from_dict(json.loads((FORMAL/'feature_normalization.json').read_text()))
  random.seed(CFG['seed']);np.random.seed(CFG['seed']);torch.manual_seed(CFG['seed']);torch.cuda.manual_seed_all(CFG['seed']);architecture={'width':32,'modes':24,'depth':4};model=JilongGlobalOperatorV2(len(feature_names()),**architecture).to(dev);model_ref[0]=model
  if state.current_stage==PipelineStage.CAPACITY_PROBE.value:
   probes=capacity_probe(model,_capacity_loss(store,static,tr,norm,dev));apply_capacity_result(state,probes);done(state,PipelineStage.CAPACITY_PROBE);save_state(FORMAL/'run_state.json',state)
  opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4);sched=build_scheduler(opt,500,state.total_planned_updates)
  if (FORMAL/'last.pt').exists():ck=restore_checkpoint(FORMAL/'last.pt',model,opt,sched);checkpoint_agreement(state,ck)
  while state.current_stage!=PipelineStage.FREEZE.value:
   if state.current_stage==PipelineStage.ARCHITECTURE.value:
    state.architecture=architecture;_advance(state);save_state(FORMAL/'run_state.json',state);continue
   if state.current_stage in {x['stage'] for x in state.effective_curriculum}:
    next_stage=next_effective_stage(state,state.current_stage);_stage_training(model,opt,sched,state,store,val_subset_store,val_subset_rows,static,tr,norm,dev,state.current_stage);enter_stage(model,opt,sched,tr,norm,state,next_stage);continue
   if state.current_stage==PipelineStage.VAL_CONFIRM.value:
    best=FORMAL/'best_candidate.pt'
    if not best.exists():raise RuntimeError('BEST_CANDIDATE_MISSING')
    restore_checkpoint(best,model,opt,sched);records,summary=validate_all_val(model,all_val_store,tr,norm,dev,rows=all_val_rows,steps=144);RESULTS.mkdir(parents=True,exist_ok=True);__import__('pandas').DataFrame(records).to_csv(RESULTS/'val_full_metrics.csv',index=False);(ROOT/'reports/GLOBAL_V2_VAL_CONFIRMATION.json').write_text(json.dumps(summary,indent=2));state.best_val_score=summary['J_val'];_advance(state);save_resume_pair(model,opt,sched,tr,norm,state);continue
   raise RuntimeError('UNSUPPORTED_CORE_STAGE '+state.current_stage)
  source=FORMAL/'best_candidate.pt';freeze_best_candidate(source,FORMAL/'best.pt',FORMAL/'FREEZE_MANIFEST.json',{'config_hash':state.config_hash,'architecture':state.architecture,'best_val_score':state.best_val_score});done(state,PipelineStage.FREEZE);save_state(FORMAL/'run_state.json',state)
  if stop_after_freeze:return
 except Exception as exc:
  record_failure(ROOT,state,exc,FORMAL/'last.pt');save_state(FORMAL/'run_state.json',state);raise
def main(a):
 if a.dry_run:print(json.dumps({'stage_graph':[x.value for x in CORE],'feature_count':len(feature_names()),'formal_training_started':False},indent=2));return
 if a.plan:print(json.dumps({'planned_stages':[x.value for x in PipelineStage],'architecture':{'width':32,'modes':24,'depth':4},'formal_training_started':False},indent=2));return
 if a.preflight:
  required=[ROOT/'data/downloads/park_v2/inputs/upper30h.npz',ROOT/'data/downloads/park_v2/inputs/upper30h.json',ROOT/'data/downloads/park_v2/inputs/upper30h_transects.json',ROOT/'outputs/park_v2_h0/spinup/final_state.npz']
  scenarios=sorted((ROOT/'outputs/scenarios').glob('JILONG_EI_*'));missing=[str(path) for path in required if not path.exists()]
  for case in scenarios:
   if not all((case/name).exists() for name in ('series.csv','quality.json','final_state.npz')) or len(list((case/'frames').glob('state_*s.npz')))!=145:missing.append(str(case))
  if len(scenarios)!=200:missing.append(f'scenario_directories:{len(scenarios)}/200')
  report={'status':'CODE_PREFLIGHT' if not missing else 'RUNTIME_ARTIFACTS_MISSING','cuda_available':torch.cuda.is_available(),'feature_count':len(feature_names()),'formal_training_started':False,'missing_artifacts':missing};(ROOT/'reports/GLOBAL_V2_FORMAL_PREFLIGHT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));return
 if a.formal:formal(a.stop_after_freeze);return
 raise SystemExit('Use --dry-run or --formal')
if __name__=='__main__':
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--dry-run',action='store_true');g.add_argument('--plan',action='store_true');g.add_argument('--preflight',action='store_true');g.add_argument('--formal',action='store_true');p.add_argument('--stop-after-freeze',action='store_true');main(p.parse_args())
