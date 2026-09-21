"""Single-state-machine V2 training-core launcher; --formal is deliberately not invoked in code review."""
from __future__ import annotations
import argparse,csv,gc,hashlib,json,math,random,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.pipeline import PipelineState,PipelineStage,load_state,save_state,done,checkpoint_agreement,record_failure,apply_capacity_result,next_effective_stage,build_scheduler,training_sample_for_step
from src.global_operator_v2.dataset import FrameStore,scenario_rows,feature_names,static_tensor
from src.global_operator_v2.frame_adapter import static_and_exogenous
from src.global_operator_v2.model import JilongGlobalOperatorV2
from src.global_operator_v2.trainer import restore_checkpoint,save_checkpoint,rollout_loss,train_stage,is_improvement
from src.global_operator_v2.validation import select_fixed_val_subset,validate_one_step,validate_all_val,validate_horizon_ladder,validate_persistence_baseline
from src.global_operator_v2.evaluation import freeze_best_candidate
from src.global_operator_v2.transforms import PhysicalTransform,FeatureNormalizer,DeltaNormalization,fit_training_transforms,fit_delta_normalization
CFG=json.loads((ROOT/'configs/GLOBAL_OPERATOR_V2.json').read_text());FORMAL=ROOT/'models/global_operator_v2';RESULTS=ROOT/'results/global_operator_v2'
model_ref=[None]
CORE=(PipelineStage.PRECHECK,PipelineStage.FIT_NORMALIZERS,PipelineStage.CAPACITY_PROBE,PipelineStage.ARCHITECTURE,PipelineStage.STAGE_A,PipelineStage.STAGE_B,PipelineStage.STAGE_C,PipelineStage.STAGE_D,PipelineStage.VAL_CONFIRM,PipelineStage.FREEZE)
def cfg_hash():return hashlib.sha256(json.dumps(CFG,sort_keys=True).encode()).hexdigest()
def initial_state():return PipelineState(CFG['pipeline_version'],cfg_hash())
def safe_append_csv(path,row):
 """Telemetry cannot be allowed to terminate an optimizer on Windows locks."""
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);error=None
 for _ in range(4):
  try:
   new=not path.exists()
   with path.open('a',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=row.keys())
    if new:writer.writeheader()
    writer.writerow(row)
   return True
  except PermissionError as exc:
   error=exc;time.sleep(.2)
 fallback=path.with_name(f'{path.stem}_fallback_{int(time.time())}.jsonl')
 with fallback.open('a',encoding='utf-8') as handle:handle.write(json.dumps({'warning':'TELEMETRY_PERMISSION_FALLBACK','target':str(path),'error':str(error),'row':row},default=str)+'\n')
 print(f'WARNING TELEMETRY_PERMISSION_FALLBACK {path}',flush=True)
 return False
def sample_for_global_step(seed,step,n,max_t):
 return training_sample_for_step(seed,step,n,1,n_time_states=max_t+1)
def save_last(model,opt,sched,tr,norm,state):
 save_checkpoint(path=FORMAL/'last.pt',model=model,optimizer=opt,scheduler=sched,step=state.global_step,transform=tr,config=CFG,normalizer=norm,stage=state.current_stage,best_metric=state.best_val_score,stage_step=state.stage_step,stage_updates_total=state.stage_updates_total,architecture=state.architecture,best_checkpoint_path=state.best_checkpoint_path,stage_best_score=state.stage_best_score)
def save_resume_pair(model,opt,sched,tr,norm,state):
 save_last(model,opt,sched,tr,norm,state);save_state(FORMAL/'run_state.json',state)
def rollback_state_to_last_checkpoint(state):
 """Keep run_state resumable if an exception lands between checkpoint cadences."""
 path=FORMAL/'last.pt'
 if not path.exists():return state
 checkpoint=torch.load(path,map_location='cpu',weights_only=False)
 state.current_stage=checkpoint['stage_name'];state.stage_step=checkpoint['stage_step'];state.global_step=checkpoint['global_step'];state.stage_updates_total=checkpoint['stage_updates_total'];state.architecture=checkpoint['architecture']
 state.best_val_score=checkpoint.get('best_val_score',checkpoint.get('best_metric'));state.best_checkpoint_path=checkpoint.get('best_checkpoint_path');state.stage_best_score=checkpoint.get('stage_best_score')
 checkpoint_agreement(state,checkpoint)
 return state
def enter_stage(model,opt,sched,tr,norm,state,stage):
 state.current_stage=stage;state.stage_step=0;state.stage_best_score=None
 config=next((item for item in state.effective_curriculum if item['stage']==stage),None)
 if config:state.stage_updates_total=config['updates']
 save_resume_pair(model,opt,sched,tr,norm,state)
def stage_completion_is_valid(state,stage,history_path=None,checkpoint_path=None):
 """A fully checkpointed stage is resumably complete only after finite validation."""
 stage=stage.value if isinstance(stage,PipelineStage) else stage
 if state.current_stage!=stage or state.stage_updates_total<1 or state.stage_step<state.stage_updates_total:return False
 history_path=Path(history_path or RESULTS/'validation_history.csv');checkpoint_path=Path(checkpoint_path or FORMAL/'last.pt')
 if not stage_has_finite_candidate(state,stage,history_path) or not checkpoint_path.exists():return False
 checkpoint_agreement(state,torch.load(checkpoint_path,map_location='cpu',weights_only=False))
 return True
def stage_has_finite_candidate(state,stage,history_path=None):
 stage=stage.value if isinstance(stage,PipelineStage) else stage;history_path=Path(history_path or RESULTS/'validation_history.csv')
 if not history_path.exists():return False
 with history_path.open(newline='') as handle:
  return any(row.get('stage')==stage and row.get('validation_status')=='FINITE' and int(row.get('global_step',-1))<=state.global_step for row in csv.DictReader(handle))
def advance_completed_stage(model,opt,sched,tr,norm,state):
 """Enter the next effective stage without replaying completed optimizer updates."""
 stage=state.current_stage
 if not stage_completion_is_valid(state,stage):return False
 next_stage=next_effective_stage(state,stage)
 if next_stage is None:raise RuntimeError('COMPLETED_STAGE_HAS_NO_SUCCESSOR '+stage)
 enter_stage(model,opt,sched,tr,norm,state,next_stage)
 return True
def probe_one_k(model,k,loss_fn=None):
 """One formal-grid forward/backward capacity measurement, without an update."""
 if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED_FOR_CAPACITY_PROBE')
 torch.cuda.reset_peak_memory_stats();model.zero_grad(set_to_none=True)
 try:
  if loss_fn is None: raise RuntimeError('CAPACITY_PROBE_INPUT_REQUIRED')
  loss=loss_fn(k);loss.backward();torch.cuda.synchronize()
  peak=torch.cuda.max_memory_allocated()/1024**3;model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
  return {'K':k,'status':'SAFE' if peak<=7.2 else 'UNSAFE','peak_vram_gib':peak,'safe':peak<=7.2}
 except torch.cuda.OutOfMemoryError:
  model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();return {'K':k,'status':'OOM','peak_vram_gib':None,'safe':False}
def capacity_probe(model,loss_fn=None,probe_fn=probe_one_k):
 before={} if model is None else {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
 out=[]
 for k in (1,2,4,6):
  result=probe_fn(model,k,loss_fn);out.append(result)
  if model is not None:model.zero_grad(set_to_none=True)
  gc.collect();torch.cuda.empty_cache()
  if not result['safe']:break
 if model is not None and any(not torch.equal(before[name],value.detach().cpu()) for name,value in model.state_dict().items()):raise RuntimeError('CAPACITY_PROBE_MUTATED_MODEL')
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
def _capacity_loss(store,static,tr,norm,delta_norm,device):
 row=store.rows.iloc[0];previous,current,_,params,times,_=store.sample(0,0);tensor=lambda x:torch.from_numpy(np.asarray(x)).unsqueeze(0).to(device);previous,current,params=tensor(previous),tensor(current),tensor(params);active=static[:,1:2]
 def closure(k):
  targets=[tensor(store.frame(row,10*(i+1))) for i in range(k)]
  return rollout_loss(model_ref[0],tr,norm,previous,current,targets,static,params,torch.tensor([times],device=device),active,900.,gradient_checkpointing=True,delta_normalization=delta_norm,weighting={**CFG['stabilization']['teacher_weighting'],'change_threshold':delta_norm.change_threshold})[0]
 return closure
def _save_subset(rows,train_rows):
 subset=select_fixed_val_subset(rows,8,20260920,train_rows);path=ROOT/'configs/global_operator_v2_val_subset.csv';subset.to_csv(path,index=False);return subset
def _record_validation(state,summary,is_global_best,is_stage_best):
 RESULTS.mkdir(parents=True,exist_ok=True);row={'global_step':state.global_step,'stage':state.current_stage,'trajectory_h_rel_l2':summary.get('trajectory_h_rel_l2'),'trajectory_momentum_rel_l2':summary.get('trajectory_momentum_rel_l2'),'mean_wet_iou':summary.get('mean_wet_iou'),'debris_front_mae_km':summary.get('debris_front_mae_km'),'mixture_volume_relative_error':summary.get('mixture_volume_relative_error'),'J_val':summary.get('J_val'),'validation_status':summary.get('validation_status'),'first_nonfinite_step':summary.get('first_nonfinite_step'),'is_global_best':is_global_best,'is_stage_best':is_stage_best}
 safe_append_csv(RESULTS/'validation_history.csv',row)
def _record_horizons(state,stage,rows):
 for row in rows:safe_append_csv(RESULTS/'horizon_history.csv',{'global_step':state.global_step,'stage':stage,**row})
def finalize_periodic_validation(state,stage,horizons,summary,is_global_best,is_stage_best,worse=None):
 """Persist periodic evidence; only an empty finite-candidate stage is terminal."""
 _record_horizons(state,stage,horizons);_record_validation(state,summary,is_global_best,is_stage_best)
 if worse is not None:
  better=5-worse;warning=worse>=3
  safe_append_csv(RESULTS/'persistence_comparison_history.csv',{'global_step':state.global_step,'stage':stage,'primary_metrics_better_than_persistence':better,'persistence_baseline_warning':warning})
  if warning:print(f'WARNING PERSISTENCE_BASELINE_WARNING primary_metrics_better_than_persistence={better}/5',flush=True)
 return bool(summary.get('finite_rollout'))
def ensure_stage_has_finite_candidate(stage_has_finite):
 if not stage_has_finite:raise RuntimeError('NO_FINITE_VALIDATION_CANDIDATE')
def _stage_training(model,opt,sched,state,store,val_subset_store,val_subset_rows,static,tr,norm,delta_norm,device,stage,persistence_summary):
 assert set(val_subset_rows.scenario_id)==set(val_subset_store.rows.scenario_id)
 config=next(item for item in state.effective_curriculum if item['stage']==stage);state.stage_updates_total=config['updates'];save_resume_pair(model,opt,sched,tr,norm,state);active=static[:,1:2];stage_has_finite=[stage_has_finite_candidate(state,stage)]
 burn_choices=CFG['stabilization']['burn_in'][stage]
 def batch(step):
  sample=training_sample_for_step(CFG['seed'],step,len(store.rows),config['k'],burn_in_choices=burn_choices,sampler=CFG['stabilization']['sampler']);row=store.rows.iloc[sample['scenario_index']];t=sample['time_s'];burn=sample['burn_in_steps'];previous,current,_,params,times,_=store.sample(sample['scenario_index'],t);tensor=lambda x:torch.from_numpy(np.asarray(x)).unsqueeze(0).to(device)
  teacher_current=tensor(store.frame(row,t+10*burn));targets=[tensor(store.frame(row,t+10*(burn+j+1))) for j in range(config['k'])];burn_targets=[tensor(store.frame(row,t+10*(j+1))) for j in range(burn)]
  return tensor(previous),tensor(current),targets,tensor(params),torch.tensor([times],device=device),burn,burn_targets,teacher_current,sample
 def loss(batch):
  previous,current,targets,params,times,burn,burn_targets,teacher_current,_=batch;return rollout_loss(model,tr,norm,previous,current,targets,static,params,times,active,900.,gradient_checkpointing=True,burn_in_steps=burn,burn_targets=burn_targets,teacher_current=teacher_current,delta_normalization=delta_norm,weighting={**CFG['stabilization']['teacher_weighting'],'change_threshold':delta_norm.change_threshold})[:2]
 def updated(current,details,cadence):
  if current.global_step%250==0:save_resume_pair(model,opt,sched,tr,norm,current)
  if current.global_step%50==0:
   latest=training_sample_for_step(CFG['seed'],current.global_step-1,len(store.rows),config['k'],burn_in_choices=burn_choices,sampler=CFG['stabilization']['sampler']);value=lambda key:details[key].detach().cpu().item();delta=value('delta');state_loss_value=value('state');wet_value=value('wet');integral_value=value('integral');row={'global_step':current.global_step,'stage':stage,'stage_step':current.stage_step,'K':config['k'],'burn_in_steps':latest['burn_in_steps'],'seed_time_s':latest['seed_time_s'],'gradient_start_time_s':latest['gradient_start_time_s'],'sampler_stratum':latest['sampler_stratum'],'total_loss':value('total_loss'),'multi_loss':value('multi'),'one_step_anchor':value('one_step_anchor'),'delta_loss':delta,'delta_h':value('delta_h'),'delta_hu':value('delta_hu'),'delta_hv':value('delta_hv'),'delta_c':value('delta_c'),'delta_ice':value('delta_ice'),'delta_dz':value('delta_dz'),'weighted_delta_contribution':.5*delta,'weighted_state_contribution':state_loss_value,'weighted_wet_contribution':.1*wet_value,'weighted_integral_contribution':.05*integral_value,'state_loss':state_loss_value,'wet_loss':wet_value,'integral_loss':integral_value,'amplitude_guard':value('amplitude_guard'),'max_channel_saturation_fraction':value('max_channel_saturation_fraction') if 'max_channel_saturation_fraction' in details else 0.,'learning_rate':opt.param_groups[0]['lr'],'step_seconds':float(details.get('step_seconds',float('nan'))),'peak_vram_gib':float(torch.cuda.max_memory_allocated()/1024**3)}
   for name in ('h','hu','hv','c','ice','dz'):row[f'raw_delta_over_bound_rms_{name}']=value(f'raw_delta_over_bound_rms_{name}') if f'raw_delta_over_bound_rms_{name}' in details else 0.;row[f'saturation_fraction_{name}']=value(f'saturation_fraction_{name}') if f'saturation_fraction_{name}' in details else 0.
   if row['weighted_delta_contribution']>100*(row['weighted_state_contribution']+row['weighted_wet_contribution']+row['weighted_integral_contribution']):print('WARNING DELTA_OBJECTIVE_DOMINANCE_WARNING',flush=True)
   if row['max_channel_saturation_fraction']>.2:print('WARNING DELTA_SATURATION_WARNING',flush=True)
   safe_append_csv(RESULTS/'training_log.csv',row)
  if current.global_step%500==0:validate_one_step(model,val_subset_store,tr,norm,device,val_subset_rows,times_s=(120,300,600,900,1200))
  if current.global_step%1000==0:
   records,summary=validate_all_val(model,val_subset_store,tr,norm,device,rows=val_subset_rows,steps=144);horizons=validate_horizon_ladder(model,val_subset_store,tr,norm,device,val_subset_rows)
   if not summary.get('finite_rollout'):
    finalize_periodic_validation(current,stage,horizons,summary,False,False)
    return
   stage_has_finite[0]=True;score=summary['J_val'];global_best=current.best_val_score;global_improved=is_improvement(score,global_best);stage_improved=is_improvement(score,current.stage_best_score)
   primary=('trajectory_h_rel_l2','trajectory_momentum_rel_l2','mean_wet_iou','mixture_volume_relative_error','debris_front_mae_km');worse=sum((summary[k]>=persistence_summary[k] if k!='mean_wet_iou' else summary[k]<=persistence_summary[k]) for k in primary)
   summary['primary_metrics_better_than_persistence']=len(primary)-worse
   if global_improved:
    current.best_val_score=score;current.best_checkpoint_path=str(FORMAL/'best_candidate.pt');save_checkpoint(path=FORMAL/'best_candidate.pt',model=model,optimizer=opt,scheduler=sched,step=current.global_step,transform=tr,config=CFG,normalizer=norm,stage=current.current_stage,best_metric=score,stage_step=current.stage_step,stage_updates_total=current.stage_updates_total,architecture=current.architecture,best_checkpoint_path=current.best_checkpoint_path,stage_best_score=current.stage_best_score);save_state(FORMAL/'run_state.json',current)
   if stage_improved:
    current.stage_best_score=score;save_checkpoint(path=FORMAL/f'{stage.lower()}_best.pt',model=model,optimizer=opt,scheduler=sched,step=current.global_step,transform=tr,config=CFG,normalizer=norm,stage=current.current_stage,best_metric=score,stage_step=current.stage_step,stage_updates_total=current.stage_updates_total,architecture=current.architecture,best_checkpoint_path=current.best_checkpoint_path,stage_best_score=current.stage_best_score);save_state(FORMAL/'run_state.json',current)
   finalize_periodic_validation(current,stage,horizons,summary,global_improved,stage_improved,worse)
 train_stage(model,opt,sched,state,config['updates'],batch,loss,on_update=updated)
 ensure_stage_has_finite_candidate(stage_has_finite[0])
 save_resume_pair(model,opt,sched,tr,norm,state)
def formal(stop_after_freeze=False):
 state=load_state(FORMAL/'run_state.json',initial_state());model=None
 try:
  rows=scenario_rows('TRAIN');all_val_rows=scenario_rows('VAL');val_subset_rows=_save_subset(all_val_rows,rows);store=FrameStore(rows);val_subset_store=FrameStore(val_subset_rows);all_val_store=FrameStore(all_val_rows);static_np,_=static_and_exogenous(ROOT/'data/downloads/park_v2/inputs/upper30h.npz');static=torch.from_numpy(static_np).unsqueeze(0).to('cuda');dev=torch.device('cuda')
  if state.current_stage==PipelineStage.PRECHECK.value:
   if len(rows)!=160 or len(feature_names())!=35 or not torch.cuda.is_available():raise RuntimeError('FORMAL_PRECHECK_FAILED')
   done(state,PipelineStage.PRECHECK);save_state(FORMAL/'run_state.json',state)
  if state.current_stage==PipelineStage.FIT_NORMALIZERS.value:
   tr,norm=fit_training_transforms(rows,store,static_np);delta_norm=fit_delta_normalization(rows,store,static_np,tr,CFG['seed'],**CFG['stabilization']['delta_normalization'],bound_quantile=CFG['stabilization']['delta_bound']['quantile'],safety_factor=CFG['stabilization']['delta_bound']['safety_factor'],change_quantile=CFG['stabilization']['teacher_weighting']['change_quantile']);
   if delta_norm.coverage<CFG['stabilization']['delta_bound']['minimum_coverage'] or any(item['coverage_all_samples']<CFG['stabilization']['delta_bound']['minimum_coverage'] for item in delta_norm.per_channel.values()):raise RuntimeError('DELTA_BOUND_COVERAGE_INSUFFICIENT')
   FORMAL.mkdir(parents=True,exist_ok=True);(FORMAL/'transform.json').write_text(json.dumps(tr.to_dict()));(FORMAL/'feature_normalization.json').write_text(json.dumps(norm.to_dict()));(FORMAL/'delta_normalization.json').write_text(json.dumps(delta_norm.to_dict(),indent=2));done(state,PipelineStage.FIT_NORMALIZERS);save_state(FORMAL/'run_state.json',state)
  else:
   tr=PhysicalTransform.from_dict(json.loads((FORMAL/'transform.json').read_text()));norm=FeatureNormalizer.from_dict(json.loads((FORMAL/'feature_normalization.json').read_text()));delta_norm=DeltaNormalization.from_dict(json.loads((FORMAL/'delta_normalization.json').read_text()))
  random.seed(CFG['seed']);np.random.seed(CFG['seed']);torch.manual_seed(CFG['seed']);torch.cuda.manual_seed_all(CFG['seed']);architecture={'width':32,'modes':24,'depth':4};model=JilongGlobalOperatorV2(len(feature_names()),**architecture,delta_bounds=delta_norm.bounds).to(dev);model.delta_normalization=delta_norm.to_dict();model_ref[0]=model
  if state.current_stage==PipelineStage.CAPACITY_PROBE.value:
   probes=capacity_probe(model,_capacity_loss(store,static,tr,norm,delta_norm,dev));apply_capacity_result(state,probes);done(state,PipelineStage.CAPACITY_PROBE);save_state(FORMAL/'run_state.json',state)
  if not (RESULTS/'persistence_baseline.csv').exists():
   baseline_records,persistence_summary=validate_persistence_baseline(val_subset_store,tr,norm,dev,val_subset_rows,144);RESULTS.mkdir(parents=True,exist_ok=True);__import__('pandas').DataFrame(baseline_records).to_csv(RESULTS/'persistence_baseline.csv',index=False)
  else:
   _,persistence_summary=validate_persistence_baseline(val_subset_store,tr,norm,dev,val_subset_rows,144)
  torch.cuda.reset_peak_memory_stats();opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4);sched=build_scheduler(opt,500,state.total_planned_updates)
  if (FORMAL/'last.pt').exists():ck=restore_checkpoint(FORMAL/'last.pt',model,opt,sched);checkpoint_agreement(state,ck)
  while state.current_stage!=PipelineStage.FREEZE.value:
   if state.current_stage==PipelineStage.ARCHITECTURE.value:
    state.architecture=architecture;_advance(state);save_state(FORMAL/'run_state.json',state);continue
   if state.current_stage in {x['stage'] for x in state.effective_curriculum}:
    if advance_completed_stage(model,opt,sched,tr,norm,state):continue
    next_stage=next_effective_stage(state,state.current_stage);_stage_training(model,opt,sched,state,store,val_subset_store,val_subset_rows,static,tr,norm,delta_norm,dev,state.current_stage,persistence_summary);enter_stage(model,opt,sched,tr,norm,state,next_stage);continue
   if state.current_stage==PipelineStage.VAL_CONFIRM.value:
    best=FORMAL/'best_candidate.pt'
    if not best.exists():raise RuntimeError('BEST_CANDIDATE_MISSING')
    restore_checkpoint(best,model,opt,sched);records,summary=validate_all_val(model,all_val_store,tr,norm,dev,rows=all_val_rows,steps=144)
    if not summary.get('finite_rollout'):raise RuntimeError('NO_FINITE_VALIDATION_CANDIDATE')
    RESULTS.mkdir(parents=True,exist_ok=True);__import__('pandas').DataFrame(records).to_csv(RESULTS/'val_full_metrics.csv',index=False);(ROOT/'reports/GLOBAL_V2_VAL_CONFIRMATION.json').write_text(json.dumps(summary,indent=2));state.best_val_score=summary['J_val'];_advance(state);save_resume_pair(model,opt,sched,tr,norm,state);continue
   raise RuntimeError('UNSUPPORTED_CORE_STAGE '+state.current_stage)
  source=FORMAL/'best_candidate.pt';freeze_best_candidate(source,FORMAL/'best.pt',FORMAL/'FREEZE_MANIFEST.json',{'config_hash':state.config_hash,'architecture':state.architecture,'best_val_score':state.best_val_score});done(state,PipelineStage.FREEZE);save_state(FORMAL/'run_state.json',state)
  if stop_after_freeze:return
 except Exception as exc:
  record_failure(ROOT,state,exc,FORMAL/'last.pt');rollback_state_to_last_checkpoint(state);save_state(FORMAL/'run_state.json',state);raise
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
