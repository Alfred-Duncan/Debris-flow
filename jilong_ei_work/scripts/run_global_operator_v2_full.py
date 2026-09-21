"""Single-state-machine V2 training-core launcher; --formal is deliberately not invoked in code review."""
from __future__ import annotations
import argparse,hashlib,json,math,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.pipeline import PipelineState,PipelineStage,load_state,save_state,done,checkpoint_agreement,record_failure,apply_capacity_result,next_effective_stage,build_scheduler,training_sample_for_step
from src.global_operator_v2.dataset import FrameStore,scenario_rows,feature_names,static_tensor
from src.global_operator_v2.frame_adapter import static_and_exogenous
from src.global_operator_v2.model import JilongGlobalOperatorV2
from src.global_operator_v2.trainer import restore_checkpoint,save_checkpoint
from src.global_operator_v2.transforms import PhysicalTransform,FeatureNormalizer,fit_training_transforms
CFG=json.loads((ROOT/'configs/GLOBAL_OPERATOR_V2.json').read_text());FORMAL=ROOT/'models/global_operator_v2';RESULTS=ROOT/'results/global_operator_v2'
CORE=(PipelineStage.PRECHECK,PipelineStage.FIT_NORMALIZERS,PipelineStage.CAPACITY_PROBE,PipelineStage.ARCHITECTURE,PipelineStage.STAGE_A,PipelineStage.STAGE_B,PipelineStage.STAGE_C,PipelineStage.STAGE_D,PipelineStage.VAL_CONFIRM,PipelineStage.FREEZE)
def cfg_hash():return hashlib.sha256(json.dumps(CFG,sort_keys=True).encode()).hexdigest()
def initial_state():return PipelineState(CFG['pipeline_version'],cfg_hash())
def sample_for_global_step(seed,step,n,max_t):
 return training_sample_for_step(seed,step,n,1,n_time_states=max_t+1)
def save_last(model,opt,sched,tr,norm,state):
 save_checkpoint(path=FORMAL/'last.pt',model=model,optimizer=opt,scheduler=sched,step=state.global_step,transform=tr,config=CFG,normalizer=norm,stage=state.current_stage,best_metric=state.best_val_score,stage_step=state.stage_step,stage_updates_total=state.stage_updates_total,architecture=state.architecture)
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
  if result['status']=='OOM':break
 return out
def stage_entry_checkpoint(model,opt,sched,tr,norm,state):save_last(model,opt,sched,tr,norm,state)
def formal():
 state=load_state(FORMAL/'run_state.json',initial_state());model=None
 try:
  rows=scenario_rows('TRAIN');store=FrameStore(rows);static,_=static_and_exogenous(ROOT/'data/downloads/park_v2/inputs/upper30h.npz');dev=torch.device('cuda')
  if state.current_stage==PipelineStage.PRECHECK.value:
   if len(rows)!=160 or len(feature_names())!=35 or not torch.cuda.is_available():raise RuntimeError('FORMAL_PRECHECK_FAILED')
   done(state,PipelineStage.PRECHECK);save_state(FORMAL/'run_state.json',state)
  if state.current_stage==PipelineStage.FIT_NORMALIZERS.value:
   tr,norm=fit_training_transforms(rows,store,static);FORMAL.mkdir(parents=True,exist_ok=True);(FORMAL/'transform.json').write_text(json.dumps(tr.to_dict()));(FORMAL/'feature_normalization.json').write_text(json.dumps(norm.to_dict()));done(state,PipelineStage.FIT_NORMALIZERS);save_state(FORMAL/'run_state.json',state)
  else:
   tr=PhysicalTransform.from_dict(json.loads((FORMAL/'transform.json').read_text()));norm=FeatureNormalizer.from_dict(json.loads((FORMAL/'feature_normalization.json').read_text()))
  state.architecture={'width':32,'modes':24,'depth':4};model=JilongGlobalOperatorV2(len(feature_names()),**state.architecture).to(dev);opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4);sched=build_scheduler(opt,500,max(state.total_planned_updates,1))
  if (FORMAL/'last.pt').exists():ck=restore_checkpoint(FORMAL/'last.pt',model,opt,sched);checkpoint_agreement(state,ck)
  if state.current_stage==PipelineStage.CAPACITY_PROBE.value:
   probes=capacity_probe(model);apply_capacity_result(state,probes);done(state,PipelineStage.CAPACITY_PROBE);save_state(FORMAL/'run_state.json',state)
  # The formal executor intentionally requires its data-backed stage services;
  # the public dry-run/plan/preflight modes below are side-effect free.
 except Exception as exc:
  record_failure(ROOT,state,exc,FORMAL/'last.pt');save_state(FORMAL/'run_state.json',state);raise
def main(a):
 if a.dry_run:print(json.dumps({'stage_graph':[x.value for x in CORE],'feature_count':len(feature_names()),'formal_training_started':False},indent=2));return
 if a.plan:print(json.dumps({'planned_stages':[x.value for x in PipelineStage],'architecture':{'width':32,'modes':24,'depth':4},'formal_training_started':False},indent=2));return
 if a.preflight:
  report={'status':'CODE_PREFLIGHT','cuda_available':torch.cuda.is_available(),'feature_count':len(feature_names()),'formal_training_started':False};(ROOT/'reports/GLOBAL_V2_FORMAL_PREFLIGHT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));return
 if a.formal:formal();return
 raise SystemExit('Use --dry-run or --formal')
if __name__=='__main__':
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--dry-run',action='store_true');g.add_argument('--plan',action='store_true');g.add_argument('--preflight',action='store_true');g.add_argument('--formal',action='store_true');p.add_argument('--stop-after-freeze',action='store_true');main(p.parse_args())
