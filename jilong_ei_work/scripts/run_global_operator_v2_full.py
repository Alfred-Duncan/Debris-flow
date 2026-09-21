"""Single-state-machine V2 training-core launcher; --formal is deliberately not invoked in code review."""
from __future__ import annotations
import argparse,hashlib,json,math,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.pipeline import PipelineState,PipelineStage,load_state,save_state,done,checkpoint_agreement,record_failure,capacity_schedule
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
 r=np.random.default_rng(np.random.SeedSequence([seed,step]));return int(r.integers(n)),int(r.integers(max_t))
def save_last(model,opt,sched,tr,norm,state):
 save_checkpoint(path=FORMAL/'last.pt',model=model,optimizer=opt,scheduler=sched,step=state.global_step,transform=tr,config=CFG,normalizer=norm,stage=state.current_stage,best_metric=state.best_val_score,stage_step=state.stage_step,stage_updates_total=state.stage_updates_total,architecture=state.architecture)
def capacity_probe(model,*args):
 out=[]
 for k in (1,2,4,6):
  try:
   # Full-grid actual forward/backward is executed only from approved --formal.
   raise NotImplementedError('capacity operation injected by approved formal executor')
  except torch.cuda.OutOfMemoryError: out.append({'K':k,'status':'OOM','safe':False});model.zero_grad(set_to_none=True);torch.cuda.empty_cache();break
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
  state.architecture={'width':32,'modes':24,'depth':4};model=JilongGlobalOperatorV2(len(feature_names()),**state.architecture).to(dev);opt=torch.optim.AdamW(model.parameters(),lr=5e-4);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
  if (FORMAL/'last.pt').exists():ck=restore_checkpoint(FORMAL/'last.pt',model,opt,sched);checkpoint_agreement(state,ck)
  if state.current_stage==PipelineStage.CAPACITY_PROBE.value:
   probes=capacity_probe(model);state.effective_curriculum=[(x.value,k,u) for x,k,u in capacity_schedule(probes)];done(state,PipelineStage.CAPACITY_PROBE);save_state(FORMAL/'run_state.json',state)
  # Training handlers are intentionally reached only in approved formal execution.
  for stage in (PipelineStage.ARCHITECTURE,PipelineStage.STAGE_A,PipelineStage.STAGE_B,PipelineStage.STAGE_C,PipelineStage.STAGE_D,PipelineStage.VAL_CONFIRM,PipelineStage.FREEZE):
   if state.current_stage==stage.value: raise RuntimeError(f'FORMAL_HANDLER_REQUIRED:{stage.value}')
 except Exception as exc:
  record_failure(ROOT,state,exc,FORMAL/'last.pt');save_state(FORMAL/'run_state.json',state);raise
def main(a):
 if a.dry_run:print(json.dumps({'stage_graph':[x.value for x in CORE],'feature_count':len(feature_names()),'formal_training_started':False},indent=2));return
 if a.formal:formal();return
 raise SystemExit('Use --dry-run or --formal')
if __name__=='__main__':
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--dry-run',action='store_true');g.add_argument('--formal',action='store_true');main(p.parse_args())
