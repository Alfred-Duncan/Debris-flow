"""Atomic, idempotent orchestration state machine for formal V2 execution."""
from __future__ import annotations
import json,os,tempfile,traceback
from dataclasses import dataclass,asdict,field
from enum import Enum
from pathlib import Path
class PipelineStage(str,Enum):
 PRECHECK='PRECHECK';FIT_NORMALIZERS='FIT_NORMALIZERS';CAPACITY_PROBE='CAPACITY_PROBE';ARCHITECTURE='ARCHITECTURE';STAGE_A='STAGE_A';STAGE_B='STAGE_B';STAGE_C='STAGE_C';STAGE_D='STAGE_D';VAL_CONFIRM='VAL_CONFIRM';FREEZE='FREEZE';LEGACY_TEST='LEGACY_TEST';FINAL_HOLDOUT_GENERATE='FINAL_HOLDOUT_GENERATE';FINAL_HOLDOUT_EVALUATE='FINAL_HOLDOUT_EVALUATE';H0='H0';RUNTIME='RUNTIME';FIGURES='FIGURES';REPORT='REPORT';COMPLETE='COMPLETE'
ORDER=list(PipelineStage)
@dataclass
class PipelineState:
 pipeline_version:str;config_hash:str;architecture:dict|None=None;current_stage:str='PRECHECK';stage_step:int=0;global_step:int=0;stage_updates_total:int=0;best_val_score:float|None=None;best_checkpoint_path:str|None=None;effective_curriculum:list[dict]=field(default_factory=list);max_safe_k:int|None=None;total_planned_updates:int=0;stage_best_score:float|None=None;completed_stages:list[str]=field(default_factory=list);final_holdout_generated:bool=False;final_holdout_unsealed:bool=False;legacy_test_done:bool=False;final_holdout_done:bool=False;h0_done:bool=False;runtime_done:bool=False;final_report_done:bool=False
def atomic_json(path:Path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.tmp');os.close(fd);Path(tmp).write_text(json.dumps(obj,indent=2));os.replace(tmp,path)
def load_state(path:Path,default:PipelineState):return PipelineState(**json.loads(path.read_text())) if path.exists() else default
def save_state(path:Path,state:PipelineState):atomic_json(path,asdict(state))
def done(state:PipelineState,stage:PipelineStage):
 if stage.value not in state.completed_stages:state.completed_stages.append(stage.value)
 i=ORDER.index(stage);state.current_stage=ORDER[min(i+1,len(ORDER)-1)].value;state.stage_step=0
def checkpoint_agreement(state,ck):
 fields=('stage_name','stage_step','global_step','stage_updates_total','config_hash','architecture')
 bad={k:(getattr(state,'current_stage' if k=='stage_name' else k,None),ck.get(k)) for k in fields if (getattr(state,'current_stage' if k=='stage_name' else k,None)!=ck.get(k))}
 if bad:raise RuntimeError(f'RUN_STATE_CHECKPOINT_MISMATCH {bad}')
def record_failure(root:Path,state,exc,last=None):atomic_json(root/'reports/GLOBAL_V2_LAST_FAILURE.json',{'stage':state.current_stage,'exception_type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'resume_possible':True,'last_checkpoint':str(last) if last else None})
def shutdown_allowed(root:Path,state):
 needed=[root/'models/global_operator_v2/best.pt',root/'reports/GLOBAL_OPERATOR_V2.json',root/'results/global_operator_v2/final_holdout_evaluation.lock.json',root/'reports/GLOBAL_V2_H0.json',root/'results/global_operator_v2/runtime_benchmark.csv']
 return state.current_stage=='COMPLETE' and state.final_report_done and state.final_holdout_done and state.h0_done and state.runtime_done and all(p.exists() for p in needed)
def capacity_schedule(probe):
 safe=[int(x['K']) for x in probe if x.get('status','SAFE')=='SAFE' and x.get('safe',False)]
 m=max(safe,default=0)
 if m<1:raise RuntimeError('NO_SAFE_CAPACITY')
 stages=[('STAGE_A',1,2000),('STAGE_B',2,2000),('STAGE_C',4,4000),('STAGE_D',6,6000)]
 return [{'stage':stage,'k':k,'updates':updates} for stage,k,updates in stages if m>=k]
def apply_capacity_result(state,probe):
 curriculum=capacity_schedule(probe)
 safe=[int(x['K']) for x in probe if x.get('status','SAFE')=='SAFE' and x.get('safe',False)]
 state.effective_curriculum=curriculum
 state.max_safe_k=max(safe)
 state.total_planned_updates=sum(item['updates'] for item in curriculum)
 return state
def effective_stage_names(state):
 return [item['stage'] for item in state.effective_curriculum]
def stage_config(state,stage_name):
 stage_name=stage_name.value if isinstance(stage_name,PipelineStage) else stage_name
 return next((item for item in state.effective_curriculum if item['stage']==stage_name),None)
def next_effective_stage(state,current_stage):
 current_stage=current_stage.value if isinstance(current_stage,PipelineStage) else current_stage
 stages=effective_stage_names(state)
 if current_stage=='ARCHITECTURE':return stages[0] if stages else 'VAL_CONFIRM'
 if current_stage in stages:
  index=stages.index(current_stage)
  return stages[index+1] if index+1<len(stages) else 'VAL_CONFIRM'
 if current_stage=='VAL_CONFIRM':return 'FREEZE'
 return None
