"""Regression coverage for completed-stage resume and non-terminal diagnostics."""
from __future__ import annotations
import csv,importlib.util,sys,tempfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.pipeline import PipelineState,checkpoint_agreement
from src.global_operator_v2.transforms import FeatureNormalizer,PhysicalTransform

def load_runner():
 spec=importlib.util.spec_from_file_location('runner_resume_quality',ROOT/'scripts/run_global_operator_v2_full.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def test_completed_stage_resume(runner):
 with tempfile.TemporaryDirectory() as directory:
  root=Path(directory);old_formal,old_results=runner.FORMAL,runner.RESULTS;runner.FORMAL=root/'models';runner.RESULTS=root/'results'
  runner.RESULTS.mkdir(parents=True);state=PipelineState('v',runner.cfg_hash(),current_stage='STAGE_A',stage_step=2000,global_step=2000,stage_updates_total=2000,effective_curriculum=[{'stage':'STAGE_A','k':1,'updates':2000},{'stage':'STAGE_B','k':2,'updates':2000}])
  with (runner.RESULTS/'validation_history.csv').open('w',newline='') as handle:csv.DictWriter(handle,fieldnames=('stage','global_step','validation_status')).writeheader();csv.DictWriter(handle,fieldnames=('stage','global_step','validation_status')).writerow({'stage':'STAGE_A','global_step':2000,'validation_status':'FINITE'})
  model=torch.nn.Linear(1,1);opt=torch.optim.AdamW(model.parameters(),lr=.01);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.);tr=PhysicalTransform(1,1,1,.05);norm=FeatureNormalizer(0,1,1,1,[0]*5,[1]*5)
  runner.save_resume_pair(model,opt,sched,tr,norm,state)
  assert runner.advance_completed_stage(model,opt,sched,tr,norm,state) is True
  assert (state.current_stage,state.stage_step,state.global_step)==('STAGE_B',0,2000)
  checkpoint_agreement(state,torch.load(runner.FORMAL/'last.pt',weights_only=False));runner.FORMAL,runner.RESULTS=old_formal,old_results

def test_persistence_warning_and_periodic_nonfinite(runner):
 events=[];runner._record_horizons=lambda *args:events.append('horizons');runner._record_validation=lambda *args:events.append('validation');runner.safe_append_csv=lambda path,row:events.append(row)
 state=PipelineState('v','h',current_stage='STAGE_A',global_step=2000)
 assert runner.finalize_periodic_validation(state,'STAGE_A',[{'horizon':1}],{'finite_rollout':True},False,False,3) is True
 assert events[:2]==['horizons','validation'] and events[2]['primary_metrics_better_than_persistence']==2 and events[2]['persistence_baseline_warning'] is True
 events.clear();assert runner.finalize_periodic_validation(state,'STAGE_A',[{'horizon':1}],{'finite_rollout':False,'validation_status':'NONFINITE_ROLLOUT'},False,False) is False
 assert events[:2]==['horizons','validation'] and events[2]['momentum_guard_activation_fraction']==0.

def test_no_finite_candidate_is_only_stage_end_error(runner):
 try:runner.ensure_stage_has_finite_candidate(False)
 except RuntimeError as error:assert str(error)=='NO_FINITE_VALIDATION_CANDIDATE'
 else:raise AssertionError('empty finite-candidate stage was accepted')

def main():
 runner=load_runner();test_completed_stage_resume(runner);test_persistence_warning_and_periodic_nonfinite(runner);test_no_finite_candidate_is_only_stage_end_error(runner);print('PASS resume and quality policy')
if __name__=='__main__':main()
