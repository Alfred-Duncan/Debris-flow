"""Regression: step-2000 quality stops must never erase diagnostics."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.pipeline import PipelineState

def load_runner():
 spec=importlib.util.spec_from_file_location('runner_telemetry',ROOT/'scripts/run_global_operator_v2_full.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def invoke(runner,worse):
 events=[];runner._record_horizons=lambda *args:events.append('horizons');runner._record_validation=lambda *args:events.append('validation')
 state=PipelineState('v','h',current_stage='STAGE_A',global_step=2000)
 try:runner.finalize_periodic_validation(state,'STAGE_A',[{'horizon':1}],{'finite_rollout':True},False,False,worse)
 except RuntimeError as exc:return events,str(exc)
 return events,None

def main():
 runner=load_runner();events,error=invoke(runner,3);assert events==['horizons','validation'] and error=='EARLY_TRAINING_QUALITY_FAILURE PERSISTENCE_COLLAPSE'
 events,error=invoke(runner,2);assert events==['horizons','validation'] and error is None
 print('PASS telemetry ordering')
if __name__=='__main__':main()
