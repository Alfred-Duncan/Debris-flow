"""Regression: periodic quality telemetry never terminates the curriculum."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.pipeline import PipelineState

def load_runner():
 spec=importlib.util.spec_from_file_location('runner_telemetry',ROOT/'scripts/run_global_operator_v2_full.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def invoke(runner,worse):
 events=[];runner._record_horizons=lambda *args:events.append('horizons');runner._record_validation=lambda *args:events.append('validation');runner.safe_append_csv=lambda path,row:events.append(row)
 state=PipelineState('v','h',current_stage='STAGE_A',global_step=2000)
 return events,runner.finalize_periodic_validation(state,'STAGE_A',[{'horizon':1}],{'finite_rollout':True},False,False,worse)

def main():
 runner=load_runner();events,finite=invoke(runner,3);assert events[:2]==['horizons','validation'] and events[2]['primary_metrics_better_than_persistence']==2 and events[2]['persistence_baseline_warning'] is True and finite is True
 events,finite=invoke(runner,2);assert events[:2]==['horizons','validation'] and events[2]['primary_metrics_better_than_persistence']==3 and events[2]['persistence_baseline_warning'] is False and finite is True
 print('PASS telemetry ordering')
if __name__=='__main__':main()
