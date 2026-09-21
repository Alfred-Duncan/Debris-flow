"""Mocked PRECHECK-to-FREEZE transition and capacity fallback tests."""
from __future__ import annotations
import importlib.util,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.pipeline import PipelineState,apply_capacity_result
from src.global_operator_v2.trainer import is_improvement
spec=importlib.util.spec_from_file_location('runner',ROOT/'scripts/run_global_operator_v2_full.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
def main():
    called=[]
    def mocked_probe(model,k,loss):
        called.append(k);return {'K':k,'status':'OOM' if k==4 else 'SAFE','safe':k<4}
    probes=runner.capacity_probe(None,lambda k:None,mocked_probe);state=apply_capacity_result(PipelineState('v','h',current_stage='ARCHITECTURE'),probes)
    assert called==[1,2,4] and state.total_planned_updates==4000
    assert runner.core_stage_path(state)==['ARCHITECTURE','STAGE_A','STAGE_B','VAL_CONFIRM','FREEZE']
    best=None
    for score in (.8,.6,.75):
        if is_improvement(score,best):best=score
    assert best==.6
if __name__=='__main__':main()
