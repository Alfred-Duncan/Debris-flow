"""Static pre-review audit for EngineeringROI-v1 execution hardening."""
from __future__ import annotations
import inspect,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.local_corrector.engineering_roi import COMPONENTS,STRATEGIES,compute_roi_components,load_engineering_roi_config
from src.local_corrector.engineering_roi_execution import ABLATION_METHODS,CORE_METHODS
def main():
    config,sha=load_engineering_roi_config(ROOT/'configs/engineering_roi_v1.json');assert config['weights']=={name:.25 for name in COMPONENTS}
    assert not ({'truth','target','future','local_correction_scales'}&set(inspect.signature(compute_roi_components).parameters))
    module=(ROOT/'src/local_corrector/engineering_roi.py').read_text();execution=(ROOT/'src/local_corrector/engineering_roi_execution.py').read_text();evaluator=(ROOT/'scripts/evaluate_engineering_roi_v1.py').read_text();suite=(ROOT/'scripts/run_engineering_roi_v1_suite.py').read_text()
    for text in (module,execution,evaluator,suite):
        for forbidden in ('SetAwareSelector','MacroPolicy','BeamBC','RV-PI','optimizer','.backward('):assert forbidden not in text
    assert set(STRATEGIES)=={'random_all','random_support','dynamic_only','front_only','support_risk_only','section_only','engineering_roi','engineering_roi_no_diversity'} and len(CORE_METHODS)==10 and len(ABLATION_METHODS)==5
    for required in ('run_output_path','cached_engineering_method','RESUME_MANIFEST_MISMATCH','SHARD_MANIFEST_MISMATCH','DUPLICATE_SCENARIO_ACROSS_SHARDS','INCOMPLETE_VAL_SHARD_COVERAGE','SELECTION_TIMELINE_INCOMPLETE'):assert required in execution
    for required in ('LOCAL_NORMALIZATION_NOT_TRAIN_ONLY',"local_checkpoint.get('update')!=4000",'--source-code-sha','roi_scoring_runtime_ms','support_guard_runtime_ms','frozen_global',"scenario_rows('VAL')"):
        assert required in evaluator
    assert 'TEST' not in evaluator and 'HOLDOUT' not in evaluator and '--scope' not in evaluator and 'ENGINEERING_ROI_V1_VAL.json' not in evaluator
    for name in ('final_method_summary.csv','final_case_metrics.csv','final_station_metrics.csv','selection_timeline.csv','roi_component_summary.csv','budget_usage_summary.csv','ablation_summary.csv','runtime_summary.csv','engineering_roi_val_report.json'):
        assert name in suite
    assert 'SUITE_INCOMPLETE' in suite and '--aggregate-only' in suite and 'FrozenGlobal' in suite
    print(json.dumps({'status':'PASS','config_sha256':sha,'scientific_definition':'UNCHANGED','execution_hardening':'PASS'},indent=2))
if __name__=='__main__':main()
