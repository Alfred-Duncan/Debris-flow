"""Static pre-review audit for final EngineeringROI-v1 execution hardening."""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.local_corrector.engineering_roi import COMPONENTS,STRATEGIES,load_engineering_roi_config
from src.local_corrector.engineering_roi_execution import ABLATION_METHODS,CORE_METHODS
def main():
    config,sha=load_engineering_roi_config(ROOT/'configs/engineering_roi_v1.json');assert config['weights']=={name:.25 for name in COMPONENTS}
    module=(ROOT/'src/local_corrector/engineering_roi.py').read_text();execution=(ROOT/'src/local_corrector/engineering_roi_execution.py').read_text();evaluator=(ROOT/'scripts/evaluate_engineering_roi_v1.py').read_text();suite=(ROOT/'scripts/run_engineering_roi_v1_suite.py').read_text();provenance=(ROOT/'src/common/provenance.py').read_text();tests=(ROOT/'scripts/test_engineering_roi_v1.py').read_text()
    for text in (module,execution,evaluator,suite):
        for forbidden in ('SetAwareSelector','MacroPolicy','BeamBC','RV-PI','optimizer','.backward('):assert forbidden not in text
    assert set(STRATEGIES)=={'random_all','random_support','dynamic_only','front_only','support_risk_only','section_only','engineering_roi','engineering_roi_no_diversity'} and len(CORE_METHODS)==10 and len(ABLATION_METHODS)==5
    for required in ('full_scenario_ids','assigned_scenario_ids','shard_count','shard_index','global_index_mod_shard_count','validate_shard_manifests','CASE_TRANSACTION_INVALID','COMMITTED.json','manifest_identity_hash','RUN_COMPLETE.json'):
        assert required in execution
    for required in ('sha256_file','global_checkpoint_sha256','local_checkpoint_sha256','local_normalization_sha256','LOCAL_NORMALIZATION_NOT_TRAIN_ONLY',"local_checkpoint.get('update')!=4000",'cuda_sync','timed_cuda','global_forward_runtime_ms',"scenario_rows('VAL')"):
        assert required in evaluator
    for required in ('--resume','suite_run_plan','SKIP_COMPLETE','SUITE_METHOD_IDENTITY_MISMATCH','formal_code_sha','scientific_definition'):
        assert required in suite
    assert 'METHOD_OUTPUT_INCOMPLETE' in execution
    for required in ('compute_support_candidates','compute_dynamic_component','compute_front_component','compute_support_risk_component','compute_section_component',"strategy=='random_all'", "strategy=='random_support'"):
        assert required in module
    for required in ('front_metadata','compute_support_risk_component','compute_section_component','compute_dynamic_component','suite_run_plan','sha256_file','timed_cuda'):
        assert required in tests
    assert 'def sha256_file' in provenance and 'stream.read(chunk_size)' in provenance and 'TEST' not in evaluator and 'HOLDOUT' not in evaluator and '--scope' not in evaluator
    print(json.dumps({'status':'PASS','scientific_definition':'UNCHANGED','execution_hardening':'FINAL_PASS','formal_val_ready':True,'config_sha256':sha},indent=2))
if __name__=='__main__':main()
