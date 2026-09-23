"""Static pre-review audit for deterministic EngineeringROI-v1; no rollout."""
from __future__ import annotations
import inspect,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.local_corrector.engineering_roi import COMPONENTS,STRATEGIES,compute_roi_components,load_engineering_roi_config

def main():
    config,sha=load_engineering_roi_config(ROOT/'configs/engineering_roi_v1.json');assert config['budgets']==[.05,.10,.20] and config['normalization']=='positive_percentile_rank'
    assert config['weights']=={name:.25 for name in COMPONENTS} and config['candidate_rule']=='support_overlap_and_positive_score' and config['diversity_rule']=='nonadjacent_first_then_fill'
    assert config['wet_threshold_m']==.05 and config['truth_in_roi_scoring'] is False and config['local_model']=='best_update_4000'
    assert not ({'truth','target','future','local_correction_scales'}&set(inspect.signature(compute_roi_components).parameters))
    module=(ROOT/'src/local_corrector/engineering_roi.py').read_text();evaluator=(ROOT/'scripts/evaluate_engineering_roi_v1.py').read_text()
    for forbidden in ('SetAwareSelector','MacroPolicy','BeamBC','RV-PI','policy iteration','optimizer','.backward('):assert forbidden not in module and forbidden not in evaluator
    assert set(STRATEGIES)=={'random_all','random_support','dynamic_only','front_only','support_risk_only','section_only','engineering_roi','engineering_roi_no_diversity'}
    for name in ('final_method_summary.csv','final_case_metrics.csv','final_station_metrics.csv','selection_timeline.csv','roi_component_summary.csv','budget_usage_summary.csv','ablation_summary.csv','runtime_summary.csv','engineering_roi_val_report.json'):
        assert name in evaluator
    assert "scenario_rows('VAL')" in evaluator and "checkpoint.get('update')!=4000" in evaluator and 'ENGINEERING_ROI_V1_VAL.json' in evaluator and '--scope' not in evaluator and 'TEST' not in evaluator and 'HOLDOUT' not in evaluator
    print(json.dumps({'status':'PASS','config_sha256':sha,'strategies':len(STRATEGIES),'components':list(COMPONENTS)},indent=2))
if __name__=='__main__':main()
