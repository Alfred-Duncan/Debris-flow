"""Synthetic EngineeringROI-v1 execution-hardening regression tests."""
from __future__ import annotations
import inspect,json,sys,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.oracle_refinement import PatchLayout,select_random
from src.local_corrector.engineering_roi import (COMPONENTS,_reference_compute_roi_components,build_roi_static_metadata,build_section_mask,compute_roi_components,load_engineering_roi_config,positive_percentile_rank,select_engineering_roi)
from src.local_corrector.engineering_roi_execution import (ABLATION_METHODS,CORE_METHODS,cached_engineering_method,canonical_method_label,partition_rows,run_output_path,validate_shard_coverage)
from scripts.run_engineering_roi_v1_suite import METHOD_SPECS,aggregate,requested_methods

def raises(code,fn):
    try:fn()
    except RuntimeError as error:assert code in str(error);return
    raise AssertionError(code)
def fields(shape=(16,16)):
    current=torch.rand(1,6,*shape);provisional=torch.rand(1,6,*shape);encoded_current=torch.randn(1,6,*shape);encoded_provisional=torch.randn(1,6,*shape);active=torch.ones(1,1,*shape,dtype=torch.bool);route=np.tile(np.arange(shape[1])*1000.,(shape[0],1));section=torch.zeros(*shape,dtype=torch.bool);section[1,1]=True;return current,provisional,encoded_current,encoded_provisional,active,route,section
def main():
    cfg,sha=load_engineering_roi_config(ROOT/'configs/engineering_roi_v1.json');assert cfg['weights']=={name:.25 for name in COMPONENTS} and cfg['budgets']==[.05,.10,.20]
    with tempfile.TemporaryDirectory() as temp:
        bad=dict(cfg);bad['wet_threshold_m']=.06;path=Path(temp)/'bad.json';path.write_text(json.dumps(bad));raises('ENGINEERING_ROI_CONFIG_INVALID',lambda:load_engineering_roi_config(path));bad=dict(cfg);bad['weights']=dict(cfg['weights'],dynamic=.30);path.write_text(json.dumps(bad));raises('ENGINEERING_ROI_CONFIG_INVALID',lambda:load_engineering_roi_config(path));bad=dict(cfg);bad['budgets']=[.1];path.write_text(json.dumps(bad));raises('ENGINEERING_ROI_CONFIG_INVALID',lambda:load_engineering_roi_config(path))
    layout=PatchLayout(np.ones((16,16),bool),4,4);current,provisional,encoded_current,encoded_provisional,active,route,section=fields();metadata=build_roi_static_metadata(layout,active,route,section)
    # Vectorized components match the retained loop reference within tolerance.
    fast=compute_roi_components(current,provisional,encoded_current,encoded_provisional,[1,2,3,4,5,6],metadata,cfg);slow=_reference_compute_roi_components(current,provisional,encoded_current,encoded_provisional,[1,2,3,4,5,6],metadata,cfg)
    assert np.array_equal(fast['support_overlap'],slow['support_overlap']) and np.array_equal(fast['candidates'],slow['candidates'])
    for name in COMPONENTS:assert np.allclose(fast['raw'][name],slow['raw'][name],rtol=1e-6,atol=1e-7) and np.allclose(fast['ranks'][name],slow['ranks'][name],rtol=1e-6,atol=1e-7)
    assert np.allclose(fast['base_score'],slow['base_score'],rtol=1e-6,atol=1e-7)
    # Scores use no teacher/target/future/local-scale inputs; static tensors are cached in metadata.
    names=set(inspect.signature(compute_roi_components).parameters);assert not ({'truth','target','future','local_correction_scales','route_chainage_m','section_mask'}&names) and metadata.route_tensor.device==metadata.active.device and metadata.section_mask.device==metadata.active.device
    ranks=positive_percentile_rank([0,1,1,2,np.nan]);assert ranks[0]==0 and ranks[1]==ranks[2] and ranks[3]==1
    # Same seed is deterministic; RandomAll preserves existing seed semantics and RandomSupport uses sorted candidate IDs.
    selected,_=select_engineering_roi(current,provisional,encoded_current,encoded_provisional,[1]*6,metadata,cfg,.2,'random_all',123);assert [p.patch_id for p in selected]==[p.patch_id for p in select_random(layout,layout.count_for_budget(.2),123)]
    a,_=select_engineering_roi(current,provisional,encoded_current,encoded_provisional,[1]*6,metadata,cfg,.2,'random_support',123);b,_=select_engineering_roi(current,provisional,encoded_current,encoded_provisional,[1]*6,metadata,cfg,.2,'random_support',123);assert [p.patch_id for p in a]==[p.patch_id for p in b]
    # Shard paths and globally fixed indices cover each scenario exactly once.
    rows=pd.DataFrame({'scenario_id':[f'V{i}' for i in range(20)]});shards=[partition_rows(rows,index,2) for index in range(2)];assert run_output_path(Path('x'))==Path('x') and run_output_path(Path('x'),1,2)==Path('x/shards/shard_1') and sorted(sum([list(part.__global_case_index) for part in shards],[]))==list(range(20));validate_shard_coverage([{'scenario_id':value} for part in shards for value in part.scenario_id],rows.scenario_id);raises('DUPLICATE_SCENARIO_ACROSS_SHARDS',lambda:validate_shard_coverage([{'scenario_id':'V0'},{'scenario_id':'V0'}],rows.scenario_id));raises('INCOMPLETE_VAL_SHARD_COVERAGE',lambda:validate_shard_coverage([{'scenario_id':value} for value in rows.scenario_id[:-1]],rows.scenario_id))
    # Canonical labels and full suite plan are exact; Frozen appears exactly once.
    assert canonical_method_label('frozen_global')=='FrozenGlobal' and canonical_method_label('engineering_roi',.1)=='EngineeringROI_B10' and tuple(METHOD_SPECS)==CORE_METHODS+ABLATION_METHODS and (CORE_METHODS+ABLATION_METHODS).count('FrozenGlobal')==1
    class Args: methods=None;core_only=False;ablations_only=False
    assert requested_methods(Args())==CORE_METHODS+ABLATION_METHODS
    # Synthetic suite aggregation combines every isolated run, including the B10 ablation table.
    with tempfile.TemporaryDirectory() as temp:
        suite_root=Path(temp);common={'config_sha256':'cfg','code_sha':'code','global_checkpoint_provenance':{'checkpoint':'global'},'local_checkpoint_update':4000}
        for label in CORE_METHODS+ABLATION_METHODS:
            run=suite_root/'runs'/label;run.mkdir(parents=True);report={'method_label':label,'manifest':common,'summary':{'rmse':1.},'runtime':{'total_runtime_seconds':1.,'mean_selected_patch_count':0.,'mean_active_coverage':0.}};(run/'method_report.json').write_text(json.dumps(report));pd.DataFrame([{'method':label,'scenario_id':'V0','rmse':1.}]).to_csv(run/'final_case_metrics.csv',index=False);pd.DataFrame([{'method':label,'scenario_id':'V0','station':'S'}]).to_csv(run/'final_station_metrics.csv',index=False);pd.DataFrame([{'method':label,'scenario_id':'V0','time_s':10,'mean_selected_dynamic_rank':.1,'mean_selected_front_rank':.2,'mean_selected_support_risk_rank':.3,'mean_selected_section_rank':.4,'mean_selected_base_score':.25,'selected_section_patch_count':1}]).to_csv(run/'selection_timeline.csv',index=False)
        aggregate(suite_root);assert len(pd.read_csv(suite_root/'final_method_summary.csv'))==15 and set(pd.read_csv(suite_root/'ablation_summary.csv').method)=={'FrozenGlobal','EngineeringROI_B10',*ABLATION_METHODS}
    # Manifest, completed list, case table, station table, and timeline are case-atomic.
    manifest={'method_label':'EngineeringROI_B10','strategy':'engineering_roi','budget':.1,'scope':'VAL_ONLY','config_sha256':'a','code_sha':'b','global_checkpoint_provenance':{'x':1},'local_checkpoint_update':4000,'local_normalization_fit_scope':'TRAIN_ONLY','support_guard_rule':'current_or_global_provisional_wet','seed_base':20260920,'patch_layout_rows':16,'patch_layout_cols':16,'eligible_patch_count':182}
    calls=[]
    def runner(one):
        scenario=str(one.iloc[0].scenario_id);calls.append(scenario);return ([{'scenario_id':scenario,'method':'EngineeringROI_B10'}],[{'scenario_id':scenario,'station':'s'}],[{'scenario_id':scenario,'time_s':step} for step in range(144)],{'total_runtime_seconds':1.,'roi_steps_ms':[1.]*144,'local_steps_ms':[1.]*144,'guard_steps_ms':[1.]*144})
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);small=rows.iloc[:2];cached_engineering_method(root,manifest,small,runner,True,False);cached_engineering_method(root,manifest,small,runner,True,True);assert calls==['V0','V1'] and len(pd.read_csv(root/'progress/selection_timeline.csv'))==288
        altered=dict(manifest);altered['code_sha']='changed';raises('RESUME_MANIFEST_MISMATCH',lambda:cached_engineering_method(root,altered,small,runner,True,True));(root/'progress/completed_cases.json').write_text(json.dumps(['V0']));raises('CACHE_INCONSISTENT',lambda:cached_engineering_method(root,manifest,small,runner,True,True))
    # Static execution order and no training/non-VAL branch.
    source=(ROOT/'scripts/evaluate_engineering_roi_v1.py').read_text();assert source.index('select_engineering_roi(')<source.index('truth_current=')<source.index('metric.add(') and source.index('apply_learned_correction(')<source.index('apply_support_guard(')<source.index('apply_momentum_state_guard(') and 'optimizer' not in source and '.backward(' not in source and "scenario_rows('VAL')" in source and 'TEST' not in source and 'HOLDOUT' not in source
    suite_source=(ROOT/'scripts/run_engineering_roi_v1_suite.py').read_text();assert suite_source.index('if args.aggregate_only:return aggregate(root)')<suite_source.index('subprocess.run(')
    production=(ROOT/'src/local_corrector/engineering_roi.py').read_text();assert 'for patch in metadata.layout.eligible' not in production[production.index('def compute_roi_components'):production.index('def _reference_compute_roi_components')]
    print('PASS EngineeringROI-v1 vectorization, shard, manifest, atomic cache, suite, and VAL-only synthetic checks')
if __name__=='__main__':main()
