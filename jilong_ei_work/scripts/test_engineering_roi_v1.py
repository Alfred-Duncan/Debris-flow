"""Synthetic-only regression tests for final EngineeringROI-v1 execution hardening."""
from __future__ import annotations
import inspect,json,sys,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common.provenance import sha256_file
from src.global_operator_v2.oracle_refinement import PatchLayout
import src.local_corrector.engineering_roi as roi
from src.local_corrector.engineering_roi_execution import (ABLATION_METHODS,CORE_METHODS,cached_engineering_method,create_or_validate_manifest,manifest_identity_hash,partition_rows,run_output_path,validate_shard_coverage,validate_shard_manifests,validate_complete_method,write_run_complete)
from scripts.run_engineering_roi_v1_suite import ALL_METHODS,METHOD_SPECS,aggregate,requested_methods,suite_run_plan

def raises(code,fn):
    try:fn()
    except RuntimeError as error:assert code in str(error);return
    raise AssertionError(code)

def manifest(label='EngineeringROI_B10',strategy='engineering_roi',assigned=('V0','V1'),shard_index=0,shard_count=0):
    full=[f'V{i}' for i in range(20)] if len(assigned)==20 else list(assigned)
    return {'method_label':label,'strategy':strategy,'budget':None if strategy=='frozen_global' else .1,'scope':'VAL_ONLY','config_sha256':'config','engineering_roi_config_sha256':'config','code_sha':'formal','global_checkpoint_sha256':'global','global_checkpoint_provenance':{'step':1},'local_checkpoint_sha256':'local','local_checkpoint_provenance':{'update':4000},'local_checkpoint_update':4000,'local_normalization_sha256':'normalization','local_normalization_fit_scope':'TRAIN_ONLY','support_guard_rule':'current_or_global_provisional_wet','support_guard_version':'SupportGuard-v1','seed_base':20260920,'patch_layout_rows':4,'patch_layout_cols':4,'eligible_patch_count':16,'full_scenario_ids':full,'assigned_scenario_ids':list(assigned),'shard_count':shard_count,'shard_index':shard_index,'partition_rule':'global_index_mod_shard_count'}

def fields(shape=(4,8)):
    current=torch.zeros(1,6,*shape);provisional=torch.zeros(1,6,*shape);encoded_current=torch.zeros(1,6,*shape);encoded_provisional=torch.zeros(1,6,*shape);active=torch.ones(1,1,*shape,dtype=torch.bool);route=np.tile(np.arange(shape[1])*1000.,(shape[0],1));section=torch.zeros(*shape,dtype=torch.bool);section[0,0]=True;layout=PatchLayout(np.ones(shape,bool),2,2);metadata=roi.build_roi_static_metadata(layout,active,route,section);return current,provisional,encoded_current,encoded_provisional,metadata

def runner(one,frozen=False):
    scenario=str(one.iloc[0].scenario_id);timeline=[] if frozen else [{'scenario_id':scenario,'time_s':step,'method':'M'} for step in range(144)]
    return [{'scenario_id':scenario,'method':'M'}],[{'scenario_id':scenario,'station':'S'}],timeline,{'total_runtime_seconds':1.,'case_wall_runtime_seconds':1.,'global_steps_ms':[1.]*144,'roi_steps_ms':[1.]*144,'local_steps_ms':[1.]*144,'guard_steps_ms':[1.]*144}

def complete_run(root,label,strategy):
    run=root/'runs'/label;run.mkdir(parents=True);rows=[f'V{i}' for i in range(20)];data=manifest(label,strategy,rows);(run/'run_manifest.json').write_text(json.dumps(data));cases=[{'method':label,'scenario_id':value,'metric':1.} for value in rows];stations=[{'method':label,'scenario_id':value,'station':'S'} for value in rows];timeline=[] if strategy=='frozen_global' else [{'method':label,'scenario_id':value,'time_s':step,'mean_selected_dynamic_rank':.1,'mean_selected_front_rank':.2,'mean_selected_support_risk_rank':.3,'mean_selected_section_rank':.4,'mean_selected_base_score':.25,'selected_section_patch_count':1} for value in rows for step in range(144)]
    pd.DataFrame(cases).to_csv(run/'final_case_metrics.csv',index=False);pd.DataFrame(stations).to_csv(run/'final_station_metrics.csv',index=False);pd.DataFrame(timeline,columns=['method','scenario_id','time_s','mean_selected_dynamic_rank','mean_selected_front_rank','mean_selected_support_risk_rank','mean_selected_section_rank','mean_selected_base_score','selected_section_patch_count']).to_csv(run/'selection_timeline.csv',index=False);pd.DataFrame([{'method':label,'total_worker_wall_runtime_seconds':1.,'mean_selected_patch_count':0.,'mean_active_coverage':0.}]).to_csv(run/'runtime_summary.csv',index=False);report={'method_label':label,'manifest':data,'summary':{'metric':1.},'runtime':{'total_worker_wall_runtime_seconds':1.,'mean_selected_patch_count':0.,'mean_active_coverage':0.},'timing_parts':[]};(run/'method_report.json').write_text(json.dumps(report));write_run_complete(run,data,cases,timeline,strategy!='frozen_global');return run

def main():
    cfg,_=roi.load_engineering_roi_config(ROOT/'configs/engineering_roi_v1.json')
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'bad.json'
        for key,value in (('wet_threshold_m',.06),('budgets',[.1])):
            bad=dict(cfg);bad[key]=value;path.write_text(json.dumps(bad));raises('ENGINEERING_ROI_CONFIG_INVALID',lambda:roi.load_engineering_roi_config(path))
        bad=dict(cfg);bad['weights']=dict(cfg['weights'],dynamic=.30);path.write_text(json.dumps(bad));raises('ENGINEERING_ROI_CONFIG_INVALID',lambda:roi.load_engineering_roi_config(path))
    current,provisional,encoded_current,encoded_provisional,metadata=fields();scales=[1,2,3,4,5,6]
    # Scientific boundaries: front strict thresholds, ±1000m zone, support-risk, section denominator, and global scales.
    provisional[0,0,0,3]=.10;provisional[0,3,0,3]=.9;_,chainage,available=roi.compute_front_component(provisional,torch.ones_like(current[:,:1],dtype=torch.bool),metadata,cfg);assert not available
    provisional[0,0,0,3]=.11;provisional[0,3,0,3]=.05;_,_,available=roi.compute_front_component(provisional,torch.ones_like(current[:,:1],dtype=torch.bool),metadata,cfg);assert not available
    provisional[0,3,0,3]=.06;front_metadata=roi.build_roi_static_metadata(PatchLayout(np.ones((4,8),bool),4,8),metadata.active,np.tile(np.arange(8)*1000.,(4,1)),metadata.section_mask);front,chainage,available=roi.compute_front_component(provisional,torch.ones_like(current[:,:1],dtype=torch.bool),front_metadata,cfg);assert available and chainage==3000 and front[1]==0 and front[2]==1 and front[4]==1 and front[5]==0
    provisional.zero_();current.zero_();provisional[0,0,0,0]=.05;provisional[0,0,0,1]=.15;current[0,0,0,2]=.01;provisional[0,0,0,2]=.05;risk=roi.compute_support_risk_component(current,provisional,metadata,cfg);assert risk.sum()>0
    support,overlap=roi.compute_support_candidates(current,provisional,metadata,.05);assert overlap.sum()>0 and not bool(support[0,0,3,7])
    supported=torch.zeros_like(support);supported[0,0,0,0]=True;section=roi.compute_section_component(supported,metadata);assert section[0]==1
    encoded_provisional[0,:,0,0]=torch.tensor(scales);dynamic=roi.compute_dynamic_component(encoded_current,encoded_provisional,scales,torch.ones_like(support),metadata);assert dynamic[0]>0;dynamic2=roi.compute_dynamic_component(encoded_current,encoded_provisional,[2*x for x in scales],torch.ones_like(support),metadata);assert dynamic2[0]<dynamic[0]
    assert np.array_equal(roi.positive_percentile_rank([0,-1,np.nan,1,1,2]),[0,0,0,2/3,2/3,1])
    # Full vectorized scorer remains numerically and selection-equivalent to the loop reference.
    torch.manual_seed(4);current,provisional,encoded_current,encoded_provisional,metadata=fields();current.uniform_();provisional.uniform_();encoded_current.normal_();encoded_provisional.normal_();fast=roi.compute_roi_components(current,provisional,encoded_current,encoded_provisional,scales,metadata,cfg);slow=roi._reference_compute_roi_components(current,provisional,encoded_current,encoded_provisional,scales,metadata,cfg)
    for name in roi.COMPONENTS:assert np.allclose(fast['raw'][name],slow['raw'][name],rtol=1e-6,atol=1e-7) and np.allclose(fast['ranks'][name],slow['ranks'][name],rtol=1e-6,atol=1e-7)
    assert np.allclose(fast['base_score'],slow['base_score'],rtol=1e-6,atol=1e-7) and np.allclose(fast['base_score'],.25*fast['ranks']['dynamic']+.25*fast['ranks']['predicted_front']+.25*fast['ranks']['support_risk']+.25*fast['ranks']['engineering_section']);selected,_=roi.select_engineering_roi(current,provisional,encoded_current,encoded_provisional,scales,metadata,cfg,.1,'engineering_roi',123);chosen,_,_=roi._diverse(np.flatnonzero(slow['candidates']),slow['base_score'],slow['patches'],metadata.layout.count_for_budget(.1));assert [patch.patch_id for patch in selected]==[slow['patches'][index].patch_id for index in chosen]
    ties,first,second=roi._diverse(np.arange(8),np.ones(8),front_metadata.layout.eligible,4);assert ties==[0,2,4,6] and first==4 and second==0
    random_support,_=roi.select_engineering_roi(current,provisional,encoded_current,encoded_provisional,scales,metadata,cfg,.1,'random_support',123);assert all(fast['support_overlap'][next(index for index,patch in enumerate(fast['patches']) if patch.patch_id==value.patch_id)]>0 for value in random_support)
    # Strategy paths invoke only their allowed scoring primitives.
    originals={name:getattr(roi,name) for name in ('compute_roi_components','compute_dynamic_component','compute_front_component','compute_support_risk_component','compute_section_component')};calls={name:0 for name in originals}
    def wrapped(name):
        def call(*args,**kwargs):calls[name]+=1;return originals[name](*args,**kwargs)
        return call
    for name in originals:setattr(roi,name,wrapped(name))
    try:
        roi.select_engineering_roi(current,provisional,encoded_current,encoded_provisional,scales,metadata,cfg,.1,'random_all',1);assert calls['compute_roi_components']==0 and sum(calls[name] for name in calls if name!='compute_roi_components')==0
        roi.select_engineering_roi(current,provisional,encoded_current,encoded_provisional,scales,metadata,cfg,.1,'random_support',1);assert calls['compute_roi_components']==0 and sum(calls[name] for name in calls if name!='compute_roi_components')==0
        for strategy,needed in (('dynamic_only','compute_dynamic_component'),('front_only','compute_front_component'),('support_risk_only','compute_support_risk_component'),('section_only','compute_section_component')):
            calls.update({name:0 for name in calls});roi.select_engineering_roi(current,provisional,encoded_current,encoded_provisional,scales,metadata,cfg,.1,strategy,1);assert calls[needed]==1 and sum(value for name,value in calls.items() if name not in (needed,'compute_roi_components'))==0
        calls.update({name:0 for name in calls});roi.select_engineering_roi(current,provisional,encoded_current,encoded_provisional,scales,metadata,cfg,.1,'engineering_roi',1);assert calls['compute_roi_components']==1
    finally:
        for name,value in originals.items():setattr(roi,name,value)
    rows=pd.DataFrame({'scenario_id':[f'V{i}' for i in range(20)]});shards=[partition_rows(rows,index,2) for index in range(2)];assert run_output_path(Path('x'))==Path('x') and run_output_path(Path('x'),1,2)==Path('x/shards/shard_1') and sorted(sum([list(part.__global_case_index) for part in shards],[]))==list(range(20));validate_shard_coverage([{'scenario_id':value} for part in shards for value in part.scenario_id],rows.scenario_id)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);small=rows.iloc[:2];data=manifest(assigned=tuple(small.scenario_id));cached_engineering_method(root,data,small,runner,True,False,1);cached_engineering_method(root,data,small,runner,True,True,1);assert len(list((root/'progress/cases').iterdir()))==2
        orphan=root/'progress/.tmp/V2.orphan';orphan.mkdir(parents=True);cached_engineering_method(root,data,small,runner,True,True,1);assert not orphan.exists();(root/'progress/cases/V0/selection_timeline.csv').write_text('scenario_id,time_s\nV0,0\n');raises('CASE_TRANSACTION_INVALID',lambda:cached_engineering_method(root,data,small,runner,True,True,1))
        frozen=Path(temp)/'frozen';frozen_rows=small.iloc[:1];frozen_data=manifest('FrozenGlobal','frozen_global',(str(frozen_rows.iloc[0].scenario_id),));cached_engineering_method(frozen,frozen_data,frozen_rows,lambda one:runner(one,True),False,False,1)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);full=[f'V{i}' for i in range(8)]
        for index in range(2):
            assigned=full[index::2];data=manifest(assigned=assigned,shard_index=index,shard_count=2);data['full_scenario_ids']=full;path=root/'shards'/f'shard_{index}';path.mkdir(parents=True);(path/'run_manifest.json').write_text(json.dumps(data))
        validate_shard_manifests(root,2);wrong=manifest(assigned=full[0::4],shard_index=0,shard_count=4);wrong['full_scenario_ids']=full;raises('RESUME_MANIFEST_MISMATCH',lambda:create_or_validate_manifest(root/'shards/shard_0',wrong,True))
    with tempfile.TemporaryDirectory() as temp:
        file_a=Path(temp)/'a';file_b=Path(temp)/'b';file_a.write_bytes(b'A');file_b.write_bytes(b'A');first=sha256_file(file_a);assert first==sha256_file(file_b);file_a.write_bytes(b'B');assert first!=sha256_file(file_a)
    from scripts import evaluate_engineering_roi_v1 as evaluator
    synchronized=[];original_sync=torch.cuda.synchronize;torch.cuda.synchronize=lambda device:synchronized.append(str(device))
    try:
        assert evaluator.timed_cuda(lambda:'cpu',torch.device('cpu'))[0]=='cpu';assert not synchronized;assert evaluator.timed_cuda(lambda:'cuda',torch.device('cuda'))[0]=='cuda';assert len(synchronized)==2
    finally:torch.cuda.synchronize=original_sync
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);complete_run(root,'FrozenGlobal','frozen_global');partial=root/'runs/RandomAll_B05';partial.mkdir(parents=True);partial_data=manifest('RandomAll_B05','random_all',['V0']);(partial/'run_manifest.json').write_text(json.dumps(partial_data));(partial/'progress').mkdir();plan=dict(suite_run_plan(root,('FrozenGlobal','RandomAll_B05','RandomAll_B10'),'formal',True));assert plan=={'FrozenGlobal':'skip','RandomAll_B05':'resume','RandomAll_B10':'fresh'};bad=json.loads((root/'runs/FrozenGlobal/RUN_COMPLETE.json').read_text());bad['code_sha']='bad';(root/'runs/FrozenGlobal/RUN_COMPLETE.json').write_text(json.dumps(bad));raises('SUITE_METHOD_IDENTITY_MISMATCH',lambda:suite_run_plan(root,('FrozenGlobal',),'formal',True))
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        for label,(strategy,_) in METHOD_SPECS.items():complete_run(root,label,strategy)
        aggregate(root);assert len(pd.read_csv(root/'final_method_summary.csv'))==15 and set(pd.read_csv(root/'ablation_summary.csv').method)=={'FrozenGlobal','EngineeringROI_B10',*ABLATION_METHODS}
    class Args: methods=None;core_only=False;ablations_only=False
    assert requested_methods(Args())==CORE_METHODS+ABLATION_METHODS
    source=(ROOT/'scripts/evaluate_engineering_roi_v1.py').read_text();execution=source[source.index('@torch.no_grad()'):];guard_source=source[source.index('def support_guard_pipeline'):source.index('def manifest_for')];assert execution.index('select_engineering_roi(')<execution.index('truth_current=')<execution.index('metric.add(') and execution.index('apply_learned_correction(')<execution.index('support_guard_pipeline(') and guard_source.index('apply_support_guard(')<guard_source.index('apply_momentum_state_guard(') and 'optimizer' not in source and '.backward(' not in source and "scenario_rows('VAL')" in source and 'TEST' not in source and 'HOLDOUT' not in source
    assert 'for patch in metadata.layout.eligible' not in inspect.getsource(roi.compute_roi_components) and 'local_correction_scales' not in inspect.signature(roi.compute_roi_components).parameters
    print('PASS final EngineeringROI-v1 synthetic semantics, transactions, shard identity, suite resume, and timing checks')
if __name__=='__main__':main()
