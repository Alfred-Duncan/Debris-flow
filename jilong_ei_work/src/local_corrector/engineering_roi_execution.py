"""Reproducible manifests, shard identity, and crash-safe case bundles."""
from __future__ import annotations
import hashlib,json,os,shutil,tempfile
from pathlib import Path
from typing import Callable
import pandas as pd

CANONICAL={'frozen_global':'FrozenGlobal','random_all':'RandomAll','random_support':'RandomSupport','engineering_roi':'EngineeringROI','dynamic_only':'DynamicOnly','front_only':'FrontOnly','support_risk_only':'SupportRiskOnly','section_only':'SectionOnly','engineering_roi_no_diversity':'EngineeringROI_NoDiversity'}
CORE_METHODS=('FrozenGlobal','RandomAll_B05','RandomAll_B10','RandomAll_B20','RandomSupport_B05','RandomSupport_B10','RandomSupport_B20','EngineeringROI_B05','EngineeringROI_B10','EngineeringROI_B20')
ABLATION_METHODS=('DynamicOnly_B10','FrontOnly_B10','SupportRiskOnly_B10','SectionOnly_B10','EngineeringROI_NoDiversity_B10')
EXPERIMENT_MANIFEST_KEYS=('method_label','strategy','budget','scope','config_sha256','engineering_roi_config_sha256','code_sha','global_checkpoint_sha256','global_checkpoint_provenance','local_checkpoint_sha256','local_checkpoint_provenance','local_checkpoint_update','local_normalization_sha256','local_normalization_fit_scope','support_guard_rule','support_guard_version','seed_base','patch_layout_rows','patch_layout_cols','eligible_patch_count','full_scenario_ids','shard_count','partition_rule')
SHARD_MANIFEST_KEYS=('shard_index','assigned_scenario_ids')
MANIFEST_KEYS=EXPERIMENT_MANIFEST_KEYS+SHARD_MANIFEST_KEYS

def canonical_method_label(strategy: str,budget: float|None=None)->str:
    if strategy not in CANONICAL:raise ValueError('CANONICAL_METHOD_INVALID')
    if strategy=='frozen_global':
        if budget is not None:raise ValueError('FROZEN_GLOBAL_HAS_NO_BUDGET')
        return 'FrozenGlobal'
    if budget not in (.05,.10,.20):raise ValueError('CANONICAL_BUDGET_INVALID')
    return f'{CANONICAL[strategy]}_B{int(round(budget*100)):02d}'

def run_output_path(root: Path,shard_index: int=0,shard_count: int=0)->Path:
    if shard_count==0:
        if shard_index!=0:raise RuntimeError('INVALID_SHARD_INDEX')
        return root
    if not 0<=shard_index<shard_count:raise RuntimeError('INVALID_SHARD_INDEX')
    return root/'shards'/f'shard_{shard_index}'

def partition_rows(rows,shard_index: int=0,shard_count: int=0):
    complete=rows.copy().reset_index(drop=True);complete['__global_case_index']=range(len(complete))
    if shard_count==0:
        if shard_index!=0:raise RuntimeError('INVALID_SHARD_INDEX')
        return complete
    if not 0<=shard_index<shard_count:raise RuntimeError('INVALID_SHARD_INDEX')
    return complete.iloc[shard_index::shard_count].reset_index(drop=True)

def _atomic_text(path: Path,text: str):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(text);os.replace(temporary,path)

def manifest_identity_hash(manifest: dict)->str:
    return hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def create_or_validate_manifest(run_out: Path,expected: dict,resume: bool):
    path=run_out/'run_manifest.json'
    if path.exists():
        actual=json.loads(path.read_text())
        if any(actual.get(key)!=expected.get(key) for key in MANIFEST_KEYS):raise RuntimeError('RESUME_MANIFEST_MISMATCH')
        if not resume:raise RuntimeError('PROGRESS_EXISTS_USE_RESUME')
    else:_atomic_text(path,json.dumps(expected,indent=2,sort_keys=True))
    return path

def _expected_assignment(manifest,index):
    full=list(map(str,manifest['full_scenario_ids']));count=int(manifest['shard_count'])
    return full if count==0 else full[index::count]

def validate_shard_manifests(run_root: Path,shard_count: int):
    if shard_count<=0:raise RuntimeError('SHARD_MANIFEST_MISMATCH')
    manifests=[]
    for index in range(shard_count):
        path=run_root/'shards'/f'shard_{index}'/'run_manifest.json'
        if not path.exists():raise RuntimeError('SHARD_MANIFEST_MISMATCH')
        manifests.append(json.loads(path.read_text()))
    first=manifests[0]
    if any(manifest.get('shard_count')!=shard_count or any(manifest.get(key)!=first.get(key) for key in EXPERIMENT_MANIFEST_KEYS) for manifest in manifests):raise RuntimeError('SHARD_MANIFEST_MISMATCH')
    indices=[manifest.get('shard_index') for manifest in manifests]
    if sorted(indices)!=list(range(shard_count)):raise RuntimeError('INVALID_SHARD_ASSIGNMENT')
    assigned=[]
    for manifest in manifests:
        index=int(manifest['shard_index']);values=list(map(str,manifest.get('assigned_scenario_ids',[])))
        if manifest.get('partition_rule')!='global_index_mod_shard_count' or values!=_expected_assignment(manifest,index):raise RuntimeError('INVALID_SHARD_ASSIGNMENT')
        assigned.extend(values)
    if len(assigned)!=len(set(assigned)):raise RuntimeError('DUPLICATE_SCENARIO_ACROSS_SHARDS')
    if set(assigned)!=set(map(str,first['full_scenario_ids'])):raise RuntimeError('INCOMPLETE_VAL_SHARD_COVERAGE')
    return manifests

def validate_shard_coverage(case_rows,expected_ids):
    ids=[str(row['scenario_id']) for row in case_rows]
    if len(ids)!=len(set(ids)):raise RuntimeError('DUPLICATE_SCENARIO_ACROSS_SHARDS')
    if set(ids)!=set(map(str,expected_ids)):raise RuntimeError('INCOMPLETE_VAL_SHARD_COVERAGE')

def _records(path: Path):
    try:return pd.read_csv(path).to_dict('records')
    except pd.errors.EmptyDataError:return []

def _recover_orphan_temps(progress: Path):
    temporary=progress/'.tmp'
    if temporary.exists():
        for path in temporary.iterdir():
            if path.is_dir():shutil.rmtree(path)
            else:path.unlink()

def _validate_bundle(path: Path,scenario: str,identity: str,requires_timeline: bool,expected_station_count: int|None):
    marker=path/'COMMITTED.json';case_path=path/'case_metrics.json';station_path=path/'station_metrics.csv';timeline_path=path/'selection_timeline.csv';timing_path=path/'timing.json'
    if not all(item.exists() for item in (marker,case_path,station_path,timeline_path,timing_path)):raise RuntimeError('CASE_TRANSACTION_INVALID')
    committed=json.loads(marker.read_text());case=json.loads(case_path.read_text());stations=_records(station_path);timeline=_records(timeline_path);timing=json.loads(timing_path.read_text())
    if committed.get('scenario_id')!=scenario or committed.get('manifest_identity_hash')!=identity or str(case.get('scenario_id'))!=scenario:raise RuntimeError('CASE_TRANSACTION_INVALID')
    if not stations or {str(row.get('scenario_id')) for row in stations}!={scenario} or expected_station_count is not None and len(stations)!=expected_station_count:raise RuntimeError('CASE_TRANSACTION_INVALID')
    if requires_timeline:
        if len(timeline)!=144 or {str(row.get('scenario_id')) for row in timeline}!={scenario}:raise RuntimeError('CASE_TRANSACTION_INVALID')
    elif timeline:raise RuntimeError('CASE_TRANSACTION_INVALID')
    if not isinstance(timing,dict):raise RuntimeError('CASE_TRANSACTION_INVALID')
    return case,stations,timeline,timing,committed

def read_committed_cases(run_out: Path,manifest: dict,requires_timeline: bool,expected_station_count: int|None=None):
    progress=run_out/'progress';identity=manifest_identity_hash(manifest);cases=[];stations=[];timeline=[];timings=[]
    for scenario in map(str,manifest['assigned_scenario_ids']):
        path=progress/'cases'/scenario
        if not path.exists():continue
        case,station_rows,timeline_rows,timing,marker=_validate_bundle(path,scenario,identity,requires_timeline,expected_station_count)
        case['_global_case_index']=marker['global_case_index'];cases.append(case);stations.extend(station_rows);timeline.extend(timeline_rows);timings.append(timing)
    cases.sort(key=lambda row:row['_global_case_index']);return cases,stations,timeline,timings

def _write_bundle(progress: Path,manifest: dict,row,new_cases,new_stations,new_timeline,timing,requires_timeline,expected_station_count):
    scenario=str(row.scenario_id);identity=manifest_identity_hash(manifest)
    if len(new_cases)!=1 or str(new_cases[0].get('scenario_id'))!=scenario or expected_station_count is not None and len(new_stations)!=expected_station_count or requires_timeline and len(new_timeline)!=144 or not requires_timeline and new_timeline:raise RuntimeError('CASE_TRANSACTION_INVALID')
    temp_root=progress/'.tmp';temp_root.mkdir(parents=True,exist_ok=True);temporary=Path(tempfile.mkdtemp(prefix=f'{scenario}.',dir=temp_root));final=progress/'cases'/scenario
    try:
        (temporary/'case_metrics.json').write_text(json.dumps(new_cases[0],sort_keys=True,allow_nan=True));pd.DataFrame(new_stations).to_csv(temporary/'station_metrics.csv',index=False);pd.DataFrame(new_timeline).to_csv(temporary/'selection_timeline.csv',index=False);(temporary/'timing.json').write_text(json.dumps(timing,sort_keys=True,allow_nan=True));(temporary/'COMMITTED.json').write_text(json.dumps({'scenario_id':scenario,'global_case_index':int(row['__global_case_index']),'manifest_identity_hash':identity},sort_keys=True))
        _validate_bundle(temporary,scenario,identity,requires_timeline,expected_station_count)
        final.parent.mkdir(parents=True,exist_ok=True)
        if final.exists():raise RuntimeError('CASE_TRANSACTION_INVALID')
        os.replace(temporary,final)
    except Exception:
        if temporary.exists():shutil.rmtree(temporary)
        raise

def cached_engineering_method(run_out: Path,manifest: dict,rows,runner: Callable,requires_timeline: bool,resume: bool,expected_station_count: int|None=None):
    """Directory rename is the per-case commit point; bundles are authoritative."""
    if '__global_case_index' not in rows.columns:
        rows=rows.copy();indices={value:index for index,value in enumerate(map(str,manifest['full_scenario_ids']))};rows['__global_case_index']=[indices[str(value)] for value in rows.scenario_id]
    create_or_validate_manifest(run_out,manifest,resume);progress=run_out/'progress';_recover_orphan_temps(progress)
    if (progress/'case_metrics.csv').exists() and not (progress/'cases').exists():raise RuntimeError('CASE_TRANSACTION_INVALID')
    cases,stations,timeline,timings=read_committed_cases(run_out,manifest,requires_timeline,expected_station_count);completed={str(row['scenario_id']) for row in cases}
    for _,row in rows.iterrows():
        scenario=str(row.scenario_id)
        if scenario in completed:continue
        new_cases,new_stations,new_timeline,timing=runner(row.to_frame().T.reset_index(drop=True));_write_bundle(progress,manifest,row,new_cases,new_stations,new_timeline,timing,requires_timeline,expected_station_count)
        cases,stations,timeline,timings=read_committed_cases(run_out,manifest,requires_timeline,expected_station_count);completed={str(item['scenario_id']) for item in cases};_atomic_text(progress/'completed_cases.json',json.dumps([str(item['scenario_id']) for item in cases],indent=2))
    if completed!=set(map(str,manifest['assigned_scenario_ids'])):raise RuntimeError('CASE_TRANSACTION_INVALID')
    return cases,stations,timeline,timings

def write_run_complete(run_out: Path,manifest: dict,cases,timeline,requires_timeline: bool):
    assigned=list(map(str,manifest['assigned_scenario_ids']))
    if {str(row['scenario_id']) for row in cases}!=set(assigned) or len(cases)!=len(assigned):raise RuntimeError('METHOD_OUTPUT_INCOMPLETE')
    if requires_timeline and (len(timeline)!=144*len(assigned) or any(len(group)!=144 for _,group in pd.DataFrame(timeline).groupby('scenario_id'))):raise RuntimeError('METHOD_OUTPUT_INCOMPLETE')
    if not requires_timeline and timeline:raise RuntimeError('METHOD_OUTPUT_INCOMPLETE')
    marker={'method_label':manifest['method_label'],'scope':manifest['scope'],'scenario_count':len(assigned),'scenario_ids':assigned,'config_sha256':manifest['config_sha256'],'code_sha':manifest['code_sha'],'global_checkpoint_sha256':manifest['global_checkpoint_sha256'],'local_checkpoint_sha256':manifest['local_checkpoint_sha256'],'local_normalization_sha256':manifest['local_normalization_sha256'],'completed_at_result_schema_version':1}
    _atomic_text(run_out/'RUN_COMPLETE.json',json.dumps(marker,indent=2,sort_keys=True));return marker

def validate_complete_method(run_out: Path,label: str,code_sha: str|None=None):
    manifest_path=run_out/'run_manifest.json';marker_path=run_out/'RUN_COMPLETE.json'
    if not manifest_path.exists() or not marker_path.exists():raise RuntimeError('METHOD_OUTPUT_INCOMPLETE')
    manifest=json.loads(manifest_path.read_text());marker=json.loads(marker_path.read_text());assigned=list(map(str,manifest.get('assigned_scenario_ids',[])))
    expected={'method_label':label,'scope':'VAL_ONLY','scenario_count':len(assigned),'scenario_ids':assigned,'config_sha256':manifest.get('config_sha256'),'code_sha':manifest.get('code_sha'),'global_checkpoint_sha256':manifest.get('global_checkpoint_sha256'),'local_checkpoint_sha256':manifest.get('local_checkpoint_sha256'),'local_normalization_sha256':manifest.get('local_normalization_sha256'),'completed_at_result_schema_version':1}
    if marker!=expected or manifest.get('method_label')!=label or code_sha is not None and manifest.get('code_sha')!=code_sha:raise RuntimeError('SUITE_METHOD_IDENTITY_MISMATCH')
    required=('final_case_metrics.csv','final_station_metrics.csv','selection_timeline.csv','runtime_summary.csv','method_report.json')
    if any(not (run_out/name).exists() for name in required):raise RuntimeError('METHOD_OUTPUT_INCOMPLETE')
    cases=_records(run_out/'final_case_metrics.csv');stations=_records(run_out/'final_station_metrics.csv');timeline=_records(run_out/'selection_timeline.csv')
    if len(cases)!=len(assigned) or {str(row.get('scenario_id')) for row in cases}!=set(assigned) or not stations:raise RuntimeError('METHOD_OUTPUT_INCOMPLETE')
    if manifest.get('strategy')=='frozen_global':
        if timeline:raise RuntimeError('METHOD_OUTPUT_INCOMPLETE')
    elif len(timeline)!=144*len(assigned) or any(len(group)!=144 for _,group in pd.DataFrame(timeline).groupby('scenario_id')):raise RuntimeError('METHOD_OUTPUT_INCOMPLETE')
    return manifest
