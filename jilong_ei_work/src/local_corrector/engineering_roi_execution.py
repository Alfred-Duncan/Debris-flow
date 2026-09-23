"""Manifest, cache, canonical naming, and shard helpers for EngineeringROI."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable
import pandas as pd

CANONICAL = {'frozen_global':'FrozenGlobal','random_all':'RandomAll','random_support':'RandomSupport','engineering_roi':'EngineeringROI','dynamic_only':'DynamicOnly','front_only':'FrontOnly','support_risk_only':'SupportRiskOnly','section_only':'SectionOnly','engineering_roi_no_diversity':'EngineeringROI_NoDiversity'}
CORE_METHODS=('FrozenGlobal','RandomAll_B05','RandomAll_B10','RandomAll_B20','RandomSupport_B05','RandomSupport_B10','RandomSupport_B20','EngineeringROI_B05','EngineeringROI_B10','EngineeringROI_B20')
ABLATION_METHODS=('DynamicOnly_B10','FrontOnly_B10','SupportRiskOnly_B10','SectionOnly_B10','EngineeringROI_NoDiversity_B10')
MANIFEST_KEYS=('method_label','strategy','budget','scope','config_sha256','code_sha','global_checkpoint_provenance','local_checkpoint_provenance','local_checkpoint_update','local_normalization_fit_scope','support_guard_rule','support_guard_version','seed_base','patch_layout_rows','patch_layout_cols','eligible_patch_count')

def canonical_method_label(strategy: str, budget: float | None = None) -> str:
    if strategy not in CANONICAL:raise ValueError('CANONICAL_METHOD_INVALID')
    if strategy=='frozen_global':
        if budget is not None:raise ValueError('FROZEN_GLOBAL_HAS_NO_BUDGET')
        return 'FrozenGlobal'
    if budget not in (.05,.10,.20):raise ValueError('CANONICAL_BUDGET_INVALID')
    return f'{CANONICAL[strategy]}_B{int(round(budget*100)):02d}'

def run_output_path(root: Path, shard_index: int=0, shard_count: int=0) -> Path:
    if shard_count==0:return root
    if not 0<=shard_index<shard_count:raise RuntimeError('INVALID_SHARD_INDEX')
    return root/'shards'/f'shard_{shard_index}'

def partition_rows(rows, shard_index: int=0, shard_count: int=0):
    complete=rows.copy();complete['__global_case_index']=range(len(complete))
    return complete if shard_count==0 else complete.iloc[shard_index::shard_count].reset_index(drop=True)

def _atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text);tmp.replace(path)

def _atomic_csv(path: Path, rows):
    frame=pd.DataFrame(rows);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');frame.to_csv(tmp,index=False);tmp.replace(path)

def create_or_validate_manifest(run_out: Path, expected: dict, resume: bool):
    path=run_out/'run_manifest.json'
    if path.exists():
        actual=json.loads(path.read_text())
        if any(actual.get(key)!=expected.get(key) for key in MANIFEST_KEYS):raise RuntimeError('RESUME_MANIFEST_MISMATCH')
        if not resume:raise RuntimeError('PROGRESS_EXISTS_USE_RESUME')
    else:
        _atomic_text(path,json.dumps(expected,indent=2,sort_keys=True))
    return path

def _records(path: Path):
    if not path.exists():return []
    try:return pd.read_csv(path).to_dict('records')
    except pd.errors.EmptyDataError:return []

def _completed(path: Path):
    return json.loads(path.read_text()) if path.exists() else []

def cached_engineering_method(run_out: Path, manifest: dict, rows, runner: Callable, requires_timeline: bool, resume: bool):
    """Case-atomic cache. A case becomes complete only after all three tables."""
    create_or_validate_manifest(run_out,manifest,resume);progress=run_out/'progress';completed_path=progress/'completed_cases.json';cases_path=progress/'case_metrics.csv';stations_path=progress/'station_metrics.csv';timeline_path=progress/'selection_timeline.csv';timing_path=progress/'timing_parts.jsonl'
    completed=_completed(completed_path);cases=_records(cases_path);stations=_records(stations_path);timeline=_records(timeline_path);timings=[json.loads(line) for line in timing_path.read_text().splitlines() if line] if timing_path.exists() else []
    complete_set=set(map(str,completed));case_set={str(row['scenario_id']) for row in cases};station_set={str(row['scenario_id']) for row in stations}
    if complete_set!=case_set or complete_set!=station_set or len(timings)!=len(completed):raise RuntimeError('CACHE_INCONSISTENT')
    if requires_timeline:
        timeline_counts=pd.DataFrame(timeline).groupby('scenario_id').size().to_dict() if timeline else {}
        if set(map(str,timeline_counts))!=complete_set or any(count!=144 for count in timeline_counts.values()):raise RuntimeError('CACHE_INCONSISTENT')
    for _,row in rows.iterrows():
        scenario=str(row.scenario_id)
        if scenario in complete_set:continue
        one=row.to_frame().T.reset_index(drop=True);new_cases,new_stations,new_timeline,timing=runner(one)
        if len(new_cases)!=1 or str(new_cases[0].get('scenario_id'))!=scenario:raise RuntimeError('CACHE_INCONSISTENT')
        if requires_timeline and len(new_timeline)!=144:raise RuntimeError('SELECTION_TIMELINE_INCOMPLETE')
        cases.extend(new_cases);stations.extend(new_stations);timeline.extend(new_timeline);timings.append(timing)
        _atomic_csv(cases_path,cases);_atomic_csv(stations_path,stations)
        if requires_timeline:_atomic_csv(timeline_path,timeline)
        _atomic_text(timing_path,'\n'.join(json.dumps(item,sort_keys=True) for item in timings)+'\n')
        completed.append(scenario);_atomic_text(completed_path,json.dumps(completed,indent=2))
        complete_set.add(scenario)
    return cases,stations,timeline,timings

def validate_shard_manifests(run_root: Path, shard_count: int):
    manifests=[]
    for index in range(shard_count):
        path=run_root/'shards'/f'shard_{index}'/'run_manifest.json'
        if not path.exists():raise RuntimeError('SHARD_MANIFEST_MISMATCH')
        manifests.append(json.loads(path.read_text()))
    first=manifests[0]
    if any(any(manifest.get(key)!=first.get(key) for key in MANIFEST_KEYS) for manifest in manifests[1:]):raise RuntimeError('SHARD_MANIFEST_MISMATCH')
    return manifests

def validate_shard_coverage(case_rows, expected_ids):
    ids=[str(row['scenario_id']) for row in case_rows]
    if len(ids)!=len(set(ids)):raise RuntimeError('DUPLICATE_SCENARIO_ACROSS_SHARDS')
    if set(ids)!=set(map(str,expected_ids)):raise RuntimeError('INCOMPLETE_VAL_SHARD_COVERAGE')
