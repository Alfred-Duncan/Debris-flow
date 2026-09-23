"""Future multi-method EngineeringROI-v1 suite planner and deterministic aggregator."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.local_corrector.engineering_roi_execution import ABLATION_METHODS,CORE_METHODS,validate_complete_method

METHOD_SPECS={'FrozenGlobal':('frozen_global',None),'RandomAll_B05':('random_all',.05),'RandomAll_B10':('random_all',.10),'RandomAll_B20':('random_all',.20),'RandomSupport_B05':('random_support',.05),'RandomSupport_B10':('random_support',.10),'RandomSupport_B20':('random_support',.20),'EngineeringROI_B05':('engineering_roi',.05),'EngineeringROI_B10':('engineering_roi',.10),'EngineeringROI_B20':('engineering_roi',.20),'DynamicOnly_B10':('dynamic_only',.10),'FrontOnly_B10':('front_only',.10),'SupportRiskOnly_B10':('support_risk_only',.10),'SectionOnly_B10':('section_only',.10),'EngineeringROI_NoDiversity_B10':('engineering_roi_no_diversity',.10)}
ALL_METHODS=CORE_METHODS+ABLATION_METHODS
IDENTITY_KEYS=('config_sha256','engineering_roi_config_sha256','code_sha','global_checkpoint_sha256','local_checkpoint_sha256','local_normalization_sha256','support_guard_version','full_scenario_ids','patch_layout_rows','patch_layout_cols','eligible_patch_count')

def requested_methods(args):
    if args.methods:return tuple(args.methods)
    if args.core_only:return CORE_METHODS
    if args.ablations_only:return ABLATION_METHODS
    return ALL_METHODS

def suite_run_plan(root: Path,labels,source_code_sha: str,resume: bool):
    """Return (label, fresh|resume|skip); this is deliberately subprocess-free."""
    plan=[]
    for label in labels:
        run=root/'runs'/label
        if not run.exists():plan.append((label,'fresh'));continue
        if (run/'RUN_COMPLETE.json').exists():
            validate_complete_method(run,label,source_code_sha);plan.append((label,'skip'));continue
        manifest_path=run/'run_manifest.json'
        if not resume:raise RuntimeError('PROGRESS_EXISTS_USE_RESUME')
        if not manifest_path.exists():raise RuntimeError('SUITE_METHOD_IDENTITY_MISMATCH')
        manifest=json.loads(manifest_path.read_text())
        if manifest.get('method_label')!=label or manifest.get('code_sha')!=source_code_sha or not (run/'progress').exists():raise RuntimeError('SUITE_METHOD_IDENTITY_MISMATCH')
        plan.append((label,'resume'))
    return plan

def _read(path):
    try:return pd.read_csv(path)
    except pd.errors.EmptyDataError:return pd.DataFrame()

def _sort(frame,columns):
    present=[column for column in columns if column in frame.columns]
    return frame.sort_values(present,kind='stable').reset_index(drop=True) if present else frame

def aggregate(root: Path):
    missing=[label for label in ALL_METHODS if not (root/'runs'/label/'method_report.json').exists()]
    if missing:raise RuntimeError('SUITE_INCOMPLETE '+','.join(missing))
    manifests=[];reports=[]
    for label in ALL_METHODS:
        run=root/'runs'/label;manifests.append(validate_complete_method(run,label));reports.append(json.loads((run/'method_report.json').read_text()))
    first=manifests[0]
    if any(any(manifest.get(key)!=first.get(key) for key in IDENTITY_KEYS) for manifest in manifests[1:]):raise RuntimeError('SUITE_METHOD_IDENTITY_MISMATCH')
    case=_sort(pd.concat([_read(root/'runs'/label/'final_case_metrics.csv') for label in ALL_METHODS],ignore_index=True),['method','scenario_id'])
    station=_sort(pd.concat([_read(root/'runs'/label/'final_station_metrics.csv') for label in ALL_METHODS],ignore_index=True),['method','scenario_id','station'])
    timeline_frames=[_read(root/'runs'/label/'selection_timeline.csv') for label in ALL_METHODS];timeline=_sort(pd.concat([frame for frame in timeline_frames if not frame.empty],ignore_index=True) if any(not frame.empty for frame in timeline_frames) else pd.DataFrame(),['method','scenario_id','time_s'])
    order={label:index for index,label in enumerate(ALL_METHODS)}
    summary=pd.DataFrame([{'method':report['method_label'],**report['summary']} for report in reports]);runtime=pd.DataFrame([{'method':report['method_label'],**report['runtime']} for report in reports]);summary['_order']=summary.method.map(order);runtime['_order']=runtime.method.map(order);summary=summary.sort_values('_order').drop(columns='_order');runtime=runtime.sort_values('_order').drop(columns='_order');ablation=summary[summary.method.isin(('FrozenGlobal','EngineeringROI_B10')+ABLATION_METHODS)].copy()
    root.mkdir(parents=True,exist_ok=True);summary.to_csv(root/'final_method_summary.csv',index=False);case.to_csv(root/'final_case_metrics.csv',index=False);station.to_csv(root/'final_station_metrics.csv',index=False);timeline.to_csv(root/'selection_timeline.csv',index=False);runtime.to_csv(root/'runtime_summary.csv',index=False);ablation.to_csv(root/'ablation_summary.csv',index=False)
    method_frame=pd.DataFrame({'method':ALL_METHODS});budget=method_frame.merge(runtime[['method','mean_selected_patch_count','mean_active_coverage']],on='method',how='left').fillna(0.);component_columns=['mean_selected_dynamic_rank','mean_selected_front_rank','mean_selected_support_risk_rank','mean_selected_section_rank','mean_selected_base_score','selected_section_patch_count'];components=timeline.groupby('method')[component_columns].mean().reset_index() if len(timeline) else pd.DataFrame(columns=['method',*component_columns]);method_frame.merge(components,on='method',how='left').to_csv(root/'roi_component_summary.csv',index=False);budget.to_csv(root/'budget_usage_summary.csv',index=False)
    report={'methods':list(ALL_METHODS),'formal_code_sha':first['code_sha'],'config_sha256':first['config_sha256'],'global_checkpoint_sha256':first['global_checkpoint_sha256'],'local_checkpoint_sha256':first['local_checkpoint_sha256'],'local_normalization_sha256':first['local_normalization_sha256'],'support_guard_version':first['support_guard_version'],'scientific_definition':'EngineeringROI-v1'};(root/'engineering_roi_val_report.json').write_text(json.dumps(report,indent=2,sort_keys=True))

def main(args):
    root=ROOT/'results'/'engineering_roi_v1'
    if args.aggregate_only:return aggregate(root)
    if not args.source_code_sha:raise RuntimeError('SOURCE_CODE_SHA_REQUIRED')
    labels=requested_methods(args);invalid=[label for label in labels if label not in METHOD_SPECS]
    if invalid:raise RuntimeError('UNKNOWN_SUITE_METHOD '+','.join(invalid))
    for label,state in suite_run_plan(root,labels,args.source_code_sha,args.resume):
        if state=='skip':print('SKIP_COMPLETE '+label);continue
        strategy,budget=METHOD_SPECS[label];command=[sys.executable,str(ROOT/'scripts/evaluate_engineering_roi_v1.py'),'--strategy',strategy,'--output-dir',f'engineering_roi_v1/runs/{label}','--source-code-sha',args.source_code_sha]
        if budget is not None:command+=['--budget',str(budget)]
        if state=='resume':command+=['--resume']
        subprocess.run(command,check=True)
    if tuple(labels)==ALL_METHODS:aggregate(root)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--methods',nargs='+');parser.add_argument('--core-only',action='store_true');parser.add_argument('--ablations-only',action='store_true');parser.add_argument('--aggregate-only',action='store_true');parser.add_argument('--resume',action='store_true');parser.add_argument('--source-code-sha');main(parser.parse_args())
