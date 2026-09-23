"""Future multi-method EngineeringROI-v1 suite planner and aggregator."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.local_corrector.engineering_roi_execution import ABLATION_METHODS,CORE_METHODS,canonical_method_label

METHOD_SPECS={'FrozenGlobal':('frozen_global',None),'RandomAll_B05':('random_all',.05),'RandomAll_B10':('random_all',.10),'RandomAll_B20':('random_all',.20),'RandomSupport_B05':('random_support',.05),'RandomSupport_B10':('random_support',.10),'RandomSupport_B20':('random_support',.20),'EngineeringROI_B05':('engineering_roi',.05),'EngineeringROI_B10':('engineering_roi',.10),'EngineeringROI_B20':('engineering_roi',.20),'DynamicOnly_B10':('dynamic_only',.10),'FrontOnly_B10':('front_only',.10),'SupportRiskOnly_B10':('support_risk_only',.10),'SectionOnly_B10':('section_only',.10),'EngineeringROI_NoDiversity_B10':('engineering_roi_no_diversity',.10)}

def requested_methods(args):
    if args.methods:return tuple(args.methods)
    if args.core_only:return CORE_METHODS
    if args.ablations_only:return ABLATION_METHODS
    return CORE_METHODS+ABLATION_METHODS

def aggregate(root: Path):
    missing=[label for label in CORE_METHODS+ABLATION_METHODS if not (root/'runs'/label/'method_report.json').exists()]
    if missing:raise RuntimeError('SUITE_INCOMPLETE '+','.join(missing))
    reports=[json.loads((root/'runs'/label/'method_report.json').read_text()) for label in CORE_METHODS+ABLATION_METHODS];manifests=[report['manifest'] for report in reports]
    first=manifests[0]
    if any(manifest['config_sha256']!=first['config_sha256'] or manifest['code_sha']!=first['code_sha'] or manifest['global_checkpoint_provenance']!=first['global_checkpoint_provenance'] or manifest['local_checkpoint_update']!=4000 for manifest in manifests[1:]):raise RuntimeError('SHARD_MANIFEST_MISMATCH')
    case=pd.concat([pd.read_csv(root/'runs'/report['method_label']/'final_case_metrics.csv') for report in reports],ignore_index=True);station=pd.concat([pd.read_csv(root/'runs'/report['method_label']/'final_station_metrics.csv') for report in reports],ignore_index=True);timeline=pd.concat([pd.read_csv(root/'runs'/report['method_label']/'selection_timeline.csv') for report in reports if (root/'runs'/report['method_label']/'selection_timeline.csv').exists()],ignore_index=True)
    summary=pd.DataFrame([{'method':report['method_label'],**report['summary']} for report in reports]);runtime=pd.DataFrame([{'method':report['method_label'],**report['runtime']} for report in reports]);ablation=summary[summary.method.isin(('FrozenGlobal','EngineeringROI_B10')+ABLATION_METHODS)].copy()
    root.mkdir(parents=True,exist_ok=True);summary.to_csv(root/'final_method_summary.csv',index=False);case.to_csv(root/'final_case_metrics.csv',index=False);station.to_csv(root/'final_station_metrics.csv',index=False);timeline.to_csv(root/'selection_timeline.csv',index=False);runtime.to_csv(root/'runtime_summary.csv',index=False);ablation.to_csv(root/'ablation_summary.csv',index=False)
    method_frame=pd.DataFrame({'method':[report['method_label'] for report in reports]});budget_usage=method_frame.merge(runtime[['method','mean_selected_patch_count','mean_active_coverage']],on='method',how='left').fillna(0.);component_columns=['mean_selected_dynamic_rank','mean_selected_front_rank','mean_selected_support_risk_rank','mean_selected_section_rank','mean_selected_base_score','selected_section_patch_count']
    components=timeline.groupby('method')[component_columns].mean().reset_index() if len(timeline) else pd.DataFrame(columns=['method',*component_columns]);components=method_frame.merge(components,on='method',how='left').fillna(0.);budget_usage.to_csv(root/'budget_usage_summary.csv',index=False);components.to_csv(root/'roi_component_summary.csv',index=False)
    (root/'engineering_roi_val_report.json').write_text(json.dumps({'methods':[report['method_label'] for report in reports],'config_sha256':first['config_sha256'],'code_sha':first['code_sha']},indent=2))

def main(args):
    root=ROOT/'results'/'engineering_roi_v1'
    if args.aggregate_only:return aggregate(root)
    if not args.source_code_sha:raise RuntimeError('SOURCE_CODE_SHA_REQUIRED')
    labels=requested_methods(args)
    invalid=[label for label in labels if label not in METHOD_SPECS]
    if invalid:raise RuntimeError('UNKNOWN_SUITE_METHOD '+','.join(invalid))
    for label in labels:
        strategy,budget=METHOD_SPECS[label];command=[sys.executable,str(ROOT/'scripts/evaluate_engineering_roi_v1.py'),'--strategy',strategy,'--output-dir',f'engineering_roi_v1/runs/{label}','--source-code-sha',args.source_code_sha]
        if budget is not None:command+=['--budget',str(budget)]
        subprocess.run(command,check=True)
    if tuple(labels)==CORE_METHODS+ABLATION_METHODS:aggregate(root)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--methods',nargs='+');parser.add_argument('--core-only',action='store_true');parser.add_argument('--ablations-only',action='store_true');parser.add_argument('--aggregate-only',action='store_true');parser.add_argument('--source-code-sha');main(parser.parse_args())
