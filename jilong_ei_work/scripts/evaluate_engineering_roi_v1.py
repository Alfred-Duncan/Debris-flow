"""Future VAL-only evaluator for frozen deterministic EngineeringROI-v1.

This program is intentionally supplied for review and a later authorized run.
It contains no training path and no non-VAL scope option.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.dataset import FrameStore,INPUT,build_features,scenario_rows
from src.global_operator_v2.frame_adapter import static_and_exogenous
from src.global_operator_v2.losses import project_physical
from src.global_operator_v2.metrics import load_transects
from src.global_operator_v2.oracle_refinement import PatchLayout,StreamingMetrics
from src.local_corrector.engineering_roi import (STRATEGIES,build_section_mask,load_engineering_roi_config,select_engineering_roi)
from src.local_corrector.model import JilongLocalCorrector
from src.local_corrector.support_guard import apply_support_guard
from src.local_corrector.trainer import apply_learned_correction,apply_momentum_state_guard
from scripts.evaluate_local_corrector_support_guard import cached_method,combine_timings
from scripts.run_global_v2_oracle_refinement import append_station,load_model,predict,station_row,station_summary,summarize

SEED=20260920

def tensor(value, device): return torch.from_numpy(np.asarray(value,np.float32)).unsqueeze(0).to(device)

def code_sha():
    for directory in (ROOT,*ROOT.parents):
        if (directory/'.git').exists():
            try:return subprocess.check_output(['git','rev-parse','HEAD'],cwd=directory,encoding='utf-8').strip()
            except Exception:pass
    source_commit=ROOT/'SOURCE_CODE_COMMIT.txt'
    return source_commit.read_text().strip() if source_commit.exists() else 'UNKNOWN'

@torch.no_grad()
def engineering_roi_execute(label, strategy, budget, rows, layout, global_model, local_model, transform, normalizer,
                             delta, normalization, static, active, route, section_mask, z0, transects, device, config):
    """One or more complete cases; target frames are read only after prediction."""
    store=FrameStore(rows);cases=[];stations=[];timeline=[];started=time.perf_counter();roi_seconds=0.;local_seconds=0.;guard_seconds=0.;steps=0
    for local_case_index,(_,row) in enumerate(rows.iterrows()):
        case_index=int(row.get('__global_case_index',local_case_index));index=int(np.where(store.rows.scenario_id.eq(row.scenario_id))[0][0])
        previous,current,_,params,time0,_=store.sample(index,0);previous,current,params=tensor(previous,device),tensor(current,device),tensor(params,device)
        metric=StreamingMetrics(active,route,delta.change_threshold);predicted_station=[];teacher_station=[]
        append_station(predicted_station,0,station_row(current,z0,transects));append_station(teacher_station,0,station_row(current,z0,transects));coverage=[];selected_counts=[]
        for step in range(144):
            provisional=predict(global_model,previous,current,static,params,time0+step/144.,transform,normalizer,active)
            encoded_current,encoded_provisional=transform.encode(current),transform.encode(provisional)
            started_roi=time.perf_counter()
            selected,selection=select_engineering_roi(current,provisional,encoded_current,encoded_provisional,active,route,section_mask,delta.scales,layout,config,budget,strategy,SEED+case_index*1000+step)
            roi_seconds+=time.perf_counter()-started_roi
            features=build_features(previous,current,static,params,torch.tensor([time0+step/144.],device=device),transform,normalizer)
            if selected:
                started_local=time.perf_counter()
                raw_t,_,_=apply_learned_correction(local_model,features,encoded_provisional,encoded_current,selected,layout,delta.scales,normalization['scales'],normalization['bounds'],active,global_model,apply_momentum_guard=False)
                local_seconds+=time.perf_counter()-started_local
            else: raw_t=encoded_provisional
            raw_physical=project_physical(transform.decode(raw_t),active);started_guard=time.perf_counter()
            corrected_t,guard=apply_support_guard(encoded_provisional,raw_t,current,provisional,active,local_corrected_physical=raw_physical)
            corrected=project_physical(transform.decode(apply_momentum_state_guard(corrected_t,global_model)),active);guard_seconds+=time.perf_counter()-started_guard
            # The selection and corrected prediction are complete above.  Only now
            # are target frames materialized for production metrics.
            truth_current=tensor(store.frame(row,step*10),device);truth_next=tensor(store.frame(row,(step+1)*10),device)
            metric.add(corrected,truth_current,truth_next,transform);append_station(predicted_station,(step+1)*10,station_row(corrected,z0,transects));append_station(teacher_station,(step+1)*10,station_row(truth_next,z0,transects))
            selection|={'method':label,'scenario_id':row.scenario_id,'time_s':(step+1)*10,'budget':f'B{int(budget*100):02d}','selected_patch_ids':';'.join(map(str,[patch.patch_id for patch in selected])),'roi_scoring_runtime_ms':1000.*roi_seconds/max(steps+1,1)}
            timeline.append(selection);coverage.append(layout.selected_active_fraction(selected));selected_counts.append(len(selected));previous,current=current,corrected;steps+=1
        cases.append(metric.result()|{'method':label,'scenario_id':row.scenario_id,'budget_fraction':budget,'budget_max_count':layout.count_for_budget(budget),'selected_patch_count':float(np.mean(selected_counts)),'active_cell_coverage_fraction':float(np.mean(coverage))})
        for item in station_summary(predicted_station,teacher_station,transects):stations.append(item|{'method':label,'scenario_id':row.scenario_id})
        print(f'{label} {row.scenario_id} complete',flush=True)
    return cases,stations,timeline,{'method':label,'runtime_seconds':time.perf_counter()-started,'roi_seconds':roi_seconds,'local_seconds':local_seconds,'guard_seconds':guard_seconds,'steps':steps,
        'roi_scoring_ms_per_step':1000.*roi_seconds/max(steps,1),'local_correction_ms_per_step':1000.*local_seconds/max(steps,1),'support_guard_ms_per_step':1000.*guard_seconds/max(steps,1)}

def write_future_outputs(out, label, cases, stations, timeline, timing, config_sha):
    out.mkdir(parents=True,exist_ok=True);summary=summarize(cases,stations);frame=pd.DataFrame(timeline)
    pd.DataFrame([{'method':label,**summary}]).to_csv(out/'final_method_summary.csv',index=False);pd.DataFrame(cases).to_csv(out/'final_case_metrics.csv',index=False);pd.DataFrame(stations).to_csv(out/'final_station_metrics.csv',index=False);frame.to_csv(out/'selection_timeline.csv',index=False)
    components=('mean_selected_dynamic_rank','mean_selected_front_rank','mean_selected_support_risk_rank','mean_selected_section_rank','mean_selected_base_score')
    pd.DataFrame([{'method':label,**{name:float(frame[name].mean()) for name in components}}]).to_csv(out/'roi_component_summary.csv',index=False)
    pd.DataFrame([{'method':label,'budget_max_count':int(frame.budget_max_count.iloc[0]),'mean_selected_count':float(frame.selected_count.mean()),'mean_active_area_coverage':float(frame.actual_active_fraction.mean())}]).to_csv(out/'budget_usage_summary.csv',index=False)
    pd.DataFrame([{'method':label,'strategy':label.split('_B')[0]}]).to_csv(out/'ablation_summary.csv',index=False);pd.DataFrame([timing]).to_csv(out/'runtime_summary.csv',index=False)
    report={'version':'EngineeringROI-v1','scope':'VAL_ONLY','config_sha256':config_sha,'code_sha':code_sha(),'global_checkpoint':'models/global_operator_v2/best.pt','local_checkpoint_update':4000,'support_guard':'current_or_global_provisional_wet','method':label,'summary':summary,'runtime':timing}
    (out/'engineering_roi_val_report.json').write_text(json.dumps(report,indent=2,allow_nan=True));return report

def merge_shards(args):
    root=ROOT/'results'/args.output_dir;parts=[root/'shards'/f'shard_{index}' for index in range(args.shard_count)]
    if any(not (part/'final_case_metrics.csv').exists() for part in parts):raise RuntimeError('ENGINEERING_ROI_SHARD_MISSING')
    cases=pd.concat([pd.read_csv(part/'final_case_metrics.csv') for part in parts],ignore_index=True);stations=pd.concat([pd.read_csv(part/'final_station_metrics.csv') for part in parts],ignore_index=True);timeline=pd.concat([pd.read_csv(part/'selection_timeline.csv') for part in parts],ignore_index=True)
    reports=[json.loads((part/'engineering_roi_val_report.json').read_text()) for part in parts];label=str(cases.method.iloc[0]);timing={'method':label,'worker_sum_runtime_seconds':sum(float(report['runtime']['runtime_seconds']) for report in reports),'roi_scoring_ms_per_step':sum(float(report['runtime']['roi_seconds']) for report in reports)*1000/max(sum(int(report['runtime']['steps']) for report in reports),1),'local_correction_ms_per_step':sum(float(report['runtime']['local_seconds']) for report in reports)*1000/max(sum(int(report['runtime']['steps']) for report in reports),1),'support_guard_ms_per_step':sum(float(report['runtime']['guard_seconds']) for report in reports)*1000/max(sum(int(report['runtime']['steps']) for report in reports),1)}
    report=write_future_outputs(root,label,cases.to_dict('records'),stations.to_dict('records'),timeline.to_dict('records'),timing,reports[0]['config_sha256'])
    (ROOT/'reports/ENGINEERING_ROI_V1_VAL.json').write_text(json.dumps(report,indent=2,allow_nan=True))

def main(args):
    if args.merge_shards:return merge_shards(args)
    if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED')
    config,config_sha=load_engineering_roi_config(ROOT/'configs/engineering_roi_v1.json');rows=scenario_rows('VAL')
    if len(rows)!=20 or not set(rows['split']).issubset({'VAL'}):raise RuntimeError('VAL_SCOPE_REQUIRED')
    rows=rows.copy();rows['__global_case_index']=np.arange(len(rows))
    if args.limit:rows=rows.iloc[:args.limit].reset_index(drop=True)
    if args.shard_count:
        if not 0<=args.shard_index<args.shard_count:raise RuntimeError('INVALID_SHARD_INDEX')
        rows=rows.iloc[args.shard_index::args.shard_count].reset_index(drop=True)
    out=ROOT/'results'/args.output_dir
    if not args.resume and (out/'progress').exists():raise RuntimeError('PROGRESS_EXISTS_USE_RESUME')
    device=torch.device('cuda');global_model,transform,normalizer,delta,_=load_model(device);static_np,_=static_and_exogenous(INPUT);static=tensor(static_np,device);active=static[:,1:2];layout=PatchLayout(static_np[1])
    model_dir=ROOT/'models/local_corrector_v1_1';normalization=json.loads((model_dir/'correction_normalization.json').read_text());checkpoint=torch.load(model_dir/'best.pt',map_location=device,weights_only=False)
    if checkpoint.get('update')!=4000:raise RuntimeError('LOCAL_CHECKPOINT_MUST_BE_BEST_4000')
    local=JilongLocalCorrector(47).to(device);local.load_state_dict(checkpoint['model']);local.eval()
    with np.load(INPUT) as data:route=np.asarray(data['route_chainage_m'],np.float32)
    transects=load_transects(ROOT/'data/downloads/park_v2/inputs/upper30h_transects.json');section_mask=build_section_mask(transects,static_np.shape[-2:]);z0=static_np[0];label=f'{args.strategy}_B{int(args.budget*100):02d}'
    runner=lambda one: engineering_roi_execute(label,args.strategy,args.budget,one,layout,global_model,local,transform,normalizer,delta,normalization,static,active,route,section_mask,z0,transects,device,config)
    cases,stations,timeline,parts=cached_method(label,rows,out/'progress',runner);timing=combine_timings(label,parts)|{'roi_scoring_ms_per_step':sum(float(item.get('roi_seconds',0.)) for item in parts)*1000/max(sum(int(item.get('steps',0)) for item in parts),1),'local_correction_ms_per_step':sum(float(item.get('local_seconds',0.)) for item in parts)*1000/max(sum(int(item.get('steps',0)) for item in parts),1),'support_guard_ms_per_step':sum(float(item.get('guard_seconds',0.)) for item in parts)*1000/max(sum(int(item.get('steps',0)) for item in parts),1)}
    report=write_future_outputs(out,label,cases,stations,timeline,timing,config_sha)
    if not args.shard_count:(ROOT/'reports/ENGINEERING_ROI_V1_VAL.json').write_text(json.dumps(report,indent=2,allow_nan=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--strategy',choices=STRATEGIES,default='engineering_roi');parser.add_argument('--budget',choices=(.05,.10,.20),type=float,default=.10);parser.add_argument('--limit',type=int);parser.add_argument('--shard-index',type=int,default=0);parser.add_argument('--shard-count',type=int,default=0);parser.add_argument('--merge-shards',action='store_true');parser.add_argument('--output-dir',default='engineering_roi_v1');parser.add_argument('--resume',action='store_true');main(parser.parse_args())
