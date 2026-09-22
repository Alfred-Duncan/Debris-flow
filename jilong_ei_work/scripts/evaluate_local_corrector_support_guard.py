"""VAL-only SupportGuard application-gate evaluation; no training or selection."""
from __future__ import annotations
import argparse, json, math, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.dataset import FrameStore,INPUT,build_features,scenario_rows
from src.global_operator_v2.frame_adapter import static_and_exogenous
from src.global_operator_v2.losses import project_physical
from src.global_operator_v2.metrics import load_transects
from src.global_operator_v2.oracle_refinement import PatchLayout,StreamingMetrics,select_random
from src.local_corrector.model import JilongLocalCorrector
from src.local_corrector.support_guard import apply_support_guard
from src.local_corrector.trainer import apply_learned_correction,apply_momentum_state_guard
from scripts.run_global_v2_oracle_refinement import (append_station,execute_method,load_model,predict,
    station_row,station_summary,summarize)

SEED=20260920
LOWER_NUMERICAL=('trajectory_h_rel_l2','trajectory_momentum_rel_l2','change_region_h_rel_l2','change_region_momentum_rel_l2')
LOWER_ENGINEERING=('false_positive_wet_fraction','wet_overprediction_ratio','mixture_volume_relative_error','debris_front_mae_km','peak_Q_relative_error')
FREEZE_KEYS=('change_region_h_rel_l2','change_region_momentum_rel_l2','mean_wet_iou','false_positive_wet_fraction','mixture_volume_relative_error','debris_front_mae_km','peak_Q_relative_error')

def tensor(x,device): return torch.from_numpy(np.asarray(x,np.float32)).unsqueeze(0).to(device)
def mean(values):
    clean=[float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(clean)) if clean else float('nan')

@torch.no_grad()
def local_execute(label,budget,guarded,rows,layout,global_model,local_model,transform,normalizer,delta,normalization,static,active,route,z0,transects,device):
    """Streaming raw/guarded Local Corrector rollout. Truth remains metrics-only."""
    store=FrameStore(rows);cases=[];stations=[];telemetry=[];count=layout.count_for_budget(budget);guard_seconds=0.;steps=0
    started=time.perf_counter()
    for local_case_index,(_,row) in enumerate(rows.iterrows()):
        case_index=int(row.get('__global_case_index',local_case_index))
        index=int(np.where(store.rows.scenario_id.eq(row.scenario_id))[0][0]);previous,current,_,params,time0,_=store.sample(index,0)
        previous,current,params=tensor(previous,device),tensor(current,device),tensor(params,device)
        metric=StreamingMetrics(active,route,delta.change_threshold);pred_series=[];truth_series=[]
        truth0=tensor(store.frame(row,0),device);append_station(pred_series,0,station_row(truth0,z0,transects));append_station(truth_series,0,station_row(truth0,z0,transects))
        coverage=[];total={'support_cell_count':0,'active_cell_count':0,'blocked_wet_creation_count':0,'blocked_local_change_fraction':[],'steps_with_blocked':0}
        for step in range(144):
            # Teacher tensors below are intentionally only consumed by StreamingMetrics/station reporting.
            truth_next=tensor(store.frame(row,(step+1)*10),device);truth_current=tensor(store.frame(row,step*10),device)
            provisional=predict(global_model,previous,current,static,params,time0+step/144.,transform,normalizer,active)
            selected=select_random(layout,count,SEED+case_index*1000+step)
            features=build_features(previous,current,static,params,torch.tensor([time0+step/144.],device=device),transform,normalizer)
            provisional_t=transform.encode(provisional)
            if count:
                raw_t,_,_=apply_learned_correction(local_model,features,provisional_t,transform.encode(current),selected,layout,
                    delta.scales,normalization['scales'],normalization['bounds'],active,global_model,apply_momentum_guard=not guarded)
            else:
                raw_t=provisional_t
            if guarded:
                raw_physical=project_physical(transform.decode(raw_t),active)
                before=time.perf_counter()
                corrected_t,record=apply_support_guard(provisional_t,raw_t,current,provisional,active,
                    local_corrected_physical=raw_physical)
                corrected_t=apply_momentum_state_guard(corrected_t,global_model)
                guard_seconds+=time.perf_counter()-before
                record|={'method':label,'scenario_id':row.scenario_id,'time_s':(step+1)*10,'budget':f'B{int(budget*100):02d}'}
                telemetry.append(record);total['support_cell_count']+=record['support_cell_count'];total['active_cell_count']+=int(active.sum().item())
                total['blocked_wet_creation_count']+=record['blocked_wet_creation_count'];total['blocked_local_change_fraction'].append(record['blocked_local_change_fraction'])
                total['steps_with_blocked']+=int(record['blocked_wet_creation_count']>0)
            corrected=project_physical(transform.decode(corrected_t if guarded else raw_t),active)
            metric.add(corrected,truth_current,truth_next,transform);append_station(pred_series,(step+1)*10,station_row(corrected,z0,transects));append_station(truth_series,(step+1)*10,station_row(truth_next,z0,transects))
            coverage.append(layout.selected_active_fraction(selected));previous,current=current,corrected;steps+=1
        result=metric.result()|{'method':label,'scenario_id':row.scenario_id,'budget_fraction':budget,'selected_patch_count':count,'active_cell_coverage_fraction':float(np.mean(coverage))};cases.append(result)
        if guarded:
            telemetry_case={'method':label,'scenario_id':row.scenario_id,'budget':f'B{int(budget*100):02d}',
                'mean_support_fraction':total['support_cell_count']/max(total['active_cell_count'],1),
                'mean_blocked_new_wet_cells_per_step':total['blocked_wet_creation_count']/144.,
                'total_blocked_new_wet_cells':total['blocked_wet_creation_count'],
                'fraction_steps_with_blocked_new_wet':total['steps_with_blocked']/144.,
                'mean_blocked_local_change_fraction':mean(total['blocked_local_change_fraction'])}
            telemetry.append(telemetry_case)
        for item in station_summary(pred_series,truth_series,transects):stations.append(item|{'method':label,'scenario_id':row.scenario_id})
        print(f'{label} {row.scenario_id} complete',flush=True)
    return cases,stations,telemetry,{'method':label,'runtime_seconds':time.perf_counter()-started,'guard_seconds':guard_seconds,'steps':steps,
        'guard_overhead_ms_per_step':1000.*guard_seconds/max(steps,1)}

def compare(raw,guarded,budget):
    names=('trajectory_h_rel_l2','trajectory_momentum_rel_l2','change_region_h_rel_l2','change_region_momentum_rel_l2','mean_wet_iou','newly_wet_precision','newly_wet_recall','false_positive_wet_fraction','wet_overprediction_ratio','mixture_volume_relative_error','debris_front_mae_km','arrival_MAE_s','peak_Q_relative_error','peak_stage_absolute_error','wet_width_relative_error')
    return [{'budget':budget,'metric':key,'raw_local':raw.get(key,float('nan')),'support_guard':guarded.get(key,float('nan')),
             'absolute_difference':guarded.get(key,float('nan'))-raw.get(key,float('nan')),
             'relative_difference':(guarded.get(key,float('nan'))-raw.get(key,float('nan')))/abs(raw.get(key,float('nan'))) if abs(raw.get(key,float('nan')))>1e-12 else float('nan')} for key in names]

def preservation_recovery(frozen,raw,guarded,budget):
    rows=[]
    for key in LOWER_NUMERICAL:
        improvement=frozen[key]-raw[key];preservation=(frozen[key]-guarded[key])/improvement if improvement>0 else float('nan')
        rows.append({'budget':budget,'metric':key,'kind':'numerical_preservation','frozen':frozen[key],'raw_local':raw[key],'support_guard':guarded[key],'ratio':preservation})
    for key in LOWER_ENGINEERING:
        degradation=raw[key]-frozen[key];recovery=(raw[key]-guarded[key])/degradation if degradation>0 else float('nan')
        rows.append({'budget':budget,'metric':key,'kind':'engineering_recovery','frozen':frozen[key],'raw_local':raw[key],'support_guard':guarded[key],'ratio':recovery})
    if raw['mean_wet_iou']<frozen['mean_wet_iou']:
        recovery=(guarded['mean_wet_iou']-raw['mean_wet_iou'])/(frozen['mean_wet_iou']-raw['mean_wet_iou'])
    else: recovery=float('nan')
    rows.append({'budget':budget,'metric':'mean_wet_iou','kind':'engineering_recovery','frozen':frozen['mean_wet_iou'],'raw_local':raw['mean_wet_iou'],'support_guard':guarded['mean_wet_iou'],'ratio':recovery})
    return rows

def summarize_guard_telemetry(records):
    """Collapse per-step telemetry without retaining any trajectory tensor."""
    frame=pd.DataFrame(records)
    if frame.empty or 'time_s' not in frame:return []
    steps=frame[frame.time_s.notna()]
    output=[]
    for method,group in steps.groupby('method'):
        total=float(group.blocked_wet_creation_count.sum());count=len(group)
        output.append({'method':method,'scenario_id':'ALL_VAL','time_s':float('nan'),'budget':str(group.budget.iloc[0]),
            'mean_support_fraction':float(group.support_fraction.mean()),'mean_blocked_new_wet_cells_per_step':total/max(count,1),
            'total_blocked_new_wet_cells':int(total),'fraction_steps_with_blocked_new_wet':float((group.blocked_wet_creation_count>0).mean()),
            'mean_blocked_local_change_fraction':float(group.blocked_local_change_fraction.mean())})
    return output

def classify(frozen,methods,preservation):
    strong=True;improved_support=False
    for budget in ('B10','B20'):
        raw,guarded=methods[f'LearnedRandom_{budget}'],methods[f'LearnedSupportGuard_{budget}']
        support_ok=(guarded['false_positive_wet_fraction']<=raw['false_positive_wet_fraction'] and guarded['wet_overprediction_ratio']<=raw['wet_overprediction_ratio'])
        improved_support|=support_ok
        med=float(np.nanmedian([r['ratio'] for r in preservation if r['budget']==budget and r['kind']=='numerical_preservation']))
        strong &= support_ok and guarded['mixture_volume_relative_error']<=raw['mixture_volume_relative_error'] and (guarded['debris_front_mae_km']<=raw['debris_front_mae_km'] or guarded['peak_Q_relative_error']<=raw['peak_Q_relative_error']) and med>=.70
    effect='STRONG' if strong else ('PARTIAL' if improved_support else 'FAIL')
    freeze=effect in {'STRONG','PARTIAL'} and any(sum((methods[f'LearnedSupportGuard_{budget}'][key]>frozen[key]) if key=='mean_wet_iou' else (methods[f'LearnedSupportGuard_{budget}'][key]<frozen[key]) for key in FREEZE_KEYS)>=4 for budget in ('B10','B20'))
    return effect,'YES' if freeze else 'NO'

def write_outputs(out,methods,all_cases,all_stations,telemetry,timings,checkpoint):
    preservation=[];comparison=[]
    for code in ('B10','B20'):
        raw,guard=methods[f'LearnedRandom_{code}'],methods[f'LearnedSupportGuard_{code}'];comparison+=compare(raw,guard,code);preservation+=preservation_recovery(methods['FrozenGlobal'],raw,guard,code)
    effect,ready=classify(methods['FrozenGlobal'],methods,preservation);telemetry_summary=summarize_guard_telemetry(telemetry);telemetry=list(telemetry)+telemetry_summary
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([{'method':name,**data} for name,data in methods.items()]).to_csv(out/'final_method_summary.csv',index=False);pd.DataFrame(all_cases).to_csv(out/'final_case_metrics.csv',index=False);pd.DataFrame(all_stations).to_csv(out/'final_station_metrics.csv',index=False);pd.DataFrame(telemetry).to_csv(out/'support_guard_telemetry.csv',index=False);pd.DataFrame(comparison).to_csv(out/'raw_vs_guarded.csv',index=False);pd.DataFrame(preservation).to_csv(out/'preservation_recovery.csv',index=False)
    report={'baseline_sha':'bdcbb22ace3daa3655ef0e502910b3a325828f76','local_checkpoint':'best.pt','local_checkpoint_update':checkpoint['update'],'scope':'VAL_ONLY','support_rule':'active AND (current_h >= 0.05 OR global_provisional_h >= 0.05)','methods':methods,'support_guard_telemetry':telemetry_summary,'runtime':timings,'SUPPORT_GUARD_EFFECT':effect,'LOCAL_CORRECTOR_READY_TO_FREEZE':ready}
    (out/'support_guard_report.json').write_text(json.dumps(report,indent=2,allow_nan=True));return report

def merge_shards(args):
    out=ROOT/'results'/args.output_dir; shard_root=out/'shards';parts=[]
    for index in range(args.shard_count):
        path=shard_root/f'shard_{index}'
        if not (path/'final_case_metrics.csv').exists():raise RuntimeError(f'MISSING_SHARD_{index}')
        parts.append(path)
    cases=pd.concat([pd.read_csv(path/'final_case_metrics.csv') for path in parts],ignore_index=True);stations=pd.concat([pd.read_csv(path/'final_station_metrics.csv') for path in parts],ignore_index=True)
    telemetry=pd.concat([pd.read_csv(path/'support_guard_telemetry.csv') for path in parts],ignore_index=True).to_dict('records')
    reports=[json.loads((path/'support_guard_report.json').read_text()) for path in parts];methods={}
    for method in cases.method.unique():methods[method]=summarize(cases[cases.method.eq(method)].to_dict('records'),stations[stations.method.eq(method)].to_dict('records'))
    timings=[]
    for method in methods:
        group=[item for report in reports for item in report['runtime'] if item['method']==method]
        timings.append({'method':method,'worker_sum_runtime_seconds':sum(float(item['runtime_seconds']) for item in group),
            'guard_overhead_ms_per_step':sum(float(item.get('guard_seconds',0.)) for item in group)*1000/max(sum(int(item.get('steps',0)) for item in group),1)})
    report=write_outputs(out,methods,cases.to_dict('records'),stations.to_dict('records'),telemetry,timings,{'update':4000})
    (ROOT/'reports/LOCAL_CORRECTOR_SUPPORT_GUARD.json').write_text(json.dumps(report,indent=2,allow_nan=True));print(json.dumps({'status':'PASS','merged_shards':args.shard_count,'SUPPORT_GUARD_EFFECT':report['SUPPORT_GUARD_EFFECT']},indent=2))

def main(args):
    if args.merge_shards:return merge_shards(args)
    if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED')
    device=torch.device('cuda');rows=scenario_rows('VAL')
    if len(rows)!=20 or not set(rows['split']).issubset({'VAL'}) or rows.scenario_id.str.contains('TEST|H0|HOLDOUT',case=False).any():raise RuntimeError('VAL_SCOPE_REQUIRED')
    rows=rows.copy();rows['__global_case_index']=np.arange(len(rows))
    if args.limit: rows=rows.iloc[:args.limit].reset_index(drop=True)
    if args.shard_count:
        if not 0<=args.shard_index<args.shard_count:raise RuntimeError('INVALID_SHARD_INDEX')
        rows=rows.iloc[args.shard_index::args.shard_count].reset_index(drop=True)
    global_model,transform,normalizer,delta,_=load_model(device);static_np,_=static_and_exogenous(INPUT);static=tensor(static_np,device);active=static[:,1:2];layout=PatchLayout(static_np[1])
    model_dir=ROOT/'models/local_corrector_v1_1';normalization=json.loads((model_dir/'correction_normalization.json').read_text())
    checkpoint=torch.load(model_dir/'best.pt',map_location=device,weights_only=False)
    if checkpoint.get('update')!=4000:raise RuntimeError('LOCAL_CHECKPOINT_MUST_BE_BEST_4000')
    if normalization.get('fit_scope')!='TRAIN_ONLY':raise RuntimeError('LOCAL_NORMALIZATION_NOT_TRAIN_ONLY')
    local=JilongLocalCorrector(47).to(device);local.load_state_dict(checkpoint['model']);local.eval()
    with np.load(INPUT) as data:route=np.asarray(data['route_chainage_m'],np.float32)
    z0=static_np[0];transects=load_transects(ROOT/'data/downloads/park_v2/inputs/upper30h_transects.json')
    methods={};all_cases=[];all_stations=[];telemetry=[];timings=[]
    t=time.perf_counter();frozen_cases,frozen_stations,_,_=execute_method('FrozenGlobal',0.,'FrozenGlobal',rows,layout,global_model,transform,normalizer,delta,static,active,route,z0,transects,device);timings.append({'method':'FrozenGlobal','runtime_seconds':time.perf_counter()-t,'guard_overhead_ms_per_step':0.});methods['FrozenGlobal']=summarize(frozen_cases,frozen_stations);all_cases+=frozen_cases;all_stations+=frozen_stations
    for budget in (.10,.20):
        code=f'B{int(budget*100):02d}';raw=f'LearnedRandom_{code}';guard=f'LearnedSupportGuard_{code}'
        cases,stations,_,timing=local_execute(raw,budget,False,rows,layout,global_model,local,transform,normalizer,delta,normalization,static,active,route,z0,transects,device);methods[raw]=summarize(cases,stations);all_cases+=cases;all_stations+=stations;timings.append(timing)
        cases,stations,tel,timing=local_execute(guard,budget,True,rows,layout,global_model,local,transform,normalizer,delta,normalization,static,active,route,z0,transects,device);methods[guard]=summarize(cases,stations);all_cases+=cases;all_stations+=stations;telemetry+=tel;timings.append(timing)
        perfect=f'RandomPerfect_{code}';t=time.perf_counter();cases,stations,_,_=execute_method(perfect,budget,'RandomPerfect',rows,layout,global_model,transform,normalizer,delta,static,active,route,z0,transects,device,case_index_column='__global_case_index');timings.append({'method':perfect,'runtime_seconds':time.perf_counter()-t,'guard_overhead_ms_per_step':0.});methods[perfect]=summarize(cases,stations);all_cases+=cases;all_stations+=stations
    out=ROOT/'results'/args.output_dir
    report=write_outputs(out,methods,all_cases,all_stations,telemetry,timings,checkpoint)
    if not args.shard_count:(ROOT/'reports/LOCAL_CORRECTOR_SUPPORT_GUARD.json').write_text(json.dumps(report,indent=2,allow_nan=True))
    print(json.dumps({'status':'PASS','SUPPORT_GUARD_EFFECT':report['SUPPORT_GUARD_EFFECT'],'LOCAL_CORRECTOR_READY_TO_FREEZE':report['LOCAL_CORRECTOR_READY_TO_FREEZE'],'output':str(out)},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);parser.add_argument('--shard-index',type=int,default=0);parser.add_argument('--shard-count',type=int,default=0);parser.add_argument('--output-dir',default='local_corrector_support_guard');parser.add_argument('--merge-shards',action='store_true');main(parser.parse_args())
