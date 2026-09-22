"""Run the VAL-only OraclePerfect local-refinement feasibility study.

OraclePerfect and RandomPerfect use target fields for replacement and are
non-deployable diagnostics.  The corrected state is fed back autoregressively.
No checkpoint, training input, or physics output is written by this script.
"""
from __future__ import annotations
import argparse, csv, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.dataset import FrameStore, scenario_rows, feature_names, build_features, INPUT
from src.global_operator_v2.frame_adapter import static_and_exogenous
from src.global_operator_v2.losses import project_physical
from src.global_operator_v2.metrics import station_metrics, arrival_from_series, load_transects
from src.global_operator_v2.model import JilongGlobalOperatorV2
from src.global_operator_v2.oracle_refinement import (PatchLayout, StreamingMetrics, STATE_NAMES, concentration_fractions,
    correct_cores, oracle_patch_scores, select_oracle, select_random)
from src.global_operator_v2.trainer import restore_checkpoint
from src.global_operator_v2.transforms import PhysicalTransform, FeatureNormalizer, DeltaNormalization

FORMAL=ROOT/'models/global_operator_v2'; OUT=ROOT/'results/oracle_refinement_v2'; REPORT=ROOT/'reports/ORACLE_REFINEMENT_FEASIBILITY_V2.json'
BASELINE_KEYS=('trajectory_h_rel_l2','trajectory_momentum_rel_l2','mean_wet_iou','mixture_volume_relative_error','debris_front_mae_km')
DISCLAIMER='OraclePerfect uses ground-truth target fields for both patch ranking and perfect local replacement. It is an upper-bound feasibility diagnostic only. RandomPerfect also uses truth replacement and is non-deployable.'

def finite_mean(values):
    values=[float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.mean(values)) if values else float('nan')

def load_model(device):
    checkpoint=torch.load(FORMAL/'best.pt',map_location='cpu',weights_only=False)
    transform=PhysicalTransform.from_dict(checkpoint['transform']); normalizer=FeatureNormalizer.from_dict(checkpoint['feature_normalizer'])
    delta=DeltaNormalization.from_dict(checkpoint['delta_normalization'])
    envelope_path=FORMAL/'momentum_state_envelope.json'; guard=None
    if envelope_path.exists():
        envelope=json.loads(envelope_path.read_text()); guard=[float(envelope['hu']['guard']),float(envelope['hv']['guard'])]
    model=JilongGlobalOperatorV2(len(feature_names()),**checkpoint['architecture'],delta_bounds=delta.bounds,momentum_state_bounds=guard).to(device)
    restore_checkpoint(FORMAL/'best.pt',model); model.eval()
    return model,transform,normalizer,delta,checkpoint

@torch.no_grad()
def predict(model, previous, current, static, params, time_fraction, transform, normalizer, active):
    encoded=transform.encode(current)
    features=build_features(previous,current,static,params,torch.tensor([time_fraction],device=current.device),transform,normalizer)
    return project_physical(transform.decode(model(features,encoded)),active)

def station_row(state, z0, transects):
    raw=state[0].detach().cpu().numpy(); z=(z0+raw[5]).astype(np.float32)
    return {name:station_metrics(raw,z,transect) for name,transect in transects.items()}

def append_station(series, time_s, values):
    row={'time_s':time_s}
    for name, metrics in values.items():
        for key,value in metrics.items(): row[f'{name}_{key}']=value
    series.append(row)

def station_summary(predicted, teacher, transects):
    rows=[]
    for name in transects:
        p,t=pd.DataFrame(predicted),pd.DataFrame(teacher); truth_arrival=arrival_from_series(t,name); pred_arrival=arrival_from_series(p,name)
        if truth_arrival is None: arrival_error=float('nan'); missed=False
        elif pred_arrival is None: arrival_error=1440.-truth_arrival; missed=True
        else: arrival_error=abs(pred_arrival-truth_arrival); missed=False
        def peak(col): return float(p[f'{name}_{col}'].abs().max()),float(t[f'{name}_{col}'].abs().max())
        q,qt=peak('Q');qd,qdt=peak('Qdebris');h,ht=peak('hmax');width,width_t=peak('wet_width_m')
        stage_p=float(p[f'{name}_stage'].max(skipna=True));stage_t=float(t[f'{name}_stage'].max(skipna=True))
        rows.append({'station':name,'arrival_time_s':pred_arrival,'truth_arrival_time_s':truth_arrival,'arrival_error_s':arrival_error,'missed_arrival':missed,
                     'peak_Q_relative_error':abs(q-qt)/max(abs(qt),1.),'peak_Qdebris_relative_error':abs(qd-qdt)/max(abs(qdt),1.),
                     'peak_hmax_relative_error':abs(h-ht)/max(abs(ht),1e-6),'peak_stage_absolute_error':abs(stage_p-stage_t) if math.isfinite(stage_p) and math.isfinite(stage_t) else float('nan'),
                     'wet_width_relative_error':abs(width-width_t)/max(abs(width_t),1.)})
    return rows

def aggregate_station(rows):
    keys=('arrival_error_s','peak_Q_relative_error','peak_Qdebris_relative_error','peak_hmax_relative_error','peak_stage_absolute_error','wet_width_relative_error')
    return {({'arrival_error_s':'arrival_MAE_s'}.get(key,key)):finite_mean([row[key] for row in rows]) for key in keys}|{'missed_arrival_count':int(sum(bool(row['missed_arrival']) for row in rows)),'truth_no_arrival_count':int(sum(row['truth_arrival_time_s'] is None for row in rows))}

def execute_method(label, budget, kind, rows, layout, model, transform, normalizer, delta, static, active, route, z0, transects, device,
                   case_index_column: str | None = None):
    """One streaming method through all VAL cases; no predicted frame is retained."""
    store=FrameStore(rows); per_case=[]; station_rows=[]; concentration=[]; timeline=[]
    for local_case_index,(_,row) in enumerate(rows.iterrows()):
        case_index=int(row[case_index_column]) if case_index_column is not None else local_case_index
        index=int(np.where(store.rows.scenario_id.eq(row.scenario_id))[0][0]);previous,current,_,params,time,_=store.sample(index,0)
        tensor=lambda x:torch.from_numpy(np.asarray(x)).unsqueeze(0).to(device)
        previous,current,params=tensor(previous),tensor(current),tensor(params)
        metric=StreamingMetrics(active,route,delta.change_threshold); predicted_station=[];teacher_station=[]
        truth0=tensor(store.frame(row,0)); append_station(predicted_station,0,station_row(truth0,z0,transects));append_station(teacher_station,0,station_row(truth0,z0,transects))
        coverages=[];count=layout.count_for_budget(budget)
        for step in range(144):
            truth_next=tensor(store.frame(row,(step+1)*10));truth_current=tensor(store.frame(row,step*10))
            if kind=='Persistence': provisional=current; selected=()
            else:
                provisional=predict(model,previous,current,static,params,time+step/144.,transform,normalizer,active)
                if kind in {'OraclePerfect','SmoothOracleV2'}:
                    scores,total,masses=oracle_patch_scores(transform.encode(provisional),transform.encode(truth_next),provisional,truth_next,active,layout,delta.scales)
                    selected=select_oracle(layout,scores,count)
                    fractions=concentration_fractions(scores); concentration.append({'method':label,'scenario_id':row.scenario_id,'time_s':(step+1)*10,'total_normalized_error':total,**fractions,**masses})
                elif kind=='RandomPerfect': selected=select_random(layout,count,20260920+case_index*1000+step)
                else: selected=()
            corrected=project_physical(correct_cores(provisional,truth_next,selected,layout,active,smooth=(kind=='SmoothOracleV2')),active) if selected else provisional
            metric.add(corrected,truth_current,truth_next,transform)
            append_station(predicted_station,(step+1)*10,station_row(corrected,z0,transects));append_station(teacher_station,(step+1)*10,station_row(truth_next,z0,transects))
            if kind!='Persistence':
                coverage=layout.selected_active_fraction(selected);coverages.append(coverage)
                timeline.append({'method':label,'scenario_id':row.scenario_id,'time_s':(step+1)*10,'budget_fraction':budget,'selected_patch_count':len(selected),'eligible_patch_count':len(layout.eligible),'active_cell_coverage_fraction':coverage})
            previous,current=current,corrected
        result=metric.result();result.update({'method':label,'scenario_id':row.scenario_id,'budget_fraction':budget,'selected_patch_count':count if kind!='Persistence' else 0,'active_cell_coverage_fraction':finite_mean(coverages)})
        per_case.append(result)
        for station in station_summary(predicted_station,teacher_station,transects):station.update({'method':label,'scenario_id':row.scenario_id});station_rows.append(station)
        print(f'{label} {row.scenario_id} complete',flush=True)
    return per_case,station_rows,concentration,timeline

def summarize(rows,station_rows):
    numeric=defaultdict(list)
    for row in rows:
        for key,value in row.items():
            if key not in {'method','scenario_id'} and not key.startswith('__') and isinstance(value,(int,float,np.floating)):numeric[key].append(value)
    out={key:finite_mean(values) for key,values in numeric.items()}
    # Dynamic-region and newly-wet metrics are explicitly one global
    # case-time-cell accumulation, never an average of small-mask ratios.
    for name in STATE_NAMES:
        num=sum(float(row[f'__change_num_{name}']) for row in rows);den=sum(float(row[f'__change_den_{name}']) for row in rows)
        out[f'change_region_{name}_rel_l2']=math.sqrt(num/max(den,1e-12))
    momentum_num=sum(float(row['__change_num_hu'])+float(row['__change_num_hv']) for row in rows);momentum_den=sum(float(row['__change_den_hu'])+float(row['__change_den_hv']) for row in rows)
    out['change_region_momentum_rel_l2']=math.sqrt(momentum_num/max(momentum_den,1e-12))
    intersection=sum(int(row['__new_intersection']) for row in rows);union=sum(int(row['__new_union']) for row in rows);pred=sum(int(row['__new_pred']) for row in rows);truth=sum(int(row['__new_truth']) for row in rows)
    out['newly_wet_iou']=intersection/union if union else float('nan');out['newly_wet_precision']=intersection/pred if pred else float('nan');out['newly_wet_recall']=intersection/truth if truth else float('nan')
    out.update(aggregate_station(station_rows));return out

def relative_improvements(summary, frozen):
    reduction=lambda key:100*(frozen[key]-summary[key])/abs(frozen[key]) if math.isfinite(summary.get(key,float('nan'))) and abs(frozen[key])>1e-12 else float('nan')
    return {'front_MAE_reduction_percent':reduction('debris_front_mae_km'),'volume_error_reduction_percent':reduction('mixture_volume_relative_error'),
            'change_h_error_reduction_percent':reduction('change_region_h_rel_l2'),'change_momentum_error_reduction_percent':reduction('change_region_momentum_rel_l2'),
            'newly_wet_IoU_gain':summary['newly_wet_iou']-frozen['newly_wet_iou'], 'arrival_MAE_reduction_percent':reduction('arrival_MAE_s'),'peakQ_error_reduction_percent':reduction('peak_Q_relative_error')}

def reproduce(model,transform,normalizer,delta,static,active,route,z0,transects,device,rows):
    records,stations,_,_=execute_method('FrozenGlobal',0.,'FrozenGlobal',rows,PatchLayout(active[0,0].cpu().numpy()),model,transform,normalizer,delta,static,active,route,z0,transects,device)
    summary=summarize(records,stations); reference=json.loads((ROOT/'reports/GLOBAL_V2_VAL_CONFIRMATION.json').read_text())
    mismatch={key:(summary[key],reference[key]) for key in BASELINE_KEYS if abs(summary[key]-reference[key])>1e-5}
    if mismatch:raise RuntimeError('FROZEN_REPRODUCTION_FAILED '+json.dumps(mismatch))
    return records,stations,summary

def main(args):
    if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED')
    torch.manual_seed(20260920);np.random.seed(20260920);device=torch.device('cuda')
    rows=scenario_rows('VAL')
    if len(rows)!=20 or not set(rows['split']).issubset({'VAL'}) or rows.scenario_id.str.contains('H0|TEST',case=False).any():raise RuntimeError('VAL_SCOPE_REQUIRED')
    model,transform,normalizer,delta,checkpoint=load_model(device); static_np,_=static_and_exogenous(INPUT);static=torch.from_numpy(static_np).unsqueeze(0).to(device);active=static[:,1:2]
    with np.load(INPUT) as data:route=np.asarray(data['route_chainage_m'],np.float32)
    z0=static_np[0];transects=load_transects(ROOT/'data/downloads/park_v2/inputs/upper30h_transects.json');layout=PatchLayout(active[0,0].detach().cpu().numpy(),16,16)
    frozen_cases,frozen_stations,frozen_summary=reproduce(model,transform,normalizer,delta,static,active,route,z0,transects,device,rows)
    if args.reproduce_only:print(json.dumps({'frozen_reproduction':'PASS',**{k:frozen_summary[k] for k in BASELINE_KEYS}},indent=2));return
    all_case=list(frozen_cases);all_station=list(frozen_stations);all_concentration=[];all_timeline=[]; summaries={'FrozenGlobal':frozen_summary}
    methods=[('Persistence',0.,'Persistence'),('RandomPerfect_B05',.05,'RandomPerfect'),('RandomPerfect_B10',.10,'RandomPerfect'),('RandomPerfect_B20',.20,'RandomPerfect'),('OraclePerfectV2_B05',.05,'OraclePerfect'),('OraclePerfectV2_B10',.10,'OraclePerfect'),('OraclePerfectV2_B20',.20,'OraclePerfect')]
    for label,budget,kind in methods:
        cases,stations,concentration,timeline=execute_method(label,budget,kind,rows,layout,model,transform,normalizer,delta,static,active,route,z0,transects,device)
        all_case.extend(cases);all_station.extend(stations);all_concentration.extend(concentration);all_timeline.extend(timeline);summaries[label]=summarize(cases,stations)
    for label,summary in summaries.items():summary['relative_to_frozen']=relative_improvements(summary,frozen_summary) if label!='FrozenGlobal' else {}
    OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(all_case).to_csv(OUT/'case_metrics.csv',index=False)
    dynamic=[{k:v for k,v in row.items() if k in {'method','scenario_id'} or k.startswith('change_region_') or k.startswith('newly_wet') or k.startswith('front_zone_')} for row in all_case]
    pd.DataFrame(dynamic).to_csv(OUT/'dynamic_region_metrics.csv',index=False);pd.DataFrame(all_station).to_csv(OUT/'station_metrics.csv',index=False);pd.DataFrame(all_concentration).to_csv(OUT/'error_concentration.csv',index=False);pd.DataFrame(all_timeline).to_csv(OUT/'patch_budget_timeline.csv',index=False)
    decomposition=[]
    for row in all_concentration:
        total=max(float(row['total_normalized_error']),1e-30);decomposition.append({k:row[k] for k in ('method','scenario_id','time_s')}|{'true_wet_error_fraction':row['error_mass_true_wet']/total,'false_positive_error_fraction':row['error_mass_false_positive_wet']/total,'both_dry_error_fraction':row['error_mass_both_dry']/total})
    pd.DataFrame(decomposition).to_csv(OUT/'error_support_decomposition.csv',index=False)
    method_rows=[]
    for label,summary in summaries.items():method_rows.append({'method':label,**{k:v for k,v in summary.items() if k!='relative_to_frozen'}})
    pd.DataFrame(method_rows).to_csv(OUT/'method_summary.csv',index=False)
    oracle_concentration={}
    for key in ('top5_fraction','top10_fraction','top20_fraction'):
        values=[row[key] for row in all_concentration];oracle_concentration[key]={'mean':finite_mean(values),'median':float(np.median(values)) if values else float('nan')}
    support={key:{'mean':finite_mean([row[key] for row in decomposition]),'median':float(np.median([row[key] for row in decomposition]))} for key in ('true_wet_error_fraction','false_positive_error_fraction','both_dry_error_fraction')}
    report={'study':'JILONG ORACLE LOCAL-REFINEMENT FEASIBILITY STUDY V2','disclaimer':DISCLAIMER,'frozen_reproduction':'PASS','checkpoint':{'global_step':checkpoint['global_step'],'stage':checkpoint['stage_name'],'architecture':checkpoint['architecture']},'layout':{'grid':'16x16','eligible_patches':len(layout.eligible),'active_cell_count':layout.active_cell_count,'pad_shape':[layout.pad_height,layout.pad_width],'core_shape':[layout.core_height,layout.core_width],'budgets':{f'{int(b*100)}%':layout.count_for_budget(b) for b in (.05,.10,.20)}},'error_concentration':oracle_concentration,'error_support_decomposition':support,'methods':summaries}
    REPORT.write_text(json.dumps(report,indent=2,allow_nan=True));print(json.dumps({'frozen_reproduction':'PASS','eligible_patches':len(layout.eligible),'report':str(REPORT)},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--reproduce-only',action='store_true');main(parser.parse_args())
