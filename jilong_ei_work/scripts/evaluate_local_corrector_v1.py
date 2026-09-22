"""VAL-only closed-loop evaluation for Local Corrector V1."""
from __future__ import annotations
import argparse, json, math, sys
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
from src.local_corrector.trainer import apply_learned_correction
from scripts.run_global_v2_oracle_refinement import (aggregate_station,append_station,execute_method,
    load_model,predict,station_row,station_summary,summarize)

SEED=20260920

def tensor(x,device):return torch.from_numpy(np.asarray(x,np.float32)).unsqueeze(0).to(device)

@torch.no_grad()
def learned_execute(label,budget,rows,layout,global_model,local_model,transform,normalizer,delta,normalization,static,active,route,z0,transects,device):
    store=FrameStore(rows);cases=[];stations=[];timeline=[];count=layout.count_for_budget(budget)
    for case_index,(_,row) in enumerate(rows.iterrows()):
        index=int(np.where(store.rows.scenario_id.eq(row.scenario_id))[0][0]);previous,current,_,params,time,_=store.sample(index,0)
        previous,current,params=tensor(previous,device),tensor(current,device),tensor(params,device)
        metric=StreamingMetrics(active,route,delta.change_threshold);pred_series=[];truth_series=[]
        truth0=tensor(store.frame(row,0),device);append_station(pred_series,0,station_row(truth0,z0,transects));append_station(truth_series,0,station_row(truth0,z0,transects))
        coverage=[]
        for step in range(144):
            truth_next=tensor(store.frame(row,(step+1)*10),device);truth_current=tensor(store.frame(row,step*10),device)
            provisional=predict(global_model,previous,current,static,params,time+step/144.,transform,normalizer,active)
            selected=select_random(layout,count,SEED+case_index*1000+step)
            features=build_features(previous,current,static,params,torch.tensor([time+step/144.],device=device),transform,normalizer)
            corrected_t,_,_=apply_learned_correction(local_model,features,transform.encode(provisional),transform.encode(current),selected,layout,delta.scales,normalization['scales'],normalization['bounds'],active,global_model)
            corrected=project_physical(transform.decode(corrected_t),active)
            metric.add(corrected,truth_current,truth_next,transform);append_station(pred_series,(step+1)*10,station_row(corrected,z0,transects));append_station(truth_series,(step+1)*10,station_row(truth_next,z0,transects))
            coverage.append(layout.selected_active_fraction(selected));timeline.append({'method':label,'scenario_id':row.scenario_id,'time_s':(step+1)*10,'budget_fraction':budget,'selected_patch_ids':';'.join(map(str,[p.patch_id for p in selected]))})
            previous,current=current,corrected
        result=metric.result()|{'method':label,'scenario_id':row.scenario_id,'budget_fraction':budget,'selected_patch_count':count,'active_cell_coverage_fraction':float(np.mean(coverage))};cases.append(result)
        for item in station_summary(pred_series,truth_series,transects):stations.append(item|{'method':label,'scenario_id':row.scenario_id})
        print(f'{label} {row.scenario_id} complete',flush=True)
    return cases,stations,timeline

def gap(frozen,learned,perfect):
    errors=('change_region_h_rel_l2','change_region_momentum_rel_l2','false_positive_wet_fraction','mixture_volume_relative_error','debris_front_mae_km','peak_Q_relative_error')
    return {key:(frozen[key]-learned[key])/(frozen[key]-perfect[key]) if abs(frozen[key]-perfect[key])>1e-12 else float('nan') for key in errors}|{'mean_wet_iou':(learned['mean_wet_iou']-frozen['mean_wet_iou'])/(perfect['mean_wet_iou']-frozen['mean_wet_iou']) if abs(perfect['mean_wet_iou']-frozen['mean_wet_iou'])>1e-12 else float('nan')}

def main(args):
    if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED')
    device=torch.device('cuda');rows=scenario_rows('VAL')
    if len(rows)!=20 or not set(rows['split']).issubset({'VAL'}) or rows.scenario_id.str.contains('TEST|H0|HOLDOUT',case=False).any():raise RuntimeError('VAL_SCOPE_REQUIRED')
    global_model,transform,normalizer,delta,_=load_model(device);static_np,_=static_and_exogenous(INPUT);static=tensor(static_np,device);active=static[:,1:2];layout=PatchLayout(static_np[1])
    normalization=json.loads((ROOT/'models/local_corrector_v1/correction_normalization.json').read_text())
    if normalization.get('fit_scope')!='TRAIN_ONLY':raise RuntimeError('LOCAL_NORMALIZATION_NOT_TRAIN_ONLY')
    local=JilongLocalCorrector(47).to(device);checkpoint=torch.load(ROOT/'models/local_corrector_v1/best.pt',map_location=device,weights_only=False);local.load_state_dict(checkpoint['model']);local.eval()
    with np.load(INPUT) as data:route=np.asarray(data['route_chainage_m'],np.float32)
    z0=static_np[0];transects=load_transects(ROOT/'data/downloads/park_v2/inputs/upper30h_transects.json')
    methods={};all_cases=[];all_stations=[];all_timeline=[]
    frozen_cases,frozen_stations,_,_=execute_method('FrozenGlobal',0.,'FrozenGlobal',rows,layout,global_model,transform,normalizer,delta,static,active,route,z0,transects,device);methods['FrozenGlobal']=summarize(frozen_cases,frozen_stations);all_cases+=frozen_cases;all_stations+=frozen_stations
    for budget in (.10,.20):
        label=f'LearnedRandom_B{int(budget*100):02d}';cases,stations,timeline=learned_execute(label,budget,rows,layout,global_model,local,transform,normalizer,delta,normalization,static,active,route,z0,transects,device);methods[label]=summarize(cases,stations);all_cases+=cases;all_stations+=stations;all_timeline+=timeline
        perfect=f'RandomPerfect_B{int(budget*100):02d}';cases,stations,_,timeline2=execute_method(perfect,budget,'RandomPerfect',rows,layout,global_model,transform,normalizer,delta,static,active,route,z0,transects,device);methods[perfect]=summarize(cases,stations);all_cases+=cases;all_stations+=stations;all_timeline+=timeline2
    out=ROOT/'results/local_corrector_v1';out.mkdir(parents=True,exist_ok=True);pd.DataFrame(all_cases).to_csv(out/'final_case_metrics.csv',index=False);pd.DataFrame(all_stations).to_csv(out/'final_station_metrics.csv',index=False);pd.DataFrame(all_timeline).to_csv(out/'patch_ids_timeline.csv',index=False)
    pd.DataFrame([{'method':name,**values} for name,values in methods.items()]).to_csv(out/'final_method_summary.csv',index=False)
    gaps=[]
    for amount in ('10','20'):gaps.append({'budget':f'B{amount}',**gap(methods['FrozenGlobal'],methods[f'LearnedRandom_B{amount}'],methods[f'RandomPerfect_B{amount}'])})
    pd.DataFrame(gaps).to_csv(out/'gap_recovery.csv',index=False);(out/'final_report.json').write_text(json.dumps({'normalization_scope':'TRAIN_ONLY','methods':methods,'gap_recovery':gaps,'local_parameters':local.parameter_count},indent=2,allow_nan=True));print(json.dumps({'status':'PASS','output':str(out),'local_parameters':local.parameter_count},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');main(parser.parse_args())
