"""Future VAL-only frozen EngineeringROI-v1 evaluator; never a training path."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common.provenance import resolve_code_sha
from src.global_operator_v2.dataset import FrameStore,INPUT,build_features,scenario_rows
from src.global_operator_v2.frame_adapter import static_and_exogenous
from src.global_operator_v2.losses import project_physical
from src.global_operator_v2.metrics import load_transects
from src.global_operator_v2.oracle_refinement import PatchLayout,StreamingMetrics
from src.local_corrector.engineering_roi import (STRATEGIES,build_roi_static_metadata,build_section_mask,load_engineering_roi_config,select_engineering_roi)
from src.local_corrector.engineering_roi_execution import (cached_engineering_method,canonical_method_label,partition_rows,run_output_path,validate_shard_coverage,validate_shard_manifests)
from src.local_corrector.model import JilongLocalCorrector
from src.local_corrector.support_guard import apply_support_guard
from src.local_corrector.trainer import apply_learned_correction,apply_momentum_state_guard
from scripts.run_global_v2_oracle_refinement import append_station,load_model,predict,station_row,station_summary,summarize
SEED=20260920
TIMELINE_COLUMNS=('method','scenario_id','time_s','budget','selected_patch_ids','budget_fraction','budget_max_count','candidate_count','selected_count','actual_active_fraction','predicted_front_chainage_m','front_available','mean_selected_dynamic_rank','mean_selected_front_rank','mean_selected_support_risk_rank','mean_selected_section_rank','mean_selected_base_score','selected_section_patch_count','first_pass_count','second_pass_count','roi_scoring_runtime_ms','local_correction_runtime_ms','support_guard_runtime_ms')

def tensor(value,device):return torch.from_numpy(np.asarray(value,np.float32)).unsqueeze(0).to(device)
def quantile(values,q):return float(np.quantile(values,q)) if values else float('nan')
def checkpoint_provenance(path,checkpoint):return {'path':str(path),'global_step':checkpoint.get('global_step'),'stage_name':checkpoint.get('stage_name'),'architecture':checkpoint.get('architecture'),'source_code_sha':checkpoint.get('source_code_sha',checkpoint.get('code_sha',checkpoint.get('source_commit','UNKNOWN')))}

def manifest_for(label,strategy,budget,rows,config_sha,code_sha,global_provenance,local_provenance,normalization,layout):
    return {'schema_version':1,'method_label':label,'strategy':strategy,'budget':budget,'scope':'VAL_ONLY','scenario_ids':list(map(str,rows.scenario_id)),'config_path':'configs/engineering_roi_v1.json','config_sha256':config_sha,'code_sha':code_sha,'global_checkpoint_path':'models/global_operator_v2/best.pt','global_checkpoint_provenance':global_provenance,'local_checkpoint_path':'models/local_corrector_v1_1/best.pt','local_checkpoint_provenance':local_provenance,'local_checkpoint_update':4000,'local_normalization_path':'models/local_corrector_v1_1/correction_normalization.json','local_normalization_fit_scope':normalization.get('fit_scope'),'support_guard_rule':'current_or_global_provisional_wet','support_guard_version':'SupportGuard-v1','seed_base':SEED,'patch_layout_rows':layout.rows,'patch_layout_cols':layout.cols,'eligible_patch_count':len(layout.eligible),'created_by':'evaluate_engineering_roi_v1.py'}

@torch.no_grad()
def execute_cases(label,strategy,budget,rows,layout,metadata,global_model,local,transform,normalizer,delta,normalization,static,route,z0,transects,device):
    store=FrameStore(rows);cases=[];stations=[];timeline=[];parts=[]
    for local_index,(_,row) in enumerate(rows.iterrows()):
        case_index=int(row.get('__global_case_index',local_index));index=int(np.where(store.rows.scenario_id.eq(row.scenario_id))[0][0]);previous,current,_,params,time0,_=store.sample(index,0);previous,current,params=tensor(previous,device),tensor(current,device),tensor(params,device)
        metric=StreamingMetrics(metadata.active,route,delta.change_threshold);pred=[];teacher=[];append_station(pred,0,station_row(current,z0,transects));append_station(teacher,0,station_row(current,z0,transects));cover=[];counts=[];roi_steps=[];local_steps=[];guard_steps=[];started=time.perf_counter()
        for step in range(144):
            provisional=predict(global_model,previous,current,static,params,time0+step/144.,transform,normalizer,metadata.active);encoded_current,encoded_provisional=transform.encode(current),transform.encode(provisional)
            if strategy=='frozen_global':selected=();selection={'budget_fraction':0.,'budget_max_count':0,'candidate_count':0,'selected_count':0,'actual_active_fraction':0.,'predicted_front_chainage_m':float('nan'),'front_available':False,'mean_selected_dynamic_rank':0.,'mean_selected_front_rank':0.,'mean_selected_support_risk_rank':0.,'mean_selected_section_rank':0.,'mean_selected_base_score':0.,'selected_section_patch_count':0,'first_pass_count':0,'second_pass_count':0};roi_step=local_step=guard_step=0.;corrected=provisional
            else:
                begin=time.perf_counter();selected,selection=select_engineering_roi(current,provisional,encoded_current,encoded_provisional,delta.scales,metadata,CONFIG,budget,strategy,SEED+case_index*1000+step);roi_step=time.perf_counter()-begin
                if selected:
                    features=build_features(previous,current,static,params,torch.tensor([time0+step/144.],device=device),transform,normalizer);begin=time.perf_counter();raw_t,_,_=apply_learned_correction(local,features,encoded_provisional,encoded_current,selected,layout,delta.scales,normalization['scales'],normalization['bounds'],metadata.active,global_model,apply_momentum_guard=False);local_step=time.perf_counter()-begin
                else:raw_t=encoded_provisional;local_step=0.
                raw_physical=project_physical(transform.decode(raw_t),metadata.active);begin=time.perf_counter();guarded,_=apply_support_guard(encoded_provisional,raw_t,current,provisional,metadata.active,local_corrected_physical=raw_physical);corrected=project_physical(transform.decode(apply_momentum_state_guard(guarded,global_model)),metadata.active);guard_step=time.perf_counter()-begin
            # Prediction and deterministic patch allocation are complete before any teacher frame is read.
            truth_current=tensor(store.frame(row,step*10),device);truth_next=tensor(store.frame(row,(step+1)*10),device);metric.add(corrected,truth_current,truth_next,transform);append_station(pred,(step+1)*10,station_row(corrected,z0,transects));append_station(teacher,(step+1)*10,station_row(truth_next,z0,transects))
            if strategy!='frozen_global':timeline.append(selection|{'method':label,'scenario_id':row.scenario_id,'time_s':(step+1)*10,'budget':f'B{int(budget*100):02d}','selected_patch_ids':';'.join(map(str,[patch.patch_id for patch in selected])),'roi_scoring_runtime_ms':1000*roi_step,'local_correction_runtime_ms':1000*local_step,'support_guard_runtime_ms':1000*guard_step})
            cover.append(layout.selected_active_fraction(selected));counts.append(len(selected));roi_steps.append(roi_step);local_steps.append(local_step);guard_steps.append(guard_step);previous,current=current,corrected
        cases.append(metric.result()|{'method':label,'scenario_id':row.scenario_id,'budget_fraction':budget or 0.,'budget_max_count':layout.count_for_budget(budget) if budget is not None else 0,'selected_patch_count':float(np.mean(counts)),'active_cell_coverage_fraction':float(np.mean(cover))});stations.extend([item|{'method':label,'scenario_id':row.scenario_id} for item in station_summary(pred,teacher,transects)]);parts.append({'method':label,'total_runtime_seconds':time.perf_counter()-started,'roi_steps_ms':[1000*x for x in roi_steps],'local_steps_ms':[1000*x for x in local_steps],'guard_steps_ms':[1000*x for x in guard_steps]})
    return cases,stations,timeline,parts[0] if len(parts)==1 else {'method':label,'parts':parts}

def runtime_summary(parts):
    flat=lambda key:[value for part in parts for value in part.get(key,[])];roi,local,guard=flat('roi_steps_ms'),flat('local_steps_ms'),flat('guard_steps_ms')
    return {'total_runtime_seconds':sum(part['total_runtime_seconds'] for part in parts),'trajectory_runtime_mean_seconds':float(np.mean([part['total_runtime_seconds'] for part in parts])),'roi_scoring_mean_ms_step':float(np.mean(roi)) if roi else 0.,'roi_scoring_median_ms_step':quantile(roi,.5) if roi else 0.,'roi_scoring_p95_ms_step':quantile(roi,.95) if roi else 0.,'local_correction_mean_ms_step':float(np.mean(local)) if local else 0.,'support_guard_mean_ms_step':float(np.mean(guard)) if guard else 0.}

def write_method_outputs(run_out,label,cases,stations,timeline,parts,manifest):
    summary=summarize(cases,stations);frame=pd.DataFrame(timeline,columns=TIMELINE_COLUMNS);runtime=runtime_summary(parts);runtime.update({'mean_selected_patch_count':float(frame.selected_count.mean()) if len(frame) else 0.,'mean_active_coverage':float(frame.actual_active_fraction.mean()) if len(frame) else 0.});run_out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([{'method':label,**summary}]).to_csv(run_out/'final_method_summary.csv',index=False);pd.DataFrame(cases).to_csv(run_out/'final_case_metrics.csv',index=False);pd.DataFrame(stations).to_csv(run_out/'final_station_metrics.csv',index=False);frame.to_csv(run_out/'selection_timeline.csv',index=False);pd.DataFrame([{'method':label,**runtime}]).to_csv(run_out/'runtime_summary.csv',index=False)
    report={'method_label':label,'manifest':manifest,'summary':summary,'runtime':runtime,'timing_parts':parts};(run_out/'method_report.json').write_text(json.dumps(report,indent=2,allow_nan=True));return report

def merge_shards(args):
    root=ROOT/'results'/args.output_dir;manifests=validate_shard_manifests(root,args.shard_count);parts=[root/'shards'/f'shard_{index}' for index in range(args.shard_count)];cases=pd.concat([pd.read_csv(path/'final_case_metrics.csv') for path in parts],ignore_index=True);stations=pd.concat([pd.read_csv(path/'final_station_metrics.csv') for path in parts],ignore_index=True);timeline=pd.concat([pd.read_csv(path/'selection_timeline.csv') for path in parts],ignore_index=True);full=scenario_rows('VAL');validate_shard_coverage(cases.to_dict('records'),full.scenario_id);label=manifests[0]['method_label'];timeline=timeline.sort_values(['scenario_id','time_s'])
    if manifests[0]['strategy']!='frozen_global' and any(len(group)!=144 for _,group in timeline.groupby('scenario_id')):raise RuntimeError('SELECTION_TIMELINE_INCOMPLETE')
    reports=[json.loads((path/'method_report.json').read_text()) for path in parts];report=write_method_outputs(root,label,cases.to_dict('records'),stations.to_dict('records'),timeline.to_dict('records'),[item for value in reports for item in value.get('timing_parts',[])],manifests[0]);return report

def main(args):
    global CONFIG
    if args.merge_shards:return merge_shards(args)
    if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED')
    CONFIG,config_sha=load_engineering_roi_config(ROOT/'configs/engineering_roi_v1.json');strategy=args.strategy;budget=None if strategy=='frozen_global' else args.budget;label=canonical_method_label(strategy,budget);root=ROOT/'results'/args.output_dir if args.output_dir else ROOT/'results'/'engineering_roi_v1'/'runs'/label;run_out=run_output_path(root,args.shard_index,args.shard_count)
    full=scenario_rows('VAL')
    if len(full)!=20 or not set(full['split']).issubset({'VAL'}):raise RuntimeError('VAL_SCOPE_REQUIRED')
    rows=partition_rows(full,args.shard_index,args.shard_count);device=torch.device('cuda');global_model,transform,normalizer,delta,global_checkpoint=load_model(device);static_np,_=static_and_exogenous(INPUT);static=tensor(static_np,device);active=static[:,1:2];layout=PatchLayout(static_np[1]);model_dir=ROOT/'models/local_corrector_v1_1';normalization=json.loads((model_dir/'correction_normalization.json').read_text())
    if normalization.get('fit_scope')!='TRAIN_ONLY':raise RuntimeError('LOCAL_NORMALIZATION_NOT_TRAIN_ONLY')
    local_checkpoint=torch.load(model_dir/'best.pt',map_location=device,weights_only=False)
    if local_checkpoint.get('update')!=4000:raise RuntimeError('LOCAL_CHECKPOINT_MUST_BE_BEST_4000')
    local=JilongLocalCorrector(47).to(device);local.load_state_dict(local_checkpoint['model']);local.eval();route=np.asarray(np.load(INPUT)['route_chainage_m'],np.float32);transects=load_transects(ROOT/'data/downloads/park_v2/inputs/upper30h_transects.json');metadata=build_roi_static_metadata(layout,active,route,build_section_mask(transects,static_np.shape[-2:]),device);code_sha=resolve_code_sha(args.source_code_sha,ROOT,require=True);manifest=manifest_for(label,strategy,budget,rows,config_sha,code_sha,checkpoint_provenance(ROOT/'models/global_operator_v2/best.pt',global_checkpoint),checkpoint_provenance(model_dir/'best.pt',local_checkpoint),normalization,layout)
    runner=lambda one:execute_cases(label,strategy,budget,one,layout,metadata,global_model,local,transform,normalizer,delta,normalization,static,route,static_np[0],transects,device)
    cases,stations,timeline,parts=cached_engineering_method(run_out,manifest,rows,runner,strategy!='frozen_global',args.resume);write_method_outputs(run_out,label,cases,stations,timeline,parts,manifest)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--strategy',choices=('frozen_global',)+STRATEGIES,default='engineering_roi');parser.add_argument('--budget',choices=(.05,.10,.20),type=float,default=.10);parser.add_argument('--shard-index',type=int,default=0);parser.add_argument('--shard-count',type=int,default=0);parser.add_argument('--merge-shards',action='store_true');parser.add_argument('--output-dir');parser.add_argument('--resume',action='store_true');parser.add_argument('--source-code-sha');main(parser.parse_args())
