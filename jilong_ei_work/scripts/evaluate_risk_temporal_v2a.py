"""Development-only v2A evaluator. It invokes the frozen v1 selector/metrics unchanged."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import evaluate_engineering_roi_v2 as base
from src.common.provenance import resolve_code_sha
from src.global_operator_v2.dataset import INPUT,scenario_rows
from src.global_operator_v2.frame_adapter import static_and_exogenous
from src.global_operator_v2.metrics import load_transects
from src.global_operator_v2.oracle_refinement import PatchLayout
from src.local_corrector.engineering_roi import build_roi_static_metadata,build_section_mask
from src.local_corrector.engineering_roi_execution import cached_engineering_method,partition_rows,run_output_path,write_run_complete
from src.local_corrector.model import JilongLocalCorrector
from src.local_corrector.soft_support_guard import apply_soft_support_guard
from src.local_corrector.trainer import apply_learned_correction,apply_momentum_state_guard

EXTRA=("fraction_zone_A","fraction_zone_B","fraction_zone_C","local_delta_norm_A","local_delta_norm_B","new_wet_created_A","new_wet_created_B","softened_local_change_fraction","newly_wet_fraction")
def transition(previous,current,provisional,encoded_current,encoded_provisional,metadata,config,budget,state,method,local,layout,delta,normalization,static,params,time_value,transform,normalizer,global_model,device):
 (selected,next_state,selection),roi=base.timed_cuda(lambda:base.select_engineering_roi_v2(current,provisional,metadata,{"wet_threshold_m":.05,"shallow_margin_upper_h_m":.15,"budgets":[.05,.1,.2],"temporal_refresh_steps":1,"temporal_fill_from_excluded":False},budget,state,temporal_refresh=True),device)
 def write():
  if not selected:return encoded_provisional
  f=base.build_features(previous,current,static,params,torch.tensor([time_value],device=current.device),transform,normalizer)
  return apply_learned_correction(local,f,encoded_provisional,encoded_current,selected,layout,delta.scales,normalization['scales'],normalization['bounds'],metadata.active,global_model,apply_momentum_guard=False)[0]
 raw,local_s=base.timed_cuda(write,device); raw_physical=base.project_physical(transform.decode(raw),metadata.active)
 def guard():
  soft,t=apply_soft_support_guard(encoded_provisional,raw,current,provisional,metadata.active,transform,radius_cells=config['soft_support']['dilation_radius_cells'],alpha_front=config['soft_support']['alpha_front']);return apply_momentum_state_guard(soft,global_model),t
 (momentum,tel),support_s=base.timed_cuda(guard,device); (physical,depth_tel),depth_seconds=base.timed_cuda(lambda:(base.project_physical(transform.decode(momentum),metadata.active),base._zero_depth_telemetry()),device)
 return physical,selected,next_state,selection|tel|depth_tel,{"roi_scoring_runtime_ms":1000*roi,"local_correction_runtime_ms":1000*local_s,"support_guard_runtime_ms":1000*support_s,"depth_guard_runtime_ms":1000*depth_seconds}
def main(a):
 if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED')
 raw=(ROOT/'configs/risk_temporal_v2a_soft_support.json').read_bytes();config=json.loads(raw);assert config['version']=='RiskTemporal-v2A'; split='VAL' if a.split=='val' else 'TEST';full=scenario_rows(split)
 if len(full)!=20:raise RuntimeError('V2A_EXACTLY_20_REQUIRED')
 rows=partition_rows(full,a.shard_index,a.shard_count);device=torch.device('cuda');model,transform,normalizer,delta,gck=base.load_model(device);static_np,_=static_and_exogenous(INPUT);static=base.tensor(static_np,device);layout=PatchLayout(static_np[1]);mdir=ROOT/'models/local_corrector_v1_1';norm=json.loads((mdir/'correction_normalization.json').read_text());lck=torch.load(mdir/'best.pt',map_location=device,weights_only=False);local=JilongLocalCorrector(47).to(device);local.load_state_dict(lck['model']);local.eval();route=np.asarray(np.load(INPUT)['route_chainage_m'],np.float32);transects=load_transects(ROOT/'data/downloads/park_v2/inputs/upper30h_transects.json');metadata=build_roi_static_metadata(layout,static[:,1:2],route,build_section_mask(transects,static_np.shape[-2:]),device);label='RiskTemporal_v2A_B10';root=ROOT/'results/risk_temporal_v2'/a.run_tag/a.split;out=run_output_path(root,a.shard_index,a.shard_count);sha=resolve_code_sha(a.source_code_sha,ROOT,require=True);manifest=base.manifest_for(label,'RiskTemporal',.1,full,rows,hashlib.sha256(raw).hexdigest(),sha,base._checkpoint_provenance(ROOT/'models/global_operator_v2/best.pt',gck),base._checkpoint_provenance(mdir/'best.pt',lck),norm,layout,a.shard_index,a.shard_count,'configs/risk_temporal_v2a_soft_support.json');manifest.update({'scope':'VAL_DEVELOPMENT' if a.split=='val' else 'ENGINEERING_STRESS_TEST','v2a_radius_cells':1,'v2a_alpha_front':.5})
 old=base.v2_transition;base.v2_transition=transition
 try: cases,stations,timeline,timing=cached_engineering_method(out,manifest,rows,lambda one:base.execute_cases(label,'RiskTemporal',.1,one,layout,metadata,config,model,local,transform,normalizer,delta,norm,static,route,static_np[0],transects,device),True,a.resume,expected_station_count=len(transects))
 finally: base.v2_transition=old
 base.write_method_outputs(out,label,cases,stations,timeline,timing,manifest);pd.DataFrame(timeline,columns=base.TIMELINE_COLUMNS+EXTRA).sort_values(['scenario_id','time_s']).to_csv(out/'selection_timeline.csv',index=False);runtime=pd.read_csv(out/'runtime_summary.csv');frame=pd.DataFrame(timeline);runtime['mean_zone_A']=frame.fraction_zone_A.mean();runtime['mean_zone_B']=frame.fraction_zone_B.mean();runtime['mean_zone_C']=frame.fraction_zone_C.mean();runtime['mean_softened_local_change_fraction']=frame.softened_local_change_fraction.mean();runtime.to_csv(out/'runtime_summary.csv',index=False);write_run_complete(out,manifest,cases,timeline,True);print(json.dumps({'status':'PASS','split':a.split,'output':str(out)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--split',choices=('val','stress'),required=True);p.add_argument('--source-code-sha',required=True);p.add_argument('--run-tag',default='v2A_soft_support');p.add_argument('--resume',action='store_true');p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shard-count',type=int,default=0);main(p.parse_args())
