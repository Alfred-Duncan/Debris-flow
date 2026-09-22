"""Run only the conditional SmoothOracleV2 B10/B20 seam diagnostic."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,torch,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.dataset import scenario_rows,INPUT
from src.global_operator_v2.frame_adapter import static_and_exogenous
from src.global_operator_v2.metrics import load_transects
from src.global_operator_v2.oracle_refinement import PatchLayout
from scripts.run_global_v2_oracle_refinement import load_model,execute_method,summarize,DISCLAIMER
def main():
 if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED')
 device=torch.device('cuda');rows=scenario_rows('VAL')
 if len(rows)!=20 or not set(rows.split).issubset({'VAL'}):raise RuntimeError('VAL_SCOPE_REQUIRED')
 model,tr,norm,delta,checkpoint=load_model(device);static_np,_=static_and_exogenous(INPUT);static=torch.from_numpy(static_np).unsqueeze(0).to(device);active=static[:,1:2]
 with np.load(INPUT) as data:route=np.asarray(data['route_chainage_m'],np.float32)
 layout=PatchLayout(active[0,0].cpu().numpy());transects=load_transects(ROOT/'data/downloads/park_v2/inputs/upper30h_transects.json');out=ROOT/'results/oracle_refinement_v2';out.mkdir(parents=True,exist_ok=True)
 all_cases=[];all_stations=[];all_concentration=[];all_timeline=[];summaries={}
 for label,budget in (('SmoothOracleV2_B10',.10),('SmoothOracleV2_B20',.20)):
  cases,stations,concentration,timeline=execute_method(label,budget,'SmoothOracleV2',rows,layout,model,tr,norm,delta,static,active,route,static_np[0],transects,device)
  all_cases+=cases;all_stations+=stations;all_concentration+=concentration;all_timeline+=timeline;summaries[label]=summarize(cases,stations)
 pd.DataFrame(all_cases).to_csv(out/'smooth_case_metrics.csv',index=False);pd.DataFrame(all_stations).to_csv(out/'smooth_station_metrics.csv',index=False);pd.DataFrame(all_concentration).to_csv(out/'smooth_error_concentration.csv',index=False);pd.DataFrame(all_timeline).to_csv(out/'smooth_patch_budget_timeline.csv',index=False)
 (ROOT/'reports/SMOOTH_ORACLE_V2.json').write_text(json.dumps({'disclaimer':DISCLAIMER,'checkpoint':{'global_step':checkpoint['global_step'],'stage':checkpoint['stage_name']},'methods':summaries},indent=2));print(json.dumps({'complete':True,'methods':list(summaries)},indent=2))
if __name__=='__main__':main()
