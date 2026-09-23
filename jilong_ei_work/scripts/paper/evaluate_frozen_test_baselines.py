"""Run only pre-existing frozen baselines on the already-open TEST split."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts import evaluate_engineering_roi_v2 as v2
from src.global_operator_v2.dataset import INPUT,scenario_rows
from src.global_operator_v2.frame_adapter import static_and_exogenous
from src.global_operator_v2.metrics import load_transects
from src.global_operator_v2.oracle_refinement import PatchLayout
from src.local_corrector.engineering_roi import build_roi_static_metadata,build_section_mask
from src.local_corrector.engineering_roi_execution import cached_engineering_method,partition_rows,write_run_complete
from src.local_corrector.engineering_roi_v2 import load_engineering_roi_v2_config
from src.local_corrector.model import JilongLocalCorrector
def main(a):
 if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED')
 full=scenario_rows('TEST');assert len(full)==20 and set(full.split)=={'TEST'}; rows=partition_rows(full,0,0);dev=torch.device('cuda');cfg,sha=load_engineering_roi_v2_config(ROOT/'configs/engineering_roi_v2.json');model,trans,normer,delta,gck=v2.load_model(dev);st,_=static_and_exogenous(INPUT);static=v2.tensor(st,dev);layout=PatchLayout(st[1]);mdir=ROOT/'models/local_corrector_v1_1';norm=json.loads((mdir/'correction_normalization.json').read_text());lck=torch.load(mdir/'best.pt',map_location=dev,weights_only=False);local=JilongLocalCorrector(47).to(dev);local.load_state_dict(lck['model']);local.eval();route=np.asarray(np.load(INPUT)['route_chainage_m'],np.float32);tr=load_transects(ROOT/'data/downloads/park_v2/inputs/upper30h_transects.json');meta=build_roi_static_metadata(layout,static[:,1:2],route,build_section_mask(tr,st.shape[-2:]),dev);label='SupportRisk_Base_B10';out=ROOT/'paper_results/test/baselines/supportrisk_base_b10';man=v2.manifest_for(label,'SupportRisk_Base',.1,full,rows,sha,'146184d1fbbf366c0c9c40066a56753b398b10f4',v2._checkpoint_provenance(ROOT/'models/global_operator_v2/best.pt',gck),v2._checkpoint_provenance(mdir/'best.pt',lck),norm,layout,0,0,'configs/engineering_roi_v2.json');man['scope']='FROZEN_BASELINE_TEST';cases,stations,timeline,timing=cached_engineering_method(out,man,rows,lambda one:v2.execute_cases(label,'SupportRisk_Base',.1,one,layout,meta,cfg,model,local,trans,normer,delta,norm,static,route,st[0],tr,dev),True,a.resume,expected_station_count=len(tr));v2.write_method_outputs(out,label,cases,stations,timeline,timing,man);write_run_complete(out,man,cases,timeline,True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');main(p.parse_args())
