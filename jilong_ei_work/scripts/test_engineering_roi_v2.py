import sys,torch,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.oracle_refinement import PatchLayout
from src.local_corrector.engineering_roi import build_roi_static_metadata
from src.local_corrector.engineering_roi_v2 import *
from src.local_corrector.depth_envelope_guard import apply_depth_envelope_guard
def main():
 c=load_engineering_roi_v2_config(ROOT/'configs/engineering_roi_v2.json');a=torch.ones(1,1,4,4,dtype=torch.bool);m=build_roi_static_metadata(PatchLayout(np.ones((4,4),bool),2,2),a,np.zeros((4,4)),torch.zeros(4,4,dtype=torch.bool));cur=torch.zeros(1,6,4,4);pro=cur.clone();pro[0,0,0,0]=.05;sel,state,t=select_engineering_roi_v2(cur,pro,m,c,.05,initial_temporal_state());assert t['selected_patch_consecutive_overlap_count']==0;out,x=apply_depth_envelope_guard(torch.full_like(cur,.1),torch.full_like(pro,.2),torch.full_like(pro,.3),a);assert out[:,0].max()==.2 and torch.equal(out[:,1:],torch.full_like(pro[:,1:],.3));print('PASS EngineeringROI-v2 synthetic safety tests')
if __name__=='__main__':main()
