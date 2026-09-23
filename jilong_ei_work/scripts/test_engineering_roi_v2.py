import inspect,sys,torch,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.oracle_refinement import PatchLayout
from src.local_corrector.engineering_roi import build_roi_static_metadata
from src.local_corrector.engineering_roi_v2 import *
from src.local_corrector.depth_envelope_guard import apply_depth_envelope_guard
def main():
 c=load_engineering_roi_v2_config(ROOT/'configs/engineering_roi_v2.json');a=torch.ones(1,1,4,4,dtype=torch.bool);m=build_roi_static_metadata(PatchLayout(np.ones((4,4),bool),2,2),a,np.zeros((4,4)),torch.zeros(4,4,dtype=torch.bool));cur=torch.zeros(1,6,4,4);pro=cur.clone();pro[0,0]=.1
 s,state,t=select_engineering_roi_v2(cur,pro,m,c,.1,initial_temporal_state());s2,state2,t2=select_engineering_roi_v2(cur,pro,m,c,.1,state);assert not({p.patch_id for p in s}&{p.patch_id for p in s2}) and t2['selected_patch_consecutive_overlap_count']==0;assert 'truth' not in inspect.signature(select_engineering_roi_v2).parameters
 low=torch.full_like(cur,.1);high=torch.full_like(cur,.2);raw=torch.full_like(cur,.3);out,tele=apply_depth_envelope_guard(low,high,raw,a);assert out[:,0].max()==.2 and torch.equal(out[:,1:],raw[:,1:]) and tele['depth_guard_activation_fraction']==1.;inside=torch.full_like(cur,.15);assert apply_depth_envelope_guard(low,high,inside,a)[0][:,0].min()==.15
 print('PASS EngineeringROI-v2 temporal refresh, SupportRisk-only, depth-envelope tests')
if __name__=='__main__':main()
