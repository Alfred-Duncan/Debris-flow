"""Future VAL-only V2 execution path; not invoked during code review."""
from __future__ import annotations
import argparse
import torch
from src.local_corrector.engineering_roi_v2 import initial_temporal_state,select_engineering_roi_v2
from src.local_corrector.depth_envelope_guard import apply_depth_envelope_guard
METHODS={'SupportRisk_Base':(.10,False,False),'SupportRisk_Temporal':(.10,True,False),'SupportRisk_DepthGuard':(.10,False,True),'EngineeringROI_v2':(None,True,True)}
def validate_args(method,budget):
 if method not in METHODS or budget not in (.05,.1,.2):raise ValueError('ENGINEERING_ROI_V2_METHOD_INVALID')
 if method!='EngineeringROI_v2' and budget!=.1:raise ValueError('ENGINEERING_ROI_V2_ABLATION_B10_ONLY')
def v2_step(current,provisional,metadata,config,budget,state,method,local_proposal,active):
 _,temporal,depth=METHODS[method];selected,new_state,telemetry=select_engineering_roi_v2(current,provisional,metadata,config,budget,state,temporal_refresh=temporal)
 # Caller applies frozen Local, existing SupportGuard, then existing MomentumGuard before this h-only envelope.
 corrected=local_proposal;guard={'depth_guard_activation_fraction':0.,'depth_guard_mean_abs_clip_m':0.,'depth_guard_max_abs_clip_m':0.}
 if depth:corrected,guard=apply_depth_envelope_guard(current,provisional,corrected,active)
 return corrected,selected,new_state,telemetry|guard
def main(args):
 validate_args(args.method,args.budget);raise RuntimeError('ENGINEERING_ROI_V2_NOT_AUTHORIZED_FOR_ROLLOUT')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--method',required=True,choices=METHODS);p.add_argument('--budget',required=True,type=float);main(p.parse_args())
