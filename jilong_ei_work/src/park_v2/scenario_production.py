"""Committed shared Park-v2 scenario-production helpers for formal and holdout cases."""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
INPUT=ROOT/'data/downloads/park_v2/inputs/upper30h.npz';RESTART=ROOT/'outputs/park_v2_h0/spinup/final_state.npz';SOLVER=ROOT/'src/park_v2/solver_sparse_v2.py'
def case_config(row):
 return {'scenario_id':row.scenario_id,'split':row.split,'design_index':int(row.design_index),'sampling_seed':20260920,'source':{'volume_scale':float(row.volume_scale),'volume_m3':float(row.volume_m3),'ice_fraction':float(row.ice_fraction),'rock_fraction':float(row.rock_fraction),'release_duration_s':30,'release_speed_m_s':0,'treatment':'progressive_at_rest'},'variable_parameters':{'erosion_K':float(row.erosion_K),'n_debris':float(row.n_debris),'dep_tau_s':float(row.dep_tau_s)}}
def solver_cmd(row,case):
 return [sys.executable,str(SOLVER),'--inputs',str(INPUT),'--restart',str(RESTART),'--out',str(case),'--release','--release-c0','1','--release-volume-scale',str(row.volume_scale),'--release-ice-fraction',str(row.ice_fraction),'--release-duration','30','--melt-eff','1','--melt-tau','0','--river-temp-c','12','--erosion-k',str(row.erosion_K),'--erosion-uc','6','--erosion-c','.7','--erodible-depth','5','--t-end','1440','--frame-dt','5','--save-frames','--mu-s','0','--n-w','.0208','--n-d',str(row.n_debris),'--cfl','.45','--inflow-scale','1','--dep-uc','1.833','--dep-tau',str(row.dep_tau_s),'--release-speed','0','--rock-temp-c','3','--q-air','300','--melt-energy','--device','cuda']
def run_one_case(row,case):
 import subprocess,numpy as np
 case.mkdir(parents=True,exist_ok=True);(case/'config.json').write_text(json.dumps(case_config(row),indent=2));subprocess.run(solver_cmd(row,case),cwd=ROOT,check=True)
 frames=case/'frames'; raw=[]
 for p in frames.glob('frame_*.npz'):
  with np.load(p) as z: raw.append((float(z['time_s']),p))
 keep=[]
 for t in range(10,1441,10):
  _,pick=min(raw,key=lambda q:abs(q[0]-t));keep.append(pick)
 for _,p in raw:
  if p not in keep:p.unlink()
 for t,p in zip(range(10,1441,10),keep):p.rename(frames/f'state_{t:04d}s.npz')
 write_initial_ml_frame_from_restart(frames/'state_0000s.npz')
 q={'status':'PASS' if (case/'final_state.npz').exists() and len(keep)==144 else 'FAIL','frames_saved':145}
 if q['status']!='PASS':raise RuntimeError(f'quality failed: {row.scenario_id}')
 (case/'quality.json').write_text(json.dumps(q,indent=2));(case/'DONE.json').write_text(json.dumps({'status':'PASS','scenario_id':row.scenario_id}))
def write_initial_ml_frame_from_restart(path):
 import numpy as np
 with np.load(RESTART) as s:
  h=np.asarray(s['h'],np.float32);wet=h>.05;idx=np.flatnonzero(wet.ravel());take=lambda x:np.asarray(x).ravel()[idx]
  np.savez_compressed(path,time_s=np.array(0.),idx=idx.astype(np.int32),h=take(h).astype(np.float16),hu=take(s['hu']).astype(np.float32),hv=take(s['hv']).astype(np.float32),c=(take(s['hc'])/np.maximum(take(h),1e-3)).astype(np.float16),ice=(take(s['hi'])/np.maximum(take(h),1e-3)).astype(np.float16),speed=(np.hypot(take(s['hu']),take(s['hv']))/np.maximum(take(h),1e-3)).astype(np.float16),dz_idx=np.array([],np.int32),dz=np.array([],np.float16))
