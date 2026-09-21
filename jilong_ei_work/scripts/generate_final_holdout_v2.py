"""Formal-only sealed Park-v2 final-holdout generator; never imported by training."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import qmc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.park_v2.scenario_production import case_config,run_one_case
DESIGN=ROOT/'configs/scenario_design';OUT=ROOT/'outputs/final_holdout'
def sobol_220():
 s=qmc.Sobol(d=5,scramble=True,seed=20260920);u=s.random_base2(8)[:220];log=lambda lo,hi,x:np.exp(np.log(lo)+x*(np.log(hi)-np.log(lo)))
 d=pd.DataFrame({'volume_scale':.8+.4*u[:,0],'ice_fraction':.1+.2*u[:,1],'erosion_K':log(.002,.010,u[:,2]),'n_debris':.012+.018*u[:,3],'dep_tau_s':log(150,2400,u[:,4])});d['volume_m3']=35420000*d.volume_scale;d['rock_fraction']=1-d.ice_fraction;d['sampling_seed']=20260920;d['design_index']=np.arange(1,221);d['split']=np.where(d.design_index.mod(10).eq(9),'VAL',np.where(d.design_index.mod(10).eq(0),'TEST','TRAIN'));return d
def reproduce_existing_design_check():
 expected=pd.read_csv(DESIGN/'JILONG_EI_SCENARIOS.csv');got=sobol_220().iloc[:200].copy();cols=['volume_scale','ice_fraction','erosion_K','n_debris','dep_tau_s','design_index'];return bool(len(expected)==200 and all(np.allclose(expected[c].to_numpy(float),got[c].to_numpy(float),rtol=0,atol=1e-12) for c in cols))
def write_reproduction_report():
 ok=reproduce_existing_design_check();p=ROOT/'reports/GLOBAL_V2_SOBOL_REPRODUCTION.json';p.write_text(json.dumps({'status':'PASS' if ok else 'FAIL','first_200_reproduced':ok,'seed':20260920,'design_only_not_generated':True},indent=2));return ok
def final_design():
 if not reproduce_existing_design_check():raise RuntimeError('FIRST_200_REPRODUCED failed; refusing holdout design')
 d=sobol_220().iloc[200:220].copy();d['scenario_id']=[f'FINAL_HOLDOUT_{i:03d}' for i in range(1,21)];d['split']='FINAL_HOLDOUT';return d[['scenario_id','volume_scale','volume_m3','ice_fraction','rock_fraction','erosion_K','n_debris','dep_tau_s','split','sampling_seed','design_index']]
def generate():
 d=final_design();DESIGN.mkdir(parents=True,exist_ok=True);d.to_csv(DESIGN/'JILONG_EI_FINAL_HOLDOUT.csv',index=False);OUT.mkdir(parents=True,exist_ok=True)
 for _,row in d.iterrows():
  case=OUT/row.scenario_id
  if (case/'DONE.json').exists():continue
  run_one_case(row,case)
 (OUT/'SEALED.json').write_text(json.dumps({'status':'SEALED','count':20,'first_200_reproduced':True},indent=2));return d
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--formal',action='store_true');a=p.parse_args()
 if not a.formal:raise SystemExit('Refusing holdout generation without --formal')
 generate()
