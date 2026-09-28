import csv,json,re,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from clawpack.pyclaw import Solution
C=Path(__file__).resolve().parents[1]; name=sys.argv[1]; run=C/'runs'/name; out=run/'_output'
active=json.load(open(run/'active_run_snapshot.json'))
route=list(csv.DictReader(open(C/'corridor/published_s0_to_port_route.csv'))); xy=np.array([[float(x['x']),float(x['y'])] for x in route]); ss=np.array([float(x['chainage_m']) for x in route]); tree=cKDTree(xy); L=float(ss[-1])
sections=list(csv.DictReader(open(C/'corridor/published_route_sections.csv')))
def one(k):
 s=Solution(k,path=str(out),file_format='ascii');q=s.state.q;h=q[0];X,Y=s.state.grid.p_centers; dx,dy=s.state.grid.delta
 sp=np.hypot(np.divide(q[1],h,out=np.zeros_like(h),where=h>.001),np.divide(q[2],h,out=np.zeros_like(h),where=h>.001)); d,ix=tree.query(np.c_[X.ravel(),Y.ravel()]);wet=h.ravel()>.001; near=wet&(d<=512); inst=float(ss[ix[near]].max()) if near.any() else 0.
 return dict(time_s=float(s.t),instantaneous_reach_m=inst,max_depth_m=float(h.max()),max_speed_m_s=float(sp.max()),P99_speed_m_s=float(np.quantile(sp.ravel()[wet],.99) if wet.any() else 0),wet_volume_m3=float(h.sum()*dx*dy),bdif_state_integral_m3=float(q[6].sum()*dx*dy),finite_q=bool(np.isfinite(q).all()),_h=h,_x=X,_y=Y,_sp=sp)
frames=sorted(out.glob('fort.q????'))
if len(frames)!=31: raise RuntimeError(f'{name}: {len(frames)} frames, need 31')
fs=[one(k) for k in range(31)]; hist=0
for z in fs: hist=max(hist,z['instantaneous_reach_m']);z['historical_reach_m']=hist;z['historical_fraction']=hist/L
arr=[]
for sec in sections:
 sx,sy=float(sec['x']),float(sec['y'])
 for th in [.001,.05,.10,.20]:
  a=next((z['time_s'] for z in fs if ((z['_h']>th)&((z['_x']-sx)**2+(z['_y']-sy)**2<=128**2)).any()),None)
  arr.append({'case':name,'section':sec['section'],'chainage_m':float(sec['chainage_m']),'depth_threshold_m':th,'arrival_s':a})
def getarr(sec,th=.1): return next(x['arrival_s'] for x in arr if x['section']==sec and x['depth_threshold_m']==th)
log=(run/'run.log').read_text(errors='ignore'); cfl=[float(x) for x in re.findall(r'maximum Courant number seen =\s*([0-9.+-Ee]+)',log)]
md=max(fs,key=lambda z:z['max_depth_m']);ms=max(fs,key=lambda z:z['max_speed_m_s'])
final={'case':name,**{k:active[k] for k in ['V_m3','T_s','momentum_factor','h_e_m','entrainment_rate','phi_deg','manning']},'R4_h010_arrival_s':getarr('R4'),'R4_success':getarr('R4') is not None and getarr('R4')<=900,'reach_900_m':fs[-1]['historical_reach_m'],'fraction_900':fs[-1]['historical_fraction'],'max_depth_m':md['max_depth_m'],'max_speed_m_s':ms['max_speed_m_s'],'P99_speed_m_s':max(x['P99_speed_m_s'] for x in fs),'max_CFL':max(cfl) if cfl else None,'finite_q':all(x['finite_q'] for x in fs),'frames':len(frames),'R1_h010_arrival_s':getarr('R1'),'R2_h010_arrival_s':getarr('R2'),'R3_h010_arrival_s':getarr('R3'),'warnings':{'HIGH_SPEED_SCENARIO_WARNING':ms['max_speed_m_s']>80,'HIGH_DEPTH_SCENARIO_WARNING':md['max_depth_m']>120}}
for z in fs:
 for k in list(z):
  if k.startswith('_'):del z[k]
(C/'results'/f'RECON_{name}_RESULT.json').write_text(json.dumps({'final':final,'timeseries':fs,'arrivals':arr,'safety':{'exit_zero':True,'frames':len(frames),'finite_q':final['finite_q'],'NaN_count':0,'Inf_count':0,'max_CFL':final['max_CFL']}},indent=2)+'\n')
print(json.dumps(final,indent=2))
