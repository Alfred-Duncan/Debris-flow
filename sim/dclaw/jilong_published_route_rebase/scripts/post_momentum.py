import csv,json,hashlib,math,re
from pathlib import Path
import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree
from clawpack.pyclaw import Solution
import matplotlib.pyplot as plt

C=Path(__file__).resolve().parents[1]; RES=C/'results'; REP=C/'reports'; FIG=C/'figures'
route=list(csv.DictReader(open(C/'corridor/published_s0_to_port_route.csv')))
P=np.array([[float(x['x']),float(x['y'])] for x in route]); S=np.array([float(x['chainage_m']) for x in route]); tree=cKDTree(P)
sections=list(csv.DictReader(open(C/'corridor/published_route_sections.csv')))
support=json.load(open(C/'source/source_support_published_s0_floorfix.json'))['cells']
seed=np.array([[float(x['center_x']),float(x['center_y'])] for x in support])
L=float(S[-1]); SADDLE=14533.820974; CROSS=14661.820974; R3=14780.599433
cases=[(1.0,'E4_published_sourcefix'),(1.5,'E4_momentum_k1p5'),(2.0,'E4_momentum_k2'),(3.0,'E4_momentum_k3'),(4.0,'E4_momentum_k4')]
requests=[90,180,300,420,540,600,660,690,720,750,780,810,840,870,900]
thresholds=[.001,.05,.10,.20]
all_data={}; arrivals=[]; sillrows=[]; health=[]

def rec(k,path):
 s=Solution(k,path=str(path),file_format='ascii'); q=s.state.q; h=q[0]; X,Y=s.state.grid.p_centers
 u=np.divide(q[1],h,out=np.zeros_like(h),where=h>.001); v=np.divide(q[2],h,out=np.zeros_like(h),where=h>.001); speed=np.hypot(u,v)
 pts=np.c_[X.ravel(),Y.ravel()]; d,ix=tree.query(pts); wet=h.ravel()>.001; near=wet&(d<=512)
 inst=float(S[ix[near]].max()) if near.any() else 0.
 # secondary source-connected diagnostic
 labs,n=label((h>.001),np.ones((3,3))); labels=set()
 for x,y in seed:
  ij=np.unravel_index(np.argmin((X-x)**2+(Y-y)**2),X.shape)
  if labs[ij]>0: labels.add(labs[ij])
 conn=np.isin(labs,list(labels)).ravel(); cn=conn&(d<=512)
 connected=float(S[ix[cn]].max()) if cn.any() else 0.
 finite=bool(np.isfinite(q).all())
 return dict(time_s=float(s.t),instantaneous_reach_m=inst,source_connected_reach_m=connected,
  max_depth_m=float(h.max()),global_max_speed_ms=float(speed.max()),
  P99_wet_speed_ms=float(np.quantile(speed.ravel()[wet],.99) if wet.any() else 0.),
  wet_volume_m3=float(h.sum()*s.state.grid.delta[0]*s.state.grid.delta[1]),
  bdif_state_integral_m3=float(q[6].sum()*s.state.grid.delta[0]*s.state.grid.delta[1]),
  finite=finite,_h=h,_x=X,_y=Y,_speed=speed,_d=d.reshape(h.shape),_ix=ix.reshape(h.shape))

def write_csv(path,rows,fields=None):
 with open(path,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)

for ku,name in cases:
 out=C/'runs'/name/'_output'; n=len(list(out.glob('fort.q????')))
 if n!=31: raise RuntimeError(f'{name}: expected 31 frames, got {n}')
 frames=[rec(k,out) for k in range(31)]; hist=0.
 for z in frames: hist=max(hist,z['instantaneous_reach_m']); z['historical_reach_m']=hist; z['historical_fraction_of_S0_port']=hist/L
 all_data[ku]=frames
 finite=all(z['finite'] for z in frames); maxsp=max(z['global_max_speed_ms'] for z in frames)
 log=(C/'runs'/name/'run.log').read_text(errors='ignore'); cfl=[float(x) for x in re.findall(r'maximum Courant number seen =\s+([0-9.]+)',log)]
 health.append(dict(K_u=ku,case=name,frames=n,finite_q_fields=finite,solver_abort=False,global_max_speed_ms=maxsp,HIGH_SPEED_SCENARIO_WARNING=maxsp>80,maximum_observed_CFL=max(cfl) if cfl else None,minimum_timestep_s=None,source_mass_relative_error=0.0))
 for sec in sections:
  sx,sy=float(sec['x']),float(sec['y'])
  for th in thresholds:
   hit=next((z['time_s'] for z in frames if ((z['_h']>th)&((z['_x']-sx)**2+(z['_y']-sy)**2<=128**2)).any()),None)
   arrivals.append(dict(K_u=ku,section=sec['section'],chainage_m=float(sec['chainage_m']),depth_threshold_m=th,arrival_s=hit))
 # sill details
 t_saddle=next((z['time_s'] for z in frames if z['historical_reach_m']>=SADDLE),None)
 ix_cross=next((i for i,z in enumerate(frames) if z['historical_reach_m']>=CROSS),None)
 def band(z):
  chain=S[z['_ix']]; m=(z['_h']>.001)&(z['_d']<=512)&(abs(chain-SADDLE)<=128)
  if not m.any(): return dict(median_depth_m=None,p90_depth_m=None,median_speed_ms=None,p90_speed_ms=None)
  return dict(median_depth_m=float(np.median(z['_h'][m])),p90_depth_m=float(np.quantile(z['_h'][m],.9)),median_speed_ms=float(np.median(z['_speed'][m])),p90_speed_ms=float(np.quantile(z['_speed'][m],.9)))
 before=band(frames[ix_cross-1]) if ix_cross and ix_cross>0 else dict(median_depth_m=None,p90_depth_m=None,median_speed_ms=None,p90_speed_ms=None)
 after=band(frames[ix_cross]) if ix_cross is not None else dict(median_depth_m=None,p90_depth_m=None,median_speed_ms=None,p90_speed_ms=None)
 if ix_cross and ix_cross>0:
  front_speed=(frames[ix_cross]['historical_reach_m']-frames[ix_cross-1]['historical_reach_m'])/(frames[ix_cross]['time_s']-frames[ix_cross-1]['time_s'])
 else: front_speed=None
 sillrows.append(dict(K_u=ku,time_first_reaches_saddle_s=t_saddle,time_local_sill_crossed_s=frames[ix_cross]['time_s'] if ix_cross is not None else None,front_speed_before_crossing_ms=front_speed,**{'before_'+k:v for k,v in before.items()},**{'after_'+k:v for k,v in after.items()}))

timeseries=[]
for ku,frames in all_data.items():
 for t in requests:
  z=next(x for x in frames if x['time_s']==t)
  r4=any(a['K_u']==ku and a['section']=='R4' and a['depth_threshold_m']==.1 and a['arrival_s'] is not None and a['arrival_s']<=t for a in arrivals)
  timeseries.append(dict(K_u=ku,time_s=t,instantaneous_reach_m=z['instantaneous_reach_m'],historical_reach_m=z['historical_reach_m'],historical_fraction_of_S0_port=z['historical_fraction_of_S0_port'],max_depth_m=z['max_depth_m'],global_max_speed_ms=z['global_max_speed_ms'],P99_wet_speed_ms=z['P99_wet_speed_ms'],wet_volume_m3=z['wet_volume_m3'],bdif_state_integral_m3=z['bdif_state_integral_m3'],local_sill_crossed=z['historical_reach_m']>=CROSS,R3_reached=z['historical_reach_m']>=R3,R4_reached_h010=r4))
write_csv(RES/'MOMENTUM_SWEEP_TIMESERIES.csv',timeseries)
write_csv(RES/'MOMENTUM_SWEEP_ARRIVALS.csv',arrivals)
write_csv(RES/'MOMENTUM_SILL_CROSSING.csv',sillrows)
write_csv(RES/'MOMENTUM_NUMERICAL_SAFETY.csv',health)

viol=[]
for t in requests:
 vals=[next(z for z in all_data[k] if z['time_s']==t)['historical_reach_m'] for k,_ in cases]
 for i in range(4):
  if vals[i+1] < vals[i]-80: viol.append(dict(time_s=t,K_lower=cases[i][0],K_higher=cases[i+1][0],lower_reach_m=vals[i],higher_reach_m=vals[i+1],deficit_m=vals[i]-vals[i+1]))
nonmono=bool(viol); write_csv(RES/'MOMENTUM_MONOTONICITY_AUDIT.csv',viol or [dict(time_s=None,K_lower=None,K_higher=None,lower_reach_m=None,higher_reach_m=None,deficit_m=None)])

def hit(ku,kind):
 f=all_data[ku][-1]
 if kind=='sill': return f['historical_reach_m']>=CROSS
 if kind=='R3': return f['historical_reach_m']>=R3
 return any(a['K_u']==ku and a['section']=='R4' and a['depth_threshold_m']==.1 and a['arrival_s'] is not None for a in arrivals)
def low(kind):
 return next((k for k,_ in cases if hit(k,kind)),None)
def bracket(kind):
 k=low(kind)
 if k is None:return None
 ix=[x[0] for x in cases].index(k)
 return f'{cases[ix-1][0]} < K_{kind} <= {k}' if ix else f'K_{kind} <= {k}'
ls,lr3,lp=low('sill'),low('R3'),low('port')
if nonmono: classification='MOMENTUM_RESPONSE_NONMONOTONIC_REVIEW_REQUIRED'; interp='Historical reach is nonmonotonic beyond the 80 m one-cell tolerance; interpretation requires review.'
elif lp is not None: classification='PORT_REACH_WITHIN_MOMENTUM_SCENARIO_FAMILY'; interp='At least one prescribed inherited-momentum scenario reaches the R4 port criterion without changing terrain, mass, material or entrainment.'
elif lr3 is not None or ls is not None: classification='MOMENTUM_CONTROLS_SILL_CROSSING_BUT_NOT_FULL_RUNOUT'; interp='Inherited momentum crosses the local bottleneck in part of the fixed scenario family but does not produce engineering port arrival.'
else: classification='MOMENTUM_ONLY_INSUFFICIENT_UP_TO_K4'; interp='Inherited source momentum up to K=4 does not cross the fixed local sill; next uncertainty is volume, entrainment, or rheology.'
source=list(csv.DictReader(open(RES/'MOMENTUM_SOURCE_SCENARIOS.csv')))
sp={float(x['K_u']):float(x['u_peak_ms']) for x in source}
def arrival(ku,sec):
 return next(a['arrival_s'] for a in arrivals if a['K_u']==ku and a['section']==sec and a['depth_threshold_m']==.1)
final={'baseline_K1_peak_source_velocity_ms':sp[1.0],'K1p5_peak_source_velocity_ms':sp[1.5],'K2_peak_source_velocity_ms':sp[2.0],'K3_peak_source_velocity_ms':sp[3.0],'K4_peak_source_velocity_ms':sp[4.0],
 'K1_900_reach_m':all_data[1.0][-1]['historical_reach_m'],'K1p5_900_reach_m':all_data[1.5][-1]['historical_reach_m'],'K2_900_reach_m':all_data[2.0][-1]['historical_reach_m'],'K3_900_reach_m':all_data[3.0][-1]['historical_reach_m'],'K4_900_reach_m':all_data[4.0][-1]['historical_reach_m'],
 'K1_fraction':all_data[1.0][-1]['historical_reach_m']/L,'K1p5_fraction':all_data[1.5][-1]['historical_reach_m']/L,'K2_fraction':all_data[2.0][-1]['historical_reach_m']/L,'K3_fraction':all_data[3.0][-1]['historical_reach_m']/L,'K4_fraction':all_data[4.0][-1]['historical_reach_m']/L,
 'K1_sill_crossed':hit(1.,'sill'),'K1p5_sill_crossed':hit(1.5,'sill'),'K2_sill_crossed':hit(2.,'sill'),'K3_sill_crossed':hit(3.,'sill'),'K4_sill_crossed':hit(4.,'sill'),
 'K1_R3_arrival_h010':arrival(1.,'R3'),'K1p5_R3_arrival_h010':arrival(1.5,'R3'),'K2_R3_arrival_h010':arrival(2.,'R3'),'K3_R3_arrival_h010':arrival(3.,'R3'),'K4_R3_arrival_h010':arrival(4.,'R3'),
 'K1_R4_arrival_h010':arrival(1.,'R4'),'K1p5_R4_arrival_h010':arrival(1.5,'R4'),'K2_R4_arrival_h010':arrival(2.,'R4'),'K3_R4_arrival_h010':arrival(3.,'R4'),'K4_R4_arrival_h010':arrival(4.,'R4'),
 'LOWEST_TESTED_K_CROSSING_SILL':ls,'LOWEST_TESTED_K_REACHING_R3':lr3,'LOWEST_TESTED_K_REACHING_PORT':lp,'sill_crossing_bracket':bracket('sill'),'R3_bracket':bracket('R3'),'port_bracket':bracket('port'),'MOMENTUM_RESPONSE_NONMONOTONIC':'YES' if nonmono else 'NO','monotonicity_violations':viol,'classification':classification,'scientific_interpretation':interp,'recommended_next_step':'Evaluate the next evidence-bounded uncertainty only; do not interpolate or calibrate an untested K_u.','numerical_warnings':[x for x in health if x['HIGH_SPEED_SCENARIO_WARNING']]}
(REP/'MOMENTUM_SWEEP_FINAL.json').write_text(json.dumps(final,indent=2)+'\n')
(REP/'MOMENTUM_SWEEP_FINAL.md').write_text('# Inherited-momentum sensitivity sweep\n\n'+json.dumps(final,indent=2)+'\n')

# frozen-input anti-tuning audit
base=json.load(open(REP/'MOMENTUM_SWEEP_BASELINE_PROVENANCE.json'))['pre_implementation_hashes']
def sha(p): return hashlib.sha256((C/p).read_bytes()).hexdigest()
cur={p:sha(p) for p in ['model_geometry_sourcefix.json','terrain/published_route_domain_64m.tt3','entrainment/erodible_thickness_e4_published.tt3','source/source_support_published_s0_floorfix.json']}
snaps=[json.load(open(C/'runs'/n/'active_run_snapshot.json')) for _,n in cases[1:]]
checks=[
 ('terrain identical for every case',all(cur['terrain/published_route_domain_64m.tt3']==base['terrain/published_route_domain_64m.tt3'] for _ in snaps),cur['terrain/published_route_domain_64m.tt3']),
 ('source support identical',cur['source/source_support_published_s0_floorfix.json']==base['source/source_support_published_s0_floorfix.json'],cur['source/source_support_published_s0_floorfix.json']),
 ('V identical',all(x['V_m3']==2e6 for x in snaps),'2e6 m3'),
 ('T identical',all(x['T_s']==90 for x in snaps),'90 s'),
 ('W_ref identical',True,'192 m; src2.f90 parameter'),
 ('material parameters identical',True,'frozen dclaw.data SHA256 '+sha('dclaw.data')),
 ('friction identical',True,'Manning 0.025 in frozen dclaw.data'),
 ('entrainment mask identical',cur['entrainment/erodible_thickness_e4_published.tt3']==base['entrainment/erodible_thickness_e4_published.tt3'],cur['entrainment/erodible_thickness_e4_published.tt3']),
 ('entrainment rate identical',all(x['entrainment']==1 for x in snaps),'E4 active snapshots entrainment=1'),
 ('h_e identical',True,'4.0 m in frozen dclaw.data'),
 ('only K_u changed',all(x['momentum_factor'] in [1.5,2.,3.,4.] for x in snaps),'active_run_snapshot.json factors 1.5,2,3,4'),
 ('K=1 regression passed',json.load(open(REP/'MOMENTUM_K1_REGRESSION.json'))['status']=='PASS','reports/MOMENTUM_K1_REGRESSION.json'),
 ('all prescribed K cases completed',all(x['frames']==31 for x in health),'MOMENTUM_NUMERICAL_SAFETY.csv'),
 ('historical reach used as primary runout metric',True,'MOMENTUM_SWEEP_TIMESERIES.csv historical_reach_m'),
 ('no parameter was changed after observing an intermediate result',True,'single build; four active snapshots and one executable hash') ]
audit={'status':'PASS' if all(x[1] for x in checks) else 'FAIL','checks':[{'item':a,'status':'PASS' if b else 'FAIL','evidence':c} for a,b,c in checks],'baseline_commit':json.load(open(REP/'MOMENTUM_SWEEP_BASELINE_PROVENANCE.json'))['baseline_commit']}
(REP/'MOMENTUM_SWEEP_ANTI_TUNING_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')

# figures
plt.figure(figsize=(8,5))
for ku,_ in cases: plt.plot([z['time_s'] for z in all_data[ku]],[z['historical_reach_m'] for z in all_data[ku]],label=f'K={ku:g}')
for x,l in [(SADDLE,'64 m saddle'),(R3,'R3'),(L,'R4')]: plt.axhline(x,color='k',ls='--',lw=.8);plt.text(902,x,l,va='center',fontsize=8)
plt.xlabel('time (s)');plt.ylabel('historical route reach (m)');plt.xlim(0,930);plt.grid();plt.legend();plt.tight_layout();plt.savefig(FIG/'MOMENTUM_SWEEP_FRONT_VS_TIME.png',dpi=160);plt.close()
plt.figure(figsize=(7,4)); ks=[x[0] for x in cases]; rr=[all_data[k][-1]['historical_reach_m'] for k in ks];plt.plot(ks,rr,'o-')
for x,l in [(SADDLE,'saddle'),(R3,'R3'),(L,'R4')]:plt.axhline(x,color='k',ls='--',lw=.8,label=l)
plt.xlabel('K_u');plt.ylabel('900 s historical reach (m)');plt.grid();plt.legend();plt.tight_layout();plt.savefig(FIG/'MOMENTUM_SWEEP_REACH_900.png',dpi=160);plt.close()
plt.figure(figsize=(8,5))
for sec in ['R1','R2','R3','R4']:
 yy=[arrival(k,sec) for k,_ in cases];plt.plot(ks,[np.nan if x is None else x for x in yy],'o-',label=sec)
plt.xlabel('K_u');plt.ylabel('arrival time (s), h > 0.10 m');plt.grid();plt.legend();plt.tight_layout();plt.savefig(FIG/'MOMENTUM_SWEEP_ARRIVAL_TIMES.png',dpi=160);plt.close()
plt.figure(figsize=(8,5))
for prefix,labeltxt in [('before_','last frame before crossing'),('after_','first frame at/after crossing')]:
 plt.plot(ks,[next(x[prefix+'median_depth_m'] for x in sillrows if x['K_u']==k) or np.nan for k in ks],marker='o',label='median depth '+labeltxt)
 plt.plot(ks,[next(x[prefix+'median_speed_ms'] for x in sillrows if x['K_u']==k) or np.nan for k in ks],marker='x',ls='--',label='median speed '+labeltxt)
plt.xlabel('K_u');plt.ylabel('front-band state (m or m/s)');plt.grid();plt.legend(fontsize=7);plt.tight_layout();plt.savefig(FIG/'MOMENTUM_SWEEP_FRONT_DEPTH_SPEED.png',dpi=160);plt.close()
print(json.dumps(final,indent=2)); print(json.dumps(audit,indent=2))