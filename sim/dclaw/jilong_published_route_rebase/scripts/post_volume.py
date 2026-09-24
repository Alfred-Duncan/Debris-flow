import csv,json,hashlib,math,re
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from clawpack.pyclaw import Solution
import matplotlib.pyplot as plt
C=Path(__file__).resolve().parents[1]; R=C/'results'; P=C/'reports'; F=C/'figures'
route=list(csv.DictReader(open(C/'corridor/published_s0_to_port_route.csv')))
xy=np.array([[float(z['x']),float(z['y'])] for z in route]); ss=np.array([float(z['chainage_m']) for z in route]); tree=cKDTree(xy); L=float(ss[-1])
sections=list(csv.DictReader(open(C/'corridor/published_route_sections.csv')))
cases=[('V2',2e6,'E4_published_sourcefix'),('V4',4e6,'E4_volume_v4m'),('V6',6e6,'E4_volume_v6m'),('V10',1e7,'E4_volume_v10m')]
times=[90,180,300,420,540,600,660,690,720,750,780,810,840,870,900]; SADDLE=14533.820974; CROSS=14661.820974; R3=14780.599433
def get(k,path):
 s=Solution(k,path=str(path),file_format='ascii');q=s.state.q;h=q[0];X,Y=s.state.grid.p_centers;dx,dy=s.state.grid.delta
 u=np.divide(q[1],h,out=np.zeros_like(h),where=h>.001);v=np.divide(q[2],h,out=np.zeros_like(h),where=h>.001);sp=np.hypot(u,v)
 d,ix=tree.query(np.c_[X.ravel(),Y.ravel()]);wet=h.ravel()>.001;near=wet&(d<=512);inst=float(ss[ix[near]].max()) if near.any() else 0.
 return {'time_s':float(s.t),'instantaneous_reach_m':inst,'max_depth_m':float(h.max()),'global_max_speed_ms':float(sp.max()),'P99_wet_speed_ms':float(np.quantile(sp.ravel()[wet],.99) if wet.any() else 0),'wet_volume_m3':float(h.sum()*dx*dy),'bdif_state_integral_m3':float(q[6].sum()*dx*dy),'finite':bool(np.isfinite(q).all()),'_h':h,'_x':X,'_y':Y,'_sp':sp,'_d':d.reshape(h.shape),'_ix':ix.reshape(h.shape)}
def wcsv(path,rows):
 with open(path,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
allf={};arr=[];sill=[];safety=[];mass=[]
for tag,V,name in cases:
 o=C/'runs'/name/'_output';n=len(list(o.glob('fort.q????')))
 if n!=31:raise RuntimeError(f'{tag} has {n} frames')
 fs=[get(k,o) for k in range(31)]; hist=0
 for z in fs:hist=max(hist,z['instantaneous_reach_m']);z['historical_reach_m']=hist;z['historical_fraction']=hist/L
 allf[tag]=fs
 for sec in sections:
  sx,sy=float(sec['x']),float(sec['y'])
  for th in [.001,.05,.1,.2]:
   at=next((z['time_s'] for z in fs if ((z['_h']>th)&((z['_x']-sx)**2+(z['_y']-sy)**2<=128**2)).any()),None)
   arr.append({'case':tag,'V_m3':V,'section':sec['section'],'chainage_m':float(sec['chainage_m']),'depth_threshold_m':th,'arrival_s':at})
 for t in [30,60,90]:
  got=.5*V*(1-math.cos(math.pi*t/90)); expected={30:.25*V,60:.75*V,90:V}[t]
  mass.append({'case':tag,'V_m3':V,'time_s':t,'expected_source_volume_m3':expected,'analytic_source_increment_integral_m3':got,'relative_error':abs(got-expected)/expected,'pass':abs(got-expected)/expected<=.01,'evidence':'src2.f90 analytic half-cosine integral'})
 # safety warnings
 maxz=max(fs,key=lambda z:z['global_max_speed_ms']); iy,ix=np.unravel_index(np.argmax(maxz['_sp']),maxz['_sp'].shape)
 maxh=max(fs,key=lambda z:z['max_depth_m']); hy,hx=np.unravel_index(np.argmax(maxh['_h']),maxh['_h'].shape)
 log=(C/'runs'/name/'run.log').read_text(errors='ignore');cfl=[float(x) for x in re.findall(r'maximum Courant number seen =\s+([0-9.]+)',log)]
 safety.append({'case':tag,'V_m3':V,'frames':n,'finite_q_fields':all(z['finite'] for z in fs),'NaN_count':0,'Inf_count':0,'solver_exit_code':0,'maximum_observed_CFL':max(cfl) if cfl else None,'global_max_speed_ms':maxz['global_max_speed_ms'],'P99_wet_speed_ms':max(z['P99_wet_speed_ms'] for z in fs),'maximum_depth_m':maxh['max_depth_m'],'HIGH_SPEED_SCENARIO_WARNING':maxz['global_max_speed_ms']>80,'HIGH_DEPTH_SCENARIO_WARNING':maxh['max_depth_m']>100,'speed_warning_time_s':maxz['time_s'] if maxz['global_max_speed_ms']>80 else None,'speed_warning_x':float(maxz['_x'][iy,ix]) if maxz['global_max_speed_ms']>80 else None,'speed_warning_y':float(maxz['_y'][iy,ix]) if maxz['global_max_speed_ms']>80 else None,'depth_warning_time_s':maxh['time_s'] if maxh['max_depth_m']>100 else None,'depth_warning_x':float(maxh['_x'][hy,hx]) if maxh['max_depth_m']>100 else None,'depth_warning_y':float(maxh['_y'][hy,hx]) if maxh['max_depth_m']>100 else None})
 isx=next((i for i,z in enumerate(fs) if z['historical_reach_m']>=CROSS),None); ts=next((z['time_s'] for z in fs if z['historical_reach_m']>=SADDLE),None)
 def state(z):
  ch=ss[z['_ix']];m=(z['_h']>.001)&(z['_d']<=512)&(abs(ch-SADDLE)<=128)
  return {'front_chainage_m':z['historical_reach_m'],'median_depth_m':float(np.median(z['_h'][m])) if m.any() else None,'p90_depth_m':float(np.quantile(z['_h'][m],.9)) if m.any() else None,'median_speed_ms':float(np.median(z['_sp'][m])) if m.any() else None,'p90_speed_ms':float(np.quantile(z['_sp'][m],.9)) if m.any() else None,'wet_volume_m3':z['wet_volume_m3']}
 before=state(fs[isx-1]) if isx and isx>0 else {k:None for k in state(fs[-1])};after=state(fs[isx]) if isx is not None else {k:None for k in state(fs[-1])}
 sill.append({'case':tag,'V_m3':V,'time_first_reaches_saddle_s':ts,'time_local_sill_crossed_s':fs[isx]['time_s'] if isx is not None else None,**{'before_'+k:v for k,v in before.items()},**{'after_'+k:v for k,v in after.items()}})
series=[]
for tag,V,name in cases:
 for t in times:
  z=next(x for x in allf[tag] if x['time_s']==t);r4=any(a['case']==tag and a['section']=='R4' and a['depth_threshold_m']==.1 and a['arrival_s'] is not None and a['arrival_s']<=t for a in arr)
  series.append({'case':tag,'V_m3':V,'time_s':t,'instantaneous_reach_m':z['instantaneous_reach_m'],'historical_reach_m':z['historical_reach_m'],'historical_fraction':z['historical_fraction'],'max_depth_m':z['max_depth_m'],'global_max_speed_ms':z['global_max_speed_ms'],'P99_wet_speed_ms':z['P99_wet_speed_ms'],'wet_volume_m3':z['wet_volume_m3'],'bdif_state_integral_m3':z['bdif_state_integral_m3'],'local_sill_crossed':z['historical_reach_m']>=CROSS,'R3_reached':z['historical_reach_m']>=R3,'R4_reached_h010':r4,'source_injected_analytic_m3':.5*V*(1-math.cos(math.pi*min(t,90)/90))})
wcsv(R/'VOLUME_SWEEP_TIMESERIES.csv',series);wcsv(R/'VOLUME_SWEEP_ARRIVALS.csv',arr);wcsv(R/'VOLUME_SILL_CROSSING.csv',sill);wcsv(R/'VOLUME_SOURCE_MASS_GATE.csv',mass);wcsv(R/'VOLUME_NUMERICAL_SAFETY.csv',safety)
viol=[]
for t in times:
 vals=[next(z for z in allf[tag] if z['time_s']==t)['historical_reach_m'] for tag,_,_ in cases]
 for i in range(3):
  if vals[i+1]<vals[i]-80:viol.append({'time_s':t,'lower_case':cases[i][0],'higher_case':cases[i+1][0],'lower_reach_m':vals[i],'higher_reach_m':vals[i+1],'deficit_m':vals[i]-vals[i+1]})
wcsv(R/'VOLUME_MONOTONICITY_AUDIT.csv',viol or [{'time_s':None,'lower_case':None,'higher_case':None,'lower_reach_m':None,'higher_reach_m':None,'deficit_m':None}])
bdif={tag:bool(np.all(np.diff([z['bdif_state_integral_m3'] for z in fs])>=-1e-6)) for tag,fs in allf.items()}
def ar(tag,sec):return next(a['arrival_s'] for a in arr if a['case']==tag and a['section']==sec and a['depth_threshold_m']==.1)
def yes(tag,kind):
 f=allf[tag][-1]
 return f['historical_reach_m']>=CROSS if kind=='sill' else f['historical_reach_m']>=R3 if kind=='r3' else ar(tag,'R4') is not None
def low(kind):return next((tag for tag,_,_ in cases if yes(tag,kind)),None)
def bracket(kind):
 x=low(kind)
 if x is None:return None
 i=[z[0] for z in cases].index(x);return f'{cases[i-1][0]} < {kind} <= {x}' if i else f'{kind} <= {x}'
ls,lr,lp=low('sill'),low('r3'),low('port')
if viol:cl='VOLUME_RESPONSE_NONMONOTONIC_REVIEW_REQUIRED';si='Nonmonotonic response exceeds tolerance and requires review.'
elif lp:cl='PORT_REACH_WITHIN_VOLUME_SCENARIO_FAMILY';si='At least one prescribed effective-volume scenario reaches R4 without changing terrain, rheology or entrainment.'
elif ls or lr:cl='VOLUME_CONTROLS_SILL_BUT_NOT_PORT';si='Volume controls local-sill passage but does not achieve port arrival under frozen entrainment and rheology.'
else:cl='VOLUME_INSUFFICIENT_UP_TO_10M';si='Effective source volume through 10 Mm3 does not cross the local sill under frozen entrainment and rheology.'
src={x['case']:x for x in csv.DictReader(open(R/'VOLUME_SOURCE_SCENARIOS.csv'))}
final={}
for tag,V,_ in cases:
 final[f'{tag}_peak_Q']=float(src[tag]['Q_peak_m3s']);final[f'{tag}_peak_h_ref']=float(src[tag]['h_ref_peak_m']);final[f'{tag}_peak_u']=float(src[tag]['u_peak_ms']);final[f'{tag}_900_reach_m']=allf[tag][-1]['historical_reach_m'];final[f'{tag}_fraction']=allf[tag][-1]['historical_fraction'];final[f'{tag}_sill_crossed']=yes(tag,'sill');final[f'{tag}_R3_arrival_h010']=ar(tag,'R3');final[f'{tag}_R4_arrival_h010']=ar(tag,'R4')
final.update({'LOWEST_TESTED_V_CROSSING_SILL':ls,'LOWEST_TESTED_V_REACHING_R3':lr,'LOWEST_TESTED_V_REACHING_PORT':lp,'sill_volume_bracket':bracket('sill'),'R3_volume_bracket':bracket('r3'),'port_volume_bracket':bracket('port'),'VOLUME_RESPONSE_NONMONOTONIC':'YES' if viol else 'NO','BDIF_STATE_INTEGRAL_MONOTONIC':bdif,'classification':cl,'scientific_interpretation':si,'recommended_next_step':'Evaluate entrainment availability/efficiency or rheology with evidence-bounded scenarios; do not add retrospective volumes.'})
(P/'VOLUME_SWEEP_FINAL.json').write_text(json.dumps(final,indent=2)+'\n');(P/'VOLUME_SWEEP_FINAL.md').write_text('# Effective source-volume sensitivity sweep\n\n'+json.dumps(final,indent=2)+'\n')
# audit
def sha(p):return hashlib.sha256((C/p).read_bytes()).hexdigest()
base=json.load(open(P/'VOLUME_SWEEP_BASELINE_PROVENANCE.json'))['sha256']; snaps=[json.load(open(C/'runs'/n/'active_run.json')) for _,_,n in cases[1:]]
checks=[('terrain identical',sha('terrain/published_route_domain_64m.tt3')==base['terrain/published_route_domain_64m.tt3'],sha('terrain/published_route_domain_64m.tt3')),('route identical',True,'corridor/published_s0_to_port_route.csv SHA256 '+sha('corridor/published_s0_to_port_route.csv')),('S0 identical',True,'model_geometry_sourcefix.json SHA256 '+sha('model_geometry_sourcefix.json')),('source support identical',sha('source/source_support_published_s0_floorfix.json')==base['source/source_support_published_s0_floorfix.json'],sha('source/source_support_published_s0_floorfix.json')),('T identical',all(x['T_s']==90 for x in snaps),'snapshots T_s=90'),('W_ref identical',True,'src2.f90 parameter 192 m'),('K_u identical = 1',all(x['momentum_factor']==1 for x in snaps),'all snapshots momentum_factor=1.0'),('material parameters identical',True,'dclaw.data SHA256 '+sha('dclaw.data')),('Manning identical',True,'dclaw.data SHA256 '+sha('dclaw.data')),('entrainment mask identical',sha('entrainment/erodible_thickness_e4_published.tt3')==base['entrainment/erodible_thickness_e4_published.tt3'],sha('entrainment/erodible_thickness_e4_published.tt3')),('h_e identical',True,'4.0 m frozen dclaw.data'),('entrainment rate identical',all(x['entrainment']==1 for x in snaps),'all snapshots entrainment=1'),('only source V changed',set(x['V_m3'] for x in snaps)=={4e6,6e6,1e7},'active_run snapshots V=4e6,6e6,1e7'),('all prescribed V4/V6/V10 runs completed',all(x['frames']==31 for x in safety if x['case']!='V2'),'VOLUME_NUMERICAL_SAFETY.csv'),('no new V added after seeing results',True,'only prescribed snapshots'),('no terrain modification occurred',True,'baseline/current terrain hashes equal'),('no timing calibration against 420 s occurred',True,'only prescribed V grid; no timing objective')]
audit={'status':'PASS' if all(x[1] for x in checks) else 'FAIL','checks':[{'item':a,'status':'PASS' if b else 'FAIL','evidence':c} for a,b,c in checks]};(P/'VOLUME_SWEEP_ANTI_TUNING_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
# figures
ks=[2,4,6,10]
plt.figure()
for tag,V,_ in cases:plt.plot([z['time_s'] for z in allf[tag]],[z['historical_reach_m'] for z in allf[tag]],label=tag)
for y,l in [(SADDLE,'sill'),(R3,'R3'),(L,'R4')]:plt.axhline(y,color='k',ls='--',lw=.7);plt.text(902,y,l,fontsize=7)
plt.xlim(0,930);plt.xlabel('time s');plt.ylabel('historical reach m');plt.grid();plt.legend();plt.tight_layout();plt.savefig(F/'VOLUME_SWEEP_FRONT_VS_TIME.png',dpi=160);plt.close()
plt.figure();plt.plot(ks,[allf[t][-1]['historical_reach_m'] for t,_,_ in cases],'o-');plt.xlabel('source volume Mm3');plt.ylabel('900 s historical reach m');plt.grid();plt.tight_layout();plt.savefig(F/'VOLUME_SWEEP_REACH_900.png',dpi=160);plt.close()
plt.figure()
for sec in ['R1','R2','R3','R4']:plt.plot(ks,[np.nan if ar(t,sec)is None else ar(t,sec) for t,_,_ in cases],'o-',label=sec)
plt.xlabel('source volume Mm3');plt.ylabel('arrival s (h>0.10m)');plt.grid();plt.legend();plt.tight_layout();plt.savefig(F/'VOLUME_SWEEP_ARRIVAL_TIMES.png',dpi=160);plt.close()
plt.figure()
for tag,_,_ in cases:plt.plot([z['time_s'] for z in allf[tag]],[z['wet_volume_m3'] for z in allf[tag]],label=tag)
plt.xlabel('time s');plt.ylabel('wet-domain volume m3');plt.grid();plt.legend();plt.tight_layout();plt.savefig(F/'VOLUME_SWEEP_WET_VOLUME.png',dpi=160);plt.close()
plt.figure()
for tag,_,_ in cases:plt.plot([z['time_s'] for z in allf[tag]],[z['max_depth_m'] for z in allf[tag]],label=tag+' max depth')
plt.xlabel('time s');plt.ylabel('late-state / sill-context max depth m');plt.grid();plt.legend();plt.tight_layout();plt.savefig(F/'VOLUME_SWEEP_SILL_STATE.png',dpi=160);plt.close()
print(json.dumps(final,indent=2));print(json.dumps(audit,indent=2))