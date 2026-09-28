import csv,json
from pathlib import Path
import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree
from clawpack.pyclaw import Solution
import matplotlib.pyplot as plt
C=Path(__file__).parents[1];res=C/'results';rep=C/'reports';fig=C/'figures';route=list(csv.DictReader(open(C/'corridor/published_s0_to_port_route.csv')));P=np.array([[float(x['x']),float(x['y'])]for x in route]);S=np.array([float(x['chainage_m'])for x in route]);T=cKDTree(P);support=json.load(open(C/'source/source_support_published_s0_floorfix.json'))['cells'];seed=np.array([[x['center_x'],x['center_y']]for x in support]);mil=list(csv.DictReader(open(C/'corridor/published_route_sections.csv')))
def one(case,k):
 o=C/'runs'/case/'_output';s=Solution(k,path=str(o),file_format='ascii');q=s.state.q;X,Y=s.state.grid.p_centers;dx,dy=s.state.grid.delta;h=q[0];u=np.divide(q[1],h,out=np.zeros_like(h),where=h>.001);v=np.divide(q[2],h,out=np.zeros_like(h),where=h>.001);sp=np.hypot(u,v);wet=h>.001;lab,n=label(wet,np.ones((3,3)));a=set()
 for x,y in seed:
  ij=np.unravel_index(np.argmin((X-x)**2+(Y-y)**2),X.shape)
  if lab[ij]>0:a.add(lab[ij])
 w=np.isin(lab,list(a));d,ix=T.query(np.c_[X[w],Y[w]]) if w.any() else (np.array([]),np.array([],int));ok=d<=512
 if ok.any():j=np.argmax(S[ix][ok]);sel=np.where(w);inds=np.where(ok)[0];ii=inds[j];fx,fy=X[w][ii],Y[w][ii];front=float(S[ix][ii]);fd=float(d[ii])
 else:fx=fy=np.nan;front=fd=0.
 return dict(time_s=float(s.t),route_front_chainage_m=front,front_x=float(fx),front_y=float(fy),front_distance_to_route_m=fd,max_depth_m=float(h.max()),max_speed_ms=float(sp.max()),p99_wet_speed_ms=float(np.quantile(sp[w],.99)if w.any()else 0),wet_volume_m3=float(h[w].sum()*dx*dy),bdif_state_integral_m3=float(q[6].sum()*dx*dy)),(h,X,Y)
def run(case,prefix):
 a=[];cache=[]
 for k in range(31):r,c=one(case,k);a.append(r);cache.append(c)
 with open(res/f'{prefix}_front.csv','w',newline='')as f:w=csv.DictWriter(f,a[0].keys());w.writeheader();w.writerows(a)
 ar=[]
 for m in mil:
  x,y=float(m['x']),float(m['y'])
  for th in [.001,.05,.1,.2]:
   hit=None
   for r,(h,X,Y)in zip(a,cache):
    if ((h>th)&((X-x)**2+(Y-y)**2<=128**2)).any():hit=r['time_s'];break
   ar.append(dict(section=m['section'],chainage_m=m['chainage_m'],depth_threshold_m=th,arrival_s=hit))
 with open(res/f'{prefix}_arrivals.csv','w',newline='')as f:w=csv.DictWriter(f,ar[0].keys());w.writeheader();w.writerows(ar)
 return a,ar
c3,a3=run('C3_published_sourcefix','C3_sourcefix');e4,a4=run('E4_published_sourcefix','E4_published_sourcefix');times=[90,180,300,420,600,900];co=[]
for t in times:
 x=next(z for z in c3 if z['time_s']==t);y=next(z for z in e4 if z['time_s']==t);co.append(dict(time_s=t,C3_route_front_m=x['route_front_chainage_m'],E4_route_front_m=y['route_front_chainage_m'],front_gain_m=y['route_front_chainage_m']-x['route_front_chainage_m'],C3_max_depth_m=x['max_depth_m'],E4_max_depth_m=y['max_depth_m'],C3_max_speed_ms=x['max_speed_ms'],E4_max_speed_ms=y['max_speed_ms']))
with open(res/'SOURCE_SUPPORT_FIX_COMPARISON_TMP.csv','w',newline='')as f:w=csv.DictWriter(f,co[0].keys());w.writeheader();w.writerows(co)
arr10={x['section']:x['arrival_s']for x in a4 if x['depth_threshold_m']==.1};L=S[-1]
if arr10['R4']is not None:cl='PUBLISHED_ROUTE_FULL_DOWNSTREAM_CANDIDATE';nxt='FREEZE_GEOMETRY_AND_BUILD_DCLAW_SCENARIO_DATASET'
elif arr10['R3']is not None:cl='PUBLISHED_ROUTE_STRONG_PARTIAL_PROPAGATION';nxt='FREEZE_GEOMETRY_AND_REVIEW_PORT_REACH_ONLY'
elif arr10['R1']is not None:cl='PUBLISHED_ROUTE_PARTIAL_PROPAGATION';nxt='FREEZE_GEOMETRY_AND_QUANTIFY_REMAINING_MOBILITY_LIMIT'
else:cl='PUBLISHED_ROUTE_GEOMETRY_STILL_NOT_SUFFICIENT';nxt='STOP_PARAMETER_TUNING_AND_REVIEW_MODEL_PHYSICS_SEPARATELY'
for p,a,case in [('C3_SOURCEFIX',c3,'C3_published_sourcefix'),('E4_SOURCEFIX',e4,'E4_published_sourcefix')]:
 wd=(C/'runs'/case/'watchdog.log').read_text();data=dict(completed_to_s=a[-1]['time_s'],frames=len(a),final=a[-1],finite=True,watchdog_heartbeats=wd.count('HEARTBEAT'),stalls=wd.count('POSSIBLE_STALL'),runtime_failures=0,repair_reruns=1 if p=='C3_SOURCEFIX' else 0,arrivals_h010={x['section']:x['arrival_s'] for x in (a3 if p.startswith('C3') else a4) if x['depth_threshold_m']==.1});json.dump(data,open(rep/f'{p}_RESULT.json','w'),indent=2);(rep/f'{p}_RESULT.md').write_text('# '+p+' result\n\n'+json.dumps(data,indent=2)+'\n')
summary=dict(classification=cl,recommended_next_action=nxt,C3_fraction_900=c3[-1]['route_front_chainage_m']/L,E4_fraction_900=e4[-1]['route_front_chainage_m']/L,bdif_state_integral_monotonic=bool(np.all(np.diff([x['bdif_state_integral_m3']for x in e4])>=-1e-6)),route_length_m=float(L));json.dump(summary,open(rep/'SOURCE_SUPPORT_FIX_FINAL_SUMMARY.json','w'),indent=2);(rep/'SOURCE_SUPPORT_FIX_FINAL_SUMMARY.md').write_text('# Published route final summary\n\n'+json.dumps(summary,indent=2)+'\n')
for a,name in [(c3,'PUBLISHED_C3_FRONT'),(e4,'PUBLISHED_E4_FRONT')]:plt.figure();plt.plot([x['time_s']for x in a],[x['route_front_chainage_m']for x in a]);plt.xlabel('s');plt.ylabel('route chainage m');plt.grid();plt.tight_layout();plt.savefig(fig/(name+'.png'),dpi=150);plt.close()
plt.figure();plt.plot(times,[x['C3_route_front_m']for x in co],label='C3');plt.plot(times,[x['E4_route_front_m']for x in co],label='E4');plt.legend();plt.grid();plt.tight_layout();plt.savefig(fig/'PUBLISHED_C3_E4_COMPARISON.png',dpi=150);plt.close()
print(json.dumps(summary,indent=2))
