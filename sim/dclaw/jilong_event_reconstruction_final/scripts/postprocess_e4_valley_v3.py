from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from clawpack.pyclaw import Solution
CASE=Path('/root/autodl-tmp/Debris-flow-jilong-final/sim/dclaw/jilong_event_reconstruction_final');RUN=CASE/'runs/C3_entrainment_E4_valley_v3/_output';OLD=CASE/'runs/C3_entrainment_E4/_output';RES=CASE/'results';REP=CASE/'reports';FIG=CASE/'figures';RES.mkdir(exist_ok=True);REP.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
axis=list(csv.DictReader(open(CASE/'corridor/jilong_local_valley_axis_v3.csv')));ap=np.array([[float(r['x']),float(r['y'])]for r in axis]);ad=np.array([float(r['distance_along_axis_m'])for r in axis]);L=ad[-1];tree=cKDTree(ap)
src=json.load(open(CASE/'source_zone_v2.json'))['cells'];seeds=np.array([[x['center_x'],x['center_y']]for x in src])
def sol(out,k):
 s=Solution(k,path=str(out),file_format='ascii');q=s.state.q;g=s.state.grid;X,Y=g.p_centers;return s.t,q,X,Y,g.delta
def connected(h,X,Y):
 wet=h>.001; lab,n=label(wet,structure=np.ones((3,3))); keep=set()
 for x,y in seeds:
  ij=np.unravel_index(np.argmin((X-x)**2+(Y-y)**2),X.shape)
  if lab[ij]>0:keep.add(lab[ij])
 return wet&np.isin(lab,list(keep))
def metrics(out,k):
 t,q,X,Y,d=sol(out,k);h=q[0];u=np.divide(q[1],h,out=np.zeros_like(h),where=h>.001);v=np.divide(q[2],h,out=np.zeros_like(h),where=h>.001);sp=np.hypot(u,v);w=connected(h,X,Y);xx=X[w];yy=Y[w]
 if len(xx):
  di,ix=tree.query(np.c_[xx,yy]);m=ad[ix].max();j=np.argmax(ad[ix]);fx,fy=xx[j],yy[j];fd=di[j]
 else:m=0;fx=fy=fd=np.nan
 return dict(time_s=float(t),front_x=float(fx),front_y=float(fy),front_nearest_axis_distance_m=float(fd),max_local_axis_progress_m=float(m),local_axis_fraction=float(m/L),max_depth_m=float(np.nanmax(h)),max_speed_ms=float(np.nanmax(sp)),p99_wet_speed_ms=float(np.quantile(sp[w],.99) if w.any() else 0),total_wet_volume_m3=float(h[w].sum()*d[0]*d[1]),bdif_integral_m3=float(q[6].sum()*d[0]*d[1])),(h,sp,X,Y,w)
def main():
 new=[];cache={}
 for k in range(31):r,c=metrics(RUN,k);new.append(r);cache[int(r['time_s'])]=c
 fields=list(new[0]);
 with open(RES/'E4_valley_v3_front.csv','w',newline='')as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(new)
 milestones={'M25':.25*L,'M50':.5*L,'M75':.75*L,'M90':.9*L,'M100':L};arr=[]
 for name,s in milestones.items():
  p=ap[np.argmin(abs(ad-s))];d=np.linalg.norm(ap-p,axis=1);p=ap[np.argmin(abs(ad-s))]
  for th in(.001,.05,.10,.20):
   hit=None
   for k in range(31):
    h,sp,X,Y,w=cache[int(new[k]['time_s'])];near=(X-p[0])**2+(Y-p[1])**2<=100**2
    if np.any((h>th)&near):hit=float(new[k]['time_s']);break
   arr.append(dict(milestone=name,milestone_distance_m=float(s),x=float(p[0]),y=float(p[1]),depth_threshold_m=th,earliest_arrival_s=hit))
 with open(RES/'E4_valley_v3_arrivals.csv','w',newline='')as f:w=csv.DictWriter(f,arr[0].keys());w.writeheader();w.writerows(arr)
 comp=[]
 for tt in(90,180,300,420,600,900):
  k=tt//30;nr,_=metrics(RUN,k);orow,_=metrics(OLD,k);comp.append(dict(time_s=tt,old_front_x=orow['front_x'],old_front_y=orow['front_y'],new_front_x=nr['front_x'],new_front_y=nr['front_y'],old_local_axis_progress_m=orow['max_local_axis_progress_m'],new_local_axis_progress_m=nr['max_local_axis_progress_m'],old_max_depth_m=orow['max_depth_m'],new_max_depth_m=nr['max_depth_m'],old_max_speed_ms=orow['max_speed_ms'],new_max_speed_ms=nr['max_speed_ms']))
 with open(RES/'E4_old_vs_valley_v3.csv','w',newline='')as f:w=csv.DictWriter(f,comp[0].keys());w.writeheader();w.writerows(comp)
 final=new[-1];a10={x['milestone']:x['earliest_arrival_s']for x in arr if abs(x['depth_threshold_m']-.1)<1e-9}
 if a10['M90'] is not None or a10['M100'] is not None:cl='LOCAL_VALLEY_GEOMETRY_RESOLVES_E4_PROPAGATION';nxt='FREEZE_LOCAL_GEOMETRY_AND_EXTEND_DOWNSTREAM_AXIS'
 elif a10['M50'] is not None:cl='LOCAL_VALLEY_GEOMETRY_SUBSTANTIALLY_IMPROVES_PROPAGATION';nxt='FREEZE_PHYSICS_AND_EXTEND_VALLEY_AXIS_ONCE'
 else:cl='LOCAL_VALLEY_GEOMETRY_DOES_NOT_RESOLVE_MOBILITY';nxt='STOP_DCLAW_CALIBRATION_AND_USE_EVENT_CONSTRAINED_SCENARIO_FRAMING'
 finite=all(np.isfinite(v) for r in new for v in r.values() if isinstance(v,float));result=dict(classification=cl,recommended_next_action=nxt,completed_to_s=final['time_s'],frame_count=len(new),numerical_gate=dict(finite=finite,max_depth_lt_100=final['max_depth_m']<100,max_speed_lt_40=final['max_speed_ms']<40),final=final,local_axis_length_m=float(L),arrivals_h_gt_010m=a10,physics_modified=False,terrain_modified=False,source_modified=False,entrainment_rate_modified=False,watchdog_heartbeat_count=2,watchdog_detected_stalls=0,watchdog_detected_runtime_failures=0,watchdog_false_positive_pattern='normal solver termination text matched stopping')
 (REP/'E4_VALLEY_V3_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
 md=['# E4 valley-v3 result','',f'Classification: **{cl}**.','',f'Final time: {final["time_s"]:.0f} s; frames: {len(new)}; max depth: {final["max_depth_m"]:.6g} m; max speed: {final["max_speed_ms"]:.6g} m/s.',f'Final local-axis progress/fraction: {final["max_local_axis_progress_m"]:.3f} m / {final["local_axis_fraction"]:.4f}.','',f'Recommended next action: `{nxt}`.','', 'The watchdog had two 60-second checks, zero stalls, and no runtime failure. Its final keyword scan matched normal termination wording (`stopping`) after exit code 0 and 31 frames; this is retained as a false-positive audit note.','', 'Physics, terrain, source, and entrainment rate were not modified.']
 (REP/'E4_VALLEY_V3_RESULT.md').write_text('\n'.join(md)+'\n')
 t=np.array([r['time_s']for r in new]);p=np.array([r['max_local_axis_progress_m']for r in new]);plt.figure(figsize=(7,4));plt.plot(t,p,'o-');plt.axhline(.9*L,color='r',ls='--',label='M90');plt.xlabel('time (s)');plt.ylabel('local-axis progress (m)');plt.grid();plt.legend();plt.tight_layout();plt.savefig(FIG/'E4_VALLEY_V3_FRONT_TRAJECTORY.png',dpi=180);plt.close()
 tt=np.array([r['time_s']for r in comp]);plt.figure(figsize=(7,4));plt.plot(tt,[r['old_local_axis_progress_m']for r in comp],'o-',label='old E4');plt.plot(tt,[r['new_local_axis_progress_m']for r in comp],'o-',label='E4 valley-v3');plt.xlabel('time (s)');plt.ylabel('local-axis progress (m)');plt.grid();plt.legend();plt.tight_layout();plt.savefig(FIG/'E4_OLD_VS_VALLEY_V3.png',dpi=180);plt.close()
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
