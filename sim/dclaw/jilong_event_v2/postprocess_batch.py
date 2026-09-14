"""Unified postprocessor for the fixed Phase-2C six-case batch."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from clawpack.pyclaw.solution import Solution
from route_diagnostics import PROXY, ROUTE_DISTANCE_M, CORRIDOR_TOLERANCE_M, local_flow_direction, nearest_s
WET=.01; SECTION_WIDTH=384.; STRIP=64.
def interp(a,x,y,xp,yp):
 i=int(np.clip(np.searchsorted(x,xp)-1,0,len(x)-2));j=int(np.clip(np.searchsorted(y,yp)-1,0,len(y)-2));tx=(xp-x[i])/(x[i+1]-x[i]);ty=(yp-y[j])/(y[j+1]-y[j]);return float((1-tx)*(1-ty)*a[i,j]+tx*(1-ty)*a[i+1,j]+(1-tx)*ty*a[i,j+1]+tx*ty*a[i+1,j+1])
def one_frame(n,out):
 s=Solution(n,path=out,file_format='ascii');q=s.state.q;x=s.state.grid.dimensions[0].centers;y=s.state.grid.dimensions[1].centers;h,hu,hv=q[:3];wet=h>WET
 front=0.;
 for i,j in zip(*np.where(wet)):
  ss,d=nearest_s(x[i],y[j])
  if d<=CORRIDOR_TOLERANCE_M: front=max(front,ss)
 flow=local_flow_direction();normal=np.array([-flow[1],flow[0]]);hp=interp(h,x,y,*PROXY);hup=interp(hu,x,y,*PROXY);hvp=interp(hv,x,y,*PROXY);ps=float(np.hypot(hup,hvp)/hp) if hp>WET else 0.;Q=0.
 for offset in np.arange(-SECTION_WIDTH/2+STRIP/2,SECTION_WIDTH/2,STRIP):
  px,py=np.array(PROXY)+offset*normal;hs=interp(h,x,y,px,py)
  if hs>WET:Q+=max(0.,interp(hu,x,y,px,py)*flow[0]+interp(hv,x,y,px,py)*flow[1])*STRIP
 speed=np.zeros_like(h);speed[wet]=np.hypot(hu[wet]/h[wet],hv[wet]/h[wet])
 return dict(time_s=float(s.t),front_distance_m=front,remaining_distance_m=max(0.,ROUTE_DISTANCE_M-front),proxy_depth_m=hp,proxy_speed_ms=ps,proxy_discharge_m3s=float(Q),domain_max_speed_ms=float(speed.max()),finite=bool(np.isfinite(q).all()),corridor_wet_area_m2=float(np.count_nonzero(wet)*((x[1]-x[0])*(y[1]-y[0]))))
def process(case,meta):
 out=case/'runs'/meta['run_id']/'_output'; frames=sorted(int(p.name[-4:]) for p in out.glob('fort.q????'));values=[one_frame(n,out) for n in frames];fp=case/'runs'/meta['run_id']/'front_progress.csv'
 with fp.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['time_s','front_distance_m','remaining_distance_m']);w.writeheader();w.writerows([{k:v[k] for k in w.fieldnames} for v in values])
 peak=max(values,key=lambda v:v['proxy_discharge_m3s']);arr=next((v['time_s'] for v in values if v['proxy_depth_m']>WET),None)
 return dict(**meta,inflow_volume_m3=.5*meta['Q_peak']*meta['T_in'],stable=all(v['finite'] for v in values),domain_max_speed_ms=max(v['domain_max_speed_ms'] for v in values),max_front_distance_m=max(v['front_distance_m'] for v in values),final_front_distance_m=values[-1]['front_distance_m'],remaining_distance_to_port_proxy_m=values[-1]['remaining_distance_m'],reached_port_proxy=arr is not None,port_proxy_arrival_s=arr,port_proxy_peak_discharge_m3s=peak['proxy_discharge_m3s'],port_proxy_peak_depth_m=max(v['proxy_depth_m'] for v in values),port_proxy_peak_speed_ms=max(v['proxy_speed_ms'] for v in values),entrained_volume_if_reliable='unavailable',final_corridor_wet_area_m2=values[-1]['corridor_wet_area_m2'])
def main(case,metas):
 rows=[process(case,m) for m in metas if m.get('solver_exit_code')==0]
 for m in metas:
  if m.get('solver_exit_code')!=0: rows.append(dict(**m,inflow_volume_m3=.5*m['Q_peak']*m['T_in'],stable=False,domain_max_speed_ms='',max_front_distance_m='',final_front_distance_m='',remaining_distance_to_port_proxy_m='',reached_port_proxy=False,port_proxy_arrival_s='',port_proxy_peak_discharge_m3s='',port_proxy_peak_depth_m='',port_proxy_peak_speed_ms='',entrained_volume_if_reliable='unavailable',final_corridor_wet_area_m2=''))
 fields=['run_id','Q_peak','T_in','entrainment_enabled','entrainment_method','inflow_volume_m3','stable','runtime_s','domain_max_speed_ms','max_front_distance_m','final_front_distance_m','remaining_distance_to_port_proxy_m','reached_port_proxy','port_proxy_arrival_s','port_proxy_peak_discharge_m3s','port_proxy_peak_depth_m','port_proxy_peak_speed_ms','entrained_volume_if_reliable']
 with (case/'CANDIDATE_RUNS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:r.get(k,'') for k in fields} for r in rows])
 (case/'batch_results.json').write_text(json.dumps(rows,indent=2)+'\n');return rows
