"""Phase 2C diagnostics: one common wet threshold for port and any future station."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from clawpack.pyclaw.solution import Solution
PORT_X, PORT_Y = 340_837.9, 3_129_051.7
WET = .01
FLOW = np.array([.4115,-.9114]); NORMAL=np.array([-FLOW[1],FLOW[0]])

def interp(a,x,y,xp,yp):
 i=int(np.clip(np.searchsorted(x,xp)-1,0,len(x)-2)); j=int(np.clip(np.searchsorted(y,yp)-1,0,len(y)-2)); tx=(xp-x[i])/(x[i+1]-x[i]); ty=(yp-y[j])/(y[j+1]-y[j]); return float((1-tx)*(1-ty)*a[i,j]+tx*(1-ty)*a[i+1,j]+(1-tx)*ty*a[i,j+1]+tx*ty*a[i+1,j+1])
def frame(n,out):
 s=Solution(n,path=out,file_format='ascii'); q=s.state.q; x=s.state.grid.dimensions[0].centers; y=s.state.grid.dimensions[1].centers; h,hu,hv=q[:3]
 hp=interp(h,x,y,PORT_X,PORT_Y); hup=interp(hu,x,y,PORT_X,PORT_Y); hvp=interp(hv,x,y,PORT_X,PORT_Y)
 ps=np.hypot(hup,hvp)/hp if hp>WET else 0.
 Q=0.
 for d in np.arange(-160.,192.,64.):
  px,py=np.array([PORT_X,PORT_Y])+d*NORMAL; hs=interp(h,x,y,px,py)
  if hs>WET: Q+=max(0.,interp(hu,x,y,px,py)*FLOW[0]+interp(hv,x,y,px,py)*FLOW[1])*64.
 wet=h>WET; speed=np.zeros_like(h); speed[wet]=np.hypot(hu[wet]/h[wet],hv[wet]/h[wet])
 eroded=float(np.maximum(q[6],0).sum()*(x[1]-x[0])*(y[1]-y[0])) if q.shape[0]>=7 else None
 return dict(time_s=float(s.t),port_depth_m=hp,port_speed_ms=float(ps),port_discharge_m3s=float(Q),domain_max_speed_ms=float(speed.max()),finite=bool(np.isfinite(q).all()),entrained_volume_m3=eroded)
def main(out,summary,qpeak,duration):
 ns=sorted(int(p.name[-4:]) for p in out.glob('fort.q????')); vals=[frame(n,out) for n in ns]
 arrival=next((d['time_s'] for d in vals if d['port_depth_m']>WET),None); peak=max(vals,key=lambda z:z['port_discharge_m3s'])
 result=dict(wet_threshold_m=WET,reached_seqiong=None,seqiong_arrival_s=None,reached_port=arrival is not None,port_arrival_s=arrival,seqiong_to_port_s=None,port_peak_discharge_m3s=peak['port_discharge_m3s'],port_peak_depth_m=max(z['port_depth_m'] for z in vals),port_peak_speed_ms=max(z['port_speed_ms'] for z in vals),domain_max_speed_ms=max(z['domain_max_speed_ms'] for z in vals),inflow_volume_m3=.5*qpeak*duration,entrained_volume_if_available=vals[-1]['entrained_volume_m3'],stable_fields=all(z['finite'] for z in vals),time_series=vals)
 summary.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({k:v for k,v in result.items() if k!='time_series'},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);p.add_argument('--qpeak',type=float,required=True);p.add_argument('--duration',type=float,required=True);a=p.parse_args();main(a.output,a.summary,a.qpeak,a.duration)
