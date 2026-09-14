"""One sequential G1-G4 V3 batch after the 32-m terrain-representation fix."""
from __future__ import annotations
import csv,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
from clawpack.pyclaw.solution import Solution
from route_diagnostics import PROXY,ROUTE_DISTANCE_M,CORRIDOR_TOLERANCE_M,nearest_s
CASE=Path(__file__).resolve().parent;WET=.01
def interp(a,x,y,xp,yp):
 i=int(np.clip(np.searchsorted(x,xp)-1,0,len(x)-2));j=int(np.clip(np.searchsorted(y,yp)-1,0,len(y)-2));tx=(xp-x[i])/(x[i+1]-x[i]);ty=(yp-y[j])/(y[j+1]-y[j]);return float((1-tx)*(1-ty)*a[i,j]+tx*(1-ty)*a[i+1,j]+(1-tx)*ty*a[i,j+1]+tx*ty*a[i+1,j+1])
def post(meta):
 out=CASE/'runs'/meta['run_id']/'_output';nums=sorted(int(p.name[-4:]) for p in out.glob('fort.q????'));vals=[];all01=[]
 for n in nums:
  s=Solution(n,path=out,file_format='ascii');q=s.state.q;x=s.state.grid.dimensions[0].centers;y=s.state.grid.dimensions[1].centers;h,hu,hv=q[:3];wet=h>WET;sp=np.zeros_like(h);sp[wet]=np.hypot(hu[wet]/h[wet],hv[wet]/h[wet]);front=0.
  for i,j in zip(*np.where(wet)):
   ss,d=nearest_s(x[i],y[j]);front=max(front,ss) if d<=CORRIDOR_TOLERANCE_M else front
  hp=interp(h,x,y,*PROXY);vals.append(dict(time_s=float(s.t),front_distance_m=front,remaining_distance_m=max(0,ROUTE_DISTANCE_M-front),proxy_depth_m=hp,finite=bool(np.isfinite(q).all()),raw=float(sp[wet].max()) if wet.any() else 0.,h01=float(sp[h>.1].max()) if (h>.1).any() else 0.,h1=float(sp[h>1].max()) if (h>1).any() else 0.));all01.extend(sp[h>.1].tolist())
 with (CASE/'runs'/meta['run_id']/'front_progress.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=['time_s','front_distance_m','remaining_distance_m']);w.writeheader();w.writerows([{k:v[k] for k in w.fieldnames} for v in vals])
 log=(CASE/'runs'/meta['run_id']/'run.log').read_text(errors='replace');cfl=[float(x) for x in re.findall(r'maximum Courant number seen =\s*([0-9.]+)',log)];a={int(v['time_s']):v['front_distance_m'] for v in vals};fronts=[v['front_distance_m'] for v in vals];return dict(**meta,solver_completed=True,finite_fields=all(v['finite'] for v in vals),maximum_CFL_from_log=max(cfl) if cfl else None,numerically_clean=all(v['finite'] for v in vals) and (not cfl or max(cfl)<=1),raw_max_speed_h_gt_001=max(v['raw'] for v in vals),max_speed_h_gt_01=max(v['h01'] for v in vals),max_speed_h_gt_1=max(v['h1'] for v in vals),p99_speed_h_gt_01=float(np.percentile(all01,99)) if all01 else 0.,front_60_s=a.get(60),front_120_s=a.get(120),front_240_s=a.get(240),front_420_s=a.get(420),front_600_s=a.get(600),max_front_distance_m=max(fronts),final_front_distance_m=fronts[-1],reached_port_proxy=any(v['proxy_depth_m']>WET for v in vals))
def main():
 start=time.monotonic();env=dict(os.environ);subprocess.run([sys.executable,'build_terrain32.py'],cwd=CASE,check=True,env=env);subprocess.run([sys.executable,'build_aux.py'],cwd=CASE,check=True,env=env);subprocess.run(['make'],cwd=CASE,check=True,env=env);M=[('G1',30000,60,0),('G2',30000,60,1),('G3',30000,90,1),('G4',30000,120,1)];metas=[]
 for rid,q,t,en in M:
  rd=CASE/'runs'/rid;shutil.rmtree(rd,ignore_errors=True);rd.mkdir(parents=True);e=dict(env,INLET_Q_PEAK=str(q),INLET_DURATION=str(t),ENTRAINMENT_ENABLED=str(en),ENTRAINMENT_METHOD='0');beg=time.monotonic()
  with (rd/'run.log').open('w') as log:
   x=subprocess.run([sys.executable,'setrun.py'],cwd=CASE,stdout=log,stderr=subprocess.STDOUT,env=e);code=x.returncode
   if code==0:shutil.rmtree(CASE/'_output',ignore_errors=True);(CASE/'.output').unlink(missing_ok=True);x=subprocess.run(['make','.output'],cwd=CASE,stdout=log,stderr=subprocess.STDOUT,env=e);code=x.returncode
  if code==0:shutil.move(str(CASE/'_output'),str(rd/'_output'))
  metas.append(dict(run_id=rid,Q_peak=q,T_in=t,entrainment_enabled=bool(en),entrainment_method=0,inflow_volume_m3=.5*q*t,runtime_s=time.monotonic()-beg,solver_exit_code=code))
 rows=[post(m) for m in metas if m['solver_exit_code']==0];fields=list(rows[0]);
 with (CASE/'G_BATCH_RUNS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 best=max(rows,key=lambda r:r['max_front_distance_m']);table='\n'.join(f"| {r['run_id']} | {r['max_front_distance_m']/1000:.3f} | {r['final_front_distance_m']/1000:.3f} | {r['raw_max_speed_h_gt_001']:.2f} / {r['max_speed_h_gt_01']:.2f} / {r['max_speed_h_gt_1']:.2f} / {r['p99_speed_h_gt_01']:.2f} | {r['maximum_CFL_from_log']:.3f} |" for r in rows);doc=f'''# Phase 2D V3 corrected-terrain rerun\n\nV3 replaces the diagnosed coarse 64-m terrain representation with one fixed **32-m** grid derived directly and continuously from the observed 12.5-m DEM. No channel was carved and no elevation was edited. Domain, inlet, hydrograph family, material parameters, and entrainment settings are otherwise unchanged.\n\n| Run | max front (km) | final front (km) | raw / h>.1 / h>1 / p99 speed (m/s) | max CFL |\n|---|---:|---:|---:|---:|\n{table}\n\nBest front: **{best['max_front_distance_m']/1000:.3f} km** ({best['run_id']}); proxy reached: {best['reached_port_proxy']}. Total G-batch wall time: {time.monotonic()-start:.2f} s.\n''';(CASE/'RESULTS_REPORT.md').write_text(doc);(CASE.parents[2]/'docs/STALL_DIAGNOSIS.md').write_text(open(CASE.parents[2]/'docs/STALL_DIAGNOSIS.md').read()+ '\n\n'+doc);(CASE/'g_batch_metadata.json').write_text(json.dumps(dict(total_wall_s=time.monotonic()-start,runs=rows),indent=2)+'\n')
if __name__=='__main__':main()
