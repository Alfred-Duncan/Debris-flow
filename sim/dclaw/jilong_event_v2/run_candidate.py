"""Execute one Phase 2C source candidate and archive exactly its outputs."""
from __future__ import annotations
import argparse,csv,json,shutil,subprocess,sys,time
from pathlib import Path
CASE=Path(__file__).resolve().parent
def call(cmd,log,env=None):
 p=subprocess.run(cmd,cwd=CASE,stdout=log,stderr=subprocess.STDOUT,text=True,env=env)
 if p.returncode: raise RuntimeError(f'failed {cmd}')
def main(run_id,qpeak,duration,entrainment):
 rd=CASE/'runs'/run_id
 if rd.exists(): raise FileExistsError(rd)
 rd.mkdir(parents=True); start=time.monotonic(); env=dict(**__import__('os').environ,INLET_Q_PEAK=str(qpeak),INLET_DURATION=str(duration),ENTRAINMENT_ENABLED=str(int(entrainment)))
 with (rd/'run.log').open('w') as log:
  call([sys.executable,'setrun.py'],log,env); (CASE/'.output').unlink(missing_ok=True)
  if (CASE/'_output').exists(): shutil.rmtree(CASE/'_output')
  call(['make','.output'],log,env); call([sys.executable,'postprocess.py','--output','_output','--summary',str(rd/'diagnostics.json'),'--qpeak',str(qpeak),'--duration',str(duration)],log,env)
 r=json.loads((rd/'diagnostics.json').read_text()); runtime=time.monotonic()-start
 row=dict(run_id=run_id,entrainment_enabled=bool(entrainment),Q_in_peak=qpeak,T_in=duration,inflow_volume=r['inflow_volume_m3'],reached_seqiong=r['reached_seqiong'],seqiong_arrival_s=r['seqiong_arrival_s'],reached_port=r['reached_port'],port_arrival_s=r['port_arrival_s'],seqiong_to_port_s=r['seqiong_to_port_s'],port_peak_discharge_m3s=r['port_peak_discharge_m3s'],port_peak_depth_m=r['port_peak_depth_m'],port_peak_speed_ms=r['port_peak_speed_ms'],domain_max_speed_ms=r['domain_max_speed_ms'],entrained_volume_if_available=r['entrained_volume_if_available'],runtime_s=runtime,stable=bool(r['stable_fields'] and r['domain_max_speed_ms']<=60.))
 cp=CASE/'CANDIDATE_RUNS.csv'; new=not cp.exists()
 with cp.open('a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=row.keys(),lineterminator='\n');
  if new:w.writeheader()
  w.writerow(row)
 shutil.move(str(CASE/'_output'),str(rd/'_output')); (rd/'candidate_row.json').write_text(json.dumps(row,indent=2)+'\n'); print(json.dumps(row,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--qpeak',type=float,required=True);p.add_argument('--duration',type=float,required=True);p.add_argument('--entrainment',type=int,choices=[0,1],required=True);a=p.parse_args();main(a.run_id,a.qpeak,a.duration,a.entrainment)
