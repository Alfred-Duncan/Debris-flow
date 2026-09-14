"""One non-interactive sequential six-case Phase-2C-fix batch."""
from __future__ import annotations
import json,os,shutil,subprocess,sys,time
from pathlib import Path
CASE=Path(__file__).resolve().parent
MATRIX=[('RUN_A',30000,60,0,0),('RUN_B',30000,60,1,0),('RUN_C',20000,90,1,0),('RUN_D',30000,90,1,0),('RUN_E',20000,120,1,0),('RUN_F',30000,120,1,0)]
def main():
 t0=time.monotonic(); env=dict(os.environ); subprocess.run([sys.executable,'build_aux.py'],cwd=CASE,check=True,env=env);subprocess.run(['make'],cwd=CASE,check=True,env=env)
 metas=[]
 for rid,q,t,en,method in MATRIX:
  rd=CASE/'runs'/rid
  if rd.exists():shutil.rmtree(rd)
  rd.mkdir(parents=True); log=rd/'run.log';e=dict(env,INLET_Q_PEAK=str(q),INLET_DURATION=str(t),ENTRAINMENT_ENABLED=str(en),ENTRAINMENT_METHOD=str(method));started=time.monotonic()
  with log.open('w') as f:
   s=subprocess.run([sys.executable,'setrun.py'],cwd=CASE,stdout=f,stderr=subprocess.STDOUT,env=e)
   code=s.returncode
   if code==0:
    shutil.rmtree(CASE/'_output',ignore_errors=True);(CASE/'.output').unlink(missing_ok=True);s=subprocess.run(['make','.output'],cwd=CASE,stdout=f,stderr=subprocess.STDOUT,env=e);code=s.returncode
  if code==0: shutil.move(str(CASE/'_output'),str(rd/'_output'))
  metas.append(dict(run_id=rid,Q_peak=q,T_in=t,entrainment_enabled=bool(en),entrainment_method=method,runtime_s=time.monotonic()-started,solver_exit_code=code,log_tail=''.join(log.read_text(errors='replace').splitlines(True)[-12:])))
 (CASE/'batch_metadata.json').write_text(json.dumps(dict(total_batch_wall_time_s=time.monotonic()-t0,runs=metas),indent=2)+'\n')
 from postprocess_batch import main as post
 post(CASE,metas)
if __name__=='__main__':main()
