from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.physics_frame_adapter import dynamic_tensor
idx=pd.read_csv(ROOT/'data/scenario_index.csv'); expected=list(range(0,1441,10)); failures=[]; checked=0
for row in idx.itertuples():
    times=[]
    for t in expected:
        p=Path(row.frames_dir)/f'state_{t:04d}s.npz'
        try:
            # Solver time is floating-point adaptive integration; retained filenames
            # denote requested 10-s checkpoints.  Accept sub-centisecond roundoff.
            a,m=dynamic_tensor(p,(921,882)); ok=a.shape==(5,921,882) and np.isfinite(a).all() and (a[0]>=0).all() and abs(m['time_s']-t)<.01
            if not ok: raise ValueError('shape/finiteness/orientation/time')
            times.append(t); checked+=1
        except Exception as e: failures.append({'scenario_id':row.scenario_id,'time_s':t,'error':str(e)})
    if times!=expected: failures.append({'scenario_id':row.scenario_id,'time_s':'sequence','error':'incomplete time sequence'})
splits=idx.groupby('split').scenario_id.nunique().to_dict(); out={'status':'PASS' if not failures and splits=={'TRAIN':160,'VAL':20,'TEST':20} else 'FAIL','split_counts':splits,'scenarios':len(idx),'frames_checked':checked,'expected_frames':len(idx)*145,'adapter':'src/physics_frame_adapter.py','shape':[5,921,882],'row0_orientation':'north (adapter metadata)','failures':failures[:50]}
(ROOT/'reports/GLOBAL_OPERATOR_DATA_AUDIT.json').write_text(json.dumps(out,indent=2)); (ROOT/'reports/GLOBAL_OPERATOR_DATA_AUDIT.md').write_text('# Global Operator data audit\n\n```json\n'+json.dumps(out,indent=2)+'\n```\n')
print(out['status'],checked,len(failures))
if failures: raise SystemExit(1)
