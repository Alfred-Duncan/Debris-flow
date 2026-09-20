"""Audit V2 state closure and sparse dz reconstruction; read-only."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.frame_adapter import dynamic_state, static_and_exogenous, physical_state_from_final

def main(limit: int=20):
    static,meta=static_and_exogenous(ROOT/'data/downloads/park_v2/inputs/upper30h.npz'); shape=tuple(meta['shape']); runs=sorted((ROOT/'outputs/scenarios').glob('JILONG_EI_*'))[:limit]
    checks=[]
    for run in runs:
        frame=run/'frames/state_1440s.npz'; final=run/'final_state.npz'
        if not frame.exists() or not final.exists(): continue
        dense,dm=dynamic_state(frame,shape); f=physical_state_from_final(final); delta=np.abs(dense[5]-f['bed_change']); checks.append({'scenario_id':run.name,'max_abs_dz_m':float(delta.max()),'mean_abs_dz_m':float(delta.mean()),'frame_dz_nonzero':dm['dz_sparse_count'],'final_dz_nonzero':int(np.count_nonzero(f['bed_change']))})
    tol=2e-6; status='PASS' if len(checks)>=20 and all(x['max_abs_dz_m']<=tol for x in checks) else 'FAIL'
    report={'status':status,'audit_cases':len(checks),'tolerance_m':tol,'dz_final_frame_vs_final_state':checks,
            'retained_frame_state':['h','hu','hv','c','ice','dz'],'reconstructable':['hc=h*c','hi=h*ice','current_bed=z_initial+dz'],
            'not_retained':['hq (remaining erodible/material variable)'],'closure_statement':'The retained state is directly reconstructable as listed, but is not claimed to be an exact Markov state because hq is absent from sparse frames.'}
    out=ROOT/'reports';out.mkdir(exist_ok=True);(out/'GLOBAL_V2_STATE_CLOSURE.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# Global Operator V2 state-closure audit','','**Status:** '+status,'', 'The V2 predictor retains `h, hu, hv, c, ice, dz`. It reconstructs `hc=h*c`, `hi=h*ice`, and `z_current=z_initial+dz`. `hq` (remaining erodible/material state) is not retained, so this is not claimed as an exact Markov closure.','','## dz cross-check','','| Cases | tolerance (m) | maximum observed (m) | Result |','|---:|---:|---:|---|',f"| {len(checks)} | {tol:.1e} | {max((x['max_abs_dz_m'] for x in checks),default=float('nan')):.3e} | {status} |"]
    (out/'GLOBAL_V2_STATE_CLOSURE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=20);main(p.parse_args().limit)
