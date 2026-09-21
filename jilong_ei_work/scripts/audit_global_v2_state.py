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
    rtol=2e-3;atol=2e-3
    # Exporter contract: cells with |dz| <= .05 m are intentionally omitted;
    # retained entries are float16, so equality to full float32 is not expected.
    semantic=[]
    for run in runs:
      frame=run/'frames/state_1440s.npz';final=run/'final_state.npz'
      if not frame.exists() or not final.exists():continue
      with np.load(frame) as a: saved=np.zeros(shape[0]*shape[1],bool);saved[np.asarray(a['dz_idx'],int)]=True;fv=np.zeros(shape[0]*shape[1],np.float32);fv[np.asarray(a['dz_idx'],int)]=np.asarray(a['dz'],np.float32)
      with np.load(final) as b: full=np.asarray(b['bed_change'],np.float32).ravel()
      retained=np.isclose(fv[saved],full[saved],rtol=rtol,atol=atol).all();omitted=(np.abs(full[~saved])<=.05+atol).all();semantic.append({'scenario_id':run.name,'retained_float16_match':bool(retained),'omitted_within_threshold':bool(omitted),'omitted_max_abs_m':float(np.abs(full[~saved]).max(initial=0))})
    status='PASS_THRESHOLD_AWARE' if len(checks)>=20 and all(x['retained_float16_match'] and x['omitted_within_threshold'] for x in semantic) else 'FAIL'
    report={'status':status,'audit_cases':len(checks),'threshold_semantics_m':.05,'storage_dtype':'float16','retained_comparison_rtol':rtol,'retained_comparison_atol_m':atol,'FULL_PHYSICAL_BED_CHANGE':'final_state.bed_change float32','RETAINED_ML_BED_CHANGE':'sparse dz_idx/dz thresholded cumulative field','threshold_audit':semantic,'dz_final_frame_vs_final_state':checks,
            'retained_frame_state':['h','hu','hv','c','ice','thresholded_dz'],'reconstructable':['hc=h*c','hi=h*ice','current_bed=z_initial+dz_retained'],
            'not_retained':['hq (heat / thermal-energy storage state)','erodible (remaining erodible-depth / material-availability state)'],'closure_statement':'The retained dataset is not an exact Markov-complete state: hq and erodible are absent. Previous/current/delta, thresholded dynamic bed, and known forcing reduce but do not eliminate partial observability. LATENT_STATE_FALLBACK remains disabled.'}
    out=ROOT/'reports';out.mkdir(exist_ok=True);(out/'GLOBAL_V2_STATE_CLOSURE.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# Global Operator V2 state-closure audit','','**Status:** '+status,'', 'Retained ML state: `h, hu, hv, c, ice, thresholded_dz`. `hc=h*c`, `hi=h*ice`, and `current_bed=z_initial+dz_retained` are reconstructible. `hq` is a heat/thermal-energy storage state; `erodible` is the separate remaining-erodible-depth field. Neither is retained, therefore this is not an exact Markov-complete state.','','## dz representation contract','','`FULL_PHYSICAL_BED_CHANGE` is float32 `final_state.bed_change`. `RETAINED_ML_BED_CHANGE` stores only `abs(dz)>0.05 m` cells as float16. Omitted below-threshold cells are expected; retained cells use rtol=2e-3, atol=0.002 m.','','| Cases | threshold (m) | Result |','|---:|---:|---|',f"| {len(checks)} | 0.05 | {status} |"]
    (out/'GLOBAL_V2_STATE_CLOSURE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=20);main(p.parse_args().limit)
