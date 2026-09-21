"""Parity check against retained Park-v2 solver series; read-only over three cases."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.frame_adapter import dynamic_state,static_and_exogenous
from src.global_operator_v2.metrics import station_metrics,debris_front,arrival_from_series,load_transects
def choose():
 r=pd.read_csv(ROOT/'data/scenario_response_summary.csv').sort_values('peak_gyirong_Q_m3_s');return [r.iloc[0].scenario_id,r.iloc[len(r)//2].scenario_id,r.iloc[-1].scenario_id]
def main():
 static,meta=static_and_exogenous(ROOT/'data/downloads/park_v2/inputs/upper30h.npz');route=np.load(ROOT/'data/downloads/park_v2/inputs/upper30h.npz')['route_chainage_m'];trs=load_transects(ROOT/'data/downloads/park_v2/inputs/upper30h_transects.json');checks=[]
 for sid in choose():
  case=ROOT/'outputs/scenarios'/sid;series=pd.read_csv(case/'series.csv');errs=[]
  front_err=[]
  for t in range(0,1441,10):
   state,_=dynamic_state(case/'frames'/f'state_{t:04d}s.npz',tuple(meta['shape']));z=static[0]+state[5];row=series.iloc[(series.time_s-t).abs().argmin()]
   for name,tr in trs.items():
    got=station_metrics(state,z,tr,meta['cell_m'])
    for k in ('Q','Qdebris','hmax','stage','wet_width_m'):
     ref=float(row[f'{name}_{k}']);val=got[k]
     if np.isfinite(ref) and np.isfinite(val):errs.append(abs(val-ref)/max(abs(ref),1.0))
   front=debris_front(state,route);ref_front=float(row['debris_front_route_km'])
   if np.isfinite(front) and np.isfinite(ref_front):front_err.append(abs(front-ref_front))
  arr={n:arrival_from_series(series,n) for n in trs};q=json.loads((case/'quality.json').read_text());arrival_ok=all(abs((arr[n] if arr[n] is not None else -1)-(q.get({'gyirong_cctv_arrival':'arrival_gyirong_s','rasuwagadhi_signal_loss':'arrival_rasu_s','syabrubesi_signal_loss':'arrival_syab_s'}[n]) if q.get({'gyirong_cctv_arrival':'arrival_gyirong_s','rasuwagadhi_signal_loss':'arrival_rasu_s','syabrubesi_signal_loss':'arrival_syab_s'}[n]) is not None else -1))<=10 for n in trs)
  checks.append({'scenario_id':sid,'max_station_relative_error':float(max(errs,default=np.inf)),'max_route_front_error_km':float(max(front_err,default=np.inf)),'arrival_parity_10s':arrival_ok})
 status='PASS' if all(x['max_station_relative_error']<=.02 and x['max_route_front_error_km']<=.03 and x['arrival_parity_10s'] for x in checks) else 'FAIL';out={'status':status,'cases':checks,'semantics':'station flux/tangent, Qdebris, hmax, stage, wet width, debris-front and arrival copied from Park-v2 solver/postprocessor; retained 10-s frames allow quantization.'}
 (ROOT/'reports/GLOBAL_V2_ENGINEERING_METRIC_PARITY.json').write_text(json.dumps(out,indent=2));(ROOT/'reports/GLOBAL_V2_ENGINEERING_METRIC_PARITY.md').write_text(f"# V2 engineering metric parity\n\n**Status:** {status}\n\nSee JSON for three existing cases and tolerance-aware retained-frame checks.\n");print(json.dumps(out,indent=2))
if __name__=='__main__':main()
