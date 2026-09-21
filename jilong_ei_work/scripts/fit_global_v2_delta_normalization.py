"""Real-data, no-optimizer sanity check for V2 transform/delta fitting."""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.dataset import FrameStore,scenario_rows
from src.global_operator_v2.frame_adapter import static_and_exogenous
from src.global_operator_v2.transforms import fit_training_transforms,fit_delta_normalization

def main():
 cfg=json.loads((ROOT/'configs/GLOBAL_OPERATOR_V2.json').read_text());rows=scenario_rows('TRAIN');store=FrameStore(rows);static,_=static_and_exogenous(ROOT/'data/downloads/park_v2/inputs/upper30h.npz');transform,_=fit_training_transforms(rows,store,static);normalization=fit_delta_normalization(rows,store,static,transform,cfg['seed'],**cfg['stabilization']['delta_normalization'],bound_quantile=cfg['stabilization']['delta_bound']['quantile'],safety_factor=cfg['stabilization']['delta_bound']['safety_factor'],change_quantile=cfg['stabilization']['teacher_weighting']['change_quantile']);report=normalization.to_dict();minimum=cfg['stabilization']['delta_bound']['minimum_coverage'];report['sanity_pass']=bool(normalization.scales[5]>cfg['stabilization']['delta_normalization']['floor'] and normalization.change_threshold>cfg['stabilization']['delta_normalization']['floor'] and all(value['coverage_all_samples']>=minimum for value in normalization.per_channel.values()));path=ROOT/'reports/GLOBAL_V2_DELTA_NORMALIZATION_SANITY.json';path.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));
 if not report['sanity_pass']:raise SystemExit('DELTA_NORMALIZATION_SANITY_FAILED')
if __name__=='__main__':main()
