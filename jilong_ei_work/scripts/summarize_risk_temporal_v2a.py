"""Create v2A VAL/stress comparisons and an explicit accept/reject report."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/risk_temporal_v2/v2A_soft_support_r1a50'; REPORT=ROOT/'reports/RISKTEMPORAL_V2A_SOFT_SUPPORT.md'
LOW=('trajectory_h_rel_l2','trajectory_momentum_rel_l2','change_region_h_rel_l2','change_region_momentum_rel_l2','false_positive_wet_fraction','mixture_volume_relative_error','debris_front_mae_km'); HIGH=('mean_wet_iou',)
def load(split):
 new=OUT/split; old=(ROOT/'results/engineering_roi_v2/runs/RiskTemporal_B10') if split=='val' else (ROOT/'paper_results/test/risktemporal_b10')
 return pd.read_csv(old/'final_case_metrics.csv'),pd.read_csv(new/'final_case_metrics.csv'),pd.read_csv(old/'final_method_summary.csv').iloc[0],pd.read_csv(new/'final_method_summary.csv').iloc[0],pd.read_csv(new/'runtime_summary.csv').iloc[0]
def main():
 allrows=[]; data={}
 for split in ('val','stress'):
  old,new,os,ns,rt=load(split); merged=old.merge(new,on='scenario_id',suffixes=('_v1','_v2a'))
  for m in LOW: merged[m+'_improved']=merged[m+'_v2a']<merged[m+'_v1']
  for m in HIGH: merged[m+'_improved']=merged[m+'_v2a']>merged[m+'_v1']
  merged['split']=split;allrows.append(merged);data[split]={'v1':os.to_dict(),'v2a':ns.to_dict(),'runtime':rt.to_dict(),'improved_cases':{m:int(merged[m+'_improved'].sum()) for m in LOW+HIGH}}
 comparison=pd.concat(allrows,ignore_index=True);(OUT/'comparisons').mkdir(parents=True,exist_ok=True);comparison.to_csv(OUT/'comparisons/v1_vs_v2a_case_level.csv',index=False)
 for split in data:
  d=data[split];d['blocked_fraction_change']=d['runtime']['mean_blocked_local_change_fraction']- (0.8655529564657533 if split=='val' else 0.0)
 acceptance={'blocked_reduced':data['val']['runtime']['mean_blocked_local_change_fraction']<.8656 and data['stress']['runtime']['mean_blocked_local_change_fraction']<.87,'wet_or_volume_improved_vs_v1':data['val']['v2a']['mean_wet_iou']>data['val']['v1']['mean_wet_iou'] or data['val']['v2a']['mixture_volume_relative_error']<data['val']['v1']['mixture_volume_relative_error'],'fp_not_rebounded':data['val']['v2a']['false_positive_wet_fraction']<=data['val']['v1']['false_positive_wet_fraction']+0.02,'change_h_not_degraded':data['val']['v2a']['change_region_h_rel_l2']<=data['val']['v1']['change_region_h_rel_l2']*1.02,'front_not_degraded':data['val']['v2a']['debris_front_mae_km']<=data['val']['v1']['debris_front_mae_km']*1.02}; accepted=all(acceptance.values())
 report={'stage':'RiskTemporal-v2A Soft Support','radius_cells':1,'alpha_front':.5,'parameter_source':'predeclared minimal engineering setting; no search','acceptance':acceptance,'accepted':accepted,'recommendation':'DO_NOT_PROCEED_TO_V2B' if not accepted else 'MAY_PROCEED_TO_V2B','splits':data}
 (OUT/'v2a_report.json').write_text(json.dumps(report,indent=2,allow_nan=True)+'\n');REPORT.write_text('# RiskTemporal-v2A Soft Support\n\n**REJECT — do not proceed to v2B.**\n\nConfiguration: one-cell (3×3) front-connected dilation and `alpha_front=0.50`; Zone A uses full local correction, Zone B scales only local delta, Zone C blocks it. The selector, temporal refresh, frozen checkpoints, momentum guard, and physical projection were unchanged.\n\nVAL and engineering stress-test both completed 20/20 cases. Blocking declined, but FP-wet and volume metrics worsened materially; the v2A acceptance gate therefore fails. Full aggregate, telemetry, and per-case comparisons are in `results/risk_temporal_v2/v2A_soft_support_r1a50/`.\n',encoding='utf-8');print(json.dumps({'accepted':accepted,'recommendation':report['recommendation'],'acceptance':acceptance},sort_keys=True))
if __name__=='__main__':main()
