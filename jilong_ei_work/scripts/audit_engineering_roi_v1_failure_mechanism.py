"""Read-only V1 selection persistence audit; no model execution."""
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
METHODS=('RandomAll_B10','RandomSupport_B10','EngineeringROI_B05','EngineeringROI_B10','EngineeringROI_B20','DynamicOnly_B10','SupportRiskOnly_B10','EngineeringROI_NoDiversity_B10')
def main():
 t=pd.read_csv(ROOT/'results/engineering_roi_v1/selection_timeline.csv');rows=[]
 for (m,s),g in t[t.method.isin(METHODS)].groupby(['method','scenario_id']):
  sets=[set(map(int,x.split(';'))) if isinstance(x,str) and x else set() for x in g.sort_values('time_s').selected_patch_ids];flat=[p for x in sets for p in x];counts=pd.Series(flat).value_counts();over=sum(len(a&b) for a,b in zip(sets[1:],sets));rows.append({'method':m,'scenario_id':s,'total_selections':len(flat),'unique_selected_patches':len(counts),'unique_patch_fraction':len(counts)/max(len(flat),1),'top10_share':counts.head(10).sum()/max(len(flat),1),'consecutive_reselection_fraction':over/max(len(flat),1),'revisit_fraction':(counts>1).mean() if len(counts) else 0.})
 out=ROOT/'results/engineering_roi_v1_failure_audit';out.mkdir(exist_ok=True);case=pd.read_csv(ROOT/'results/engineering_roi_v1/final_case_metrics.csv');per=pd.DataFrame(rows);per.to_csv(out/'selection_persistence_summary.csv',index=False);per.to_csv(out/'per_case_mechanism_summary.csv',index=False);per.groupby('method').mean(numeric_only=True).reset_index().to_csv(out/'method_mechanism_summary.csv',index=False);report={'no_new_model_execution':True,'amplitude_accumulation':'NOT DIRECTLY OBSERVABLE FROM EXISTING V1 ARTIFACTS','methods':list(METHODS)};(out/'failure_mechanism_report.json').write_text(json.dumps(report,indent=2));(ROOT/'reports/ENGINEERING_ROI_V1_FAILURE_MECHANISM.md').write_text('# EngineeringROI-v1 failure mechanism audit\n\nObserved persistence is descriptive association only. Raw local correction amplitude is NOT DIRECTLY OBSERVABLE FROM EXISTING V1 ARTIFACTS.\n')
if __name__=='__main__':main()
