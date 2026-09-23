"""Read-only V1 persistence and descriptive-association audit; never loads a model."""
import json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];METHODS=('RandomAll_B10','RandomSupport_B10','EngineeringROI_B05','EngineeringROI_B10','EngineeringROI_B20','DynamicOnly_B10','SupportRiskOnly_B10','EngineeringROI_NoDiversity_B10')
def persistence(sets):
 flat=[p for s in sets for p in s];counts=pd.Series(flat).value_counts();gaps=[];streaks=[]
 for patch in counts.index:
  hits=[i for i,s in enumerate(sets) if patch in s];gaps += [b-a for a,b in zip(hits[1:],hits)];runs=[];run=0
  for i in range(len(sets)):
   run=run+1 if patch in sets[i] else 0;runs.append(run)
  streaks.append(max(runs))
 overlap=sum(len(a&b) for a,b in zip(sets[1:],sets));return {'total_selections':len(flat),'unique_selected_patches':len(counts),'unique_patch_fraction':len(counts)/max(len(flat),1),'top1_selection_share':counts.head(1).sum()/max(len(flat),1),'top5_share':counts.head(5).sum()/max(len(flat),1),'top10_share':counts.head(10).sum()/max(len(flat),1),'consecutive_reselection_fraction':overlap/max(len(flat),1),'revisit_fraction':float((counts>1).mean()) if len(counts) else 0.,'mean_selection_count_per_used_patch':counts.mean() if len(counts) else 0.,'maximum_selection_count':counts.max() if len(counts) else 0,'mean_revisit_gap_steps':float(np.mean(gaps)) if gaps else np.nan,'mean_longest_consecutive_streak':float(np.mean(streaks)) if streaks else 0.}
def main():
 t=pd.read_csv(ROOT/'results/engineering_roi_v1/selection_timeline.csv');rows=[]
 for (method,scenario),g in t[t.method.isin(METHODS)].groupby(['method','scenario_id']):
  sets=[set(map(int,x.split(';'))) if isinstance(x,str) and x else set() for x in g.sort_values('time_s').selected_patch_ids];rows.append({'method':method,'scenario_id':scenario,**persistence(sets)})
 out=ROOT/'results/engineering_roi_v1_failure_audit';out.mkdir(exist_ok=True);per=pd.DataFrame(rows);cases=pd.read_csv(ROOT/'results/engineering_roi_v1/final_case_metrics.csv');merged=per.merge(cases,on=['method','scenario_id'],how='left');numeric=merged.select_dtypes('number');assoc=[]
 for metric in ('trajectory_h_rel_l2','mixture_volume_relative_error','peak_hmax_relative_error','peak_stage_absolute_error','peak_Q_relative_error','debris_front_mae_km','arrival_MAE_s'):
  if metric in numeric:
   for signal in ('top10_share','consecutive_reselection_fraction','mean_selection_count_per_used_patch'):
    assoc.append({'metric':metric,'signal':signal,'pearson':numeric[[metric,signal]].corr(method='pearson').iloc[0,1],'spearman':numeric[[metric,signal]].corr(method='spearman').iloc[0,1]})
 per.to_csv(out/'selection_persistence_summary.csv',index=False);merged.to_csv(out/'per_case_mechanism_summary.csv',index=False);merged.groupby('method').mean(numeric_only=True).reset_index().to_csv(out/'method_mechanism_summary.csv',index=False);report={'no_new_model_execution':True,'methods':list(METHODS),'descriptive_associations':assoc,'amplitude_accumulation':'NOT DIRECTLY OBSERVABLE FROM EXISTING V1 ARTIFACTS'};(out/'failure_mechanism_report.json').write_text(json.dumps(report,indent=2,allow_nan=True));(ROOT/'reports/ENGINEERING_ROI_V1_FAILURE_MECHANISM.md').write_text('# EngineeringROI-v1 failure mechanism audit\n\nObserved selection persistence and correlations are descriptive associations, not causal proof. Raw local correction amplitude is **NOT DIRECTLY OBSERVABLE FROM EXISTING V1 ARTIFACTS**. No new model execution occurred.\n')
if __name__=='__main__':main()
