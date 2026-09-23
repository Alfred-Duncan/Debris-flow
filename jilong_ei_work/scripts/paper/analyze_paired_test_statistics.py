"""Deterministic paired TEST statistics; positive improvement means RiskTemporal is better."""
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'paper_results/statistics';OUT.mkdir(parents=True,exist_ok=True);SEED=20260924
LOW=['trajectory_h_rel_l2','trajectory_momentum_rel_l2','change_region_h_rel_l2','change_region_momentum_rel_l2','false_positive_wet_fraction','mixture_volume_relative_error','debris_front_mae_km'];HIGH=['mean_wet_iou']
def run():
 sources={'FrozenGlobal':ROOT/'paper_results/test/frozen_global/final_case_metrics.csv','RandomAll_B10':ROOT/'paper_results/test/baselines/randomall_b10/final_case_metrics.csv','SupportRisk_Base_B10':ROOT/'paper_results/test/baselines/supportrisk_base_b10/final_case_metrics.csv','RiskTemporal_B10':ROOT/'paper_results/test/risktemporal_b10/final_case_metrics.csv'};d={k:pd.read_csv(v).set_index('scenario_id') for k,v in sources.items()};rng=np.random.default_rng(SEED);rows=[]
 for base in ('FrozenGlobal','RandomAll_B10','SupportRisk_Base_B10'):
  for m in LOW+HIGH:
   x=(d[base][m]-d['RiskTemporal_B10'][m]).to_numpy() if m in LOW else (d['RiskTemporal_B10'][m]-d[base][m]).to_numpy();boots=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(10000)]);p=1. if np.allclose(x,0) else wilcoxon(x).pvalue;rows.append({'baseline':base,'metric':m,'baseline_mean':d[base][m].mean(),'risktemporal_mean':d['RiskTemporal_B10'][m].mean(),'mean_improvement':x.mean(),'median_improvement':np.median(x),'std':x.std(ddof=1),'improved_cases':int((x>0).sum()),'worsened_cases':int((x<0).sum()),'ties':int((x==0).sum()),'improvement_fraction':float((x>0).mean()),'bootstrap_mean_ci_low':np.quantile(boots,.025),'bootstrap_mean_ci_high':np.quantile(boots,.975),'raw_p_value':p})
 out=pd.DataFrame(rows);out['holm_adjusted_p_value']=out.groupby('baseline').raw_p_value.transform(lambda s:np.minimum(1,np.maximum.accumulate(np.sort(s.to_numpy())*(len(s)-np.arange(len(s)))))[np.argsort(np.argsort(s.to_numpy()))]);out.to_csv(OUT/'test_paired_statistics.csv',index=False);(OUT/'test_paired_statistics.tex').write_text(out.to_latex(index=False),encoding='utf8')
if __name__=='__main__':run()
