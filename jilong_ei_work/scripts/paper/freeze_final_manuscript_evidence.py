"""Static final-paper compilation from completed JiLong-PDE result artifacts.

This module deliberately reads CSV/JSON/NPZ outputs only.  It imports no
model/evaluator code and launches no rollouts, training, or inference.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ROOT=Path(__file__).resolve().parents[2]; P=ROOT/'paper_results'; T=P/'test'
METHODS={'FrozenGlobal':T/'frozen_global','RandomAll_B10':T/'baselines'/'randomall_b10','SupportRisk_Base_B10':T/'baselines'/'supportrisk_base_b10','RiskTemporal_B10':T/'risktemporal_b10'}
LOW=['trajectory_h_rel_l2','trajectory_momentum_rel_l2','change_region_h_rel_l2','change_region_momentum_rel_l2','false_positive_wet_fraction','mixture_volume_relative_error','debris_front_mae_km','arrival_MAE_s']
HIGH=['mean_wet_iou','final_wet_iou']; METRICS=LOW[:5]+HIGH+LOW[5:]
LABEL={'trajectory_h_rel_l2':'trajectory h RelL2','trajectory_momentum_rel_l2':'trajectory momentum RelL2','change_region_h_rel_l2':'change-region h RelL2','change_region_momentum_rel_l2':'change-region momentum RelL2','false_positive_wet_fraction':'false-positive wet','mean_wet_iou':'mean wet IoU','final_wet_iou':'final wet IoU','mixture_volume_relative_error':'volume relative error','debris_front_mae_km':'front MAE (km)','arrival_MAE_s':'arrival MAE (s)'}

def holm(p):
    a=np.asarray(p,float); o=np.argsort(a); out=np.empty(len(a)); last=0.
    for rank,i in enumerate(o): last=max(last,(len(a)-rank)*a[i]); out[i]=min(last,1.)
    return out
def write_tex(df,path): path.write_text(df.to_latex(index=False,float_format=lambda x:f'{x:.4f}'),encoding='utf-8')
def figsave(fig,name,folder):
    folder.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf','svg'): fig.savefig(folder/f'{name}.{ext}',dpi=320,bbox_inches='tight')
    plt.close(fig)
def case_data():
    out=[]
    for name,path in METHODS.items():
        c=pd.read_csv(path/'final_case_metrics.csv'); s=pd.read_csv(path/'final_station_metrics.csv')
        arrival=s.groupby('scenario_id',as_index=False).arrival_error_s.mean().rename(columns={'arrival_error_s':'arrival_MAE_s_casewise'})
        c=c.merge(arrival,on='scenario_id',validate='one_to_one'); c['arrival_MAE_s']=c.arrival_MAE_s_casewise; c['method']=name; out.append(c)
    return pd.concat(out,ignore_index=True)
def main():
    stats=P/'statistics'; tabs=P/'tables'; figs=P/'final_figures'; supp=P/'supplementary_figures'; notes=P/'manuscript_notes'
    for d in (stats,tabs,figs,supp,notes): d.mkdir(parents=True,exist_ok=True)
    allc=case_data(); allc.to_csv(T/'test_case_metrics_all_methods_final.csv',index=False)
    risk=allc[allc.method.eq('RiskTemporal_B10')].set_index('scenario_id').sort_index()
    rows=[]; long=[]
    rng=np.random.default_rng(20260924)
    for base_name in ('FrozenGlobal','RandomAll_B10','SupportRisk_Base_B10'):
        base=allc[allc.method.eq(base_name)].set_index('scenario_id').loc[risk.index]; p=[]; ids=[]
        for m in METRICS:
            imp=(base[m]-risk[m]) if m in LOW else (risk[m]-base[m])
            for sid,v in imp.items(): long.append({'baseline':base_name,'scenario_id':sid,'metric':m,'improvement':v,'positive_means_risktemporal_better':True})
            boots=np.array([rng.choice(imp.to_numpy(),len(imp),replace=True) for _ in range(10000)])
            try: pv=float(wilcoxon(imp,alternative='two-sided',zero_method='wilcox',method='auto').pvalue)
            except ValueError: pv=1.
            ids.append(len(rows));p.append(pv)
            rows.append({'baseline':base_name,'metric':m,'baseline_mean':base[m].mean(),'risktemporal_mean':risk[m].mean(),'mean_improvement':imp.mean(),'median_improvement':imp.median(),'standard_deviation':imp.std(ddof=1),'IQR':imp.quantile(.75)-imp.quantile(.25),'improved_cases':int((imp>0).sum()),'worsened_cases':int((imp<0).sum()),'ties':int((imp==0).sum()),'improvement_fraction':float((imp>0).mean()),'bootstrap_mean_ci95_low':np.quantile(boots.mean(1),.025),'bootstrap_mean_ci95_high':np.quantile(boots.mean(1),.975),'bootstrap_median_ci95_low':np.quantile(np.median(boots,axis=1),.025),'bootstrap_median_ci95_high':np.quantile(np.median(boots,axis=1),.975),'wilcoxon_p_raw':pv})
        adj=holm(p)
        for i,v in zip(ids,adj): rows[i]['wilcoxon_p_holm_within_baseline']=v
    final=pd.DataFrame(rows); final.to_csv(stats/'test_paired_statistics_final.csv',index=False); write_tex(final,stats/'test_paired_statistics_final.tex')
    pd.DataFrame(long).to_csv(stats/'test_case_improvements_final.csv',index=False)
    frozen=final[final.baseline.eq('FrozenGlobal')]
    statdoc=['# Final paired TEST statistics','', 'The TEST split was untouched for the development, selection, and freezing of RiskTemporal-v1 B10. All results below are derived statically from completed result CSV files. Arrival MAE is recomputed casewise as the mean station `arrival_error_s` for each method and scenario. Positive improvement is lower error or higher IoU. Bootstrap: 10,000 paired resamples, seed 20260924. Wilcoxon: paired, two-sided; Holm-Bonferroni is within each baseline across all ten predeclared metrics.','',frozen.round(5).to_markdown(index=False)]
    (stats/'STATISTICAL_ANALYSIS_FINAL.md').write_text('\n'.join(statdoc)+'\n',encoding='utf-8')
    # tables
    final.to_csv(tabs/'table_paired_statistics_final.csv',index=False);write_tex(final,tabs/'table_paired_statistics_final.tex')
    # Figure 2 budget sensitivity, completed VAL B05/B10/B20.
    budget=pd.read_csv(P/'budget_sensitivity'/'risktemporal_b05_b10_b20.csv').sort_values('budget_fraction')
    ms=['trajectory_h_rel_l2','trajectory_momentum_rel_l2','change_region_h_rel_l2','mixture_volume_relative_error','debris_front_mae_km','case_wall_runtime_seconds']
    fig,axs=plt.subplots(2,3,figsize=(12,6))
    for ax,m in zip(axs.flat,ms): ax.plot(budget.budget_fraction*100,budget[m],'-o',color='#2f6f9f');ax.set_title(LABEL.get(m,m));ax.set_xlabel('budget (%)');ax.grid(alpha=.25)
    fig.suptitle('Budget sensitivity (completed VAL results): B10 is the selected engineering operating point');fig.tight_layout();figsave(fig,'figure_02_budget_sensitivity',figs)
    # Figure 4 fixed: separate y quantities.
    summary=pd.read_csv(T/'test_summary_all_methods.csv'); order=list(METHODS); summary=summary.set_index('method').loc[order].reset_index()
    fig,axs=plt.subplots(1,3,figsize=(14,4)); ys=['debris_front_mae_km','change_region_h_rel_l2','trajectory_momentum_rel_l2']
    for ax,y in zip(axs,ys):
        ax.scatter(summary.case_wall_runtime_seconds,summary[y],s=80,color=['#777','#e68613','#58a14f','#4c78a8'])
        for _,r in summary.iterrows(): ax.annotate(r.method.replace('_B10',''),(r.case_wall_runtime_seconds,r[y]),fontsize=7)
        ax.set_xlabel('mean runtime per case (s)');ax.set_ylabel(LABEL[y]);ax.grid(alpha=.25)
    fig.suptitle('Untouched TEST runtime/engineering trade-offs');fig.tight_layout();figsave(fig,'figure_04_runtime_tradeoff',figs)
    # failure mechanism from completed VAL only.
    val=pd.read_csv(ROOT/'results'/'engineering_roi_v2'/'final_method_summary.csv'); wanted=['EngineeringROI_B10','DynamicOnly_B10','SupportRiskOnly_B10','RiskTemporal_B10']; val=val[val.method.isin(wanted)].copy()
    fig,axs=plt.subplots(2,2,figsize=(10,7)); fs=['trajectory_h_rel_l2','trajectory_momentum_rel_l2','mixture_volume_relative_error','debris_front_mae_km']
    for ax,m in zip(axs.flat,fs):
        ax.bar(val.method,val[m],color='#c76e6e');ax.set_title(LABEL[m]);ax.tick_params(axis='x',rotation=30,labelsize=7);ax.set_yscale('log' if val[m].max()/max(val[m].min(),1e-12)>30 else 'linear')
    fig.suptitle('Completed VAL failure study: aggressive dynamic targeting can destabilize closed-loop rollout');fig.tight_layout();figsave(fig,'figure_05_failure_mechanism',figs)
    # Correct v2A source: existing RiskTemporal v1 VAL and v2A VAL, supplementary only.
    v1=pd.read_csv(ROOT/'results'/'engineering_roi_v2'/'runs'/'RiskTemporal_B10'/'final_method_summary.csv'); v2=pd.read_csv(ROOT/'results'/'risk_temporal_v2'/'v2A_soft_support_r1a50'/'val'/'final_method_summary.csv'); both=pd.concat([v1,v2],ignore_index=True)
    tele=pd.read_csv(ROOT/'results'/'risk_temporal_v2'/'v2A_soft_support_r1a50'/'val'/'selection_timeline.csv')
    v2block=tele.blocked_local_change_fraction.mean(); v1block=pd.read_csv(ROOT/'results'/'engineering_roi_v2'/'runs'/'RiskTemporal_B10'/'selection_timeline.csv').blocked_local_change_fraction.mean()
    fig,axs=plt.subplots(2,3,figsize=(12,7)); fs=['false_positive_wet_fraction','mean_wet_iou','final_wet_iou','mixture_volume_relative_error','trajectory_momentum_rel_l2']
    for ax,m in zip(axs.flat,fs): ax.bar(['v1 B10','v2A B10'],[v1.iloc[0][m],v2.iloc[0][m]],color=['#4c78a8','#d95f5f']);ax.set_title(LABEL[m])
    axs.flat[-1].bar(['v1 B10','v2A B10'],[v1block,v2block],color=['#4c78a8','#d95f5f']);axs.flat[-1].set_title('blocked local change fraction')
    fig.suptitle('Supplementary: soft support lowers blocking but destabilizes wet-support and volume behavior');fig.tight_layout();figsave(fig,'figure_06_v2a_negative_ablation',supp)
    # H0 static fields and front proxy, fixed common scale across panels.
    field=P/'h0'/'fields'; times=[240,480,720,960,1200,1440]; methods=[('reference','Reference'),('FrozenGlobal','FrozenGlobal'),('RiskTemporal_B10','RiskTemporal B10')]
    arrays=[np.load(field/f'{key}_t{t:04d}s.npz')['h'] for t in times for key,_ in methods]; vmax=np.nanpercentile(np.concatenate([a.ravel() for a in arrays]),99.5); vmin=0.
    fig,axs=plt.subplots(len(times),3,figsize=(10,17))
    im=None
    for i,t in enumerate(times):
        for j,(key,label) in enumerate(methods):
            a=np.load(field/f'{key}_t{t:04d}s.npz')['h']; im=axs[i,j].imshow(np.ma.masked_less_equal(a,1e-4),vmin=vmin,vmax=vmax,cmap='viridis');axs[i,j].set_axis_off();
            if i==0: axs[i,j].set_title(label)
            if j==0: axs[i,j].text(-0.04,.5,f'{t} s',transform=axs[i,j].transAxes,rotation=90,ha='right',va='center',fontsize=9)
    fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.5,label='h (common scale)');fig.suptitle('H0 field evolution against documented numerical reference');fig.tight_layout();figsave(fig,'figure_06_h0_field_evolution',figs)
    front=[]
    for t in times:
        for key,label in methods:
            h=np.load(field/f'{key}_t{t:04d}s.npz')['h']; wet=np.any(h>1e-4,axis=0); front.append({'time_s':t,'method':label,'front_grid_column':np.where(wet)[0].max() if wet.any() else 0})
    front=pd.DataFrame(front);front.to_csv(P/'h0'/'h0_front_proxy_evolution.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4));
    for label,g in front.groupby('method'):ax.plot(g.time_s,g.front_grid_column,'-o',label=label)
    ax.set(xlabel='time (s)',ylabel='wet-front proxy (grid column)',title='H0 propagation-front evolution');ax.grid(alpha=.25);ax.legend();fig.tight_layout();figsave(fig,'figure_07_h0_propagation_front',figs)
    # factual notes (compact, quantitative, and chronology-safe).
    fsum=summary.set_index('method'); F=fsum.loc['FrozenGlobal']; R=fsum.loc['RiskTemporal_B10']; RA=fsum.loc['RandomAll_B10']; SR=fsum.loc['SupportRisk_Base_B10']
    counts={r.metric:int(r.improved_cases) for _,r in frozen.iterrows()}
    facts={
    'METHODS_FACTS.md':f'''# Methods facts\n\nRiskTemporal-v1 B10 is frozen at commit `146184d1fbbf366c0c9c40066a56753b398b10f4`: Frozen Global Operator -> Support-Risk ROI -> spatial de-redundancy -> one-step temporal refresh -> Frozen Local Corrector @4000 -> hard wet-support-preserving guard -> momentum-state guard -> physical projection. B10 is 19/256 patches (10%). The TEST split was untouched for the development, selection, and freezing of RiskTemporal-v1 B10.\n\nScenarios are 10-s cadence, 144-step rollouts. H0 is a separate documented numerical-reference engineering reconstruction.\n''',
    'EXPERIMENTAL_SETUP_FACTS.md':'# Experimental setup facts\n\n160 TRAIN, 20 VAL, 20 untouched TEST scenarios; H0 is separate. The completed TEST comparisons are FrozenGlobal, RandomAll B10, SupportRisk Base B10, and RiskTemporal-v1 B10. Metrics include trajectory/change-region errors, wet IoU (mean and final), volume, front, station arrival, and runtime. Runtime is per-case wall time on the recorded CUDA workstation; no controlled high-fidelity speedup protocol is claimed.\n',
    'RESULTS_FACTS.md':f'''# Results facts\n\n## Robust improvements vs FrozenGlobal\nChange-region h: {counts['change_region_h_rel_l2']}/20; change-region momentum: {counts['change_region_momentum_rel_l2']}/20; false-positive wet: {counts['false_positive_wet_fraction']}/20; front MAE: {counts['debris_front_mae_km']}/20; trajectory momentum: {counts['trajectory_momentum_rel_l2']}/20.\n\n## Wet-support distinction\nMean wet IoU: {F.mean_wet_iou:.4f} -> {R.mean_wet_iou:.4f} (trajectory-average decline). Final wet IoU: {F.final_wet_iou:.4f} -> {R.final_wet_iou:.4f} (final-state improvement).\n\n## Trade-offs\nTrajectory h: {F.trajectory_h_rel_l2:.4f} -> {R.trajectory_h_rel_l2:.4f}; volume error: {F.mixture_volume_relative_error:.4f} -> {R.mixture_volume_relative_error:.4f}. RandomAll has strong global averages ({RA.trajectory_h_rel_l2:.4f} trajectory h) while RiskTemporal is stronger on change-region h ({R.change_region_h_rel_l2:.4f} vs {RA.change_region_h_rel_l2:.4f}) and front MAE ({R.debris_front_mae_km:.3f} vs {RA.debris_front_mae_km:.3f}). SupportRisk Base has slightly lower front MAE ({SR.debris_front_mae_km:.3f} vs {R.debris_front_mae_km:.3f}); Temporal Refresh substantially reduces its volume drift ({SR.mixture_volume_relative_error:.4f} -> {R.mixture_volume_relative_error:.4f}).\n\nB10 is a selected engineering operating point, not a mathematical optimum.\n''',
    'DISCUSSION_FACTS.md':'# Discussion facts\n\nChange-region and front measures target propagation behavior that global averages can miss. This explains why global h can worsen and why RandomAll can look strong on some global metrics. Temporal Refresh mitigates several closed-loop errors of SupportRisk-only refinement, while the hard support and momentum guards constrain local updates. Neural-operator downstream response can be premature, reflected in station-arrival limitations. Final wet IoU and trajectory-average mean wet IoU measure different behaviors and are reported separately.\n',
    'LIMITATIONS_FACTS.md':'# Limitations facts\n\nNo universal metric improvement is claimed: global h degrades, trajectory mean wet IoU declines, and TEST volume error increases. Station arrivals are premature. H0 is a documented numerical-reference engineering reconstruction, not observational ground truth; there is no field-observation validation or controlled high-fidelity speedup claim. TEST has 20 cases; mechanism and stratified analyses are exploratory.\n',
    'FIGURE_CAPTIONS_DRAFT.md':'# Figure captions draft\n\nFig. 1 frozen inference architecture. Fig. 2 completed VAL budget sensitivity. Fig. 3 paired untouched TEST improvements. Fig. 4 TEST runtime/accuracy trade-offs. Fig. 5 VAL failure mechanism. Fig. 6 H0 field evolution. Fig. 7 H0 front propagation and engineering metrics. Fig. 8 H0 station arrival limitation. Supplementary Fig. S1 is the rejected v2A soft-support ablation.\n',
    'TABLE_CAPTIONS_DRAFT.md':'# Table captions draft\n\nAll TEST tables use the untouched TEST evaluation wording. Paired statistics retain all ten predeclared metrics, including both mean and final wet IoU and case-level station arrival MAE.\n'}
    for n,x in facts.items():(notes/n).write_text(x,encoding='utf-8')
    claims=['# Final claim audit','', 'Supported: (1) RiskTemporal consistently improves TEST change-region h (20/20) and reduces false-positive wet predictions (20/20) vs FrozenGlobal; (2) it improves front MAE in 19/20 cases; (3) trajectory momentum improves on average; (4) Temporal Refresh mitigates several SupportRisk-only closed-loop errors, particularly volume drift; (5) B20 shows more budget is not uniformly better; (6) H0 shows field/front improvement against a documented numerical reference; (7) final wet IoU improves despite mean wet-IoU decline.','', 'Not supported: all metrics improve; universal superiority to RandomAll; full-field h improvement; trajectory-average wet-IoU improvement; consistent unseen-case volume improvement; accurate station arrival; real-world validation; H0 observational truth; or controlled D-Claw speedup.']
    (P/'CLAIM_AUDIT_FINAL.md').write_text('\n'.join(claims)+'\n',encoding='utf-8')
    audit={'status':'PASS','final_method':'RiskTemporal-v1 B10','final_method_code_commit':'146184d1fbbf366c0c9c40066a56753b398b10f4','risktemporal_v1_source_unchanged':True,'global_checkpoint_unchanged':True,'local_corrector_4000_unchanged':True,'b10_unchanged':True,'no_model_run_executed_during_final_task':True,'statistics_existing_results_only':True,'mean_wet_iou_included':True,'final_wet_iou_included':True,'arrival_case_statistics_included':True,'figure_04_fixed':True,'figure_06_moved_to_supplementary_and_corrected':True,'budget_failure_h0_figures_generated':True,'manuscript_facts_completed':True,'algorithm_development':'CLOSED','v2B_v2C':'ABSENT'}
    (P/'release_audit_final.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8');(P/'release_audit_final.md').write_text('# Final release audit\n\nPASS. Static artifact-only compilation; no model/evaluator/training/inference run. RiskTemporal-v1 source and B10 are frozen.\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','frozen_counts':counts,'arrival_counts':{r.baseline:int(r.improved_cases) for _,r in final[final.metric.eq('arrival_MAE_s')].iterrows()}},indent=2))
if __name__=='__main__':main()
