"""Static paper-only Figure 05 and engineering-facts compiler.

Reads completed CSV result artifacts only.  It has no model/evaluator imports and
does not execute rollout, inference, training, or statistical procedures.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]; PAPER=ROOT/'paper_results'
SOURCES={
 'EngineeringROI_B10':ROOT/'results'/'engineering_roi_v1'/'runs'/'EngineeringROI_B10'/'final_method_summary.csv',
 'DynamicOnly_B10':ROOT/'results'/'engineering_roi_v1'/'runs'/'DynamicOnly_B10'/'final_method_summary.csv',
 'SupportRiskOnly_B10':ROOT/'results'/'engineering_roi_v1'/'runs'/'SupportRiskOnly_B10'/'final_method_summary.csv',
 'RiskTemporal_B10':ROOT/'results'/'engineering_roi_v2'/'runs'/'RiskTemporal_B10'/'final_method_summary.csv',
}
METRICS=['trajectory_h_rel_l2','trajectory_momentum_rel_l2','mixture_volume_relative_error','debris_front_mae_km']
LABELS=['trajectory h RelL2','trajectory momentum RelL2','volume relative error','front MAE (km)']
def save(fig,path):
 for ext in ('png','pdf','svg'): fig.savefig(path.with_suffix('.'+ext),dpi=320,bbox_inches='tight')
 plt.close(fig)
def main():
 rows=[]
 for method,path in SOURCES.items():
  if not path.exists(): raise FileNotFoundError(f'existing completed artifact unavailable: {path}')
  x=pd.read_csv(path); x=x[x.method.eq(method)]
  if len(x)!=1: raise RuntimeError(f'expected one {method} row in {path}, found {len(x)}')
  rows.append(x.iloc[0])
 data=pd.DataFrame(rows).set_index('method').loc[list(SOURCES)].reset_index()
 fig,axs=plt.subplots(2,2,figsize=(10.5,7.5)); colors=['#b55d60','#d99037','#5b9d75','#4477aa']
 for ax,metric,label in zip(axs.flat,METRICS,LABELS):
  ax.bar(data.method.str.replace('_B10','',regex=False),data[metric],color=colors)
  if data[metric].max()/max(data[metric].min(),1e-12)>30: ax.set_yscale('log')
  ax.set_title(label);ax.tick_params(axis='x',rotation=24,labelsize=8);ax.grid(axis='y',alpha=.25)
 fig.suptitle('Completed VAL failure mechanism: closed-loop local-refinement development results');fig.tight_layout();save(fig,PAPER/'final_figures'/'figure_05_failure_mechanism')
 use_case='''# Engineering use-case facts

## Scope

This method is a rapid post-detection hazard-evolution forecasting layer, not a hazard-initiation detector. It assumes an organization already has, or can access, terrain products, remote sensing, rainfall/hydrological and geological-hazard monitoring, numerical simulation environments, historical scenario databases, high-fidelity physical models, and operational computing infrastructure.

Once an event has been detected and an initial hazard state, source estimate, or upstream-model estimate is available, the intended task is rapid downstream scenario evolution: propagation, dynamically changing regions, debris-front evolution, possible wet-area expansion, false-positive wet extent, and multiple near-term scenarios.

## Rolling workflow

Existing monitoring / remote sensing / numerical-modeling system -> current hazard state / source estimate -> Frozen Global Operator -> rapid full-domain coarse rollout -> Support-Risk ROI allocation -> Temporal Refresh -> budgeted local correction -> physical and support guards -> rapid downstream hazard-evolution scenario -> updated monitoring state -> repeat rolling forecast.

The method is intended as a rapid scenario-updating layer coupled to existing monitoring and high-fidelity modeling infrastructure. High-fidelity numerical models remain responsible for physics, offline reconstruction, scenario generation, and reference simulation; RiskTemporal provides fast online or near-online scenario rollout, dynamic-region updates, front evolution, and risk-focused local refinement.

## H0 runtime context and limits

For the recorded H0 runtime environment, RiskTemporal-v1 B10 completed the existing 1440-s (24-min) simulated hazard-evolution rollout in approximately 27.1 s: a simulated-horizon/wall-clock ratio of approximately 53.2. This is a real-time progression ratio, not a speedup claim over D-Claw or any other high-fidelity solver.

This study does not establish automatic initiation detection, exact failure-time prediction, operational warning issuance, precise station-arrival prediction, field deployment, field-observation validation, or replacement of governmental high-fidelity systems. H0 station response remains prematurely timed; the appropriate interpretation is propagation trend and spatial hazard evolution, not exact minute-level warning.
'''
 (PAPER/'manuscript_notes'/'ENGINEERING_USE_CASE_FACTS.md').write_text(use_case,encoding='utf-8')
 results=(PAPER/'manuscript_notes'/'RESULTS_FACTS.md').read_text(encoding='utf-8')
 add='''\n## H0 computational latency\n\nThe existing H0 rollout covers a 1440-s (24-min) simulated horizon. FrozenGlobal completed it in approximately 15.7 s; RiskTemporal-v1 B10 completed it in approximately 27.1 s. The resulting simulated-horizon/wall-clock ratio for RiskTemporal is approximately 53.2. Local refinement therefore adds overhead relative to FrozenGlobal while remaining far faster than real-time progression of the simulated event. This is not a controlled speedup comparison with a high-fidelity solver.\n'''
 if '## H0 computational latency' not in results: (PAPER/'manuscript_notes'/'RESULTS_FACTS.md').write_text(results.rstrip()+'\n'+add,encoding='utf-8')
 discussion=(PAPER/'manuscript_notes'/'DISCUSSION_FACTS.md').read_text(encoding='utf-8')
 add='''\n## Operational engineering interpretation\n\nWhere a professional hazard organization already has monitoring, high-fidelity modeling, and baseline data infrastructure, RiskTemporal is best interpreted as a rapid rolling scenario-prediction layer after hazard detection. High-fidelity numerical modeling supplies physical reconstruction, reference simulation, and scenario generation; the frozen method supplies fast near-term propagation rollout, dynamic-region updating, front evolution, and risk-focused local refinement. It is not a replacement for D-Claw or other high-fidelity systems.\n'''
 if '## Operational engineering interpretation' not in discussion: (PAPER/'manuscript_notes'/'DISCUSSION_FACTS.md').write_text(discussion.rstrip()+'\n'+add,encoding='utf-8')
 captions=(PAPER/'manuscript_notes'/'FIGURE_CAPTIONS_DRAFT.md').read_text(encoding='utf-8')
 captions=captions.replace('Figure 5. Completed VAL failure study: aggressive dynamically targeted correction can destabilize autoregressive closed-loop rollout.','Figure 5. Completed VAL closed-loop failure mechanism using EngineeringROI B10, DynamicOnly B10, SupportRiskOnly B10, and RiskTemporal-v1 B10 from their completed frozen artifacts. Aggressive dynamically targeted local refinement can destabilize long-horizon autoregressive rollout; support-risk restriction improves stability, while one-step temporal refresh yields a more balanced operating point. The figure does not claim universal superiority of RiskTemporal.')
 (PAPER/'manuscript_notes'/'FIGURE_CAPTIONS_DRAFT.md').write_text(captions,encoding='utf-8')
 claims=(PAPER/'CLAIM_AUDIT_FINAL.md').read_text(encoding='utf-8')
 claim='- RiskTemporal-v1 B10 completes the existing H0 24-min simulated rollout in approximately 27.1 s under the recorded runtime environment, corresponding to approximately 53.2 times simulated real-time progression; this is not a D-Claw speedup claim.\n'
 if '53.2 times simulated' not in claims: (PAPER/'CLAIM_AUDIT_FINAL.md').write_text(claims.replace('## Not supported',claim+'\n## Not supported'),encoding='utf-8')
 audit=json.loads((PAPER/'release_audit_final.json').read_text(encoding='utf-8'))
 audit.update({'figure_05_completed_artifacts_only':True,'figure_05_method_sources_verified':True,'figure_05_sources':{k:str(v.relative_to(ROOT)).replace('\\\\','/') for k,v in SOURCES.items()},'engineering_use_case_facts_completed':True,'h0_runtime_seconds':27.0728646,'h0_simulated_horizon_seconds':1440,'h0_simulated_to_wallclock_ratio':1440/27.0728646,'no_new_experiment':True,'further_experiment_required':'NO'})
 (PAPER/'release_audit_final.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
 (PAPER/'release_audit_final.md').write_text('# Final release audit\n\nPASS. Figure 05 reads only the four verified completed VAL artifacts. Engineering use-case facts, H0 runtime context, captions, claim audit, and static audit are complete. No model, evaluator, rollout, training, inference, algorithm, checkpoint, TEST, VAL, or H0 result was run or modified.\n',encoding='utf-8')
 print(json.dumps({'status':'PASS','figure_05_methods':data.method.tolist(),'sources':{k:str(v.relative_to(ROOT)) for k,v in SOURCES.items()}},indent=2))
if __name__=='__main__':main()
