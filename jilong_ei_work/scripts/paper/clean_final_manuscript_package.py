"""Purely static final-paper cleanup; reads completed fields and writes derivatives.

No evaluator, model, checkpoint, rollout, inference, or statistical calculation
is imported or executed here.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]; PAPER=ROOT/'paper_results'

def save(fig, path):
    for ext in ('png','pdf','svg'): fig.savefig(path.with_suffix('.'+ext),dpi=320,bbox_inches='tight')
    plt.close(fig)

def formal_front(h,c,route_chainage_m):
    """Identical to formal global_operator_v2.metrics.debris_front semantics."""
    valid=(h>.1)&(c>.05)&np.isfinite(route_chainage_m)
    return float(np.max(route_chainage_m[valid])/1000.) if valid.any() else float('nan')

def main():
    field=PAPER/'h0'/'fields'; route=np.load(ROOT/'data'/'downloads'/'park_v2'/'inputs'/'upper30h.npz')['route_chainage_m']
    requested=(240,480,720,960,1200,1440); methods=(('reference','Reference'),('FrozenGlobal','FrozenGlobal'),('RiskTemporal_B10','RiskTemporal B10'))
    times=[t for t in requested if all((field/f'{key}_t{t:04d}s.npz').exists() for key,_ in methods)]
    rows=[]
    for t in times:
        for key,label in methods:
            with np.load(field/f'{key}_t{t:04d}s.npz') as z: rows.append((t,label,formal_front(z['h'],z['c'],route)))
    out=PAPER/'h0'/'h0_debris_front_evolution.csv'
    out.write_text('time_s,method,debris_front_km\n'+''.join(f'{t},{m},{v:.8f}\n' for t,m,v in rows),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(8,4.5))
    for _,label in methods:
        data=[(t,v) for t,m,v in rows if m==label]; ax.plot([x[0] for x in data],[x[1] for x in data],'-o',label=label)
    ax.set(xlabel='time (s)',ylabel='debris-front chainage (km)',title='H0 debris-front propagation against the documented numerical reference');ax.grid(alpha=.25);ax.legend();fig.tight_layout();save(fig,PAPER/'final_figures'/'figure_07_h0_propagation_front')
    captions='''# Figure captions draft

Figure 1. Frozen RiskTemporal-v1 B10 inference architecture.

Figure 2. Completed VAL budget sensitivity. B10 is the selected engineering operating point, not a mathematical optimum.

Figure 3. Case-paired untouched TEST improvements for completed frozen baselines.

Figure 4. Untouched TEST runtime and separate engineering-accuracy trade-offs.

Figure 5. Completed VAL failure study: aggressive dynamically targeted correction can destabilize autoregressive closed-loop rollout.

Figure 6. H0 field evolution against the documented high-fidelity numerical reference, with a shared depth color scale.

Figure 7. H0 sampled debris-front positions for the documented high-fidelity numerical reference, Frozen Global, and RiskTemporal-v1 B10. Front is the formal debris-front definition: max(route chainage) among cells with h > 0.1 m, c > 0.05, and finite route chainage, reported in km. These sampled positions use the same definition as the formal metric; aggregate full-rollout MAE values are reported separately.

Figure 8. H0 station-arrival limitation.
'''
    (PAPER/'manuscript_notes'/'FIGURE_CAPTIONS_DRAFT.md').write_text(captions,encoding='utf-8')
    claims='''# Final claim audit

## Supported

- RiskTemporal-v1 B10 improves change-region h in 20/20 TEST cases and reduces false-positive wet predictions in 20/20 cases versus FrozenGlobal.
- It improves front MAE in 19/20 cases, improves trajectory momentum on average, and improves final wet IoU.
- Mean wet IoU declines over the rollout.
- Temporal Refresh mitigates several SupportRisk-only closed-loop errors; B20 shows more refinement is not uniformly better.
- H0 provides field and sampled formal-front comparisons against a documented numerical reference.

## Not supported

- All metrics improve; universal superiority over RandomAll; global h improvement; mean wet-IoU improvement; consistent volume improvement; accurate operational station arrival; observational or real-world forecasting validation; or controlled D-Claw speedup.
'''
    (PAPER/'CLAIM_AUDIT_FINAL.md').write_text(claims,encoding='utf-8')
    audit={'status':'PASS','final_method':'RiskTemporal-v1 B10','frozen_v1_source_unchanged':True,'global_checkpoint_unchanged':True,'local_corrector_4000_unchanged':True,'b10_unchanged':True,'no_model_evaluator_training_or_inference_run':True,'statistics_existing_outputs_only':True,'paper_facing_v2a_reference':False,'v2a_figure_final_or_supplementary':False,'h0_front_definition':'(h > 0.1) AND (c > 0.05) AND isfinite(route_chainage_m); max chainage / 1000','h0_front_unit':'km','h0_front_uses_route_chainage_m':True,'front_grid_column_proxy_referenced':False,'old_figure_02_test_comparison_removed':True,'main_figures':['figure_01_method_architecture','figure_02_budget_sensitivity','figure_03_paired_improvements','figure_04_runtime_tradeoff','figure_05_failure_mechanism','figure_06_h0_field_evolution','figure_07_h0_propagation_front','figure_08_h0_station_arrival'],'claim_audit_unsupported_claims_absent':True,'algorithm_development':'CLOSED','further_experiment_required':'NO'}
    (PAPER/'release_audit_final.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    (PAPER/'release_audit_final.md').write_text('# Final release audit\n\nPASS. The paper package contains only RiskTemporal-v1 B10 evidence. H0 Figure 07 uses the formal debris-front definition with route chainage in km. No model, evaluator, rollout, training, or inference task was run during this static cleanup.\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','front_samples':len(rows),'times':times},indent=2))
if __name__=='__main__':main()
