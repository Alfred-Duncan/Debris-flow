"""Risk-Aware Temporally Refreshed Local Refinement; not V1 weight retuning."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .engineering_roi import compute_support_candidates,compute_support_risk_component,positive_percentile_rank,_diverse
@dataclass(frozen=True)
class TemporalRefreshState: previous_selected_patch_ids:frozenset[int]=frozenset()
def initial_temporal_state():return TemporalRefreshState()
def load_engineering_roi_v2_config(path):
    config=json.loads(Path(path).read_text());expected={'version':'EngineeringROI-v2','roi_component':'support_risk','temporal_refresh_steps':1,'temporal_fill_from_excluded':False,'depth_envelope_guard':'current_global_convex_hull','truth_in_roi_scoring':False}
    if any(config.get(k)!=v for k,v in expected.items()) or config.get('budgets')!=[.05,.1,.2]:raise RuntimeError('ENGINEERING_ROI_V2_CONFIG_INVALID')
    return config
def select_engineering_roi_v2(current_physical,provisional_physical,metadata,config,budget_fraction,temporal_state):
    if budget_fraction not in config['budgets']:raise ValueError('ENGINEERING_ROI_V2_BUDGET_INVALID')
    _,overlap=compute_support_candidates(current_physical,provisional_physical,metadata,config['wet_threshold_m']);raw=compute_support_risk_component(current_physical,provisional_physical,metadata,config).detach().cpu().numpy();rank=positive_percentile_rank(raw);patches=tuple(metadata.layout.eligible);eligible=np.flatnonzero((overlap.detach().cpu().numpy()>0)&(rank>0));eligible=np.asarray([i for i in eligible if patches[i].patch_id not in temporal_state.previous_selected_patch_ids],dtype=int);maximum=metadata.layout.count_for_budget(budget_fraction);chosen,first,second=_diverse(eligible,rank,patches,maximum);selected=tuple(patches[i] for i in chosen);ids=frozenset(p.patch_id for p in selected);overlap_count=len(ids&temporal_state.previous_selected_patch_ids);return selected,TemporalRefreshState(ids),{'budget_fraction':budget_fraction,'budget_max_count':maximum,'candidate_count':len(eligible),'selected_count':len(selected),'mean_selected_support_risk_rank':float(np.mean(rank[chosen])) if chosen else 0.,'first_pass_count':first,'second_pass_count':second,'selected_patch_consecutive_overlap_count':overlap_count,'selected_patch_consecutive_overlap_fraction':overlap_count/max(len(selected),1)}
