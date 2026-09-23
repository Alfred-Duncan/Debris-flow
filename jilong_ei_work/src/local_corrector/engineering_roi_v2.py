"""Predeclared SupportRisk-only V2 selector; V1 weights are not retuned."""
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
 c=json.loads(Path(path).read_text());e={'version':'EngineeringROI-v2','wet_threshold_m':.05,'shallow_margin_upper_h_m':.15,'roi_component':'support_risk','support_risk_definition':'new_wet_or_shallow_margin','candidate_rule':'support_overlap_and_positive_support_risk','normalization':'positive_percentile_rank','spatial_diversity':'nonadjacent_first_then_fill','temporal_refresh':'exclude_previous_step_selected_patches','temporal_refresh_steps':1,'temporal_fill_from_excluded':False,'depth_envelope_guard':'current_global_convex_hull','depth_envelope_channels':['h'],'budgets':[.05,.1,.2],'global_model':'frozen','local_model':'best_update_4000','support_guard':'current_or_global_provisional_wet','truth_in_roi_scoring':False}
 if any(c.get(k)!=v for k,v in e.items()):raise RuntimeError('ENGINEERING_ROI_V2_CONFIG_INVALID')
 return c
def select_engineering_roi_v2(current_physical,provisional_physical,metadata,config,budget_fraction,temporal_state,temporal_refresh=True):
 if budget_fraction not in config['budgets']:raise ValueError('ENGINEERING_ROI_V2_BUDGET_INVALID')
 _,overlap=compute_support_candidates(current_physical,provisional_physical,metadata,config['wet_threshold_m']);raw=compute_support_risk_component(current_physical,provisional_physical,metadata,config).detach().cpu().numpy();rank=positive_percentile_rank(raw);patches=tuple(metadata.layout.eligible);all_candidates=np.flatnonzero((overlap.detach().cpu().numpy()>0)&(rank>0));candidates=np.asarray([i for i in all_candidates if not temporal_refresh or patches[i].patch_id not in temporal_state.previous_selected_patch_ids],dtype=int);chosen,first,second=_diverse(candidates,rank,patches,metadata.layout.count_for_budget(budget_fraction));selected=tuple(patches[i] for i in chosen);ids=frozenset(p.patch_id for p in selected);prior=temporal_state.previous_selected_patch_ids;return selected,TemporalRefreshState(ids),{'budget_fraction':budget_fraction,'budget_max_count':metadata.layout.count_for_budget(budget_fraction),'candidate_count':len(candidates),'pre_refresh_candidate_count':len(all_candidates),'selected_count':len(selected),'mean_selected_support_risk_rank':float(np.mean(rank[chosen])) if chosen else 0.,'first_pass_count':first,'second_pass_count':second,'selected_patch_consecutive_overlap_count':len(ids&prior),'selected_patch_consecutive_overlap_fraction':len(ids&prior)/max(len(selected),1)}
