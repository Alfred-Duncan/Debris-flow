"""Synthetic regression tests for frozen EngineeringROI-v1; no scenario data."""
from __future__ import annotations
import inspect, json, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.oracle_refinement import PatchLayout,select_random
from src.local_corrector.engineering_roi import (_diverse,build_section_mask,compute_roi_components,load_engineering_roi_config,
    positive_percentile_rank,select_engineering_roi)

def config(): return json.loads((ROOT/'configs/engineering_roi_v1.json').read_text())
def fields(shape=(8,8)):
    current=torch.zeros(1,6,*shape);provisional=torch.zeros_like(current);encoded_current=torch.zeros_like(current);encoded_provisional=torch.zeros_like(current);active=torch.ones(1,1,*shape,dtype=torch.bool);route=np.tile(np.arange(shape[1],dtype=float)*1000.,(shape[0],1));section=torch.zeros(*shape,dtype=torch.bool);return current,provisional,encoded_current,encoded_provisional,active,route,section
def raises(fn):
    try:fn()
    except RuntimeError:return
    raise AssertionError('expected clear metadata failure')

def main():
    cfg=config();layout=PatchLayout(np.ones((8,8),bool),4,4);current,provisional,encoded_current,encoded_provisional,active,route,section=fields()
    # API cannot accept teacher/target/future fields; scales affect dynamic, not Local scales.
    names=inspect.signature(compute_roi_components).parameters;assert not ({'truth','target','future','local_correction_scales'}&set(names))
    current[:,0]=.2;provisional[:,0]=.2;encoded_provisional[:,:,0:2,0:2]=2
    first=compute_roi_components(current,provisional,encoded_current,encoded_provisional,active,route,section,[1]*6,layout,cfg)
    scaled=compute_roi_components(current,provisional,encoded_current,encoded_provisional,active,route,section,[2]*6,layout,cfg)
    assert first['raw']['dynamic'][0]>first['raw']['dynamic'][1] and scaled['raw']['dynamic'][0]<first['raw']['dynamic'][0]
    # Production front is strictly provisional h>.10 and c>.05, with ±1000 m zone.
    provisional[:,0]=0;provisional[:,3]=0;provisional[:,0,1,7]=.11;provisional[:,3,1,7]=.06
    front=compute_roi_components(current,provisional,encoded_current,encoded_provisional,active,route,section,[1]*6,layout,cfg)
    assert front['front_available'] and front['predicted_front_chainage_m']==7000. and front['raw']['predicted_front'][3]>0
    provisional[:,0,1,7]=.10;front_none=compute_roi_components(current,provisional,encoded_current,encoded_provisional,active,route,section,[1]*6,layout,cfg);assert not front_none['front_available'] and np.all(front_none['raw']['predicted_front']==0)
    # New wet and shallow margins score risk; deep stable wet does not.
    current[:,0]=0;provisional[:,0]=0;provisional[:,0,0,0]=.05;provisional[:,0,0,1]=.149;provisional[:,0,7,7]=.3;current[:,0,7,7]=.3
    risk=compute_roi_components(current,provisional,encoded_current,encoded_provisional,active,route,section,[1]*6,layout,cfg);assert risk['raw']['support_risk'][0]>0 and risk['raw']['support_risk'][-1]==0
    # Fixed transect conversion is strict and includes supplied row/column points.
    mask=build_section_mask({'s':{'rows':[0,7],'cols':[1,6]}},(8,8));assert mask[0,1] and mask[7,6]
    raises(lambda:build_section_mask({'bad':{'rows':[8],'cols':[0]}},(8,8)));raises(lambda:build_section_mask({},(8,8)))
    current[:,0]=.2;provisional[:,0]=.2;provisional[:,3]=0;encoded_provisional.zero_();section[0,0]=True
    section_info=compute_roi_components(current,provisional,encoded_current,encoded_provisional,active,route,section,[1]*6,layout,cfg);assert section_info['raw']['engineering_section'][0]==1. and section_info['ranks']['engineering_section'][0]==1.
    section.zero_()
    # Positive ranks: zeros/nonfinite remain zero, ties agree, max is one, monotonicity holds.
    ranks=positive_percentile_rank([0,1,1,2,np.nan,-1]);assert ranks[0]==0 and ranks[1]==ranks[2] and ranks[3]==1 and ranks[1]<ranks[3]
    assert cfg['weights']=={'dynamic':.25,'predicted_front':.25,'support_risk':.25,'engineering_section':.25} and sum(cfg['weights'].values())==1
    # No score means no forced spend even when support overlaps the patch.
    current[:,0]=.2;provisional[:,0]=.2;provisional[:,3]=0;encoded_provisional.zero_();empty,_=select_engineering_roi(current,provisional,encoded_current,encoded_provisional,active,route,section,[1]*6,layout,cfg,.2,'engineering_roi',1);assert not empty
    # Deterministic IDs, eligibility, seed-identical random_all, and random_support-only membership.
    encoded_provisional[:,:,0:2,0:2]=2;one,t1=select_engineering_roi(current,provisional,encoded_current,encoded_provisional,active,route,section,[1]*6,layout,cfg,.2,'engineering_roi',77);two,t2=select_engineering_roi(current,provisional,encoded_current,encoded_provisional,active,route,section,[1]*6,layout,cfg,.2,'engineering_roi',77)
    assert [p.patch_id for p in one]==[p.patch_id for p in two] and all(p in layout.eligible for p in one) and t1['selected_count']==t2['selected_count']
    all_random,_=select_engineering_roi(current,provisional,encoded_current,encoded_provisional,active,route,section,[1]*6,layout,cfg,.2,'random_all',77);assert [p.patch_id for p in all_random]==[p.patch_id for p in select_random(layout,layout.count_for_budget(.2),77)]
    supported,_=select_engineering_roi(current,provisional,encoded_current,encoded_provisional,active,route,section,[1]*6,layout,cfg,.2,'random_support',77);info=compute_roi_components(current,provisional,encoded_current,encoded_provisional,active,route,section,[1]*6,layout,cfg);assert all(info['support_overlap'][list(info['patch_ids']).index(p.patch_id)]>0 for p in supported)
    # Diversity: first pass avoids 8-neighbours where possible, then second pass fills fixed budget; ties use patch id.
    large=PatchLayout(np.ones((16,16),bool),4,4);indices=np.array([0,1,2,5,10]);scores=np.zeros(len(large.eligible));scores[indices]=[5.,4.,3.,2.,1.];picked,first_count,second_count=_diverse(indices,scores,large.eligible,3);assert first_count==3 and second_count==0 and all(max(abs(large.eligible[a].row_id-large.eligible[b].row_id),abs(large.eligible[a].col_id-large.eligible[b].col_id))>=2 for n,a in enumerate(picked) for b in picked[n+1:])
    scores=np.zeros(len(large.eligible));scores[[0,1,2]]=1.;picked,first_count,second_count=_diverse(np.array([0,1,2]),scores,large.eligible,3);assert [large.eligible[i].patch_id for i in picked]==[0,2,1] and first_count==2 and second_count==1
    # Budget fractions keep PatchLayout semantics and no-diversity shares fusion scores.
    assert [large.count_for_budget(v) for v in cfg['budgets']]==[1,2,4]
    source=(ROOT/'scripts/evaluate_engineering_roi_v1.py').read_text();assert source.index('select_engineering_roi(')<source.index('truth_current=')<source.index('metric.add(')
    assert source.index('apply_learned_correction(')<source.index('apply_support_guard(')<source.index('apply_momentum_state_guard(')
    assert 'previous,current=current,corrected' in source and 'optimizer' not in source and '.backward(' not in source and "scenario_rows('VAL')" in source
    print('PASS EngineeringROI-v1 synthetic scoring, selection, guard order, dispatch contract, and VAL-only structure')
if __name__=='__main__':main()
