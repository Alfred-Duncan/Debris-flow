"""Low-cost behavioural regression tests for oracle-refinement primitives."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.oracle_refinement import PatchLayout, correct_cores, oracle_patch_scores, select_oracle, select_random, concentration_fractions

def main():
    active=np.zeros((17,19),bool);active[:15,:18]=True;layout=PatchLayout(active,16,16)
    assert layout.pad_height==32 and layout.pad_width==32 and len(layout.eligible)>0
    # B=0 correction is exactly identity; B=100% matches target on all active cells.
    pred=torch.zeros((1,6,17,19));truth=torch.ones_like(pred);mask=torch.from_numpy(active)[None,None]
    assert torch.equal(correct_cores(pred,truth,(),layout,mask),pred)
    all_selected=tuple(layout.eligible);full=correct_cores(pred,truth,all_selected,layout,mask)
    assert torch.equal(full*mask,truth*mask)
    # Same budget gives same number of patch calls and deterministic oracle ties use patch id.
    k=layout.count_for_budget(.20);scores=np.arange(len(layout.eligible),dtype=float)
    assert len(select_oracle(layout,scores,k))==len(select_random(layout,k,20260920))==k
    assert select_oracle(layout,np.ones(len(layout.eligible)),k)==tuple(layout.eligible[:k])
    # Inactive padding/cells never score; teacher-only relevant change is required.
    current=torch.zeros_like(pred);target=torch.zeros_like(pred);target[:,:,0,0]=1
    value,total=oracle_patch_scores(pred,target,current,target,mask,layout,[1]*6)
    assert total>0 and np.isclose(value.sum(),total)
    assert concentration_fractions(np.array([1.,3.,6.]))['top50_fraction']==.9
    print('PASS oracle refinement primitives')
if __name__=='__main__':main()
