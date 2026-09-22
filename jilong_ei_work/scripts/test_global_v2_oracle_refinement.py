"""Low-cost behavioural regression tests for oracle-refinement primitives."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.oracle_refinement import PatchLayout, StreamingMetrics, correct_cores, oracle_patch_scores, select_oracle, select_random, concentration_fractions

class IdentityTransform:
    def encode(self,x): return x

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
    value,total,_=oracle_patch_scores(pred,target,pred,target,mask,layout,[1]*6)
    assert total>0 and np.isclose(value.sum(),total)
    assert concentration_fractions(np.array([1.,3.,6.]))['top50_fraction']==.9
    # Teacher-only change/newly-wet semantics: static dry space cannot claim
    # an artificial IoU=1, and prediction magnitude cannot create a mask.
    m=StreamingMetrics(mask,np.zeros((17,19),np.float32),.1);cur=torch.zeros_like(pred);target=torch.zeros_like(pred);target[:,0,0,0]=.06;target[:,1:,0,0]=1.
    noisy=torch.ones_like(pred)*100
    m.add(torch.zeros_like(pred),cur,target,IdentityTransform());r=m.result()
    assert r['__new_truth']==1 and r['__new_intersection']==0 and r['newly_wet_iou']==0.
    assert r['__change_den_h']>0 and r['__change_den_h']<target[:,0].numel()
    m_noisy=StreamingMetrics(mask,np.zeros((17,19),np.float32),.1);m_noisy.add(noisy,cur,target,IdentityTransform())
    assert m_noisy.result()['__change_den_h']==r['__change_den_h']
    # Symmetric score catches false-positive wetness, momentum and dz even
    # when truth is dry; dry composition alone remains intentionally ignored.
    dry=torch.zeros_like(pred);wet=dry.clone();wet[:,0,0,0]=.2
    _,total,_=oracle_patch_scores(wet,dry,wet,dry,mask,layout,[1]*6);assert total>0
    momentum=dry.clone();momentum[:,1,0,0]=1;_,total,_=oracle_patch_scores(momentum,dry,momentum,dry,mask,layout,[1]*6);assert total>0
    dz=dry.clone();dz[:,5,0,0]=1;_,total,_=oracle_patch_scores(dz,dry,dz,dry,mask,layout,[1]*6);assert total>0
    composition=dry.clone();composition[:,3:5,0,0]=1;_,total,_=oracle_patch_scores(composition,dry,composition,dry,mask,layout,[1]*6);assert total==0
    wet_truth=dry.clone();wet_truth[:,0,0,0]=.1;_,total,_=oracle_patch_scores(composition,wet_truth,wet_truth,wet_truth,mask,layout,[1]*6);assert total>0
    # The runner's explicit state handoff is the closed-loop contract; guard
    # it structurally without loading the 8-GB runtime dataset in unit tests.
    runner=(ROOT/'scripts/run_global_v2_oracle_refinement.py').read_text()
    assert 'previous,current=current,corrected' in runner and "scenario_rows('VAL')" in runner and 'VAL_SCOPE_REQUIRED' in runner
    print('PASS oracle refinement primitives')
if __name__=='__main__':main()
