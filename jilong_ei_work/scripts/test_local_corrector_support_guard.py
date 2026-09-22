"""Small deterministic regression tests for the inference-only SupportGuard."""
from __future__ import annotations
import inspect,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.losses import project_physical
from src.global_operator_v2.oracle_refinement import PatchLayout,select_random
from src.local_corrector.patches import apply_core
from src.local_corrector.support_guard import apply_support_guard,support_mask
from src.local_corrector.trainer import apply_momentum_state_guard

def main():
    active=torch.tensor([[[[1,1,1,0]]]],dtype=torch.bool);current=torch.zeros(1,6,1,4);provisional=torch.zeros_like(current);raw=torch.arange(24,dtype=torch.float32).reshape(1,6,1,4)
    raw[:,0,0,:3]=.10
    # dry/dry blocks all six channels; inactive never writes.
    guarded,telemetry=apply_support_guard(provisional,raw,current,provisional,active,local_corrected_physical=raw)
    assert torch.equal(guarded[:,:,:,:3],provisional[:,:,:,:3]) and telemetry['blocked_wet_creation_count']==3
    # Current wet and Global provisional wet each independently allow all channels.
    current[:,0,0,1]=.06;provisional[:,0,0,2]=.06;guarded,_=apply_support_guard(provisional,raw,current,provisional,active,local_corrected_physical=raw)
    assert torch.equal(guarded[:,:,0,1],raw[:,:,0,1]) and torch.equal(guarded[:,:,0,2],raw[:,:,0,2]) and torch.equal(guarded[:,:,0,3],provisional[:,:,0,3])
    # Truth is not an input and cannot alter support; B0 proposal exactly retains frozen provisional.
    assert torch.equal(support_mask(current,provisional,active),support_mask(current,provisional,active));b0,_=apply_support_guard(provisional,provisional,current,provisional,active);assert torch.equal(b0,provisional)
    # Core-only patch writeback leaves every halo/non-core location untouched.
    layout=PatchLayout(np.ones((4,4),bool),2,2);patch=(layout.patches[0],);base=torch.zeros(1,6,4,4);corr=torch.ones(1,6,2,2);out=apply_core(base,corr,patch,layout,torch.ones(1,1,4,4));assert out[:,:,0:2,0:2].eq(1).all() and out[:,:,2:,2:].eq(0).all()
    # Existing absolute momentum guard remains after gating and projection enforces physical constraints.
    model=type('M',(),{'momentum_state_guard':True,'momentum_state_bounds':torch.tensor([.2,.3])})();state=torch.zeros(1,6,1,1);state[:,1]=9;state[:,2]=-9;state[:,0]=-1;state[:,3]=2;state[:,4]=3
    state=apply_momentum_state_guard(state,model);physical=project_physical(state,torch.ones(1,1,1,1));assert physical[:,1].abs().max()<=.2 and physical[:,2].abs().max()<=.3 and physical[:,0].min()>=0 and (physical[:,4]<=physical[:,3]).all()
    # Same deterministic IDs for raw, guarded, and RandomPerfect at a given seed/budget.
    ids=lambda: [p.patch_id for p in select_random(layout,layout.count_for_budget(.5),20260920+1000+17)]
    assert ids()==ids()==ids()
    source=(ROOT/'scripts/evaluate_local_corrector_support_guard.py').read_text();assert 'optimizer' not in source and '.backward(' not in source and "scenario_rows('VAL')" in source
    print('PASS SupportGuard dry/wet, six-channel, core, guard, projection, B0, IDs, inference-only VAL')
if __name__=='__main__':main()
