"""Regression tests for the five V1.1 repair requirements."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.oracle_refinement import PatchLayout
from src.local_corrector.patches import local_features
from src.local_corrector.v1_1 import restore_local_rng,snapshot_local_rng,true_front_patch

def main():
    # Global delta scale controls the coarse transition feature independently
    # of Local target normalization.
    layout=PatchLayout(np.ones((116,112),bool),2,2);p=(layout.patches[0],);g=torch.zeros(1,35,116,112);pro=torch.ones(1,6,116,112);cur=torch.zeros_like(pro)
    a=local_features(g,pro,cur,[1]*6,p,layout);b=local_features(g,pro,cur,[2]*6,p,layout)
    assert not torch.equal(a,b) and torch.equal(a,local_features(g,pro,cur,[1]*6,p,layout))
    # A high-wet upstream patch must not replace a known downstream front zone.
    active=torch.ones(1,1,116,112);truth=torch.zeros(1,6,116,112);truth[:,:,0:58,0:56]=1;truth[:,3,58:116,56:112]=.1;truth[:,0,58:116,56:112]=.2
    route=np.add.outer(np.arange(116,dtype=float)*1000.,np.arange(112,dtype=float)*100.);front=true_front_patch(layout,truth,active,route);assert front.patch_id==3
    rng=np.random.default_rng(20260920);state=snapshot_local_rng(rng);first=rng.integers(1<<30,size=20);resumed=restore_local_rng(state);assert np.array_equal(first,resumed.integers(1<<30,size=20))
    print('PASS V1.1 scale separation, true front sampler, exact local RNG')
if __name__=='__main__':main()
