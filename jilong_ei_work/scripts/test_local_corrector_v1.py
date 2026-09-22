"""Fast Local Corrector V1 invariants; no scenario data or VAL access."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.oracle_refinement import PatchLayout, select_random
from src.local_corrector.model import JilongLocalCorrector
from src.local_corrector.normalization import bounded, fit
from src.local_corrector.patches import HALO, apply_core, core, core_active, extract, local_features


def main():
    torch.manual_seed(1)
    model=JilongLocalCorrector(47); x=torch.randn(2,47,74,72)
    assert torch.equal(model(x),torch.zeros(2,6,74,72)); assert model.parameter_count>0
    active=np.ones((921,882),bool);active[-20:,-20:]=False;layout=PatchLayout(active)
    assert (layout.pad_height,layout.pad_width,layout.core_height,layout.core_width)==(928,896,58,56)
    assert len(layout.eligible)>0 and HALO==8
    edge=layout.patches[-1]; field=torch.randn(1,47,921,882); footprint=extract(field,edge,layout)
    assert footprint.shape==(1,47,74,72) and core(footprint,layout).shape[-2:]==(58,56)
    logical_active=core_active(torch.from_numpy(active)[None,None],(edge,),layout)
    assert not logical_active[:,:,51:,:].any() and not logical_active[:,:,:,42:].any()
    features=local_features(field[:,:35],field[:,35:41],field[:,35:41],np.ones(6),(edge,),layout)
    assert features.shape==(1,47,74,72)
    provisional=torch.zeros(1,6,921,882);correction=torch.ones(1,6,58,56)
    written=apply_core(provisional,correction,(edge,),layout,torch.from_numpy(active)[None,None])
    assert written[:,:,edge.r0:edge.r1,edge.c0:edge.c1].sum()>0 and written[:,:,:edge.r0,:].sum()==0
    assert written[:,:,-20:,-20:].sum()==0
    d=fit([torch.randn(10000).numpy() for _ in range(6)],['TRAIN_001'],[0])
    assert d['fit_scope']=='TRAIN_ONLY' and min(v['coverage'] for v in d['per_channel'].values())>=.999
    assert torch.isfinite(bounded(torch.randn(1,6,2,2),d['scales'],d['bounds'])).all()
    assert select_random(layout,0,1)==() and len(select_random(layout,8,1))==8
    print('PASS local-corrector geometry, zero head, core-only writeback, normalization')
if __name__=='__main__':main()
