"""CPU-only reproducibility tests for sampler, capacity fallback and scheduler."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.pipeline import PipelineState,apply_capacity_result,build_scheduler,training_sample_for_step

def test_stateless_sampler():
    for k in (1,2,4,6):
        continuous=[training_sample_for_step(20260920,step,17,k) for step in range(1000)]
        resumed=[training_sample_for_step(20260920,step,17,k) for step in range(500,1000)]
        assert continuous[500:]==resumed
        assert all(x['time_s']+10*k<=1440 for x in continuous)

def test_capacity_oom_fallback():
    probe=[{'K':1,'status':'SAFE','safe':True},{'K':2,'status':'SAFE','safe':True},{'K':4,'status':'OOM','safe':False}]
    state=apply_capacity_result(PipelineState('v','h'),probe)
    assert state.total_planned_updates==4000 and state.max_safe_k==2 and 'STAGE_D' in state.skipped_stages

def test_scheduler_resume():
    def make():
        parameter=torch.nn.Parameter(torch.tensor(1.));optimizer=torch.optim.AdamW([parameter],lr=5e-4,weight_decay=1e-4);return optimizer,build_scheduler(optimizer,500,4000)
    full_opt,full=make();full_values=[]
    for step in range(2501):full_opt.step();full.step();full_values.append(full_opt.param_groups[0]['lr'])
    opt,scheduler=make()
    for _ in range(1000):opt.step();scheduler.step()
    saved=(opt.state_dict(),scheduler.state_dict());opt2,scheduler2=make();opt2.load_state_dict(saved[0]);scheduler2.load_state_dict(saved[1])
    values=[]
    for step in range(1000,2501):opt2.step();scheduler2.step();values.append((step,opt2.param_groups[0]['lr']))
    assert all(abs(lr-full_values[step])<1e-15 for step,lr in values if step in (1001,1500,2000,2500))

if __name__=='__main__':test_stateless_sampler();test_capacity_oom_fallback();test_scheduler_resume()
