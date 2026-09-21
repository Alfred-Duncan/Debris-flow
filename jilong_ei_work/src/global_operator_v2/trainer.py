"""Closed-loop trainer primitives used by the formal launcher and bounded smoke tests."""
from __future__ import annotations
import random
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from .dataset import build_features
from .losses import project_physical, step_loss, amplitude_guard, rollout_objective

CURRICULUM=(("A",2000,1),("B",2000,2),("C",4000,4),("D",6000,6))

def choose_amp(device: torch.device):
    # RVPI's spectral layer uses complex einsum after rfft2.  Its current CUDA
    # implementation rejects both ComplexBFloat16 and ComplexHalf, so enabling
    # autocast would crash or silently change the numerical operator.  Keep the
    # feature gate explicit; a future real-valued spectral kernel may return
    # BF16 first and FP16+GradScaler second here.
    return False,None

def save_checkpoint(path, model, optimizer, scheduler, step, transform, config, normalizer=None, stage=None, best_metric=None, stage_step=None, stage_updates_total=None, architecture=None):
    torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict() if scheduler else None,
                "checkpoint_version":"2.2","step":step,"stage_name":stage,"stage_step":stage_step,"stage_updates_total":stage_updates_total,"global_step":step,"best_metric":best_metric,"best_checkpoint_path":None,"transform":transform.to_dict(),"feature_normalizer":normalizer.to_dict() if normalizer else None,"config":config,"config_hash":__import__('hashlib').sha256(__import__('json').dumps(config,sort_keys=True).encode()).hexdigest(),"architecture":architecture,"torch_rng":torch.get_rng_state(),"numpy_rng":np.random.get_state(),"python_rng":random.getstate(),
                "cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},path)

def restore_checkpoint(path, model, optimizer=None, scheduler=None):
    ck=torch.load(path,map_location="cpu",weights_only=False); model.load_state_dict(ck["model"])
    if optimizer and ck.get("optimizer"):optimizer.load_state_dict(ck["optimizer"])
    if scheduler and ck.get("scheduler"):scheduler.load_state_dict(ck["scheduler"])
    torch.set_rng_state(ck["torch_rng"]);np.random.set_state(ck["numpy_rng"]);random.setstate(ck["python_rng"])
    if torch.cuda.is_available() and ck.get("cuda_rng"):torch.cuda.set_rng_state_all(ck["cuda_rng"])
    return ck

def rollout_loss(model, transform, normalizer, previous, current, targets, static, params, times, active, cell_area, amp_dtype=None, gradient_checkpointing=True):
    """Closed loop: targets are future distinct GT frames, inputs after k1 are predictions."""
    losses=[];amps=[];pred=current;steps=len(targets)
    for k in range(steps):
        features=build_features(previous,pred,static,params,times+(.0069444*k),transform,normalizer)
        encoded=transform.encode(pred)
        fn=lambda f,e:model(f,e)
        with torch.autocast(device_type=pred.device.type,enabled=amp_dtype is not None,dtype=amp_dtype): out=checkpoint(fn,features,encoded,use_reentrant=False) if gradient_checkpointing and len(targets)>=2 else fn(features,encoded)
        nextp=project_physical(transform.decode(out));teacher_current=current if k==0 else targets[k-1];term,detail=step_loss(out,transform.encode(targets[k]),teacher_current,nextp,targets[k],active,cell_area); losses.append(term);amps.append(amplitude_guard(out,transform.encode(targets[k]),active));previous,pred=pred,nextp
    total,parts=rollout_objective(losses,amps);return total,{**parts,**detail},pred
