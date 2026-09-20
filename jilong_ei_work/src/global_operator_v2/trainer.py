"""Closed-loop trainer primitives used by the formal launcher and bounded smoke tests."""
from __future__ import annotations
import random
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from .dataset import build_features
from .losses import project_physical, v2_loss

CURRICULUM=(("A",2000,1),("B",2000,2),("C",4000,4),("D",6000,6))

def choose_amp(device: torch.device):
    # RVPI's spectral layer uses complex einsum after rfft2.  Its current CUDA
    # implementation rejects both ComplexBFloat16 and ComplexHalf, so enabling
    # autocast would crash or silently change the numerical operator.  Keep the
    # feature gate explicit; a future real-valued spectral kernel may return
    # BF16 first and FP16+GradScaler second here.
    return False,None

def save_checkpoint(path, model, optimizer, scheduler, step, transform, config):
    torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict() if scheduler else None,
                "step":step,"transform":transform.to_dict(),"config":config,"torch_rng":torch.get_rng_state(),"numpy_rng":np.random.get_state(),"python_rng":random.getstate(),
                "cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},path)

def restore_checkpoint(path, model, optimizer=None, scheduler=None):
    ck=torch.load(path,map_location="cpu",weights_only=False); model.load_state_dict(ck["model"])
    if optimizer and ck.get("optimizer"):optimizer.load_state_dict(ck["optimizer"])
    if scheduler and ck.get("scheduler"):scheduler.load_state_dict(ck["scheduler"])
    torch.set_rng_state(ck["torch_rng"]);np.random.set_state(ck["numpy_rng"]);random.setstate(ck["python_rng"])
    if torch.cuda.is_available() and ck.get("cuda_rng"):torch.cuda.set_rng_state_all(ck["cuda_rng"])
    return ck

def rollout_loss(model, transform, previous, current, targets, static, params, times, active, steps, amp_dtype=None):
    losses=[]; pred=current
    for k in range(steps):
        features=build_features(previous,pred,static,params,times+(.0069444*k))
        encoded=transform.encode(pred)
        with torch.autocast(device_type=pred.device.type,enabled=amp_dtype is not None,dtype=amp_dtype): out=model(features,encoded)
        nextp=project_physical(transform.decode(out)); total,detail=v2_loss(out,transform.encode(targets[k]),nextp,targets[k],active)
        losses.append(total); previous,pred=pred,nextp
    return torch.stack(losses).mean(),detail,pred
