from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
from src.models.wrappers import _rvpi_import

class JilongGlobalOperatorV2(nn.Module):
    """FNO predicting a physical-state increment in transformed coordinates."""
    def __init__(self, in_channels: int, width: int=32, modes: int=16, depth: int=4, delta_bounds=None, momentum_state_bounds=None):
        super().__init__(); self.in_channels=in_channels
        FNO=_rvpi_import(Path(__file__).resolve().parents[2]/"external/RVPI-PDE")
        self.fno=FNO(in_channels, 6, width, modes, depth)
        head=self.fno.project[-1]
        if not isinstance(head,nn.Conv2d):raise TypeError('RVPI FNO output head must be Conv2d')
        nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        self.bounded_residual=delta_bounds is not None
        bounds=torch.ones(6,dtype=torch.float32) if delta_bounds is None else torch.as_tensor(delta_bounds,dtype=torch.float32)
        if bounds.shape != (6,) or not torch.isfinite(bounds).all() or not (bounds>0).all():raise ValueError('delta_bounds must be six positive finite values')
        self.register_buffer("delta_bounds",bounds)
        self.momentum_state_guard=momentum_state_bounds is not None
        momentum_bounds=torch.full((2,),float('inf'),dtype=torch.float32) if momentum_state_bounds is None else torch.as_tensor(momentum_state_bounds,dtype=torch.float32)
        if momentum_bounds.shape != (2,) or (self.momentum_state_guard and (not torch.isfinite(momentum_bounds).all() or not (momentum_bounds>0).all())):raise ValueError('momentum_state_bounds must be two positive finite values')
        self.register_buffer("momentum_state_bounds",momentum_bounds)
        self.momentum_envelope_version=None
        self.last_momentum_guard_activation_fraction=0.
    def forward(self, x: torch.Tensor, encoded_current: torch.Tensor, return_diagnostics: bool=False):
        raw_delta=self.fno(x)
        if not self.bounded_residual:return (encoded_current+raw_delta,raw_delta) if return_diagnostics else encoded_current+raw_delta
        bounds=self.delta_bounds[None,:,None,None].to(dtype=raw_delta.dtype)
        # The division gives a near-linear central region while tanh guarantees
        # a finite TRAIN-derived transformed-coordinate update bound.
        bounded_delta=bounds*torch.tanh(raw_delta/bounds)
        output=encoded_current+bounded_delta
        if self.momentum_state_guard:
            state_bounds=self.momentum_state_bounds[None,:,None,None].to(dtype=output.dtype)
            unclamped=output[:,1:3];activation=(unclamped.abs()>state_bounds)
            self.last_momentum_guard_activation_fraction=float(activation.float().mean().detach().cpu())
            output=torch.cat((output[:,:1],torch.maximum(torch.minimum(unclamped,state_bounds),-state_bounds),output[:,3:]),dim=1)
        else:self.last_momentum_guard_activation_fraction=0.
        return (output,raw_delta/bounds) if return_diagnostics else output
    @property
    def parameter_count(self): return sum(p.numel() for p in self.parameters())
