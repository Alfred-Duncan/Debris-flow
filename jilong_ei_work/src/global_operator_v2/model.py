from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
from src.models.wrappers import _rvpi_import

class JilongGlobalOperatorV2(nn.Module):
    """FNO predicting a physical-state increment in transformed coordinates."""
    def __init__(self, in_channels: int, width: int=32, modes: int=16, depth: int=4, delta_bounds=None):
        super().__init__(); self.in_channels=in_channels
        FNO=_rvpi_import(Path(__file__).resolve().parents[2]/"external/RVPI-PDE")
        self.fno=FNO(in_channels, 6, width, modes, depth)
        self.bounded_residual=delta_bounds is not None
        bounds=torch.ones(6,dtype=torch.float32) if delta_bounds is None else torch.as_tensor(delta_bounds,dtype=torch.float32)
        if bounds.shape != (6,) or not torch.isfinite(bounds).all() or not (bounds>0).all():raise ValueError('delta_bounds must be six positive finite values')
        self.register_buffer("delta_bounds",bounds)
    def forward(self, x: torch.Tensor, encoded_current: torch.Tensor) -> torch.Tensor:
        raw_delta=self.fno(x)
        if not self.bounded_residual:return encoded_current+raw_delta
        bounds=self.delta_bounds[None,:,None,None].to(dtype=raw_delta.dtype)
        # The division gives a near-linear central region while tanh guarantees
        # a finite TRAIN-derived transformed-coordinate update bound.
        bounded_delta=bounds*torch.tanh(raw_delta/bounds)
        return encoded_current + bounded_delta
    @property
    def parameter_count(self): return sum(p.numel() for p in self.parameters())
