from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
from src.models.wrappers import _rvpi_import

class JilongGlobalFNO(nn.Module):
    """14-channel conditioning FNO with residual 10-s state stepping."""
    def __init__(self,width=24,modes=16,depth=3):
        super().__init__(); FNO=_rvpi_import(Path(__file__).resolve().parents[2]/'external/RVPI-PDE'); self.fno=FNO(14,5,width,modes,depth)
    def forward(self,x): return x[:,:5]+self.fno(x)
    @property
    def parameter_count(self): return sum(p.numel() for p in self.parameters())
