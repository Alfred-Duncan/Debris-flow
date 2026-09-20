from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
from src.models.wrappers import _rvpi_import

class JilongGlobalOperatorV2(nn.Module):
    """FNO predicting a physical-state increment in transformed coordinates."""
    def __init__(self, in_channels: int, width: int=32, modes: int=16, depth: int=4):
        super().__init__(); self.in_channels=in_channels
        FNO=_rvpi_import(Path(__file__).resolve().parents[2]/"external/RVPI-PDE")
        self.fno=FNO(in_channels, 6, width, modes, depth)
    def forward(self, x: torch.Tensor, encoded_current: torch.Tensor) -> torch.Tensor:
        return encoded_current + self.fno(x)
    @property
    def parameter_count(self): return sum(p.numel() for p in self.parameters())
