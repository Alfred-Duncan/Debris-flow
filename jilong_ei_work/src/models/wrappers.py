"""Project-owned wrappers around RVPI's non-RL FNO building block."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


def _rvpi_import(upstream_root: Path):
    root = str(upstream_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from rvpi.models.fno import FNO2d
    return FNO2d


class GlobalOperatorWrapper(nn.Module):
    """Untrained one-step global FNO interface for the smoke pipeline only."""
    def __init__(self, upstream_root: str | Path, static_channels: int, state_channels: int, param_channels: int = 1, width: int = 16, modes: int = 8, depth: int = 2):
        super().__init__()
        FNO2d = _rvpi_import(Path(upstream_root))
        self.model = FNO2d(static_channels + state_channels + param_channels, state_channels, width, modes, depth)

    def forward(self, static: torch.Tensor, state_t: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        if params.ndim == 2:
            params = params[:, :, None, None].expand(-1, -1, state_t.shape[-2], state_t.shape[-1])
        return self.model(torch.cat((static, state_t, params), dim=1))


class LocalCorrectorWrapper(nn.Module):
    """Optional untrained residual FNO on a manually supplied local patch."""
    def __init__(self, upstream_root: str | Path, state_channels: int, conditioning_channels: int = 1, width: int = 12, modes: int = 8, depth: int = 2):
        super().__init__()
        FNO2d = _rvpi_import(Path(upstream_root))
        self.model = FNO2d(state_channels + conditioning_channels, state_channels, width, modes, depth)

    def forward(self, pred: torch.Tensor, local_region: tuple[slice, slice], conditioning: torch.Tensor) -> torch.Tensor:
        ys, xs = local_region
        patch = pred[:, :, ys, xs]
        correction = self.model(torch.cat((patch, conditioning[:, :, ys, xs]), dim=1))
        out = pred.clone()
        out[:, :, ys, xs] = patch + correction
        return out

