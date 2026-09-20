"""Zero-preserving physical transforms; fitting is train-split-only."""
from __future__ import annotations
import torch

class PhysicalTransform:
    names = ("h", "hu", "hv", "c", "ice", "dz")
    def __init__(self, wet_scale: float = 1.0, momentum_scale: float = 1.0, dz_scale: float = 1.0):
        self.wet_scale=max(float(wet_scale), 1e-6); self.momentum_scale=max(float(momentum_scale),1e-6); self.dz_scale=max(float(dz_scale),1e-6)
    @classmethod
    def fit(cls, states: torch.Tensor, wet_threshold: float = .03) -> "PhysicalTransform":
        # quantiles intentionally ignore all-domain dry zeros for h scaling.
        wet=states[:,0][states[:,0] >= wet_threshold]
        q=lambda x: float(torch.quantile(x.abs(), .95).clamp_min(1e-6).cpu())
        return cls(q(wet) if wet.numel() else 1., q(states[:,1:3]), q(states[:,5]))
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        y=x.clone(); y[:,0]=torch.log1p(torch.clamp(x[:,0],min=0)/self.wet_scale)
        y[:,1:3]=torch.asinh(x[:,1:3]/self.momentum_scale); y[:,5]=torch.asinh(x[:,5]/self.dz_scale)
        return y
    def decode(self, y: torch.Tensor) -> torch.Tensor:
        x=y.clone(); x[:,0]=self.wet_scale*torch.expm1(y[:,0]).clamp_min(0)
        x[:,1:3]=self.momentum_scale*torch.sinh(y[:,1:3]); x[:,5]=self.dz_scale*torch.sinh(y[:,5])
        return x
    def to_dict(self): return {"h": {"kind":"log1p","scale":self.wet_scale},"hu_hv":{"kind":"asinh","scale":self.momentum_scale},"c_ice":{"kind":"identity"},"dz":{"kind":"asinh","scale":self.dz_scale}}
