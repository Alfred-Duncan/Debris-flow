from __future__ import annotations
import torch
from torch import nn
class Block(nn.Module):
 def __init__(self,w):
  super().__init__();self.a=nn.Sequential(nn.Conv2d(w,w,3,padding=1),nn.GroupNorm(8,w),nn.SiLU(),nn.Conv2d(w,w,3,padding=1),nn.GroupNorm(8,w))
 def forward(self,x):return torch.nn.functional.silu(x+self.a(x))
class JilongLocalCorrector(nn.Module):
 def __init__(self,input_channels,width=64,depth=6):
  super().__init__();self.stem=nn.Sequential(nn.Conv2d(input_channels,width,3,padding=1),nn.GroupNorm(8,width),nn.SiLU());self.blocks=nn.Sequential(*[Block(width) for _ in range(depth)]);self.head=nn.Conv2d(width,6,1);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
 def forward(self,x):return self.head(self.blocks(self.stem(x)))
 @property
 def parameter_count(self):return sum(p.numel() for p in self.parameters())
