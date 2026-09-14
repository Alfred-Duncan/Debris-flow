"""Write the fixed 2-m D-Claw entrainable-layer input on the 64-m V2 grid."""
from pathlib import Path
import numpy as np
from clawpack.geoclaw import topotools
XL, YL, DX, MX, MY = 334_021.9, 3_128_083.2, 64., 141, 261
x = np.linspace(XL, XL+MX*DX, MX+1); y = np.linspace(YL, YL+MY*DX, MY+1)
topo = topotools.Topography(); topo.x, topo.y = x, y; topo.Z = np.full((MY+1,MX+1), 2.0)
topo.write(Path(__file__).with_name("entrainable_thickness.tt3"), topo_type=3)
