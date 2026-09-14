from pathlib import Path
import os,numpy as np
from clawpack.geoclaw import topotools
XL,YL,DX,MX,MY=334021.9,3128083.2,32.,282,522
x=np.linspace(XL,XL+MX*DX,MX+1);y=np.linspace(YL,YL+MY*DX,MY+1);t=topotools.Topography();t.x,t.y,t.Z=x,y,np.full((MY+1,MX+1),float(os.environ.get('ENTRAINABLE_THICKNESS_M','2.0')));t.write(Path(__file__).with_name('entrainable_thickness.tt3'),topo_type=3)
