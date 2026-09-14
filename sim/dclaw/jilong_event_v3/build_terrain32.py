"""Observed-DEM 32-m terrain: nodata-masked 8-m primary plus 12.5-m fallback."""
from pathlib import Path
import numpy as np,rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject,Resampling
from clawpack.geoclaw import topotools
from scipy.interpolate import RegularGridInterpolator
CASE=Path(__file__).resolve().parent;PROJECT=CASE.parents[2];XL,YL,DX,MX,MY=334021.9,3128083.2,32.,282,522;XU,YU=XL+MX*DX,YL+MY*DX;x=np.linspace(XL,XU,MX+1);y=np.linspace(YL,YU,MY+1);tr=Affine.translation(XL-DX/2,YU+DX/2)*Affine.scale(DX,-DX);shape=(MY+1,MX+1)
def warp(path):
 z=np.full(shape,np.nan)
 with rasterio.open(path) as s:reproject(rasterio.band(s,1),z,src_transform=s.transform,src_crs=s.crs,src_nodata=s.nodata,dst_transform=tr,dst_crs='EPSG:32645',dst_nodata=np.nan,resampling=Resampling.bilinear)
 return z
data=PROJECT/'data/extracted';z8=warp(next(data.rglob('DongLinZangBu_basin_HMA8mDEM_fillAW3D30.tif')));z12=warp(next(data.rglob('DEM 12.5m.tif')));z=np.where(np.isfinite(z8),z8,z12)
if not np.isfinite(z).all():
 old=topotools.Topography();old.read(PROJECT/'sim/dclaw/jilong_event_v2/jilong_dem_primary_64m.tt3',topo_type=3);yy,xx=np.meshgrid(y,x,indexing='ij');fallback=RegularGridInterpolator((old.y,old.x),old.Z,bounds_error=True)(np.column_stack((yy.ravel(),xx.ravel()))).reshape(shape);z=np.where(np.isfinite(z),z,fallback)
t=topotools.Topography();t.x,t.y,t.Z=x,y,z;t.write(CASE/'jilong_dem_observed_32m.tt3',topo_type=3)
