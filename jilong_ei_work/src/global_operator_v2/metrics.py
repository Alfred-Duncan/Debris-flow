"""Metric definitions copied from Park-v2 solver_sparse_v2.py output loop."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

def station_metrics(state,z,transect,cell_m=30.):
    """Exact solver semantics: flux projects onto stored tangent, not a normal."""
    r=np.asarray(transect['rows'],int);c=np.asarray(transect['cols'],int);tx,ty=transect['tangent'];scale=float(transect.get('flux_scale',1.))
    h,hu,hv,solid,ice=state[:5]; conc=np.divide(solid,h,out=np.zeros_like(h),where=h>=1e-3) if solid.max()>1 else solid
    flux=(hu[r,c]*tx+hv[r,c]*ty)*cell_m*scale; wet=h[r,c]>.05
    return {'Q':float(flux.sum()),'Qdebris':float((flux*conc[r,c]).sum()),'hmax':float(h[r,c].max()),'stage':float((z[r,c]+h[r,c])[wet].min()) if wet.any() else float('nan'),'cmax':float(conc[r,c][wet].max()) if wet.any() else 0.,'wet_width_m':float(wet.sum()*cell_m)}
def debris_front(state,chainage):
    h,c=state[0],state[3];valid=(h>.1)&(c>.05)&np.isfinite(chainage);return float(np.max(chainage[valid])/1e3) if valid.any() else float('nan')
def flood_front(state,h_base,chainage):
    valid=(state[0]-h_base>.5)&np.isfinite(chainage);return float(np.max(chainage[valid])/1e3) if valid.any() else float('nan')
def arrival_from_series(df,station):
    q,s=df[f'{station}_Q'],df[f'{station}_stage'];hit=(q>2*q.iloc[0]+50)|(s>s.iloc[0]+.5);return float(df.loc[hit,'time_s'].iloc[0]) if hit.any() else None
def load_transects(path):return json.loads(Path(path).read_text(encoding='utf-8'))
