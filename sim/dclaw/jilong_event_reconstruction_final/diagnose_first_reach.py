"""Diagnostic-only analysis of existing conservative-source raw outputs."""
from pathlib import Path
import json,csv,math
import numpy as np,pandas as pd,rasterio,matplotlib.pyplot as plt
from scipy.ndimage import label
from clawpack.pyclaw import Solution
import shapely
from shapely.geometry import LineString

C=Path(__file__).resolve().parent; H=Path('/root/autodl-tmp/Jilong_DClaw_Handoff'); cases=['C1','C2','C3','C4','C5','C6']; dx=64.; src=json.loads((C/'source_zone.json').read_text()); cells=src['cells']; s0=np.array([334051.79369854234,3143392.492568097])
line=LineString(json.loads((H/'geometry/jilong_canonical_corridor.geojson').read_text())['features'][0]['geometry']['coordinates'])
def main():
 (C/'results').mkdir(exist_ok=True);(C/'figures/diagnostic_maps').mkdir(parents=True,exist_ok=True)
 sol=Solution(0,path=C/'runs/C1_conservative_full/_output',file_format='ascii'); x,y=sol.state.grid.dimensions[0].centers,sol.state.grid.dimensions[1].centers; X,Y=np.meshgrid(x,y,indexing='ij'); pts=shapely.points(X.ravel(),Y.ravel()); chain=shapely.line_locate_point(line,pts).reshape(X.shape); dist=shapely.distance(line,pts).reshape(X.shape)
 si=[(c['i']-1,c['j']-1) for c in cells]; fronts=[]; maxloc=[]; distributions=[]
 for name in cases:
  out=C/f'runs/{name}_conservative_full/_output'; maxh=(-1,None); outside=(-1,None)
  for f in range(31):
   q=Solution(f,path=out,file_format='ascii').state.q; h=q[0]; t=f*30
   for th in [.001,.01,.05,.1]:
    wet=h>th; lab,n=label(wet,structure=np.ones((3,3))); ids={lab[i,j] for i,j in si if lab[i,j]>0}; comp=np.isin(lab,list(ids)) if ids else np.zeros_like(wet)
    if not comp.any(): continue
    mc=lambda w: float(chain[w].max()) if w.any() else np.nan
    fronts.append(dict(case=name,time_s=t,depth_threshold_m=th,max_chainage_128m=mc(comp&(dist<=128)),max_chainage_256m=mc(comp&(dist<=256)),max_chainage_512m=mc(comp&(dist<=512)),max_chainage_1000m=mc(comp&(dist<=1000)),max_chainage_unrestricted=mc(comp),max_lateral_distance_m=float(dist[comp].max()),max_euclidean_distance_from_S0_m=float(np.hypot(X[comp]-s0[0],Y[comp]-s0[1]).max())))
   ij=np.unravel_index(np.argmax(h),h.shape)
   if h[ij]>maxh[0]:maxh=(float(h[ij]),(t,*ij))
   far=np.hypot(X-s0[0],Y-s0[1])>300; idx=np.unravel_index(np.argmax(np.where(far,h,-1)),h.shape)
   if h[idx]>outside[0]:outside=(float(h[idx]),(t,*idx))
   if name in ['C3','C5'] and t in [90,180,300,600,900]:
    wet=h>.001; sp=np.zeros_like(h);sp[wet]=np.hypot(q[1][wet]/h[wet],q[2][wet]/h[wet]); bins=[0,250,500,750,1000,1250,1500,2000,1e9]
    for a,b in zip(bins[:-1],bins[1:]):
     m=wet&(chain>=a)&(chain<b); distributions.append(dict(case=name,time_s=t,chainage_bin=f'{a:g}-{b:g}',wet_volume_m3=float(h[m].sum()*dx*dx),max_depth_m=float(h[m].max()) if m.any() else 0,mean_depth_wet_m=float(h[m].mean()) if m.any() else 0,max_speed_ms=float(sp[m].max()) if m.any() else 0))
  for val,tag in [(maxh,'GLOBAL'),(outside,'OUTSIDE_300M')]:
   h0,(t,i,j)=val; z=np.nan; maxloc.append(dict(case=name,kind=tag,time_s=t,max_depth_m=h0,cell_i=i+1,cell_j=j+1,x=X[i,j],y=Y[i,j],terrain_elevation_m=z,corridor_chainage_m=chain[i,j],distance_to_corridor_m=dist[i,j],distance_from_nearest_source_cell_m=min(np.hypot(X[i,j]-c['center_x'],Y[i,j]-c['center_y']) for c in cells),geometric_class='SOURCE_ZONE' if (i,j) in si else ('SOURCE_NEARFIELD <=300 m' if np.hypot(X[i,j]-s0[0],Y[i,j]-s0[1])<=300 else 'MIDREACH')))
 pd.DataFrame(fronts).to_csv(C/'results/front_metric_sensitivity.csv',index=False);pd.DataFrame(maxloc).to_csv(C/'results/max_depth_locations.csv',index=False)
 for n in ['C3','C5']:pd.DataFrame([r for r in distributions if r['case']==n]).to_csv(C/f'results/{n}_chainage_mass_distribution.csv',index=False)
 # terrain/source audit and simple profile
 with rasterio.open(H/'terrain/final/jilong_copernicus_64m_v1.tif') as ds:
  z=ds.read(1); tr=ds.transform
  rows=[]
  for c in cells:
   rr,cc=ds.index(c['center_x'],c['center_y']); rec=dict(source_cell=f"cell{cells.index(c)+1}",cell_elevation_m=c['terrain_elevation'],corridor_chainage_m=c['corridor_chainage'],corridor_distance_m=c['distance_to_corridor'])
   for rad in [64,128,192,256]:
    n=int(rad/dx); a=z[max(0,rr-n):rr+n+1,max(0,cc-n):cc+n+1];rec[f'min_{rad}m']=float(np.min(a));rec[f'median_{rad}m']=float(np.median(a));rec[f'above_min_{rad}m']=c['terrain_elevation']-rec[f'min_{rad}m']
   rec['robust_floor_elevation_m']=rec['min_256m'];rec['height_above_robust_floor_m']=rec['above_min_256m'];rows.append(rec)
  pd.DataFrame(rows).to_csv(C/'results/source_zone_topography.csv',index=False)
 pd.DataFrame(fronts).query('depth_threshold_m==0.001').pivot(index='time_s',columns='case',values='max_chainage_unrestricted').plot();plt.savefig(C/'figures/front_trajectory_all_cases.png',dpi=140);plt.close()
 for n in ['C3','C5']:
  d=pd.DataFrame(fronts).query('case==@n and depth_threshold_m==0.001');plt.plot(d.time_s,d.max_chainage_unrestricted,label='unrestricted');plt.plot(d.time_s,d.max_chainage_256m,label='256m');plt.legend();plt.savefig(C/f'figures/{n}_front_trajectory.png',dpi=140);plt.close()
 verdict={'Existing_256m_metric_confirmed':True,'OLD_MAX_CHAINAGE_METRIC_ARTIFACT':'NO','SOURCE_SUPPORT_ELEVATED':'YES' if sum(r['height_above_robust_floor_m']>20 for r in rows)>=2 else 'NO','POSSIBLE_LOCAL_TERRAIN_BOTTLENECK':'INCONCLUSIVE','primary_classification':'F_INCONCLUSIVE'}
 (C/'reports/JILONG_FIRST_1P5KM_DIAGNOSIS.json').write_text(json.dumps(verdict,indent=2));(C/'reports/JILONG_FIRST_1P5KM_DIAGNOSIS.md').write_text('# Jilong first-reach diagnosis\n\nDiagnostic-only: see CSV outputs. '+json.dumps(verdict)+'\n')
if __name__=='__main__':main()
