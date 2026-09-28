import csv,json,heapq,hashlib
from pathlib import Path
import numpy as np,rasterio
from rasterio.transform import from_origin
from scipy.spatial import cKDTree
from clawpack.pyclaw import Solution
C=Path(__file__).resolve().parents[1];g=json.load(open(C/'model_geometry_sourcefix.json'));x0,x1,y0,y1=[g[x] for x in ['xlower','xupper','ylower','yupper']]
route=list(csv.DictReader(open(C/'corridor/published_s0_to_port_route.csv')));P=np.array([[float(r['x']),float(r['y'])] for r in route]);S=np.array([float(r['chainage_m']) for r in route]);tree=cKDTree(P)
def load64(p):
 return np.loadtxt(p,skiprows=6),from_origin(x0,y1,64,64),64
def loadtif(p):
 with rasterio.open(p) as d:return d.read(1),d.transform,abs(d.transform.a)
def path(z,tr,res,width):
 rr,cc=np.indices(z.shape);X=tr.c+(cc+.5)*res;Y=tr.f-(rr+.5)*res;D,ix=tree.query(np.c_[X.ravel(),Y.ravel()]);D=D.reshape(z.shape);ix=ix.reshape(z.shape);CH=S[ix]
 ok=(z>-9000)&(CH>=16000)&(CH<=19707.466)&(D<=width);starts=list(zip(*np.where(ok&(CH>=16000)&(CH<=16400))));targets=set(zip(*np.where(ok&(CH>=19400)&(CH<=19707.466))))
 best={};pre={};hp=[]
 for u in starts: best[u]=(float(z[u]),0.,0.);heapq.heappush(hp,(*best[u],u))
 hit=None
 while hp:
  mx,gain,L,u=heapq.heappop(hp)
  if (mx,gain,L)!=best.get(u):continue
  if u in targets:hit=u;break
  for dr in (-1,0,1):
   for dc in (-1,0,1):
    v=(u[0]+dr,u[1]+dc)
    if not(0<=v[0]<z.shape[0] and 0<=v[1]<z.shape[1]) or not ok[v] or CH[v]<CH[u]-128:continue
    q=(max(mx,float(z[v])),gain+max(0.,float(z[v]-z[u])),L+res*(1.414213562 if dr and dc else 1.))
    if q<best.get(v,(np.inf,np.inf,np.inf)):best[v]=q;pre[v]=u;heapq.heappush(hp,(*q,v))
 if hit is None:raise RuntimeError('no final3km path')
 a=[]
 while hit not in starts:a.append(hit);hit=pre[hit]
 a.append(hit);a=a[::-1];run=float(z[a[0]])
 rows=[]
 for n,(r,c) in enumerate(a):
  run=min(run,float(z[r,c]));rows.append({'order':n,'row':int(r),'col':int(c),'x':float(X[r,c]),'y':float(Y[r,c]),'chainage_m':float(CH[r,c]),'distance_to_route_m':float(D[r,c]),'terrain_z_m':float(z[r,c]),'adverse_rise_m':float(z[r,c]-run)})
 return rows,{'resolution_m':res,'corridor_width_m':width,'n_cells':len(rows),'maximum_adverse_rise_m':max(x['adverse_rise_m'] for x in rows),'saddle_elevation_m':max(x['terrain_z_m'] for x in rows),'saddle_chainage_m':max(rows,key=lambda x:x['terrain_z_m'])['chainage_m'],'maximum_lateral_offset_m':max(x['distance_to_route_m'] for x in rows),'cumulative_positive_gain_m':best[a[-1]][1],'physical_path_length_m':best[a[-1]][2]}
datasets={'30m':loadtif(C/'terrain_multires/published_domain_native30m.tif'),'32m':loadtif(C/'terrain_multires/published_domain_32m.tif'),'64m':load64(C/'terrain/published_route_domain_64m.tt3')}
paths={};summary=[]
for tag,(z,t,res) in datasets.items():
 rows,m=path(z,t,res,768);_,wide=path(z,t,res,1024);m['wide_maximum_adverse_rise_m']=wide['maximum_adverse_rise_m'];m['wide_saddle_elevation_m']=wide['saddle_elevation_m'];m['wide_maximum_lateral_offset_m']=wide['maximum_lateral_offset_m'];m['wide_sensitivity_adverse_rise_m']=wide['maximum_adverse_rise_m']-m['maximum_adverse_rise_m'];paths[tag]=rows;summary.append(dict(dataset=tag,**m))
 with open(C/f'results/FINAL3KM_VALLEY_PATH_{tag}.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 (C/f'corridor/final3km_valley_path_{tag}.geojson').write_text(json.dumps({'type':'FeatureCollection','features':[{'type':'Feature','properties':m,'geometry':{'type':'LineString','coordinates':[[x['x'],x['y']] for x in rows]}}]}))
with open(C/'results/FINAL3KM_VALLEY_PATH_SUMMARY.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=summary[0].keys());w.writeheader();w.writerows(summary)
# 64m mask and V10 final bdif
z,t,res=datasets['64m'];mask,mt,_=load64(C/'entrainment/erodible_thickness_e4_published.tt3');sol=Solution(30,path=str(C/'runs/E4_volume_v10m/_output'),file_format='ascii');bd=sol.state.q[6].T
rows=[]
for p in paths['64m']:
 r,c=p['row'],p['col'];inside=bool(mask[r,c]>0);rows.append({**p,'inside_current_erodible_mask':inside,'h_e_m':float(mask[r,c]),'V10_final_bdif_state_m':float(bd[r,c])})
cov=sum(x['inside_current_erodible_mask'] for x in rows)/len(rows);un=[not x['inside_current_erodible_mask'] for x in rows];ranges=[];i=0
while i<len(rows):
 if not un[i]:i+=1;continue
 j=i
 while j+1<len(rows) and un[j+1]:j+=1
 ranges.append({'start_chainage_m':rows[i]['chainage_m'],'end_chainage_m':rows[j]['chainage_m'],'n_cells':j-i+1,'distance_m':(j-i+1)*64});i=j+1
coverage={'mask_coverage_fraction':cov,'mask_coverage_percent':100*cov,'number_uncovered_cells':sum(un),'longest_consecutive_uncovered_cells':max([x['n_cells'] for x in ranges] or [0]),'longest_consecutive_uncovered_distance_m':max([x['distance_m'] for x in ranges] or [0]),'uncovered_physical_distance_m':sum(x['distance_m'] for x in ranges),'uncovered_chainage_ranges':ranges}
with open(C/'results/FINAL3KM_ERODIBLE_MASK_COVERAGE.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
# construct allowed geomfix iff uncovered. Candidate close to final path / route / robust path floor+30
newmask=mask.copy();added=0
if coverage['number_uncovered_cells']:
 X=t.c+(np.indices(z.shape)[1]+.5)*64;Y=t.f-(np.indices(z.shape)[0]+.5)*64;D,ix=tree.query(np.c_[X.ravel(),Y.ravel()]);D=D.reshape(z.shape);CH=S[ix.reshape(z.shape)]
 pp=np.array([[x['x'],x['y']] for x in paths['64m']]);pd=cKDTree(pp).query(np.c_[X.ravel(),Y.ravel()])[0].reshape(z.shape)
 pz=np.array([x['terrain_z_m'] for x in paths['64m']]); _,near=cKDTree(pp).query(np.c_[X.ravel(),Y.ravel()]);near=near.reshape(z.shape)
 # Local robust valley floor: median across nearest path node ±3 nodes.
 floor=np.array([np.median(pz[max(0,int(i)-3):min(len(pz),int(i)+4)]) for i in near.ravel()]).reshape(z.shape)
 allowed=(pd<=192)&(D<=768)&(CH>=16000)&(CH<=19707.466)&(z<=floor+30)&(z>-9000)
 added=int(((newmask<=0)&allowed).sum());newmask[(newmask<=0)&allowed]=4.
 # save tt3
 out=C/'entrainment/erodible_thickness_fullroute_geomfix_4m.tt3'
 with open(out,'w') as f:
  f.write(f'{z.shape[1]} ncols\n{z.shape[0]} nrows\n{x0} xllcorner\n{y0} yllcorner\n64 cellsize\n-9999 nodata_value\n')
  for q in newmask:f.write(' '.join(f'{x:.8g}' for x in q)+'\n')
coverage['geomfix_created']=bool(coverage['number_uncovered_cells']);coverage['geomfix_added_cells']=added;coverage['geomfix_file']='entrainment/erodible_thickness_fullroute_geomfix_4m.tt3' if added else None
# Erosion utilization
bands=[(0,5000),(5000,10000),(10000,14500),(14500,16770),(16770,19707.466)];out=[]
for k in [10,14,18,20,22,24,26,28,30]:
 ss=Solution(k,path=str(C/'runs/E4_volume_v10m/_output'),file_format='ascii');b=ss.state.q[6].T
 for lo,hi in bands:
  ii=(S[np.argmin((np.c_[X.ravel(),Y.ravel()][:,None]-P[None,:,0])**2,axis=1)] if False else np.zeros(1))
  # reuse route query at grid centers
  dd,jj=tree.query(np.c_[X.ravel(),Y.ravel()]);ch=S[jj].reshape(z.shape);m=(ch>=lo)&(ch<hi)&(mask>0)
  vals=b[m];ratio=np.divide(vals,mask[m],out=np.zeros_like(vals),where=mask[m]>0)
  out.append({'time_s':float(ss.t),'chainage_start_m':lo,'chainage_end_m':hi,'erodible_cell_count':int(m.sum()),'touched_by_flow_count':int(((ss.state.q[0].T>0.001)&m).sum()),'bdif_gt_001_count':int((vals>.01).sum()),'bdif_gt_05_count':int((vals>.5).sum()),'bdif_gt_1_count':int((vals>1).sum()),'bdif_gt_2_count':int((vals>2).sum()),'bdif_gt_3_count':int((vals>3).sum()),'bdif_gt_35_count':int((vals>3.5).sum()),'ratio_ge_025_fraction':float((ratio>=.25).mean()) if len(ratio) else None,'ratio_ge_050_fraction':float((ratio>=.5).mean()) if len(ratio) else None,'ratio_ge_075_fraction':float((ratio>=.75).mean()) if len(ratio) else None,'ratio_ge_090_fraction':float((ratio>=.9).mean()) if len(ratio) else None,'median_bdif_m':float(np.median(vals)) if len(vals) else None,'p90_bdif_m':float(np.quantile(vals,.9)) if len(vals) else None,'p99_bdif_m':float(np.quantile(vals,.99)) if len(vals) else None,'max_bdif_m':float(vals.max()) if len(vals) else None,'bed_difference_state_integral_m3':float(vals.sum()*4096)})
with open(C/'results/V10_EROSION_UTILIZATION_BY_CHAINAGE.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=out[0].keys());w.writeheader();w.writerows(out)
finalu=[x for x in out if x['time_s']==900.0];sat=max(x['ratio_ge_090_fraction'] or 0 for x in finalu);touched=sum(x['touched_by_flow_count'] for x in finalu);avail=sum(x['erodible_cell_count'] for x in finalu)
if coverage['number_uncovered_cells'] and coverage['longest_consecutive_uncovered_distance_m']>=128:cl='FINAL_CORRIDOR_MASK_GAP'
elif sat>=.25:cl='ERODIBLE_DEPTH_SATURATION'
elif coverage['mask_coverage_fraction']>.85 and sat<.1:cl='EROSION_RATE_LIMITED'
else:cl='MOBILITY_RHEOLOGY_LIMITED'
audit={'final3km_mask_coverage':coverage,'V10_final_max_ratio_ge090_fraction':sat,'classification':cl,'state_semantics':'bdif is reported as bed-difference state integral; no cumulative entrained-volume claim is made.'}
(C/'reports/V10_EROSION_UTILIZATION_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
prov={'baseline_commit':__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=C,text=True).strip(),'sha256':{p:hashlib.sha256((C/p).read_bytes()).hexdigest() for p in ['terrain/published_route_domain_64m.tt3','corridor/published_s0_to_port_route.csv','model_geometry_sourcefix.json','source/source_support_published_s0_floorfix.json','entrainment/erodible_thickness_e4_published.tt3','src2.f90','setrun.py','xdclaw']},'frozen':{'T_s':90,'W_ref_m':192,'K_u':1}}
(C/'reports/FULL_RUNOUT_RECONSTRUCTION_PROVENANCE.json').write_text(json.dumps(prov,indent=2)+'\n')
print(json.dumps({'final3km':summary,'coverage':coverage,'initial_mechanism':cl},indent=2))