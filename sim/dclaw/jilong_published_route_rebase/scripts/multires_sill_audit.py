import numpy as np,csv,json,heapq,hashlib
from pathlib import Path
import rasterio
from rasterio.warp import reproject,Resampling
from rasterio.transform import from_origin
from scipy.spatial import cKDTree
C=Path(__file__).parents[1];RAW=Path('/root/autodl-tmp/Jilong_DClaw_Handoff/terrain/source/Copernicus_DSM_COG_10_N28_00_E085_00_DEM.tif');TM=C/'terrain_multires';TM.mkdir(exist_ok=True);g=json.load(open(C/'model_geometry_sourcefix.json'));x0,x1,y0,y1=g['xlower'],g['xupper'],g['ylower'],g['yupper'];
route=list(csv.DictReader(open(C/'corridor/published_s0_to_port_route.csv')));P=np.array([[float(x['x']),float(x['y'])]for x in route]);S=np.array([float(x['chainage_m'])for x in route]);tree=cKDTree(P)
def make(res,path,src=None,method=Resampling.bilinear):
 nx,ny=round((x1-x0)/res),round((y1-y0)/res);a=np.full((ny,nx),-9999.,np.float32);t=from_origin(x0,y1,res,res)
 if src is None:
  with rasterio.open(RAW)as d:reproject(rasterio.band(d,1),a,src_transform=d.transform,src_crs=d.crs,dst_transform=t,dst_crs='EPSG:32645',dst_nodata=-9999.,resampling=method)
 else:reproject(src[0],a,src_transform=src[1],src_crs='EPSG:32645',dst_transform=t,dst_crs='EPSG:32645',dst_nodata=-9999.,resampling=method)
 with rasterio.open(path,'w',driver='GTiff',height=ny,width=nx,count=1,dtype='float32',crs='EPSG:32645',transform=t,nodata=-9999.)as d:d.write(a,1)
 return a,t
n30,t30=make(30,TM/'published_domain_native30m.tif');n32,t32=make(32,TM/'published_domain_32m.tif',(n30,t30),Resampling.average)
z64=np.loadtxt(C/'terrain/published_route_domain_64m.tt3',skiprows=6);t64=from_origin(x0,y1,64,64)
def arrpath(z,t,res,label,width=768):
 rr,cc=np.indices(z.shape);X=t.c+(cc+.5)*res;Y=t.f-(rr+.5)*res;D,ix=tree.query(np.c_[X.ravel(),Y.ravel()]);D=D.reshape(z.shape);ix=ix.reshape(z.shape);CH=S[ix];ok=(z>-9000)&(CH>=13500)&(CH<=16200)&(D<=width);st=set(zip(*np.where(ok&(CH>=13700)&(CH<=14000))));ta=set(zip(*np.where(ok&(CH>=15500)&(CH<=16000))));best={};pre={};hp=[]
 for u in st:best[u]=(float(z[u]),0.,0.);heapq.heappush(hp,(*best[u],u))
 hit=None
 while hp:
  mx,ga,L,u=heapq.heappop(hp)
  if (mx,ga,L)!=best.get(u):continue
  if u in ta:hit=u;break
  for dr in(-1,0,1):
   for dc in(-1,0,1):
    a,b=u[0]+dr,u[1]+dc
    if not(0<=a<z.shape[0]and 0<=b<z.shape[1])or not ok[a,b]or CH[a,b]<CH[u]-128:continue
    q=(max(mx,float(z[a,b])),ga+max(0,float(z[a,b]-z[u])),L+res*(2**.5 if dr and dc else 1));v=(a,b)
    if q<best.get(v,(np.inf,np.inf,np.inf)):best[v]=q;pre[v]=u;heapq.heappush(hp,(*q,v))
 if hit is None:raise RuntimeError(label)
 q=[]
 while hit not in st:q.append(hit);hit=pre[hit]
 q.append(hit);q=q[::-1];run=z[q[0]];rows=[]
 for k,(r,c)in enumerate(q):run=min(run,z[r,c]);rows.append(dict(order=k,x=float(X[r,c]),y=float(Y[r,c]),chainage_m=float(CH[r,c]),distance_to_route_m=float(D[r,c]),terrain_z=float(z[r,c]),adverse_rise_m=float(z[r,c]-run)))
 with open(C/f'results/STOPPING_ZONE_DEM_PATH_{label}.csv','w',newline='')as f:w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
 (C/f'corridor/stopping_zone_valley_path_{label}.geojson').write_text(json.dumps(dict(type='FeatureCollection',features=[dict(type='Feature',properties={},geometry=dict(type='LineString',coordinates=[[x['x'],x['y']]for x in rows]))])))
 saddle=max(rows,key=lambda x:x['terrain_z']);return rows,dict(resolution_m=res,n_cells=len(rows),adverse_rise_m=max(x['adverse_rise_m']for x in rows),saddle_elevation_m=saddle['terrain_z'],saddle_chainage_m=saddle['chainage_m'],max_route_offset_m=max(x['distance_to_route_m']for x in rows))
a30,m30=arrpath(n30,t30,30,'30m');a32,m32=arrpath(n32,t32,32,'32m');a64,m64=arrpath(z64,t64,64,'64m_reaudit');_,w30=arrpath(n30,t30,30,'30m_wide',1024);_,w32=arrpath(n32,t32,32,'32m_wide',1024)
def near(a,b):
 A=np.array([[x['x'],x['y']]for x in a]);B=np.array([[x['x'],x['y']]for x in b]);d=cKDTree(B).query(A)[0];return dict(median=float(np.median(d)),p90=float(np.quantile(d,.9)),maximum=float(d.max()),fraction_over500=float((d>500).mean()))
sep=near(a30,a32);consistent=sep['fraction_over500']<=.25
# route profile 10 m samples
q=np.arange(13500,16200.1,10);coords=np.c_[np.interp(q,S,P[:,0]),np.interp(q,S,P[:,1])]
def samp(z,t):
 c=((coords[:,0]-t.c)/abs(t.a)).astype(int);r=((t.f-coords[:,1])/abs(t.e)).astype(int);return z[r,c]
prof=[];zz=[samp(n30,t30),samp(n32,t32),samp(z64,t64)]
def rise(a):return float(max(a[j]-a[max(0,j-100):j+1].min()for j in range(len(a))))
for i,(x,y)in enumerate(coords):prof.append(dict(chainage_m=q[i],x=x,y=y,z30=zz[0][i],z32=zz[1][i],z64=zz[2][i]))
with open(C/'results/MULTIRES_PUBLISHED_ROUTE_PROFILE.csv','w',newline='')as f:w=csv.DictWriter(f,prof[0].keys());w.writeheader();w.writerows(prof)
r30,r32,r64=map(rise,zz)
if not consistent:cl='MULTIRES_PATH_TOPOLOGY_INCONSISTENT';tr=False
elif r30>=20 and r32>=20 and abs(r32-r30)<=10:cl='PERSISTENT_GLO30_TOPOGRAPHIC_SILL';tr=False
elif r30<=20 and r32<=20 and m64['adverse_rise_m']-max(m30['adverse_rise_m'],m32['adverse_rise_m'])>=8 and m64['saddle_elevation_m']-max(m30['saddle_elevation_m'],m32['saddle_elevation_m'])>=5:cl='64M_COARSENING_ARTIFACT_LIKELY';tr=True
elif m64['adverse_rise_m']-m32['adverse_rise_m']>=10 and m32['adverse_rise_m']<m64['adverse_rise_m'] and m30['adverse_rise_m']<m64['adverse_rise_m']:cl='RESOLUTION_SENSITIVE_MIXED_SILL';tr=True
elif r30<10 and r32<10 and r64<10:cl='NO_MEANINGFUL_LOCAL_SILL';tr=False
else:cl='UNRESOLVED';tr=False
for p in [C/'results/MULTIRES_VALLEY_PATH_SUMMARY.csv']:
 with open(p,'w',newline='')as f:w=csv.DictWriter(f,list(m30.keys()));w.writeheader();w.writerows([m30,m32,m64])
prov=dict(native30_independent_of_64m=True,terrain32_independent_of_64m=True,accepted64_baseline=True,raw_source=str(RAW),raw_sha256=hashlib.sha256(RAW.read_bytes()).hexdigest(),native30_resolution=30,terrain32_resolution=32,domain_bounds=[x0,x1,y0,y1]);json.dump(prov,open(C/'reports/MULTIRES_TERRAIN_PROVENANCE.json','w'),indent=2);open(C/'reports/MULTIRES_TERRAIN_PROVENANCE.md','w').write('# provenance\n\n'+json.dumps(prov,indent=2))
aud=dict(R30_m=m30['adverse_rise_m'],R32_m=m32['adverse_rise_m'],R64_m=m64['adverse_rise_m'],S30_m=m30['saddle_elevation_m'],S32_m=m32['saddle_elevation_m'],S64_m=m64['saddle_elevation_m'],published_route_rise_30_m=r30,published_route_rise_32_m=r32,published_route_rise_64_m=r64,path30_max_offset_m=m30['max_route_offset_m'],path32_max_offset_m=m32['max_route_offset_m'],path64_max_offset_m=m64['max_route_offset_m'],path30_32_median_separation_m=sep['median'],path30_32_p90_separation_m=sep['p90'],path30_32_max_separation_m=sep['maximum'],R30_wide_m=w30['adverse_rise_m'],R32_wide_m=w32['adverse_rise_m'],CORRIDOR_WIDTH_SENSITIVE=bool(abs(w30['adverse_rise_m']-m30['adverse_rise_m'])>10 or abs(w32['adverse_rise_m']-m32['adverse_rise_m'])>10),classification=cl,classification_reason='predeclared lexicographic path criteria',TRIGGER_32M_SOLVER=tr);json.dump(aud,open(C/'reports/MULTIRES_SILL_AUDIT.json','w'),indent=2);open(C/'reports/MULTIRES_SILL_AUDIT.md','w').write('# multires sill audit\n\n'+json.dumps(aud,indent=2));print(json.dumps(aud,indent=2))
