import csv,json,heapq,math
from pathlib import Path
import numpy as np,pandas as pd,rasterio,matplotlib.pyplot as plt
from rasterio.windows import from_bounds
from clawpack.pyclaw.solution import Solution
R=Path('/root/autodl-tmp/Debris-flow-jilong-final');C=R/'sim/dclaw/jilong_event_reconstruction_final';H=Path('/root/autodl-tmp/Jilong_DClaw_Handoff');P=R/'local_validation/jilong_gate_a/terrain_final/jilong_glo30_valley_profile.csv';G=R/'local_validation/jilong_gate_a/corridor/jilong_canonical_corridor.geojson';S0,S1=2200.,4100.
L=np.array(json.loads(G.read_text())['features'][0]['geometry']['coordinates']);V=np.diff(L,axis=0);D=np.hypot(V[:,0],V[:,1]);CS=np.r_[0,np.cumsum(D)]
def ip(s):
 s=np.asarray(s);k=np.clip(np.searchsorted(CS,s)-1,0,len(D)-1);u=(s-CS[k])/D[k];return L[k]+V[k]*u[...,None],V[k]/D[k,None]
def pj(x,y):
 x=np.asarray(x);y=np.asarray(y);bd=np.full(x.shape,np.inf);ch=np.zeros(x.shape)
 for k in range(len(D)):
  u=np.clip(((x-L[k,0])*V[k,0]+(y-L[k,1])*V[k,1])/D[k]**2,0,1);q=(x-(L[k,0]+u*V[k,0]))**2+(y-(L[k,1]+u*V[k,1]))**2;m=q<bd;bd[m]=q[m];ch[m]=CS[k]+u[m]*D[k]
 return ch,np.sqrt(bd)
def val(ds,x,y):return float(next(ds.sample([(x,y)]))[0])
def mn(ds,x,y,r):
 w=from_bounds(x-r,y-r,x+r,y+r,ds.transform).round_offsets().round_lengths().intersection(rasterio.windows.Window(0,0,ds.width,ds.height));a=ds.read(1,window=w);rr,cc=np.indices(a.shape);xx,yy=rasterio.transform.xy(ds.window_transform(w),rr,cc,offset='center');xx=np.array(xx).reshape(a.shape);yy=np.array(yy).reshape(a.shape);a=np.where((a==ds.nodata)|((xx-x)**2+(yy-y)**2>r*r),np.inf,a);i,j=np.unravel_index(a.argmin(),a.shape);return float(a[i,j]),float(xx[i,j]),float(yy[i,j])
def rise(s,z):
 i=0;b=(-1,0,0)
 for j in range(1,len(z)):
  if z[j]-z[i]>b[0]:b=(z[j]-z[i],i,j)
  if z[j]<z[i]:i=j
 return {'start_chainage_m':float(s[b[1]]),'end_chainage_m':float(s[b[2]]),'magnitude_m':float(b[0]),'local_mean_adverse_slope':float(b[0]/(s[b[2]]-s[b[1]]))}
ds={n:rasterio.open(H/f'terrain/final/jilong_copernicus_{f}_v1.tif') for n,f in [('native','glo30_utm45'),('m32','32m'),('m64','64m')]}
s=np.arange(S0,S1+1,25.);xy,t=ip(s);o={'chainage_m':s,'corridor_x':xy[:,0],'corridor_y':xy[:,1]}
for n,d in ds.items():
 o[n+'_corridor_z']=[];o[n+'_cross_min_z']=[];o[n+'_cross_min_x']=[];o[n+'_cross_min_y']=[]
 for r in (128,256,512):o[n+f'_min_{r}_z']=[];o[n+f'_min_{r}_x']=[];o[n+f'_min_{r}_y']=[]
 for p,tt in zip(xy,t):
  o[n+'_corridor_z'].append(val(d,*p));nr=np.array([-tt[1],tt[0]]);u=np.linspace(-512,512,205);q=p+u[:,None]*nr;z=np.array([val(d,*v) for v in q]);k=np.nanargmin(z);o[n+'_cross_min_z'].append(z[k]);o[n+'_cross_min_x'].append(q[k,0]);o[n+'_cross_min_y'].append(q[k,1])
  for r in (128,256,512):
   z,x,y=mn(d,*p,r);o[n+f'_min_{r}_z'].append(z);o[n+f'_min_{r}_x'].append(x);o[n+f'_min_{r}_y'].append(y)
M=pd.DataFrame(o);rep=C/'reports';fig=C/'figures';M.to_csv(rep/'local_bottleneck_2200_4100_multires.csv',index=False)
g=pd.read_csv(P).query('station_m>=2200 and station_m<=4100');tx=np.gradient(g.x,g.station_m);ty=np.gradient(g.y,g.station_m);nn=np.hypot(tx,ty);tx/=nn;ty/=nn;off=g.robust_low_point_lateral_offset_m.values;rx=g.x.values-off*ty;ry=g.y.values+off*tx
Q=pd.DataFrame({'chainage':g.station_m,'corridor_x':g.x,'corridor_y':g.y,'robust_offset':off,'robust_low_x':rx,'robust_low_y':ry,'GateB_robust_floor_z':g.robust_valley_floor_elevation_m})
for n,name in [('native','native_GLO30_z_at_robust_low'),('m32','32m_z_at_robust_low'),('m64','64m_z_at_robust_low')]:Q[name]=[val(ds[n],x,y) for x,y in zip(rx,ry)]
Q.to_csv(rep/'local_bottleneck_existing_robust_low.csv',index=False)
# deterministic 8-connected local low path
d=ds['m64'];a=d.read(1);rr,cc=np.indices(a.shape);X,Y=rasterio.transform.xy(d.transform,rr,cc,offset='center');X=np.array(X).reshape(a.shape);Y=np.array(Y).reshape(a.shape);ch,di=pj(X,Y);ok=(di<=800)&(ch>=1900)&(ch<=4400)&np.isfinite(a)&(a!=d.nodata)
def an(sv):
 z=np.where(ok&(abs(ch-sv)<=160)&(di<=320),a,np.inf);return np.unravel_index(z.argmin(),z.shape)
st,go=an(2200),an(4100);dist=np.full(a.shape,np.inf);dist[st]=0;pr={};hp=[(0,st)]
while hp:
 c,u=heapq.heappop(hp)
 if c!=dist[u]:continue
 if u==go:break
 for dr in (-1,0,1):
  for dc in (-1,0,1):
   v=(u[0]+dr,u[1]+dc)
   if not(dr or dc) or not(0<=v[0]<a.shape[0] and 0<=v[1]<a.shape[1]) or not ok[v]:continue
   step=64*math.hypot(dr,dc);nc=c+step+40*max(0,a[v]-a[u])+0.15*step*(di[v]/800)**2
   if nc<dist[v]:dist[v]=nc;pr[v]=u;heapq.heappush(hp,(nc,v))
path=[go]
while path[-1]!=st:path.append(pr[path[-1]])
path=path[::-1];r=np.array([p[0] for p in path]);c=np.array([p[1] for p in path]);px,py,pz=X[r,c],Y[r,c],a[r,c];pc,pdist=pj(px,py);K=pd.DataFrame({'path_order':range(len(path)),'x':px,'y':py,'terrain_z':pz,'projected_chainage':pc,'distance_to_canonical_corridor':pdist})
for n in ds:K[n+'_z']=[val(ds[n],x,y) for x,y in zip(px,py)]
K.to_csv(rep/'local_bottleneck_dem_low_path.csv',index=False)
ris={n:rise(s,M[n+'_min_512_z'].values) for n in ds}
cl=np.r_[0,np.cumsum(np.hypot(np.diff(px),np.diff(py)))]
def wr(w):
 best=(-1,0,0)
 for i in range(len(pz)):
  for j in range(i+1,len(pz)):
   if cl[j]-cl[i]>w: break
   if pz[j]-pz[i]>best[0]:best=(pz[j]-pz[i],i,j)
 return {'start_path_length_m':float(cl[best[1]]),'end_path_length_m':float(cl[best[2]]),'magnitude_m':float(best[0])}
pris={str(w):wr(w) for w in (64,128,256,512)}
sol=Solution(30,path=C/'runs/C3_entrainment_E4/_output',file_format='ascii');q=sol.state.q;h,hu,hv=q[:3];xg,yg=sol.state.grid.dimensions[0].centers,sol.state.grid.dimensions[1].centers;xx,yy=np.meshgrid(xg,yg,indexing='ij');fc,fd=pj(xx,yy);wet=h>.001;i,j=np.unravel_index(np.argmax(np.where(wet,fc,-1)),h.shape);spd=math.hypot(hu[i,j]/h[i,j],hv[i,j]/h[i,j]);front={'i':i+1,'j':j+1,'x':float(xx[i,j]),'y':float(yy[i,j]),'terrain_z':val(d,xx[i,j],yy[i,j]),'h':float(h[i,j]),'speed':spd,'projected_chainage':float(fc[i,j]),'distance_to_canonical_corridor':float(fd[i,j])}
lower=pris['512']['magnitude_m']<30 and pdist.max()>150
cls='A_GEOMETRY_TRACK_MISPLACEMENT' if lower and ris['native']['magnitude_m']<100 else ('B_64M_COARSENING_BARRIER' if ris['native']['magnitude_m']<100 and ris['m32']['magnitude_m']<100 and ris['m64']['magnitude_m']>100 else ('C_REAL_SINGLE_SOURCE_GLO30_FEATURE' if all(x['magnitude_m']>100 for x in ris.values()) else 'D_MIXED_OR_UNRESOLVED'))
jp=rep/'ENTRAINMENT_E4_RESULT.json';j=json.loads(jp.read_text());j['available_layer_thickness_m']=4.;jp.write_text(json.dumps(j,indent=2)+'\n')
A={'segment_m':[2200,4100],'adverse_rise':ris,'candidate_path':{'exists':True,'min_distance_m':float(pdist.min()),'max_distance_m':float(pdist.max()),'max_adverse_rise_m':pris},'e4_front_cell':front,'e4_immediately_upstream_of_suspicious_rise':bool(2800<=2933.464<=3050),'classification':cls,'terrain_modified':False,'corridor_modified':False,'dclaw_runs_performed':0,'e4_metadata_corrected_to_4m':True};(rep/'JILONG_LOCAL_BOTTLENECK_AUDIT.json').write_text(json.dumps(A,indent=2,default=float)+'\n');(rep/'JILONG_LOCAL_BOTTLENECK_AUDIT.md').write_text('# Jilong local bottleneck audit\n\nClassification: '+cls+'\n\n'+json.dumps(A,indent=2,default=float)+'\n')
plt.figure(figsize=(11,8));plt.imshow(a,extent=d.bounds,origin='upper',cmap='terrain');plt.plot(xy[:,0],xy[:,1],'y-',label='canonical');plt.plot(rx,ry,'r-',label='Gate-B robust-low');plt.plot(px,py,'lime',label='DEM low path');plt.scatter(front['x'],front['y'],c='magenta',label='E4 front');plt.legend();plt.xlim(xy[:,0].min()-900,xy[:,0].max()+900);plt.ylim(xy[:,1].min()-900,xy[:,1].max()+900);plt.gca().set_aspect('equal');plt.savefig(fig/'JILONG_LOCAL_BOTTLENECK_2200_4100.png',dpi=250);plt.close()
plt.figure(figsize=(11,5));plt.plot(g.station_m,g.robust_valley_floor_elevation_m,label='GateB');plt.plot(s,M.native_corridor_z,label='native canonical');plt.plot(s,M.native_min_512_z,label='native min');plt.plot(s,M.m32_min_512_z,label='32m min');plt.plot(s,M.m64_min_512_z,label='64m min');plt.plot(pc,pz,label='low path');plt.axvline(2614.409);plt.axvline(2933.464);plt.legend();plt.grid();plt.savefig(fig/'JILONG_LOCAL_BOTTLENECK_PROFILE.png',dpi=250);plt.close()
print(json.dumps({'rises':ris,'maxdist':float(pdist.max()),'pathrises':pris,'cls':cls},indent=2,default=float))
