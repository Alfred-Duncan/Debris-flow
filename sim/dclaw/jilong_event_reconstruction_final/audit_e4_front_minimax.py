import json,heapq,math
from pathlib import Path
import numpy as np,pandas as pd,rasterio,matplotlib.pyplot as plt
R=Path('/root/autodl-tmp/Debris-flow-jilong-final');C=R/'sim/dclaw/jilong_event_reconstruction_final';H=Path('/root/autodl-tmp/Jilong_DClaw_Handoff');G=R/'local_validation/jilong_gate_a/corridor/jilong_canonical_corridor.geojson';F=np.array([335758.,3141052.])
L=np.array(json.loads(G.read_text())['features'][0]['geometry']['coordinates']);V=np.diff(L,axis=0);D=np.hypot(V[:,0],V[:,1]);CS=np.r_[0,np.cumsum(D)]
def proj(x,y):
 x=np.asarray(x);y=np.asarray(y);b=np.full(x.shape,np.inf);q=np.zeros(x.shape)
 for k in range(len(D)):
  u=np.clip(((x-L[k,0])*V[k,0]+(y-L[k,1])*V[k,1])/D[k]**2,0,1);d=(x-(L[k,0]+u*V[k,0]))**2+(y-(L[k,1]+u*V[k,1]))**2;m=d<b;b[m]=d[m];q[m]=CS[k]+u[m]*D[k]
 return q,np.sqrt(b)
def run(n,f):
 ds=rasterio.open(H/'terrain/final'/f);a=ds.read(1);rr,cc=np.indices(a.shape);X,Y=rasterio.transform.xy(ds.transform,rr,cc,offset='center');X=np.array(X).reshape(a.shape);Y=np.array(Y).reshape(a.shape);ch,di=proj(X,Y);ok=np.isfinite(a)&(a!=ds.nodata)&(ch>=2600)&(ch<=4300)&(di<=1000)
 z=np.where(ok,(X-F[0])**2+(Y-F[1])**2,np.inf);st=np.unravel_index(z.argmin(),z.shape);belt=ok&(ch>=4050)&(ch<=4150)&(di<=512);zm=a[belt].min();tar=belt&(a<=zm+15)
 cost=np.full(a.shape,np.inf);leng=np.full(a.shape,np.inf);md=np.full(a.shape,np.inf);cost[st]=a[st];leng[st]=0;md[st]=di[st];pre={};hp=[(a[st],0.,di[st],st)]
 while hp:
  co,le,me,u=heapq.heappop(hp)
  if (co,le,me)!=(cost[u],leng[u],md[u]):continue
  for dr in (-1,0,1):
   for dc in (-1,0,1):
    v=(u[0]+dr,u[1]+dc)
    if not(dr or dc) or not(0<=v[0]<a.shape[0] and 0<=v[1]<a.shape[1]) or not ok[v] or ch[v]<ch[u]-100:continue
    nc=max(co,a[v]);nl=le+ds.res[0]*math.hypot(dr,dc);nm=(me*le+di[v]*ds.res[0]*math.hypot(dr,dc))/nl
    if (nc,nl,nm)<(cost[v],leng[v],md[v]):cost[v]=nc;leng[v]=nl;md[v]=nm;pre[v]=u;heapq.heappush(hp,(nc,nl,nm,v))
 cand=np.where(tar,cost,np.inf);go=np.unravel_index(cand.argmin(),cand.shape);path=[go]
 while path[-1]!=st:path.append(pre[path[-1]])
 path=path[::-1];r=np.array([p[0] for p in path]);c=np.array([p[1] for p in path]);x,y,z=X[r,c],Y[r,c],a[r,c];pc,pdist=proj(x,y);rm=np.minimum.accumulate(z);ar=z-rm;k=int(np.argmax(z));ka=int(np.argmax(ar));df=pd.DataFrame({'path_order':range(len(path)),'x':x,'y':y,'terrain_z':z,'projected_chainage_m':pc,'distance_to_canonical_corridor_m':pdist,'running_min_z':rm,'adverse_rise_from_running_min_m':ar,'is_controlling_saddle':[i==k for i in range(len(path))]});df.to_csv(C/'reports'/f'minimax_path_{n}.csv',index=False)
 neigh=[]
 for dr in (-1,0,1):
  for dc in (-1,0,1):
   v=(st[0]+dr,st[1]+dc)
   if 0<=v[0]<a.shape[0] and 0<=v[1]<a.shape[1] and (dr or dc) and ch[v]>=ch[st]-100:neigh.append(a[v])
 cl='OPEN_DOWNSTREAM_GRADIENT' if any(v<=a[st]+1 for v in neigh) else 'LOCAL_DEPRESSION'
 out={'start_i':int(st[1]+1),'start_j':int(st[0]+1),'start_x':float(X[st]),'start_y':float(Y[st]),'start_z':float(a[st]),'start_distance_m':float(math.hypot(X[st]-F[0],Y[st]-F[1])),'target_candidate_count':int(tar.sum()),'target_z':float(a[go]),'highest_path_z':float(z[k]),'minimum_saddle_above_start':float(z[k]-a[st]),'maximum_adverse_rise':float(ar[ka]),'path_length':float(np.sum(np.hypot(np.diff(x),np.diff(y)))),'straight_line_displacement':float(math.hypot(x[-1]-x[0],y[-1]-y[0])),'tortuosity':float(np.sum(np.hypot(np.diff(x),np.diff(y)))/math.hypot(x[-1]-x[0],y[-1]-y[0])),'min_distance':float(pdist.min()),'median_distance':float(np.median(pdist)),'max_distance':float(pdist.max()),'min_chainage':float(pc.min()),'max_chainage':float(pc.max()),'saddle':{'x':float(x[k]),'y':float(y[k]),'z':float(z[k]),'chainage':float(pc[k]),'distance':float(pdist[k])},'max_adverse_start':{'x':float(x[np.argmin(z[:ka+1])]),'y':float(y[np.argmin(z[:ka+1])])},'max_adverse_end':{'x':float(x[ka]),'y':float(y[ka])},'neighborhood':cl,'path':df};return out,ds
outs={};dss={}
for n,f in [('native','jilong_copernicus_glo30_utm45_v1.tif'),('32m','jilong_copernicus_32m_v1.tif'),('64m','jilong_copernicus_64m_v1.tif')]:outs[n],dss[n]=run(n,f)
sadd={};ks=list(outs)
for i in range(3):
 for j in range(i+1,3):
  a,b=outs[ks[i]]['saddle'],outs[ks[j]]['saddle'];sadd[ks[i]+'-'+ks[j]]=math.hypot(a['x']-b['x'],a['y']-b['y'])
v=[outs[k]['minimum_saddle_above_start'] for k in ks];cls='A_CONTINUOUS_LOW_CONNECTION' if v[0]<=20 and v[1]<=20 and v[2]<=30 else ('B_64M_COARSENING_BARRIER' if v[0]<=20 and v[1]<=20 and v[2]>=40 else ('C_MULTIRES_MAJOR_SADDLE' if min(v)>=40 else 'D_MIXED_OR_INTERMEDIATE'))
o={'exact_front_xy':F.tolist(),'front_samples':{k:float(next(dss[k].sample([tuple(F)]))[0]) for k in ks},'paths':{k:{q:v for q,v in outs[k].items() if q!='path'} for k in ks},'saddle_separation_m':sadd,'classification':cls,'terrain_modified':False,'corridor_modified':False,'dclaw_runs':0};(C/'reports/JILONG_E4_FRONT_MINIMAX_AUDIT.json').write_text(json.dumps(o,indent=2)+'\n');(C/'reports/JILONG_E4_FRONT_MINIMAX_AUDIT.md').write_text('# E4 front minimax audit\n\n'+json.dumps(o,indent=2)+'\n')
plt.figure(figsize=(11,8));d=dss['64m'];plt.imshow(d.read(1),extent=d.bounds,origin='upper',cmap='terrain');plt.plot(L[:,0],L[:,1],'w-',label='canonical');plt.scatter(*F,c='magenta',label='E4 front')
for n,col in [('native','cyan'),('32m','yellow'),('64m','lime')]:q=outs[n]['path'];plt.plot(q.x,q.y,color=col,label=n);plt.scatter(outs[n]['saddle']['x'],outs[n]['saddle']['y'],c=col)
plt.legend();plt.xlim(334000,338000);plt.ylim(3137000,3142500);plt.gca().set_aspect('equal');plt.savefig(C/'figures/JILONG_E4_FRONT_MINIMAX_PATHS.png',dpi=220);plt.close()
plt.figure(figsize=(10,5))
for n,col in [('native','cyan'),('32m','yellow'),('64m','lime')]:q=outs[n]['path'];dd=np.r_[0,np.cumsum(np.hypot(np.diff(q.x),np.diff(q.y)))];plt.plot(dd,q.terrain_z,color=col,label=n)
plt.legend();plt.grid();plt.savefig(C/'figures/JILONG_E4_FRONT_MINIMAX_PROFILE.png',dpi=220);plt.close()
print(json.dumps(o,indent=2))
