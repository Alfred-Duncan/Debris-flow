from __future__ import annotations
import csv, heapq, json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import rowcol, xy
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

CASE=Path('/root/autodl-tmp/Debris-flow-jilong-final/sim/dclaw/jilong_event_reconstruction_final')
H=Path('/root/autodl-tmp/Jilong_DClaw_Handoff/terrain/final')
R={'native':H/'jilong_copernicus_glo30_utm45_v1.tif','32m':H/'jilong_copernicus_32m_v1.tif','64m':H/'jilong_copernicus_64m_v1.tif'}
REP=CASE/'reports';COR=CASE/'corridor';ENT=CASE/'entrainment';FIG=CASE/'figures'
for p in (REP,COR,ENT,FIG): p.mkdir(exist_ok=True)
F=np.array([335758.,3141052.]); S0=np.array([334051.79369854234,3143392.4925680971]); S4=np.array([340571.8257777202,3129640.775894154]); V=(S4-S0)/np.linalg.norm(S4-S0)
BOX=(335250.,337400.,3139300.,3141350.)
def prog(p): return float(np.dot(np.asarray(p)-F,V))
def ctr(t,r,c): return np.array(xy(t,r,c,offset='center'),float)
def write(p,a):
    with open(p,'w',newline='') as f:
        w=csv.DictWriter(f,a[0].keys());w.writeheader();w.writerows(a)
def load(path,exp=False):
    with rasterio.open(path) as d: z=d.read(1).astype(float);t=d.transform;n=d.nodata
    if n is not None:z[np.isclose(z,n)]=np.nan
    x0,x1,y0,y1=BOX
    if exp:x1+=300;y0-=300
    rr,cc=np.indices(z.shape);xx=t.c+(cc+.5)*t.a;yy=t.f+(rr+.5)*t.e
    q=(xx>=x0)&(xx<=x1)&(yy>=y0)&(yy<=y1);r0,r1=np.where(q)[0].min(),np.where(q)[0].max();c0,c1=np.where(q)[1].min(),np.where(q)[1].max()
    return z[r0:r1+1,c0:c1+1],t,r0,c0,(x0,x1,y0,y1)
def start(z,t,r0,c0):
    r,c=rowcol(t,*F);rr,cc=np.indices(z.shape);d=((rr+r0-r)**2+(cc+c0-c)**2).astype(float);d[~np.isfinite(z)]=np.inf;return np.unravel_index(np.argmin(d),z.shape)
def target(z,t,r0,c0,b,s):
    x0,x1,y0,y1=b;rr,cc=np.indices(z.shape);xx=t.c+(cc+c0+.5)*t.a;yy=t.f+(rr+r0+.5)*t.e;p=(xx-F[0])*V[0]+(yy-F[1])*V[1]
    return set(zip(*np.where(np.isfinite(z)&(p>=900)&(z<=z[s]-100)&((xx>=x1-200)|(yy<=y0+200)))))
def route(z,t,r0,c0,s,ts):
    best={s:(float(z[s]),0.,float(z[s]))};pre={};hp=[(*best[s],s)];dirs=[(a,b)for a in(-1,0,1)for b in(-1,0,1)if a or b]; hit=None
    while hp:
        mx,L,sm,u=heapq.heappop(hp)
        if (mx,L,sm)!=best.get(u):continue
        if u in ts:hit=u;break
        pu=ctr(t,u[0]+r0,u[1]+c0);pp=prog(pu)
        for dr,dc in dirs:
            a,b=u[0]+dr,u[1]+dc
            if not(0<=a<z.shape[0]and 0<=b<z.shape[1])or not np.isfinite(z[a,b]):continue
            pv=ctr(t,a+r0,b+c0)
            if prog(pv)<pp-100:continue
            q=(max(mx,float(z[a,b])),L+np.linalg.norm(pv-pu),sm+float(z[a,b]));v=(a,b)
            if q<best.get(v,(np.inf,np.inf,np.inf)):best[v]=q;pre[v]=u;heapq.heappush(hp,(*q,v))
    if hit is None:return None
    out=[]
    while True:
        out.append(hit)
        if hit==s: return out[::-1]
        hit=pre[hit]
def flood(z):
    f=z.copy();seen=np.zeros(z.shape,bool);h=[];n,m=z.shape
    for r in range(n):
        for c in(0,m-1):
            if np.isfinite(f[r,c])and not seen[r,c]: seen[r,c]=1;heapq.heappush(h,(f[r,c],r,c))
    for c in range(m):
        for r in(0,n-1):
            if np.isfinite(f[r,c])and not seen[r,c]: seen[r,c]=1;heapq.heappush(h,(f[r,c],r,c))
    while h:
        v,r,c=heapq.heappop(h)
        for dr in(-1,0,1):
            for dc in(-1,0,1):
                a,b=r+dr,c+dc
                if 0<=a<n and 0<=b<m and not seen[a,b]and np.isfinite(f[a,b]):seen[a,b]=1;f[a,b]=max(f[a,b],v);heapq.heappush(h,(f[a,b],a,b))
    return f
def records(path,z,t,r0,c0,fill=None):
    a=[];run=np.inf
    for k,(r,c)in enumerate(path):
        p=ctr(t,r+r0,c+c0);zz=float(z[r,c]);run=min(run,zz);d=dict(path_order=k,x=float(p[0]),y=float(p[1]),terrain_z=zz,downstream_progress_m=prog(p),running_min_z=run,adverse_rise_m=zz-run)
        if fill is not None:d['fill_depth_m']=float(fill[r,c]-z[r,c])
        a.append(d)
    return a
def stats(a):
    p=np.array([[x['x'],x['y']]for x in a]);z=np.array([x['terrain_z']for x in a]);L=np.linalg.norm(np.diff(p,axis=0),axis=1).sum();D=np.linalg.norm(p[-1]-p[0])
    return dict(start_elevation_m=float(z[0]),target_elevation_m=float(z[-1]),highest_path_elevation_m=float(z.max()),minimum_saddle_above_start_m=float(z.max()-z[0]),maximum_adverse_rise_m=float(max(x['adverse_rise_m']for x in a)),path_length_m=float(L),straight_line_displacement_m=float(D),tortuosity=float(L/D),final_downstream_progress_m=float(a[-1]['downstream_progress_m']))
def inter(a):
    p=np.array([[x['x'],x['y']]for x in a]);q=np.array([x['downstream_progress_m']for x in a]);k=np.r_[True,np.diff(q)>1e-6];p=p[k];q=q[k];g=np.arange(max(0,q.min()),q.max()+.1,50);return g,np.c_[np.interp(g,q,p[:,0]),np.interp(g,q,p[:,1])]
def separation(a,b):
    ga,A=inter(a);gb,B=inter(b);g=np.arange(max(ga.min(),gb.min()),min(ga.max(),gb.max())+.1,50)
    if len(g)<2:return dict(median_m=1e9,p90_m=1e9,maximum_m=1e9)
    A=np.c_[np.interp(g,ga,A[:,0]),np.interp(g,ga,A[:,1])];B=np.c_[np.interp(g,gb,B[:,0]),np.interp(g,gb,B[:,1])];d=np.linalg.norm(A-B,axis=1);return dict(median_m=float(np.median(d)),p90_m=float(np.quantile(d,.9)),maximum_m=float(d.max()))
def dense(a):
    p=np.array([[x['x'],x['y']]for x in a]);z=np.array([x['terrain_z']for x in a]);s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];q=np.r_[np.arange(0,s[-1],20),s[-1]];return np.c_[np.interp(q,s,p[:,0]),np.interp(q,s,p[:,1]),np.interp(q,s,z),q]
def full64():
    with rasterio.open(R['64m'])as d:return d.read(1).astype(float),d.transform,d.nodata
def buildmask(ax):
    z,t,nod=full64();rows=list(csv.DictReader(open(ENT/'erodible_mask_64m.csv')));old=np.zeros(z.shape,bool)
    for x in rows:old[z.shape[0]-int(x['j']),int(x['i'])-1]=bool(int(x['erodible']))
    rr,cc=np.indices(z.shape);xx=t.c+(cc+.5)*t.a;yy=t.f+(rr+.5)*t.e;p=(xx-F[0])*V[0]+(yy-F[1])*V[1];x0,x1,y0,y1=BOX;repl=(p>=-200)&(p<=prog(ax[-1,:2])+200)&(xx>=x0)&(xx<=x1)&(yy>=y0)&(yy<=y1)
    ds,ix=cKDTree(ax[:,:2]).query(np.c_[xx.ravel(),yy.ravel()]);ds=ds.reshape(z.shape);ix=ix.reshape(z.shape);new=old.copy();new[repl]=0;new[repl&np.isfinite(z)&(ds<=256)&(z<=ax[ix,2]+20)]=1
    for x in rows:x['erodible']=str(int(new[z.shape[0]-int(x['j']),int(x['i'])-1]))
    write(ENT/'erodible_mask_valley_v3_64m.csv',rows)
    with open(ENT/'erodible_thickness_e4_valley_v3.tt3','w')as f:f.write(f'{z.shape[1]} ncols\n{z.shape[0]} nrows\n{t.c} xllcorner\n{t.f+z.shape[0]*t.e} yllcorner\n{abs(t.a)} cellsize\n-9999 nodata_value\n');np.savetxt(f,np.where(new,4.,0.),fmt='%.1f')
    return dict(old_total_mask_cells=int(old.sum()),new_total_mask_cells=int(new.sum()),old_local_cells_removed=int((old&repl&~new).sum()),new_local_cells_added=int((new&repl&~old).sum()),unchanged_cells_outside_replacement_region=int((old[~repl]==new[~repl]).sum())),(z,t,old,new)
def map64(ax):
    z,t,nod=full64();out=[]
    for x,y,zz,s in ax:
        r,c=rowcol(t,x,y);o=[]
        for a in range(max(0,r-1),min(z.shape[0],r+2)):
            for b in range(max(0,c-1),min(z.shape[1],c+2)):
                d=np.linalg.norm(ctr(t,a,b)-[x,y])
                if np.isfinite(z[a,b])and d<=50:o.append((d,a,b))
        if o:
            d,a,b=min(o)
            if not out or(a,b)!=(out[-1]['row'],out[-1]['col']):out.append(dict(axis_order=len(out),row=a,col=b,x=float(ctr(t,a,b)[0]),y=float(ctr(t,a,b)[1]),terrain_z=float(z[a,b]),native_axis_distance_m=float(d),downstream_progress_m=prog(ctr(t,a,b))))
    for a,b in zip(out,out[1:]):
        if max(abs(a['row']-b['row']),abs(a['col']-b['col']))>1:raise RuntimeError('64m axis is not 8-connected')
    write(COR/'jilong_local_valley_axis_v3_64m.csv',out);return out
def figure(dat,ax,method):
    z,t,old,new=dat;rr,cc=np.indices(z.shape);xx=t.c+(cc+.5)*t.a;yy=t.f+(rr+.5)*t.e;plt.figure(figsize=(10,8));plt.imshow(z,extent=(t.c,t.c+z.shape[1]*t.a,t.f+z.shape[0]*t.e,t.f),origin='upper',cmap='terrain');plt.colorbar(label='elevation (m)');plt.scatter(xx[old],yy[old],s=3,c='cyan',alpha=.2,label='old mask');plt.scatter(xx[new],yy[new],s=3,c='magenta',alpha=.22,label='new mask');plt.plot(ax[:,0],ax[:,1],'-r',lw=2,label='NEW LOCAL VALLEY AXIS V3');plt.scatter(*F,c='white',edgecolors='k',label='actual E4 old front');plt.scatter(ax[-1,0],ax[-1,1],c='yellow',edgecolors='k',label='local outlet');plt.xlim(BOX[:2]);plt.ylim(BOX[2:]);plt.legend(fontsize=8);plt.title('Jilong local valley axis v3 — '+method);plt.tight_layout();plt.savefig(FIG/'JILONG_LOCAL_VALLEY_AXIS_V3.png',dpi=180);plt.close()
def main():
    A=dict(authoritative_e4_front_xy=F.tolist(),global_downstream_unit_vector=V.tolist(),terrain_modified=False,source_modified=False);Q={}
    for n,p in R.items():
        z,t,r0,c0,b=load(p);s=start(z,t,r0,c0);ts=target(z,t,r0,c0,b,s);ex=False
        if not ts:z,t,r0,c0,b=load(p,True);s=start(z,t,r0,c0);ts=target(z,t,r0,c0,b,s);ex=True
        rt=route(z,t,r0,c0,s,ts);a=records(rt,z,t,r0,c0);write(REP/f'valley_axis_candidate_{n}.csv',a);Q[n]=(z,t,r0,c0,b,a);A[n]=stats(a);A[n]['target_count']=len(ts);A[n]['expanded_once']=ex
    s32=separation(Q['native'][-1],Q['32m'][-1]);s64=separation(Q['native'][-1],Q['64m'][-1]);A['separation_native_32m']=s32;A['separation_native_64m']=s64
    good=A['native']['maximum_adverse_rise_m']<=35 and A['32m']['maximum_adverse_rise_m']<=35 and A['64m']['maximum_adverse_rise_m']<=45 and s32['p90_m']<=150 and s64['p90_m']<=180;method='ORIGINAL_DEM_MINIMUM_SADDLE' if good else None
    if not good:
        G={}
        for n,(z,t,r0,c0,b,a)in Q.items():
            f=flood(z);s=start(z,t,r0,c0);rt=route(f,t,r0,c0,s,target(z,t,r0,c0,b,s));g=records(rt,z,t,r0,c0,f);write(REP/f'priority_flood_route_{n}.csv',g);G[n]=g;A['priority_flood_'+n]=dict(max_fill_depth_m=max(x['fill_depth_m']for x in g),p95_fill_depth_m=float(np.quantile([x['fill_depth_m']for x in g],.95)),total_filled_cells=int(sum(x['fill_depth_m']>0 for x in g)))
        ps32=separation(G['native'],G['32m']);ps64=separation(G['native'],G['64m']);A['priority_flood_separation_native_32m']=ps32;A['priority_flood_separation_native_64m']=ps64
        if A['priority_flood_native']['max_fill_depth_m']<=25 and A['priority_flood_32m']['max_fill_depth_m']<=25 and A['priority_flood_64m']['max_fill_depth_m']<=35 and ps32['p90_m']<=150 and ps64['p90_m']<=180:method='PRIORITY_FLOOD_ROUTING_ONLY';Q['native']=(*Q['native'][:-1],G['native'])
    A['original_dem_valley_axis_pass']=good;A['method']=method
    if method is None:
        A['classification']='NO_DEFENSIBLE_CONTINUOUS_LOCAL_VALLEY_AXIS_IN_ACCEPTED_DEM';(REP/'LOCAL_VALLEY_AXIS_V3_PREFLIGHT.json').write_text(json.dumps(A,indent=2)+'\n');(REP/'LOCAL_VALLEY_AXIS_V3_PREFLIGHT.md').write_text('# Local valley axis v3 preflight\n\nNO_DEFENSIBLE_CONTINUOUS_LOCAL_VALLEY_AXIS_IN_ACCEPTED_DEM\n');print(json.dumps(A,indent=2));return
    ax=dense(Q['native'][-1]);rs=[dict(axis_order=i,x=float(x),y=float(y),terrain_z=float(z),distance_along_axis_m=float(s),downstream_progress_m=prog([x,y]),provenance='Local numerical valley axis derived from the accepted Copernicus terrain, anchored at the actual E4 front, without using the locally misaligned canonical corridor as a downstream target.')for i,(x,y,z,s)in enumerate(ax)];write(COR/'jilong_local_valley_axis_v3.csv',rs);(COR/'jilong_local_valley_axis_v3.geojson').write_text(json.dumps(dict(type='FeatureCollection',features=[dict(type='Feature',properties=dict(provenance=rs[0]['provenance'],method=method),geometry=dict(type='LineString',coordinates=[[x['x'],x['y']]for x in rs]))]),indent=2)+'\n')
    a64=map64(ax);pp=np.array([[x['x'],x['y']]for x in a64]);zz=np.array([x['terrain_z']for x in a64]);ss=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pp,axis=0),axis=1))];A['axis_64m_adverse_rise_windows_m']={str(w):max(float(zz[i:np.searchsorted(ss,ss[i]+w)].max()-zz[i:np.searchsorted(ss,ss[i]+w)].min())for i in range(len(ss)-1))for w in(64,128,256,512)};A['local_axis_length_m']=float(ax[-1,3]);A['local_axis_outlet_xy']=ax[-1,:2].tolist();A['axis_64m_start_elevation_m']=float(zz[0]);A['axis_64m_minimum_elevation_m']=float(zz.min());A['axis_64m_end_elevation_m']=float(zz[-1]);ms,dat=buildmask(ax);A['mask']=ms;A['classification']='AXIS_ACCEPTED_FOR_FINAL_E4';A['source_support_cells']=11;A['h_e_max_m']=4.;(REP/'LOCAL_VALLEY_AXIS_V3_PREFLIGHT.json').write_text(json.dumps(A,indent=2)+'\n');(REP/'LOCAL_VALLEY_AXIS_V3_PREFLIGHT.md').write_text('# Local valley axis v3 preflight\n\n'+json.dumps(A,indent=2)+'\n');figure(dat,ax,method);print(json.dumps(A,indent=2))
if __name__=='__main__':main()
