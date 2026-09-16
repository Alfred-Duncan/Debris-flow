"""Postprocess existing moving-entry outputs; this script never runs D-Claw."""
from __future__ import annotations
import csv, json, os
from heapq import heappop, heappush
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import LineString, Point
from shapely.ops import substring
from clawpack.geoclaw import topotools
from clawpack.pyclaw.solution import Solution

CASE=Path(__file__).resolve().parent; PROJECT=CASE.parents[2]
ENTRY=np.array([334051.8,3143392.5]); PROXY=np.array([340571.8257777202,3129640.775894154])
WET=.01; RHO_F,RHO_S,G=1100.,2700.,9.81; SELECT=(0,60,120,240,420,600)

def data_root():
    roots=[Path(os.environ['JILONG_DATA_ROOT']) if os.environ.get('JILONG_DATA_ROOT') else None,PROJECT/'data/extracted',PROJECT.parent/'Jilong-DebrisFlow/data/extracted']
    return next(p for p in roots if p and p.exists())

def river_line():
    shp=next((data_root()/'basic_geographic_data').rglob('四级河流.shp')); multi=gpd.read_file(shp).to_crs(32645).geometry.iloc[0]; a,b=multi.geoms
    return LineString(list(substring(a,a.project(Point(*ENTRY)),a.length).coords)+list(b.coords))

def priority_fill(z, outlets):
    """Priority-flood fill draining to terrain-defined low boundary outlets."""
    ny,nx=z.shape; out=z.copy(); seen=np.zeros_like(z,dtype=bool); parent=np.full((ny,nx,2),-1,dtype=int); heap=[]
    for j,i in outlets:
        seen[j,i]=True; heappush(heap,(out[j,i],j,i))
    for_di=tuple((dj,di) for dj in (-1,0,1) for di in (-1,0,1) if dj or di)
    while heap:
        elev,j,i=heappop(heap)
        for dj,di in for_di:
            jj,ii=j+dj,i+di
            if 0<=jj<ny and 0<=ii<nx and not seen[jj,ii]:
                seen[jj,ii]=True; parent[jj,ii]=(j,i); out[jj,ii]=max(z[jj,ii],elev); heappush(heap,(out[jj,ii],jj,ii))
    return out,parent

def d8_path(topo):
    z=topo.Z; ny,nx=z.shape; dx=float(topo.x[1]-topo.x[0]); dy=float(topo.y[1]-topo.y[0])
    # Do not use the mapped river to set the outlet.  The outlet seeds are
    # solely the lowest 1% (minimum 8) of DEM boundary cells; routing to all
    # boundaries would instead select an arbitrary nearby map edge.
    boundary=np.unique(np.r_[np.c_[np.arange(ny),np.zeros(ny,int)],np.c_[np.arange(ny),np.full(ny,nx-1,int)],np.c_[np.zeros(nx,int),np.arange(nx)],np.c_[np.full(nx,ny-1,int),np.arange(nx)]],axis=0)
    bz=np.array([z[j,i] for j,i in boundary]); valid=np.isfinite(bz) & (bz>0.)
    # Zero-valued TT3 boundary cells are NoData padding, not terrain outlets.
    valid_bz=bz[valid]; rank=min(len(valid_bz)-1,max(7,int(np.ceil(.01*len(valid_bz)))-1)); cut=np.partition(valid_bz,rank)[rank]
    outlets=[tuple(q) for q in boundary[valid & (bz<=cut)]]
    filled,parent=priority_fill(z,outlets)
    to=np.full((ny,nx,2),-1,dtype=int); drop=np.zeros((ny,nx))
    for j in range(ny):
        for i in range(nx):
            best=0.; target=(-1,-1)
            for dj in (-1,0,1):
                for di in (-1,0,1):
                    if not(dj or di): continue
                    jj,ii=j+dj,i+di
                    if 0<=jj<ny and 0<=ii<nx:
                        slope=(filled[j,i]-filled[jj,ii])/np.hypot(di*dx,dj*dy)
                        if slope>best: best,target=slope,(jj,ii)
            # On a priority-flood flat, the recorded parent gives the D8
            # connection to the selected terrain outlet; elsewhere preserve
            # the steepest filled-surface direction.
            if target==(-1,-1) and parent[j,i,0]>=0: target=tuple(parent[j,i])
            drop[j,i]=best; to[j,i]=target
    # D8 drainage accumulation, then independently select the most-drained
    # terrain cell in a 256-m neighbourhood of the supplied entry coordinate.
    accum=np.ones((ny,nx)); order=np.argsort(filled.ravel())[::-1]
    for k in order:
        j,i=np.unravel_index(k,(ny,nx)); jj,ii=to[j,i]
        if jj>=0: accum[jj,ii]+=accum[j,i]
    xx,yy=np.meshgrid(topo.x,topo.y); near=np.hypot(xx-ENTRY[0],yy-ENTRY[1])<=256.
    score=np.where(near & (to[:,:,0]>=0),accum,-np.inf); j,i=np.unravel_index(np.argmax(score),score.shape); start=np.array([topo.x[i],topo.y[j]])
    cells=[]; seen=set()
    while 0<=j<ny and 0<=i<nx and (j,i) not in seen and len(cells)<2000:
        cells.append((j,i)); seen.add((j,i)); jj,ii=to[j,i]
        if jj<0: break
        j,i=jj,ii
    coords=np.array([[topo.x[i],topo.y[j]] for j,i in cells])
    if len(coords)<2: raise RuntimeError('D8 path did not leave entry neighbourhood')
    line=LineString(coords)
    # Uniform 32-m stations support the requested 20--50-m profile spacing.
    ss=np.arange(0,min(5000.,line.length)+1e-9,32.); pts=np.array([[line.interpolate(s).x,line.interpolate(s).y] for s in ss])
    return line,ss,pts,start,accum,len(outlets),float(cut)

def centered_slope(s,z,window):
    return np.array([(np.interp(min(s[-1],q+window/2),s,z)-np.interp(max(s[0],q-window/2),s,z))/max(1.,min(s[-1],q+window/2)-max(s[0],q-window/2)) for q in s])

def valley_metrics(s,pts,zfun):
    n=len(pts); width=np.full(n,np.nan); relief=np.full(n,np.nan); bend=np.full(n,np.nan)
    for k,p in enumerate(pts):
        lo=max(0,k-2); hi=min(n-1,k+2); t=pts[hi]-pts[lo]; norm=np.linalg.norm(t)
        if norm==0: continue
        normal=np.array([-t[1],t[0]])/norm; offs=np.arange(-512.,513.,32.); cross=p+offs[:,None]*normal; cz=zfun(np.c_[cross[:,1],cross[:,0]]); center=float(zfun([[p[1],p[0]]])[0])
        # Width is the contiguous valley-floor span within 30 m of the local
        # centreline elevation; relief is the 512-m cross-section range.
        good=cz<=center+30.; mid=np.argmin(abs(offs)); left=mid; right=mid
        while left>0 and good[left-1]: left-=1
        while right<len(offs)-1 and good[right+1]: right+=1
        width[k]=offs[right]-offs[left]; relief[k]=np.nanmax(cz)-np.nanmin(cz)
        if 1<=k<n-1:
            a=pts[k]-pts[k-1]; b=pts[k+1]-pts[k]; bend[k]=np.degrees(np.arccos(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1)))
    return width,relief,bend

def route_front(h,x,y,line):
    wet=np.argwhere(h>WET); front=0.
    for i,j in wet:
        p=Point(float(x[i]),float(y[j])); d=line.distance(p)
        if d<=400.: front=max(front,line.project(p))
    return front

def read_case(case,river,d8line):
    out=CASE/'moving_runs'/case/'_output'; frames=sorted(int(p.name[-4:]) for p in out.glob('fort.q????')); records=[]; snapshots={}
    for n in frames:
        sol=Solution(n,path=out,file_format='ascii'); q=sol.state.q; x,y=sol.state.grid.dimensions[0].centers,sol.state.grid.dimensions[1].centers; h,hu,hv,hm=q[:4]
        wet=h>WET; speed=np.zeros_like(h); speed[wet]=np.hypot(hu[wet]/h[wet],hv[wet]/h[wet]); m=np.zeros_like(h); m[wet]=hm[wet]/h[wet]; cell=(x[1]-x[0])*(y[1]-y[0]); rho=np.where(wet,m*RHO_S+(1-m)*RHO_F,RHO_F)
        xx,yy=np.meshgrid(x,y,indexing='ij'); vol=h.sum()*cell; moving05=(h*(speed>.5)).sum()*cell; moving2=(h*(speed>2.)).sum()*cell
        comx=float((h*xx).sum()*cell/vol) if vol else np.nan; comy=float((h*yy).sum()*cell/vol) if vol else np.nan
        topo=topotools.Topography(); topo.read(CASE/'jilong_dem_12p5m_64m.tt3',topo_type=3); zfun=RegularGridInterpolator((topo.y,topo.x),topo.Z,bounds_error=False,fill_value=np.nan); zz=zfun(np.c_[yy.ravel(),xx.ravel()]).reshape(h.shape)
        comz=float((h*zz).sum()*cell/vol) if vol else np.nan; ep=float((rho*G*h*zz).sum()*cell); ek=float((.5*rho*h*speed**2).sum()*cell)
        pts=np.c_[xx[wet],yy[wet]]; vector_dist=np.array([river.distance(Point(*p)) for p in pts]) if len(pts) else np.array([np.nan]); d8dist=np.array([d8line.distance(Point(*p)) for p in pts]) if len(pts) else np.array([np.nan])
        records.append(dict(time_s=float(sol.t),wet_volume_m3=vol,moving_volume_gt_05_m3=moving05,moving_volume_gt_2_m3=moving2,com_x_m=comx,com_y_m=comy,com_elevation_m=comz,mean_speed_moving_ms=float(speed[speed>.5].mean()) if (speed>.5).any() else 0.,p90_speed_ms=float(np.percentile(speed[wet],90)) if wet.any() else 0.,p99_speed_ms=float(np.percentile(speed[wet],99)) if wet.any() else 0.,max_depth_m=float(h.max()),wet_area_m2=float(wet.sum()*cell),area_h_gt_1_m2=float((h>1).sum()*cell),area_h_gt_3_m2=float((h>3).sum()*cell),potential_energy_proxy=ep,kinetic_energy_proxy=ek,river_front_m=route_front(h,x,y,river),d8_front_m=route_front(h,x,y,d8line),wet_median_river_offset_m=float(np.nanmedian(vector_dist)),wet_median_d8_offset_m=float(np.nanmedian(d8dist)),wet_fraction_river_gt_400=float((vector_dist>400).mean())))
        if round(sol.t) in SELECT:
            snapshots[int(round(sol.t))]=dict(h=h,speed=speed,u=np.where(wet,hu/h,0),v=np.where(wet,hv/h,0),m=m,wet=wet,x=x,y=y,t=float(sol.t))
    return records,snapshots

def save_csv(case,records):
    with (CASE/f'mobility_timeseries_{case}.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=records[0].keys(),lineterminator='\n'); w.writeheader(); w.writerows(records)

def make_figure(topo,river,d8line,s,profile,allrecords,snaps):
    out=PROJECT/'outputs/reference_based'; out.mkdir(parents=True,exist_ok=True); fig,axs=plt.subplots(1,3,figsize=(17,5))
    ax=axs[0]; ax.pcolormesh(topo.x,topo.y,topo.Z,cmap='terrain',shading='auto',rasterized=True); ax.contour(topo.x,topo.y,topo.Z,levels=12,colors='k',linewidths=.25,alpha=.45); rx,ry=river.xy; dx,dy=d8line.xy; ax.plot(rx,ry,'c-',lw=1.2,label='mapped fourth-order river'); ax.plot(dx,dy,'m-',lw=1.5,label='DEM D8 valley path'); ax.plot(*ENTRY,'ko',label='entry'); ax.plot(*PROXY,'r*',ms=8,label='port proxy')
    for front,label in ((1672,'1.67 km'),(2324,'2.32 km'),(2572,'2.57 km'),(2612,'2.61 km')):
        p=river.interpolate(front); ax.plot(p.x,p.y,'wo',ms=3,mec='k'); ax.annotate(label,(p.x,p.y),fontsize=5,xytext=(2,2),textcoords='offset points')
    colors={0:'w',60:'yellow',120:'orange',240:'red',420:'purple',600:'black'}
    for t,d in snaps['M6'].items(): ax.contour(d['x'],d['y'],d['h'].T,levels=[WET],colors=[colors[t]],linewidths=1.1)
    d=snaps['M6'][240]; skip=(slice(None,None,8),slice(None,None,8)); ax.quiver(d['x'][skip[0]],d['y'][skip[1]],d['u'][skip].T,d['v'][skip].T,color='k',scale=180,alpha=.45,width=.002)
    ax.set_title('A. DEM, independent valley path, M6 wet contours'); ax.legend(fontsize=6,loc='best'); ax.set_aspect('equal')
    ax=axs[1]; ax.plot(s/1000,profile['elevation_m'],'k',label='elevation (m)'); ax.set_ylabel('elevation (m)'); ax2=ax.twinx(); ax2.plot(s/1000,-profile['slope_250m'],color='tab:red',label='downslope 250 m'); ax2.plot(s/1000,profile['valley_width_m'],color='tab:blue',label='valley width'); ax2.set_ylabel('downslope slope / width (m)'); ax.set_xlabel('DEM-path distance (km)'); ax.set_title('B. Longitudinal geomorphology'); ax.grid(alpha=.25); ax2.legend(fontsize=7,loc='best')
    ax=axs[2]
    for case in ('M3','M6'):
        r=allrecords[case]; tt=[x['time_s'] for x in r]; ax.plot(tt,[x['river_front_m']/1000 for x in r],label=f'{case} vector-front')
    ax.set_xlabel('time (s)'); ax.set_ylabel('mapped-vector front (km)'); ax3=ax.twinx(); r=allrecords['M6']; ax3.plot([x['time_s'] for x in r],[x['wet_area_m2']/1e6 for x in r],'k--',label='M6 wet area'); ax3.plot([x['time_s'] for x in r],[x['kinetic_energy_proxy']/1e12 for x in r],'g--',label='M6 KE proxy'); ax3.set_ylabel('wet area (km²) / KE proxy (10¹² J)'); ax.set_title('C. Front, spreading and mobility'); ax.legend(fontsize=7,loc='upper left'); ax3.legend(fontsize=7,loc='lower right')
    fig.tight_layout(); fig.savefig(out/'geomorphic_mobility_summary.png',dpi=180); fig.savefig(out/'geomorphic_diagnosis.png',dpi=180); plt.close(fig)

def main():
    topo=topotools.Topography(); topo.read(CASE/'jilong_dem_12p5m_64m.tt3',topo_type=3); river=river_line(); d8line,s,pts,start,accum,noutlets,outlet_cut=d8_path(topo); zfun=RegularGridInterpolator((topo.y,topo.x),topo.Z,bounds_error=False,fill_value=np.nan); elev=zfun(np.c_[pts[:,1],pts[:,0]]); width,relief,bend=valley_metrics(s,pts,zfun); offsets=np.array([river.distance(Point(*p)) for p in pts])
    profile=dict(s_m=s,x=pts[:,0],y=pts[:,1],elevation_m=elev,slope_local=np.gradient(elev,s),slope_100m=centered_slope(s,elev,100),slope_250m=centered_slope(s,elev,250),slope_500m=centered_slope(s,elev,500),valley_width_m=width,cross_valley_relief_m=relief,river_vector_offset_m=offsets,bend_angle_deg=bend)
    with (CASE/'geomorphic_profile.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=profile.keys(),lineterminator='\n'); w.writeheader(); w.writerows([{k:profile[k][i] for k in profile} for i in range(len(s))])
    allrecords={}; snaps={}
    for case in ('M1','M3','M4','M6'):
        r,snap=read_case(case,river,d8line); allrecords[case]=r; snaps[case]=snap; save_csv(case,r)
        # Server-only arrays preserve h, velocity vectors, m, remaining h, and wet masks for all required selected times.
        np.savez_compressed(CASE/f'geomorphic_fields_{case}.npz',**{f'{t}_{k}':v for t,d in snap.items() for k,v in d.items() if k not in ('x','y','t')},x=next(iter(snap.values()))['x'],y=next(iter(snap.values()))['y'])
    # M2/M5 receive full mobility time series without field snapshots.
    for case in ('M2','M5'):
        r,_=read_case(case,river,d8line); allrecords[case]=r; save_csv(case,r)
    make_figure(topo,river,d8line,s,profile,allrecords,snaps)
    near=(s>=1500)&(s<=3500); p={'d8_start_offset_m':float(np.linalg.norm(start-ENTRY)),'d8_path_length_m':float(d8line.length),'d8_outlet_seed_count':noutlets,'d8_outlet_elevation_cutoff_m':outlet_cut,'river_offset_median':float(np.median(offsets)),'river_offset_max':float(np.max(offsets)),'river_offset_gt100_m_count':int((offsets>100).sum()),'river_offset_gt200_m_count':int((offsets>200).sum()),'river_offset_gt400_m_count':int((offsets>400).sum()),'near_15_35km_elevation_change_m':float(elev[near][-1]-elev[near][0]) if near.any() else np.nan,'near_15_35km_min_slope_250m':float(np.min(-profile['slope_250m'][near])) if near.any() else np.nan,'near_15_35km_max_valley_width_m':float(np.nanmax(width[near])) if near.any() else np.nan}
    (CASE/'geomorphic_diagnosis_metrics.json').write_text(json.dumps(p,indent=2)+'\n')
    print(json.dumps(p,indent=2))

if __name__=='__main__': main()
