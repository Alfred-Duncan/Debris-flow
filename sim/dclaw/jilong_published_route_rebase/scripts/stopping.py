import numpy as np,csv,json,heapq
from pathlib import Path
from scipy.spatial import cKDTree
from clawpack.pyclaw import Solution
C=Path(__file__).parents[1];R=C/'results';REP=C/'reports';F=C/'figures';
route=list(csv.DictReader(open(C/'corridor/published_s0_to_port_route.csv')));P=np.array([[float(x['x']),float(x['y'])]for x in route]);S=np.array([float(x['chainage_m'])for x in route]);tree=cKDTree(P)
z=np.loadtxt(C/'terrain/published_route_domain_64m.tt3',skiprows=6);g=json.load(open(C/'model_geometry_sourcefix.json'));rr,cc=np.indices(z.shape);X=g['xlower']+(cc+.5)*64;Y=g['yupper']-(rr+.5)*64;D,ix=tree.query(np.c_[X.ravel(),Y.ravel()]);D=D.reshape(z.shape);ix=ix.reshape(z.shape);CH=S[ix];valid=z>-9000;reg=valid&(CH>=13500)&(CH<=16200)&(D<=768)
# lexicographic terrain-only path
start=set(zip(*np.where(reg&(CH>=13700)&(CH<=14000))));target=set(zip(*np.where(reg&(CH>=15500)&(CH<=16000))));best={};pre={};hp=[]
for u in start:best[u]=(float(z[u]),0.,0.);heapq.heappush(hp,(*best[u],u))
hit=None
while hp:
 mx,gain,L,u=heapq.heappop(hp)
 if (mx,gain,L)!=best.get(u):continue
 if u in target:hit=u;break
 for dr in(-1,0,1):
  for dc in(-1,0,1):
   a,b=u[0]+dr,u[1]+dc
   if not(0<=a<z.shape[0]and 0<=b<z.shape[1])or not reg[a,b]or CH[a,b]<CH[u]-128:continue
   q=(max(mx,float(z[a,b])),gain+max(0,float(z[a,b]-z[u])),L+64*(2**.5 if dr and dc else 1));v=(a,b)
   if q<best.get(v,(np.inf,np.inf,np.inf)):best[v]=q;pre[v]=u;heapq.heappush(hp,(*q,v))
if hit is None:raise SystemExit('no path')
path=[]
while hit not in start:path.append(hit);hit=pre[hit]
path.append(hit);path=path[::-1];rows=[];run=z[path[0]]
for k,(r,c)in enumerate(path):run=min(run,z[r,c]);rows.append(dict(order=k,row=r,col=c,x=X[r,c],y=Y[r,c],chainage_m=CH[r,c],distance_to_route_m=D[r,c],terrain_z=z[r,c],adverse_rise_m=z[r,c]-run))
with open(R/'STOPPING_ZONE_DEM_PATH.csv','w',newline='')as f:w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
(C/'corridor/stopping_zone_dem_valley_path.geojson').write_text(json.dumps(dict(type='FeatureCollection',features=[dict(type='Feature',properties={},geometry=dict(type='LineString',coordinates=[[x['x'],x['y']]for x in rows]))])))
# mask coverage
mask=list(csv.DictReader(open(C/'entrainment/erodible_mask_published_route_64m.csv')));M=np.zeros(z.shape,bool)
for x in mask:M[z.shape[0]-int(x['j']),int(x['i'])-1]=bool(int(x['erodible']))
for x in rows:x['inside_current_erodible_mask']=int(M[x['row'],x['col']]);x['h_e_current']=4 if M[x['row'],x['col']] else 0
with open(R/'STOPPING_ZONE_ERODIBLE_MASK_COVERAGE.csv','w',newline='')as f:w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
covered=np.array([x['inside_current_erodible_mask']for x in rows],bool);long=max([sum(1 for _ in grp) for val,grp in __import__('itertools').groupby(covered) if not val]or[0])*64
# E4 front band heads
out=C/'runs/E4_published_sourcefix/_output';state=[]
for k in range(20,31):
 sol=Solution(k,path=str(out),file_format='ascii');q=sol.state.q;xx,yy=sol.state.grid.p_centers;h=q[0];sp=np.hypot(np.divide(q[1],h,out=np.zeros_like(h),where=h>.001),np.divide(q[2],h,out=np.zeros_like(h),where=h>.001));dd,ii=tree.query(np.c_[xx.ravel(),yy.ravel()]);dd=dd.reshape(h.shape);ii=ii.reshape(h.shape);ch=S[ii];wet=h>.001;fm=ch[wet&(dd<=512)].max();b=wet&(dd<=512)&(ch>=fm-384)&(ch<=fm);H=z.T+h+sp**2/(2*9.81);state.append(dict(time_s=sol.t,front_chainage_m=float(fm),wet_cells=int(b.sum()),h_median=float(np.median(h[b])),h_p90=float(np.quantile(h[b],.9)),h_max=float(h[b].max()),speed_median=float(np.median(sp[b])),speed_p90=float(np.quantile(sp[b],.9)),speed_max=float(sp[b].max()),head_median=float(np.median(H[b])),head_p90=float(np.quantile(H[b],.9)),head_max=float(H[b].max())))
with open(R/'E4_STOPPING_FRONT_LOCAL_STATE.csv','w',newline='')as f:w=csv.DictWriter(f,state[0].keys());w.writeheader();w.writerows(state)
# C3 corrected historical max
hist=0;cr=[]
for k in range(31):
 sol=Solution(k,path=str(C/'runs/C3_published_sourcefix/_output'),file_format='ascii');h=sol.state.q[0];xx,yy=sol.state.grid.p_centers;dd,ii=tree.query(np.c_[xx.ravel(),yy.ravel()]);m=(h.ravel()>.001)&(dd<=512);cur=float(S[ii[m]].max())if m.any()else 0;hist=max(hist,cur);cr.append(dict(time_s=sol.t,active_connected_front_m=cur,historical_max_reached_front_m=hist))
with open(R/'C3_sourcefix_front_corrected.csv','w',newline='')as f:w=csv.DictWriter(f,cr[0].keys());w.writeheader();w.writerows(cr)
summary=dict(dem_path_max_adverse_rise_m=float(max(x['adverse_rise_m']for x in rows)),dem_path_max_lateral_offset_m=float(max(x['distance_to_route_m']for x in rows)),mask_coverage_fraction=float(covered.mean()),longest_uncovered_m=int(long),uncovered_cells=int((~covered).sum()),z_saddle_dem=float(max(x['terrain_z']for x in rows)),conditional_maskfix_justified=bool((~covered).any()),classification='ERODIBLE_MASK_TRUNCATES_VALID_VALLEY_FLOOR' if (~covered).any() else 'MOBILITY_LIMIT_ON_OPEN_VALLEY_FLOOR')
json.dump(summary,open(REP/'E4_STOPPING_MECHANISM_AUDIT.json','w'),indent=2);open(REP/'E4_STOPPING_MECHANISM_AUDIT.md','w').write('# E4 stopping mechanism audit\n\n'+json.dumps(summary,indent=2)+'\n');json.dump(dict(corrected_metric='historical max wet route chainage',final=cr[-1]),open(REP/'C3_POSTPROCESS_CORRECTION.json','w'),indent=2);open(REP/'C3_POSTPROCESS_CORRECTION.md','w').write('# C3 correction\n\n'+json.dumps(cr[-1],indent=2)+'\n');print(json.dumps(summary,indent=2))
