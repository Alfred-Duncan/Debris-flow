import csv,json,heapq,re,sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from clawpack.pyclaw import Solution
C=Path(__file__).resolve().parents[1]; OUT=C/'runs/RECON_B_rate060_LONG3600/_output'; R=C/'results'; P=C/'reports'; F=C/'figures';R4=19707.465910023548
geom=json.load(open(C/'model_geometry_sourcefix.json'));x0,x1,y0,y1=[geom[k] for k in ('xlower','xupper','ylower','yupper')]
route=list(csv.DictReader(open(C/'corridor/published_s0_to_port_route.csv')));rp=np.array([[float(a['x']),float(a['y'])] for a in route]);rs=np.array([float(a['chainage_m']) for a in route]);rtree=cKDTree(rp); r4xy=rp[-1]
# 64 m terrain (TT3 rows top-to-bottom); construct all geometry by physical coordinates.
z=np.loadtxt(C/'terrain/published_route_domain_64m.tt3',skiprows=6);nr,nc=z.shape; rr,cc=np.indices(z.shape);TX=x0+(cc+.5)*64;TY=y1-(rr+.5)*64; valid=z>-9000
# terminal direction from final 500 m route data
ix=np.where(rs>=R4-500)[0];u=rp[ix[-1]]-rp[ix[0]];u=u/np.linalg.norm(u)
# starts close to R4 and boundary targets in downstream half-plane
start=list(zip(*np.where(valid&(((TX-r4xy[0])**2+(TY-r4xy[1])**2)<=128**2))))
bound=(rr==0)|(rr==nr-1)|(cc==0)|(cc==nc-1);proj=(TX-r4xy[0])*u[0]+(TY-r4xy[1])*u[1]
target=set(zip(*np.where(valid&bound&(proj>0))))
best={};pre={};hp=[]
for q in start:best[q]=(float(z[q]),0.,0.);heapq.heappush(hp,(*best[q],q))
hit=None
while hp:
 mx,gain,L,a=heapq.heappop(hp)
 if (mx,gain,L)!=best.get(a):continue
 if a in target:hit=a;break
 for dr in (-1,0,1):
  for dc in (-1,0,1):
   if dr==dc==0:continue
   b=(a[0]+dr,a[1]+dc)
   if not(0<=b[0]<nr and 0<=b[1]<nc) or not valid[b]:continue
   q=(max(mx,float(z[b])),gain+max(0.,float(z[b]-z[a])),L+64*(2**.5 if dr and dc else 1.))
   if q<best.get(b,(np.inf,np.inf,np.inf)):best[b]=q;pre[b]=a;heapq.heappush(hp,(*q,b))
if hit is None:raise RuntimeError('no defensible downstream DEM extension')
path=[]
while hit not in start:path.append(hit);hit=pre[hit]
path.append(hit);path=path[::-1]
ex=[];d=0;last=None;run=z[path[0]]
for k,(r,c) in enumerate(path):
 if last is not None:d+=np.hypot(TX[r,c]-last[0],TY[r,c]-last[1])
 run=min(run,z[r,c]);ex.append({'order':k,'extension_chainage_m':float(d),'x':float(TX[r,c]),'y':float(TY[r,c]),'terrain_z_m':float(z[r,c]),'adverse_rise_m':float(z[r,c]-run),'signed_downstream_projection_m':float(proj[r,c])});last=(TX[r,c],TY[r,c])
with open(R/'POST_R4_DEM_EXTENSION_64m.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=ex[0]);w.writeheader();w.writerows(ex)
Pth=np.array([[q['x'],q['y']] for q in ex]);ES=np.array([q['extension_chainage_m'] for q in ex]); etree=cKDTree(Pth)
(C/'corridor/post_R4_dem_extension_64m.geojson').write_text(json.dumps({'type':'FeatureCollection','features':[{'type':'Feature','properties':{'extension_length_m':float(ES[-1]),'maximum_adverse_rise_m':float(max(q['adverse_rise_m'] for q in ex)),'boundary_endpoint_xy':[ex[-1]['x'],ex[-1]['y']],'boundary_endpoint_elevation_m':ex[-1]['terrain_z_m'],'terminal_direction_unit':u.tolist()},'geometry':{'type':'LineString','coordinates':Pth.tolist()}}]}))
# Explicit coordinate-orientation proof. q is shape (mx,my), while terrain is row/column; map by xy only.
s0=Solution(0,path=str(OUT),file_format='ascii');X0,Y0=s0.state.grid.p_centers
align={'status':'PASS','terrain_shape_row_col':list(z.shape),'q_shape_x_y':list(s0.state.q[0].shape),'terrain_top_row_y':float(TY[0,0]),'terrain_bottom_row_y':float(TY[-1,0]),'q_min_y':float(Y0.min()),'q_max_y':float(Y0.max()),'q_left_x':float(X0.min()),'q_right_x':float(X0.max()),'terrain_min_y':float(TY.min()),'terrain_max_y':float(TY.max()),'method':'All q-derived metrics use physical PyClaw X,Y centers and KD-tree mapping; no q.T-to-TT3-index alignment is used.','checks':{'q_y_range_matches_terrain_centers':bool(abs(Y0.min()-TY.min())<1e-6 and abs(Y0.max()-TY.max())<1e-6),'q_x_range_matches_terrain_centers':bool(abs(X0.min()-TX.min())<1e-6 and abs(X0.max()-TX.max())<1e-6)}}
align['status']='PASS' if all(align['checks'].values()) else 'FAIL';(P/'B60_LONGRUN_COORDINATE_ALIGNMENT.json').write_text(json.dumps(align,indent=2)+'\n')
if align['status']!='PASS':raise RuntimeError('coordinate alignment ambiguous')
secs=list(csv.DictReader(open(C/'corridor/published_route_sections.csv'))); thresholds=[.001,.05,.1,.2,.5,1.0]; labels={.001:'0001',.05:'005',.1:'010',.2:'020',.5:'050',1.0:'100'}; fs=[]; arrival={t:None for t in thresholds}; secarr={q['section']:None for q in secs}; boundary=[]
for k in range(121):
 s=Solution(k,path=str(OUT),file_format='ascii');q=s.state.q;h=q[0];X,Y=s.state.grid.p_centers;dx,dy=s.state.grid.delta
 sp=np.hypot(np.divide(q[1],h,out=np.zeros_like(h),where=h>.001),np.divide(q[2],h,out=np.zeros_like(h),where=h>.001));pts=np.c_[X.ravel(),Y.ravel()];dist,ri=rtree.query(pts);ed,ei=etree.query(pts);wet=h.ravel()>.001;near=wet&(dist<=512);pub=float(rs[ri[near]].max()) if near.any() else 0.;r4d=np.hypot(X-r4xy[0],Y-r4xy[1]); flags={t:bool(((h>t)&(r4d<=128)).any()) for t in thresholds}
 for t in thresholds:
  if flags[t] and arrival[t] is None:arrival[t]=float(s.t)
 extflags={t:((h.ravel()>t)&(ed<=512)) for t in [.001,.1]}; er={t:float(ES[ei[extflags[t]]].max()) if extflags[t].any() else 0. for t in extflags}
 # Furthest downstream wet cell is selected on physical extension station, not array index.
 fm=extflags[.001]; fi=np.flatnonzero(fm); fur=fi[np.argmax(ES[ei[fm]])] if len(fi) else int(np.argmax((X-r4xy[0])*u[0]+(Y-r4xy[1])*u[1])); fx,fy=float(X.ravel()[fur]),float(Y.ravel()[fur]
 ); edge=(np.isclose(X,x0+32)|np.isclose(X,x1-32)|np.isclose(Y,y0+32)|np.isclose(Y,y1-32)); touch=bool(((h>.1)&edge).any());
 if touch: boundary.append((float(s.t),X[(h>.1)&edge],Y[(h>.1)&edge],h[(h>.1)&edge],sp[(h>.1)&edge]))
 fs.append({'time_s':float(s.t),'instantaneous_published_route_reach_m':pub,'historical_published_route_reach_m':0.,'published_fraction':0.,**{f'R4_h{labels[t]}_reached':flags[t] for t in thresholds},'post_R4_extension_reach_h0001_m':er[.001],'post_R4_extension_reach_h010_m':er[.1],'total_diagnostic_reach_h010_m':R4+er[.1] if flags[.1] else pub,'max_depth_m':float(h.max()),'global_max_speed_ms':float(sp.max()),'P99_wet_speed_ms':float(np.quantile(sp.ravel()[wet],.99) if wet.any() else 0.),'wet_volume_m3':float(h.sum()*dx*dy),'bdif_state_integral_m3':float(q[6].sum()*dx*dy),'furthest_downstream_x':fx,'furthest_downstream_y':fy,'furthest_downstream_distance_from_R4_m':float(np.hypot(fx-r4xy[0],fy-r4xy[1])),'minimum_distance_wet_to_domain_boundary_m':float(np.minimum.reduce([X[wet.reshape(h.shape)]-(x0),x1-X[wet.reshape(h.shape)],Y[wet.reshape(h.shape)]-y0,y1-Y[wet.reshape(h.shape)] ]).min()) if wet.any() else None,'wet_h010_touches_domain_boundary':touch,'_h':h,'_sp':sp,'_X':X,'_Y':Y,'_ed':ed.reshape(h.shape),'_ei':ei.reshape(h.shape)})
hist=0
for a in fs:hist=max(hist,a['instantaneous_published_route_reach_m']);a['historical_published_route_reach_m']=hist;a['published_fraction']=hist/R4
# arrivals sections and 900 reproduction
for sec in secs:
 sx,sy=float(sec['x']),float(sec['y']);secarr[sec['section']]=next((a['time_s'] for a in fs if ((a['_h']>.1)&((a['_X']-sx)**2+(a['_Y']-sy)**2<=128**2)).any()),None)
r900=next(a for a in fs if a['time_s']==900); reproduction={'route_difference_m':abs(r900['historical_published_route_reach_m']-18208.954505154226),'maxdepth_relative_difference':abs(r900['max_depth_m']-75.08881498773077)/75.08881498773077,'R1_R2_R3_arrivals':secarr,'pass':abs(r900['historical_published_route_reach_m']-18208.954505154226)<=80 and abs(r900['max_depth_m']-75.08881498773077)/75.08881498773077<=.05 and all(abs(secarr[x]-v)<=30 for x,v in {'R1':150,'R2':270,'R3':390}.items())}
# terminal summary and selected checkpoints
terminal=[]
for t in [900,1200,1500,1800,2100,2400,2700,3000,3300,3600]:
 a=next(x for x in fs if x['time_s']==t);m=(a['_h']>.001)&(abs(a['_ed']-a['post_R4_extension_reach_h010_m'])<=256);terminal.append({'time_s':t,'published_reach_m':a['historical_published_route_reach_m'],'R4_h010_reached':a['R4_h010_reached'],'post_R4_extension_reach_h010_m':a['post_R4_extension_reach_h010_m'],'front_band_median_depth_m':float(np.median(a['_h'][m])) if m.any() else None,'front_band_P90_depth_m':float(np.quantile(a['_h'][m],.9)) if m.any() else None,'front_band_median_speed_ms':float(np.median(a['_sp'][m])) if m.any() else None,'front_band_P90_speed_ms':float(np.quantile(a['_sp'][m],.9)) if m.any() else None,'wet_volume_m3':a['wet_volume_m3'],'boundary_touch':a['wet_h010_touches_domain_boundary']})
with open(R/'B60_LONGRUN_TIMESERIES.csv','w',newline='') as f:
 rows=[]
 for a in fs:
  rows.append({k:v for k,v in a.items() if not k.startswith('_')})
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
with open(R/'B60_LONGRUN_TERMINAL_MOTION.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=terminal[0]);w.writeheader();w.writerows(terminal)
# R4 arrival detail
if arrival[.1] is not None:
 a=next(x for x in fs if x['time_s']==arrival[.1]);m=(a['_h']>.001)&((a['_X']-r4xy[0])**2+(a['_Y']-r4xy[1])**2<=256**2);q=Solution(int(arrival[.1]/30),path=str(OUT),file_format='ascii').state.q;u0=np.divide(q[1],q[0],out=np.zeros_like(q[0]),where=q[0]>.001);v0=np.divide(q[2],q[0],out=np.zeros_like(q[0]),where=q[0]>.001); row={'arrival_time_s':arrival[.1],'median_depth_m':float(np.median(a['_h'][m])),'P90_depth_m':float(np.quantile(a['_h'][m],.9)),'max_depth_m':float(a['_h'][m].max()),'median_speed_ms':float(np.median(a['_sp'][m])),'P90_speed_ms':float(np.quantile(a['_sp'][m],.9)),'max_speed_ms':float(a['_sp'][m].max()),'wet_area_m2':float(m.sum()*4096),'mean_flow_direction_unit':[float(u0[m].mean()),float(v0[m].mean())]};
 with open(R/'B60_R4_ARRIVAL_STATE.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=row);w.writeheader();w.writerow(row)
last=fs[-1];a300=next(a for a in fs if a['time_s']==3300);a600=next(a for a in fs if a['time_s']==3000);metric=lambda a: (R4+a['post_R4_extension_reach_h010_m']) if a['R4_h010_reached'] else a['historical_published_route_reach_m'];adv300=metric(last)-metric(a300);adv600=metric(last)-metric(a600); stationary=adv300<64 and adv600<128
if boundary:
 bt=boundary[0];bdetail={'first_boundary_touch_s':bt[0],'boundary_side':'mixed_or_coordinate_resolved_from_xy','x_range':[float(bt[1].min()),float(bt[1].max())],'y_range':[float(bt[2].min()),float(bt[2].max())],'depth_median':float(np.median(bt[3])),'depth_max':float(bt[3].max()),'speed_median':float(np.median(bt[4])),'speed_max':float(bt[4].max())}
else:bdetail=None
classification='DOMAIN_LIMITED_AFTER_R4' if last['R4_h010_reached'] and boundary else ('R4_REACHED_AND_NATURAL_RUNOUT_INSIDE_DOMAIN' if last['R4_h010_reached'] and stationary else ('R4_REACHED_BUT_STILL_MOVING_AT_3600' if last['R4_h010_reached'] else 'R4_NOT_REACHED_BY_3600'))
log=(C/'runs/RECON_B_rate060_LONG3600/run.log').read_text(errors='ignore');cfl=[float(x) for x in re.findall(r'maximum Courant number seen =\s+([0-9.]+)',log)]
final={'continuation_method':'EXACT_B60_RERUN_FROM_ZERO','physics_identical_to_RECON_B_rate060':'YES','B60_900_reproduction':'PASS' if reproduction['pass'] else 'FAIL','reproduction_detail':reproduction,'R1_h010_arrival_s':secarr['R1'],'R2_h010_arrival_s':secarr['R2'],'R3_h010_arrival_s':secarr['R3'],**{f'R4_h{n}_arrival_s':arrival[t] for n,t in [('0001',.001),('005',.05),('010',.1),('020',.2),('050',.5),('100',1.0)]},'R4_reached_h010':'YES' if last['R4_h010_reached'] else 'NO','post_R4_extension_length_available_m':float(ES[-1]),'post_R4_h010_reach_at_3600_m':last['post_R4_extension_reach_h010_m'],'total_diagnostic_reach_at_3600_m':metric(last),'furthest_wet_XY_at_3600':[last['furthest_downstream_x'],last['furthest_downstream_y']],'furthest_distance_from_R4_at_3600_m':last['furthest_downstream_distance_from_R4_m'],'domain_boundary_touched':'YES' if bool(boundary) else 'NO','first_domain_boundary_touch_s':bdetail['first_boundary_touch_s'] if bdetail else None,'boundary_detail':bdetail,'approximately_stationary_at_3600':'YES' if stationary else 'NO','advance_last_300s_m':adv300,'advance_last_600s_m':adv600,'max_depth_m':max(a['max_depth_m'] for a in fs),'max_speed_ms':max(a['global_max_speed_ms'] for a in fs),'P99_speed_ms':max(a['P99_wet_speed_ms'] for a in fs),'max_CFL':max(cfl) if cfl else None,'numerical_warning':'YES' if max(a['global_max_speed_ms'] for a in fs)>80 or max(a['max_depth_m'] for a in fs)>120 else 'NO','final_classification':classification,'recommended_next_step':'Extend/rebuild the computational domain before interpreting terminal runout.' if boundary else 'Freeze long-run result and assess post-R4 mobility.'}
(P/'B60_LONGRUN_FINAL.json').write_text(json.dumps(final,indent=2)+'\n');(P/'B60_LONGRUN_FINAL.md').write_text('# B60 long-duration continuation\n\n'+json.dumps(final,indent=2)+'\n')
# figures
plt.figure();plt.plot([a['time_s'] for a in fs],[a['historical_published_route_reach_m'] for a in fs],label='published route reach');plt.plot([a['time_s'] for a in fs],[metric(a) for a in fs],label='published + diagnostic extension');plt.axhline(14780.599433,ls='--',c='k',lw=.7,label='R3');plt.axhline(R4,ls='--',c='r',lw=.7,label='R4');plt.legend();plt.grid();plt.xlabel('time s');plt.ylabel('reach m');plt.tight_layout();plt.savefig(F/'B60_LONGRUN_REACH_VS_TIME.png',dpi=160);plt.close()
plt.figure();plt.imshow(last['_h'].T>0.001,origin='lower',extent=[x0,x1,y0,y1],alpha=.55,cmap='Blues');plt.plot(rp[:,0],rp[:,1],'k-',lw=.7);plt.plot(Pth[:,0],Pth[:,1],'r-',lw=1);plt.plot(*r4xy,'ro');plt.xlim(x0,x1);plt.ylim(y0,y1);plt.tight_layout();plt.savefig(F/'B60_LONGRUN_FINAL_FOOTPRINT.png',dpi=160);plt.close()
plt.figure();plt.plot(ES,[a['terrain_z_m'] for a in ex],label='DEM extension');plt.xlabel('extension chainage m');plt.ylabel('elevation m');plt.grid();plt.tight_layout();plt.savefig(F/'B60_LONGRUN_POST_R4_PROFILE.png',dpi=160);plt.close()
print(json.dumps(final,indent=2))
