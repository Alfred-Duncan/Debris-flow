import numpy as np,json,csv
from pathlib import Path
from scipy.spatial import cKDTree
C=Path(__file__).parents[1];g=json.load(open(C/'model_geometry.json'));x0,y0,mx,my=g['xlower'],g['yupper'],g['mx'],g['my'];dx=64
p=C/'terrain/published_route_domain_64m.tt3';z=np.loadtxt(p,skiprows=6);valid=z>-9000;rr,cc=np.indices(z.shape);X=x0+(cc+.5)*dx;Y=y0-(rr+.5)*dx
route=list(csv.DictReader(open(C/'corridor/published_s0_to_port_route.csv')));P=np.array([[float(q['x']),float(q['y'])]for q in route]);S=np.array([float(q['chainage_m'])for q in route]);tree=cKDTree(P);D,ix=tree.query(np.c_[X.ravel(),Y.ravel()]);D=D.reshape(z.shape);ix=ix.reshape(z.shape);CH=S[ix]
# robust local floor: cross-route band around each nearest 20m station, +/-256 normal and within 32m tangent
floor=np.full(z.shape,np.nan)
for k in range(len(P)):
 m=valid&(np.abs(CH-S[k])<=32)&(D<=256);a=np.sort(z[m]);
 if len(a)>=2:floor[ix==k]=(a[0]+a[1])/2
h=z-floor
cand=valid&(CH>=-64)&(CH<=384)&(D<=256)&np.isfinite(h)&(h<=25)
s0=P[0];dist=(X-s0[0])**2+(Y-s0[1])**2;sr,sc=np.unravel_index(np.argmin(np.where(cand,dist,np.inf)),z.shape)
sel=[];todo=[(sr,sc)];seen=set(todo)
while todo and len(sel)<12:
 r,c=todo.pop(0);sel.append((r,c))
 for dr in(-1,0,1):
  for dc in(-1,0,1):
   a,b=r+dr,c+dc
   if 0<=a<z.shape[0]and 0<=b<z.shape[1]and cand[a,b]and(a,b)not in seen:seen.add((a,b));todo.append((a,b))
if len(sel)<6:raise SystemExit('NO_DEFENSIBLE_VALLEY_FLOOR_SOURCE_SUPPORT_AT_64M')
# tangent from first valid downstream polyline segment
v=P[1]-P[0];v=v/np.linalg.norm(v);cells=[]
for r,c in sel:cells.append(dict(row=int(r+1),col=int(c+1),center_x=float(X[r,c]),center_y=float(Y[r,c]),chainage_m=float(CH[r,c]),lateral_distance_m=float(D[r,c]),terrain_z=float(z[r,c]),local_floor_z=float(floor[r,c]),height_above_local_floor_m=float(h[r,c]),tangent_x=float(v[0]),tangent_y=float(v[1])))
# robust W ref using cross-section and contiguous low cells (8 conn) through S0
n=np.array([-v[1],v[0]]);along=(X-s0[0])*v[0]+(Y-s0[1])*v[1];lat=(X-s0[0])*n[0]+(Y-s0[1])*n[1];m=valid&(abs(along)<=32)&(abs(lat)<=320);a=np.sort(z[m]);rob=(a[0]+a[1])/2;low=m&(z<=rob+20);r0,c0=np.unravel_index(np.argmin(np.where(m,z,np.inf)),z.shape);q=[(r0,c0)];got=[];seen=set(q)
while q:
 r,c=q.pop();got.append((r,c))
 for dr in(-1,0,1):
  for dc in(-1,0,1):
   A,B=r+dr,c+dc
   if 0<=A<z.shape[0]and 0<=B<z.shape[1]and low[A,B]and(A,B)not in seen:seen.add((A,B));q.append((A,B))
W=len(got)*64
if not 64<=W<=512: raise SystemExit(f'WREF_OUT_OF_RANGE {W}')
old=json.load(open(C/'source/source_support_published_s0.json'))['cells'];summ=lambda a:dict(min=float(np.min(a)),median=float(np.median(a)),max=float(np.max(a)))
out=dict(n_cells=len(cells),cell_area_m2=4096.,area_m2=len(cells)*4096.,W_ref_model_m=W,cells=cells,height_above_floor_m=summ([x['height_above_local_floor_m']for x in cells]),old_support_height_above_floor_m=summ([x['height_above_local_floor_m']for x in old]),hard_limit_m=25.)
with open(C/'source/source_support_published_s0_floorfix.csv','w',newline='')as f:w=csv.DictWriter(f,cells[0].keys());w.writeheader();w.writerows(cells)
json.dump(out,open(C/'source/source_support_published_s0_floorfix.json','w'),indent=2)
json.dump(out,open(C/'reports/SOURCE_SUPPORT_FLOORFIX_PREFLIGHT.json','w'),indent=2);(C/'reports/SOURCE_SUPPORT_FLOORFIX_PREFLIGHT.md').write_text('# Source support floor fix\n\n'+json.dumps(out,indent=2)+'\n')
g['cells']=cells;g['W_ref_model_m']=W;json.dump(g,open(C/'model_geometry_sourcefix.json','w'),indent=2)
print(json.dumps(out,indent=2))
