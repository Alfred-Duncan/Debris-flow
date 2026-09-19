from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[2]
PROFILE = ROOT / "local_validation/jilong_gate_a/terrain_final/jilong_glo30_valley_profile.csv"
TERRAIN = CASE / "jilong_copernicus_64m.tt3"
DX, HE, AUX_INDEX = 64.0, 2.0, 7
def read_tt3(path):
    with path.open() as f:
        h=[next(f).split() for _ in range(6)]
        nx,ny=int(h[0][0]),int(h[1][0]); xll,yll,cell=float(h[2][0]),float(h[3][0]),float(h[4][0]); z=np.loadtxt(f)
    if z.shape != (ny,nx): raise RuntimeError("unexpected terrain shape")
    return nx,ny,xll,yll,cell,z
def profile_track():
    raw=np.genfromtxt(PROFILE,delimiter=",",names=True,dtype=float,encoding="utf8")
    s,x,y=raw["station_m"],raw["x"],raw["y"]
    tx,ty=np.gradient(x,s),np.gradient(y,s); n=np.hypot(tx,ty); tx/=n; ty/=n
    off=raw["robust_low_point_lateral_offset_m"]
    return s,x,y,off,x+off*(-ty),y+off*tx,raw["robust_valley_floor_elevation_m"]
def main():
    ent,rep,fig=CASE/"entrainment",CASE/"reports",CASE/"figures"
    for d in (ent,rep,fig): d.mkdir(exist_ok=True)
    src=CASE/"source_zone_v2.json"; source_hash=hashlib.sha256(src.read_bytes()).hexdigest()
    tnx,tny,xll,yll,cell,ztop=read_tt3(TERRAIN)
    domain=json.loads(src.read_text()); nx,ny=int(domain["mx"]),int(domain["my"])
    if (tnx,cell,nx,ny)!=(168,DX,168,248) or not np.isfinite(ztop).all(): raise RuntimeError("accepted terrain check failed")
    s,cx,cy,off,rx,ry,floor=profile_track()
    with (ent/"robust_low_track.csv").open("w",newline="",encoding="utf8") as f:
        w=csv.writer(f); w.writerow(["chainage_m","corridor_x","corridor_y","robust_offset_m","robust_low_x","robust_low_y","robust_floor_z"]); w.writerows(zip(s,cx,cy,off,rx,ry,floor))
    ii,jj=np.meshgrid(np.arange(1,nx+1),np.arange(1,ny+1)); X=xll+(ii-.5)*DX; Y=yll+(jj-.5)*DX
    best2=np.full(X.shape,np.inf); best_s=np.zeros(X.shape); best_floor=np.zeros(X.shape)
    for k in range(len(s)-1):
        ax,ay,bx,by=rx[k],ry[k],rx[k+1],ry[k+1]; vx,vy=bx-ax,by-ay; vv=vx*vx+vy*vy
        t=np.clip(((X-ax)*vx+(Y-ay)*vy)/vv,0.,1.); ddx=X-(ax+t*vx); ddy=Y-(ay+t*vy); d2=ddx*ddx+ddy*ddy; use=d2<best2
        best2[use]=d2[use]; best_s[use]=(s[k]+t*(s[k+1]-s[k]))[use]; best_floor[use]=(floor[k]+t*(floor[k+1]-floor[k]))[use]
    dist=np.sqrt(best2); Z=ztop[-ny:,:][::-1,:]
    erod=(dist<=256.0)&(Z<=best_floor+20.0)&(best_s>=s[0])&(best_s<=s[-1]); he=np.where(erod,HE,0.0)
    with (ent/"erodible_mask_64m.csv").open("w",newline="",encoding="utf8") as f:
        w=csv.writer(f); w.writerow(["i","j","x","y","terrain_z","chainage_m","distance_to_robust_low_track_m","robust_floor_z","height_above_robust_floor_m","erodible"])
        for j in range(ny):
            for i in range(nx): w.writerow([i+1,j+1,f"{X[j,i]:.6f}",f"{Y[j,i]:.6f}",f"{Z[j,i]:.9g}",f"{best_s[j,i]:.6f}",f"{dist[j,i]:.6f}",f"{best_floor[j,i]:.6f}",f"{Z[j,i]-best_floor[j,i]:.6f}",int(erod[j,i])])
    with (ent/"erodible_thickness_e2.tt3").open("w",encoding="ascii") as f:
        f.write(f"{tnx} ncols\n{tny} nrows\n{xll:.12g} xllcorner\n{yll:.12g} yllcorner\n{DX:.12g} cellsize\n-9999 nodata_value\n"); he_top=np.zeros_like(ztop); he_top[-ny:,:]=he[::-1,:]; np.savetxt(f,he_top,fmt="%.1f")
    ncell=int(erod.sum()); area=ncell*DX*DX; maxvol=area*HE
    if not(np.isfinite(he).all() and (he>=0).all() and set(np.unique(he)).issubset({0.,HE})): raise RuntimeError("invalid h_e")
    if hashlib.sha256(src.read_bytes()).hexdigest()!=source_hash: raise RuntimeError("source was modified")
    plt.figure(figsize=(9,7)); plt.pcolormesh(X,Y,erod,cmap="Greens",shading="nearest"); plt.plot(rx,ry,"k-",lw=.6,label="Gate-B robust-low track"); plt.scatter(X[erod],Y[erod],s=4,c="lime",label="h_e=2 m cells"); plt.axis("equal"); plt.xlabel("Easting (m)"); plt.ylabel("Northing (m)"); plt.legend(loc="best"); plt.title("Deterministic E2 erodible mask, 64 m grid"); plt.tight_layout(); plt.savefig(fig/"erodible_mask_64m.png",dpi=180); plt.close()
    payload={"scenario":"C3_entrainment_E2","terrain":"jilong_copernicus_64m.tt3","terrain_shape":[nx,ny],"terrain_modified":False,"source_zone_v2_sha256":source_hash,"source_cell_count":len(json.loads(src.read_text())["cells"]),"mask_rule":"distance to continuous Gate-B robust-low track <= 256 m AND terrain_z <= robust_floor_z + 20 m AND chainage within S0-S4","h_e_aux_index_fortran":AUX_INDEX,"h_e_m":HE,"erodible_cell_count":ncell,"cell_area_m2":DX*DX,"erodible_area_m2":area,"maximum_available_erodible_volume_m3":maxvol,"finite":True,"nonnegative":True,"deterministic":True,"entrainment":{"enabled":1,"method":0,"rate":.20,"me":.62,"segregation":0}}
    (rep/"ENTRAINMENT_E2_PREFLIGHT.json").write_text(json.dumps(payload,indent=2)+"\n")
    (rep/"ENTRAINMENT_E2_PREFLIGHT.md").write_text(f"# E2 entrainment preflight\n\n- Terrain: accepted unchanged 64 m computational tt3.\n- Gate-B profile: full S0-S4 authoritative profile; robust-low points use its canonical tangent, signed normal offset, and continuous segment interpolation.\n- Rule: distance <= 256 m, z <= robust floor + 20 m, and S0-S4 chainage only.\n- Erodible cells: **{ncell}**; area: **{area:.0f} m2**; maximum available thickness-volume: **{maxvol:.0f} m3**.\n- D-Claw available-erodible-thickness aux index: **7** (Fortran one-based i_ent, Cartesian i_dig=2).\n- h_e is exactly 2.0 m in mask cells and 0 elsewhere. All values are finite and non-negative.\n- Frozen source support remains {payload['source_cell_count']} cells, hash {source_hash}.\n")
    (rep/"ENTRAINMENT_IMPLEMENTATION_AUDIT.md").write_text("# Entrainment implementation audit\n\nThe audited installed Cartesian D-Claw source sets i_dig=2 and i_ent=i_dig+5=7 (Fortran one-based). src2.f90 reads available erodible thickness as aux(i_ent,i,j) and increments q(i_bdif,i,j). The generated type-3 auxinit raster is therefore assigned to aux slot 7. E2 uses built-in entrainment=1, entrainment_method=0, entrainment_rate=0.20, and me=0.62; segregation remains disabled.\n")
    setrun=CASE/"setrun.py"; text=setrun.read_text()
    old="d.entrainment, d.entrainment_method, d.entrainment_rate, d.me = 0, 1, 0.0, .62"
    new='d.entrainment, d.entrainment_method, d.entrainment_rate, d.me = 1, 0, .20, .62\n    r.auxinitdclaw_data.auxinitfiles.append([3, 7, CASE / "entrainment/erodible_thickness_e2.tt3"])'
    if old not in text: raise RuntimeError("unexpected setrun entrainment line")
    setrun.write_text(text.replace(old,new))
    (CASE/"active_run.json").write_text(json.dumps({"case_id":"C3_entrainment_E2","V_m3":2.e6,"T_s":90.,"tfinal_s":900.,"output_interval_s":30.},indent=2)+"\n")
    print(json.dumps(payload,indent=2))
if __name__=="__main__": main()
