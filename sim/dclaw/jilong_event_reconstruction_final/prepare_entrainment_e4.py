from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np
CASE=Path(__file__).resolve().parent
DX=64.; N=928
def read_tt3(p):
    with p.open() as f:
        h=[next(f) for _ in range(6)]; z=np.loadtxt(f)
    return h,z
def main():
    ent=CASE/"entrainment"; rep=CASE/"reports"; mask=ent/"erodible_mask_64m.csv"; e2=ent/"erodible_thickness_e2.tt3"; e4=ent/"erodible_thickness_e4.tt3"
    members=[]
    with mask.open(newline="") as f:
        for r in csv.DictReader(f):
            if int(r["erodible"]): members.append((int(r["i"]),int(r["j"])))
    if len(members)!=N or len(set(members))!=N: raise RuntimeError("E2 mask membership invalid")
    header,z2=read_tt3(e2)
    if not np.isfinite(z2).all() or int((z2>0).sum())!=N or not set(np.unique(z2)).issubset({0.,2.}):raise RuntimeError("E2 h_e input inconsistent")
    z4=np.where(z2>0,4.,0.)
    with e4.open("w",encoding="ascii") as f:
        f.writelines(header);np.savetxt(f,z4,fmt="%.1f")
    _,check=read_tt3(e4)
    if not (np.isfinite(check).all() and (check>=0).all() and check.max()==4. and int((check>0).sum())==N):raise RuntimeError("E4 h_e invalid")
    source=json.loads((CASE/"source_zone_v2.json").read_text())
    if len(source["cells"])!=11:raise RuntimeError("source support changed")
    payload={"scenario":"C3_entrainment_E4","e2_mask_sha256":hashlib.sha256(mask.read_bytes()).hexdigest(),"erodible_cell_count":N,"cell_area_m2":4096.,"erodible_area_m2":N*4096.,"available_layer_thickness_m":4.,"maximum_available_erodible_volume_m3":N*4096.*4.,"h_e_aux_index_fortran":7,"h_e_all_finite":True,"h_e_nonnegative":True,"h_e_max_m":4.,"same_mask_as_E2":True,"terrain_modified":False,"source_support_cells":len(source["cells"]),"source_V_m3":2.e6,"source_T_s":90.,"entrainment_rate":.20,"entrainment_method":0,"me":.62,"segregation":0}
    (rep/"ENTRAINMENT_E4_PREFLIGHT.json").write_text(json.dumps(payload,indent=2)+"\n")
    (rep/"ENTRAINMENT_E4_PREFLIGHT.md").write_text(f"# E4 entrainment preflight\n\n- Exact E2 mask hash: {payload['e2_mask_sha256']}; membership count: {N}.\n- Cell area: 4096 m2; erodible area: {payload['erodible_area_m2']:.0f} m2.\n- h_e: 4.0 m only at the frozen E2 members; maximum available volume: {payload['maximum_available_erodible_volume_m3']:.0f} m3.\n- Aux slot: 7 (Fortran one-based). All h_e values are finite and non-negative.\n- Frozen source remains 11 cells, V=2.0e6 m3 and T=90 s. Entrainment method/rate/me remain 0/0.20/0.62.\n")
    p=CASE/"setrun.py";s=p.read_text()
    old='r.auxinitdclaw_data.auxinitfiles.append([3, 7, CASE / "entrainment/erodible_thickness_e2.tt3"])'
    new='r.auxinitdclaw_data.auxinitfiles.append([3, 7, CASE / "entrainment/erodible_thickness_e4.tt3"])'
    if old not in s:raise RuntimeError("setrun is not in frozen E2 configuration")
    p.write_text(s.replace(old,new))
    (CASE/"active_run.json").write_text(json.dumps({"case_id":"C3_entrainment_E4","V_m3":2.e6,"T_s":90.,"tfinal_s":900.,"output_interval_s":30.},indent=2)+"\n")
    print(json.dumps(payload,indent=2))
if __name__=="__main__":main()
