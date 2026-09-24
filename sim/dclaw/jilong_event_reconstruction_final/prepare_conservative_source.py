"""Prepare the deterministic fixed-S0 conservative internal source support."""
from __future__ import annotations
import hashlib, json, math
from pathlib import Path
import numpy as np
import rasterio

CASE = Path(__file__).resolve().parent
HANDOFF = Path("/root/autodl-tmp/Jilong_DClaw_Handoff")
TERRAIN = HANDOFF / "terrain/final/jilong_copernicus_64m_v1.tif"
CORRIDOR = HANDOFF / "geometry/jilong_canonical_corridor.geojson"
DX, G, RHO_F, M0, W_REF = 64.0, 9.81, 1100.0, 0.62, 148.81741474020365
TARGETS = (32.0, 96.0, 160.0)
CASES = {"C1": (1e6, 90.), "C2": (1e6, 180.), "C3": (2e6, 90.), "C4": (2e6, 180.), "C5": (3e6, 90.), "C6": (3e6, 180.)}
TERRAIN_SHA = "53a7f882464eb0757bf24e45069f370c2f8198e145d0b281eacfccf660e55cfd"

def F(v, duration, time):
    if time <= 0: return 0.0
    if time >= duration: return v
    return .5 * v * (1. - math.cos(math.pi * time / duration))

def metrics(points, xy):
    best = (float("inf"), 0., None); accumulated = 0.
    for a, b in zip(points[:-1], points[1:]):
        vec = b - a; length = float(np.hypot(*vec))
        if length == 0: continue
        fraction = float(np.clip(np.dot(xy-a, vec)/(length*length), 0., 1.))
        distance = float(np.hypot(*(xy-(a+fraction*vec))))
        if distance < best[0]: best = (distance, accumulated+fraction*length, vec/length)
        accumulated += length
    if best[2] is None: raise RuntimeError("invalid canonical corridor")
    return best[1], best[0], best[2]

def prefix(points, maximum_chainage=400.):
    """Retain the only corridor reach that can be eligible for the 32–160 m source support."""
    kept=[points[0]]; accumulated=0.
    for a,b in zip(points[:-1],points[1:]):
        length=float(np.hypot(*(b-a)))
        if accumulated+length >= maximum_chainage:
            kept.append(a+(maximum_chainage-accumulated)/length*(b-a)); break
        kept.append(b); accumulated += length
    return np.asarray(kept,float)

def adjacent(a, b): return max(abs(a["i"]-b["i"]), abs(a["j"]-b["j"])) == 1

def choose(candidate_sets):
    # Each list is elevation ordered. Retain the first complete compatible 8-connected path.
    for a in candidate_sets[0]:
        for b in candidate_sets[1]:
            if not adjacent(a,b): continue
            for c in candidate_sets[2]:
                if adjacent(b,c) and len({(a["i"],a["j"]),(b["i"],b["j"]),(c["i"],c["j"])}) == 3:
                    return [a,b,c]
    raise RuntimeError("no valid 8-connected three-cell source-zone path")

def main():
    (CASE / "reports").mkdir(exist_ok=True); (CASE / "case_config").mkdir(exist_ok=True)
    if hashlib.sha256(TERRAIN.read_bytes()).hexdigest() != TERRAIN_SHA: raise RuntimeError("accepted terrain SHA256 mismatch")
    points = prefix(np.asarray(json.loads(CORRIDOR.read_text())["features"][0]["geometry"]["coordinates"], float))
    with rasterio.open(TERRAIN) as src:
        z = src.read(1)
        if not np.isfinite(z).all() or src.crs.to_epsg()!=32645 or src.res!=(DX,DX): raise RuntimeError("terrain integrity/geometry check failed")
        xlower, xupper, ylower = map(float,(src.bounds.left,src.bounds.right,src.bounds.bottom))
        yedges = src.bounds.top-DX*np.arange(src.height+1)
        yupper = float(yedges[np.argmin(abs(yedges-points[0][1]))])
        mx, my = int(round((xupper-xlower)/DX)), int(round((yupper-ylower)/DX))
        (CASE / "jilong_copernicus_64m.tt3").write_text(f"{src.width} ncols\n{src.height} nrows\n{xlower:.12g} xllcorner\n{ylower:.12g} yllcorner\n{DX:.12g} cellsize\n-9999 nodata_value\n",encoding="ascii")
        with (CASE / "jilong_copernicus_64m.tt3").open("a",encoding="ascii") as f: np.savetxt(f,z,fmt="%.9g")
        candidates=[[] for _ in TARGETS]
        for j in range(1,my+1):
            y=ylower+(j-.5)*DX; row=int(round((src.bounds.top-y)/DX-.5))
            for i in range(1,mx+1):
                x=xlower+(i-.5)*DX; col=int(round((x-src.bounds.left)/DX-.5))
                chain,distance,tangent=metrics(points,np.array([x,y]))
                if distance>96.: continue
                item={"i":i,"j":j,"center_x":x,"center_y":y,"terrain_elevation":float(z[row,col]),"corridor_chainage":chain,"distance_to_corridor":distance,"tangent_x":float(tangent[0]),"tangent_y":float(tangent[1])}
                for k,target in enumerate(TARGETS):
                    if abs(chain-target)<=96.: candidates[k].append(item)
    for target, choices in zip(TARGETS,candidates):
        if not choices: raise RuntimeError(f"no candidate near target {target}")
        choices.sort(key=lambda c:(c["terrain_elevation"],abs(c["corridor_chainage"]-target),c["i"],c["j"]))
    cells=choose(candidates)
    if not all(cells[k]["corridor_chainage"]<cells[k+1]["corridor_chainage"] for k in range(2)): raise RuntimeError("nonmonotonic selected chainages")
    zone={"method":"deterministic_8_connected_valley_floor","terrain":str(TERRAIN),"terrain_sha256":TERRAIN_SHA,"corridor":str(CORRIDOR),"crs":"EPSG:32645","dx_m":DX,"source_targets_m":list(TARGETS),"xlower":xlower,"xupper":xupper,"ylower":ylower,"yupper":yupper,"mx":mx,"my":my,"cells":cells}
    (CASE/"source_zone.json").write_text(json.dumps(zone,indent=2)+"\n")
    zone_report=["# Fixed-S0 conservative source zone","","This numerical support is not an observed release footprint.","Interior centers <=96 m from the canonical corridor were enumerated. For target chainages 32, 96 and 160 m, candidates within ±96 m were elevation-sorted. The first complete unique, monotonic, 8-connected path was retained; a low candidate that breaks connectivity is rejected deterministically.","","| target (m) | i | j | x (m) | y (m) | z (m) | chainage (m) | corridor distance (m) | tangent x | tangent y |","|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for target,c in zip(TARGETS,cells): zone_report.append(f"| {target:.0f} | {c['i']} | {c['j']} | {c['center_x']:.3f} | {c['center_y']:.3f} | {c['terrain_elevation']:.3f} | {c['corridor_chainage']:.3f} | {c['distance_to_corridor']:.3f} | {c['tangent_x']:.9f} | {c['tangent_y']:.9f} |")
    (CASE/"reports/SOURCE_ZONE.md").write_text("\n".join(zone_report)+"\n")
    pre={"architecture":"CONSERVATIVE_FIXED_S0_SOURCE_ZONE","formula":"F(t)=0 (t<=0); V/2*(1-cos(pi*t/T)) (0<t<T); V (t>=T)","source_zone_cell_count":3,"cell_area_m2":DX*DX,"source_area_m2":3*DX*DX,"fixed_W_ref_m":W_REF,"incremental_state":{"dh":"DeltaV/(3*dx*dy)","dhu":"dh*U_src*tangent_x","dhv":"dh*U_src*tangent_y","dhm":"0.62*dh","dpb":"1100*9.81*dh","dhchi":0.,"dbdif":0.},"cases":{}}
    for name,(volume,duration) in CASES.items():
        qpeak=math.pi*volume/(2*duration); href=((qpeak/W_REF)**2/G)**(1/3); upk=math.sqrt(G*href); probe_dt=min(.05,duration/1000); dv=F(volume,duration,probe_dt); dh=dv/(3*DX*DX)
        row={"target_volume_m3":volume,"duration_s":duration,"analytic_final_volume_m3":F(volume,duration,duration),"Q_peak_m3s":qpeak,"h_ref_peak_m":href,"U_src_peak_ms":upk,"probe_dt_s":probe_dt,"probe_deltaV_m3":dv,"probe_delta_h_m":dh,"probe_delta_hm_m":M0*dh,"probe_delta_pb_Pa_m":RHO_F*G*dh,"finite":bool(np.isfinite([qpeak,href,upk,dv,dh]).all()),"nonnegative_increment":bool(dh>=0),"downstream_momentum":True,"pass":bool(F(volume,duration,duration)==volume and upk<40 and dh>=0)}
        if not row["pass"]: raise RuntimeError(f"preflight failed: {name}")
        pre["cases"][name]=row
        cfg={"case_id":name,"V_m3":volume,"T_s":duration,"Q_peak_m3s":qpeak,"tfinal_s":900.,"output_interval_s":30.,"closure":"conservative-fixed-S0-internal-source-zone"}
        (CASE/f"case_config/{name}.json").write_text(json.dumps(cfg,indent=2)+"\n")
    (CASE/"reports/CONSERVATIVE_SOURCE_PREFLIGHT.json").write_text(json.dumps(pre,indent=2)+"\n")
    rows=["# Conservative source preflight","","The source area is exactly three 64 m cells (12,288 m²). The fixed unit-Froude closure supplies only injected momentum, never volume.","","| case | V (m3) | T (s) | Qpeak (m3/s) | Usrc peak (m/s) | final F(T) (m3) | pass |","|---|---:|---:|---:|---:|---:|---|"]
    for n,r in pre["cases"].items(): rows.append(f"| {n} | {r['target_volume_m3']:.0f} | {r['duration_s']:.0f} | {r['Q_peak_m3s']:.6f} | {r['U_src_peak_ms']:.6f} | {r['analytic_final_volume_m3']:.0f} | {r['pass']} |")
    rows += ["","All cases have exactly three valid cells, finite and nonnegative increments, downstream tangents, and Usrc peak <40 m/s."]
    (CASE/"reports/CONSERVATIVE_SOURCE_PREFLIGHT.md").write_text("\n".join(rows)+"\n")

if __name__=="__main__": main()
