"""Build a provenance-preserving, local computational DEM for Gate B only.

The two source DEMs and canonical S0--S4 geometry are opened read-only.  This
script never conditions terrain: it only completes NoData and replaces the two
previously documented native-data artifact tiles with the independent 8 m DEM.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds
from shapely.geometry import LineString, Point, mapping

ROOT = Path(r"E:\Alfred\Jilong_GateA")
RAW = ROOT / "raw_extracted"
CORRIDOR = ROOT / "corridor"
REPORT = ROOT / "reports"
FIG = ROOT / "figures"
PREFLIGHT = ROOT / "terrain_preflight"
PATCHES = PREFLIGHT / "patches"
WORK = ROOT / "computational_terrain_work"
OUT = ROOT / "computational_terrain"
CRS = "EPSG:32645"
NODATA = 65535.0
S0 = Point(334051.79369854234, 3143392.492568097)
S4 = Point(340571.8257777202, 3129640.775894154)
STATION_SPACING = 50.0
HALF_WIDTH = 250.0
DOMAIN_HALF_WIDTH = 750.0
PATCH_DEFS = {
    "B04": {"start": 16300.0, "end": 16650.0, "reason": "Documented high-severity native 12.5 m artifact; independent 8 m support."},
    "B05": {"start": 18550.0, "end": 18700.0, "reason": "Documented high-severity native 12.5 m artifact; independent 8 m support."},
}


def valid(a, nodata=NODATA):
    return np.isfinite(a) & (a != nodata) & (a > 0)


def distances(line, spacing=STATION_SPACING):
    d = np.arange(0.0, line.length + spacing, spacing)
    d[-1] = line.length
    return d


def segment(line, start, end):
    """Line substring without altering the authoritative line."""
    d = np.linspace(max(0.0, start), min(line.length, end), max(2, int((end-start)/12.5)+2))
    return LineString([line.interpolate(float(x)).coords[0] for x in d])


def cross(line, d, half=HALF_WIDTH):
    p = line.interpolate(float(d)); a = line.interpolate(max(0.0, d - 25)); b = line.interpolate(min(line.length, d + 25))
    dx, dy = b.x-a.x, b.y-a.y; norm = math.hypot(dx, dy)
    nx, ny = -dy/norm, dx/norm
    return LineString([(p.x-nx*half, p.y-ny*half), (p.x+nx*half, p.y+ny*half)]), p


def sample_profile(ds, line, half=HALF_WIDTH, spacing=STATION_SPACING):
    """Read-only 50 m station profile; robust floor = median lowest five."""
    rows = []
    for d in distances(line, spacing):
        sec, p = cross(line, d, half)
        offsets = np.arange(-half, half+ds.res[0]*.51, ds.res[0])
        pts = [sec.interpolate((o+half)/(2*half), normalized=True).coords[0] for o in offsets]
        vals = np.asarray([x[0] for x in ds.sample(pts)], dtype=float)
        vv = vals[valid(vals, ds.nodata)]
        floor = float(np.median(np.sort(vv)[:5])) if len(vv) >= 5 else np.nan
        center = vals[int(np.argmin(np.abs(offsets)))]
        if not valid(np.asarray([center]), ds.nodata)[0]: center = np.nan
        rows.append({"d":float(d), "x":p.x, "y":p.y, "floor":floor, "center":float(center) if np.isfinite(center) else np.nan})
    return rows


def rises(values, d, windows=(50,100,250,500,1000)):
    ans = {}
    for w in windows:
        out=[]
        for i, x in enumerate(d):
            j=int(np.searchsorted(d,x+w,side="left"))
            out.append(values[j]-values[i] if j<len(d) and np.isfinite(values[i]) and np.isfinite(values[j]) else np.nan)
        ans[str(w)] = None if not np.isfinite(out).any() else float(np.nanmax(out))
    return ans


def candidates(rows, min_height=60.0):
    d=np.asarray([r["d"] for r in rows]); z=np.asarray([r["floor"] for r in rows]); out=[]
    flank=max(1,int(500/STATION_SPACING))
    for i in range(flank,len(z)-flank):
        if not np.isfinite(z[i]): continue
        up,down=z[i-flank:i+1],z[i:i+flank+1]
        if not(np.isfinite(up).any() and np.isfinite(down).any()): continue
        h=min(z[i]-np.nanmin(up),z[i]-np.nanmin(down))
        if h<min_height: continue
        threshold=z[i]-h*.5; left=i; right=i
        while left and np.isfinite(z[left-1]) and z[left-1]>=threshold: left-=1
        while right+1<len(z) and np.isfinite(z[right+1]) and z[right+1]>=threshold: right+=1
        if d[right]-d[left]<150: continue
        if any(abs(d[i]-q["crest_station_m"])<300 for q in out): continue
        out.append({"start_station_m":float(d[left]),"crest_station_m":float(d[i]),"end_station_m":float(d[right]),"barrier_height_m":float(h),"barrier_length_m":float(d[right]-d[left]),"x":rows[i]["x"],"y":rows[i]["y"]})
    return out


def coverage(mask, geom, transform):
    inside=geometry_mask([mapping(geom)],out_shape=mask.shape,transform=transform,invert=True)
    return 100.0*float((mask&inside).sum())/float(inside.sum()), inside


def robust_stats(x):
    x=np.asarray(x,dtype=float); med=float(np.median(x)); mad=float(np.median(np.abs(x-med)))
    return {"n_samples":int(x.size),"median_dem12_minus_dem8_m":med,"mean_dem12_minus_dem8_m":float(np.mean(x)),"NMAD_m":float(1.4826*mad),"P05_m":float(np.percentile(x,5)),"P25_m":float(np.percentile(x,25)),"P50_m":float(np.percentile(x,50)),"P75_m":float(np.percentile(x,75)),"P95_m":float(np.percentile(x,95)),"RMSE_m":float(np.sqrt(np.mean(x*x)))}


def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def write_geojson(path, geom, props):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({"type":"FeatureCollection","features":[{"type":"Feature","properties":props,"geometry":mapping(geom)}]},indent=2)+"\n",encoding="utf-8")


def resampled_profile(arr, transform, crs, nodata, line, resolution, method):
    height=max(1,int(math.ceil(arr.shape[0]*transform.a/resolution))); width=max(1,int(math.ceil(arr.shape[1]*transform.a/resolution)))
    dst=np.full((height,width),nodata,dtype="float32")
    dst_transform=Affine(resolution,0,transform.c,0,-resolution,transform.f)
    reproject(arr,dst,src_transform=transform,src_crs=crs,src_nodata=nodata,dst_transform=dst_transform,dst_crs=crs,dst_nodata=nodata,resampling=method)
    profile={"driver":"GTiff","height":height,"width":width,"count":1,"dtype":"float32","crs":crs,"transform":dst_transform,"nodata":nodata}
    with MemoryFile() as mem:
        with mem.open(**profile) as ds: ds.write(dst,1); rows=sample_profile(ds,line)
    return rows


def main():
    for p in (REPORT,FIG,PREFLIGHT,PATCHES,WORK,OUT): p.mkdir(parents=True,exist_ok=True)
    line=gpd.read_file(CORRIDOR/"jilong_canonical_corridor.geojson").to_crs(CRS).geometry.iloc[0]
    if Point(line.coords[0]).distance(S0)>.01 or Point(line.coords[-1]).distance(S4)>.01: raise RuntimeError("Canonical S0/S4 changed; stop.")
    dem12=next((RAW/"dem_12p5m").rglob("*.tif")); dem8=next((RAW/"dem_8m").rglob("*.tif"))
    with rasterio.open(dem12) as a, rasterio.open(dem8) as b:
        # Local computational domain is a snapped subset of the native target grid.
        w=from_bounds(*line.buffer(DOMAIN_HALF_WIDTH).bounds,transform=a.transform).round_offsets().round_lengths()
        w=Window(max(0,w.col_off),max(0,w.row_off),min(a.width-w.col_off,w.width),min(a.height-w.row_off,w.height))
        transform=a.window_transform(w); native=a.read(1,window=w).astype("float32")
        h,ww=native.shape; sec=np.full_like(native,-9999.0); secmask=np.zeros(native.shape,dtype="uint8")
        reproject(rasterio.band(b,1),sec,src_transform=b.transform,src_crs=b.crs,src_nodata=b.nodata,dst_transform=transform,dst_crs=a.crs,dst_nodata=-9999.0,resampling=Resampling.bilinear)
        reproject(np.ones((b.height,b.width),dtype="uint8"),secmask,src_transform=b.transform,src_crs=b.crs,src_nodata=0,dst_transform=transform,dst_crs=a.crs,dst_nodata=0,resampling=Resampling.nearest)
        secok=(secmask==1)&np.isfinite(sec)&(sec>-9000)
        nativeok=valid(native,a.nodata); union=nativeok|secok
        bands={"corridor":line,"band250":line.buffer(250),"band500":line.buffer(500),"band750":line.buffer(750)}
        cov={}
        for name,geom in bands.items():
            n,inside=coverage(nativeok,geom,transform); s,_=coverage(secok,geom,transform); u,_=coverage(union,geom,transform)
            cov[name]={"dem12_coverage_percent":n,"dem8_coverage_percent":s,"union_coverage_percent":u,"cells":int(inside.sum())}
        # The union test is deliberately completed before the derived DEM exists.
        if cov["corridor"]["union_coverage_percent"]<100 or cov["band250"]["union_coverage_percent"]<99:
            uncovered=(~union)&geometry_mask([mapping(line.buffer(250))],out_shape=native.shape,transform=transform,invert=True)
            write_geojson(PREFLIGHT/"union_coverage_uncovered.geojson",line.buffer(250),{"status":"STOP","uncovered_cells":int(uncovered.sum())})
            raise RuntimeError("Union coverage gate failed before mosaic construction")
        profile={"driver":"GTiff","height":h,"width":ww,"count":1,"dtype":"float32","crs":a.crs,"transform":transform,"nodata":-9999.0,"compress":"deflate"}
        with rasterio.open(WORK/"dem8_bilinear_on_native12_grid.tif","w",**profile) as dst: dst.write(sec,1)
        maskprof=profile.copy();maskprof.update(dtype="uint8",nodata=0)
        with rasterio.open(WORK/"dem8_validity_nearest_on_native12_grid.tif","w",**maskprof) as dst: dst.write(secok.astype("uint8"),1)

        # Five independent overlap anchors, buffered around the already accepted valley system and excluding patch cores.
        anchors={"A_upstream_B04":(15400,16050),"B_downstream_B04":(16850,17500),"C_upstream_B05":(17650,18300),"D_downstream_B05":(18880,19450),"E_stable_midreach":(9000,10500)}
        astats={}; medians=[]
        for name,(start,end) in anchors.items():
            zone=segment(line,start,end).buffer(200)
            inside=geometry_mask([mapping(zone)],out_shape=native.shape,transform=transform,invert=True)
            both=inside&nativeok&secok
            diff=native[both]-sec[both]
            # Robustly omit cliff/edge disagreement from datum comparison only.
            if len(diff):
                m=np.median(diff); mad=np.median(np.abs(diff-m)); use=np.abs(diff-m)<=max(10.0,4*1.4826*mad)
                st=robust_stats(diff[use]); st["raw_overlap_samples"]=int(len(diff)); astats[name]=st;medians.append(st["median_dem12_minus_dem8_m"])
            else: astats[name]={"n_samples":0}
        # The requested valley-floor comparison is independent of pixelwise
        # terrain relief and is deliberately evaluated before any patch exists.
        localprof={"driver":"GTiff","height":h,"width":ww,"count":1,"dtype":"float32","crs":a.crs,"transform":transform,"nodata":-9999.0}
        with MemoryFile() as mn, MemoryFile() as ms:
            with mn.open(**localprof) as nds, ms.open(**localprof) as sds:
                nlocal=np.where(nativeok,native,-9999.0).astype("float32")
                nds.write(nlocal,1); sds.write(sec,1)
                pn=sample_profile(nds,line); ps=sample_profile(sds,line)
        floor_stats={}
        for name,(start,end) in anchors.items():
            dif=np.asarray([a0["floor"]-b0["floor"] for a0,b0 in zip(pn,ps)
                            if start<=a0["d"]<=end and np.isfinite(a0["floor"]) and np.isfinite(b0["floor"])])
            floor_stats[name]=robust_stats(dif) if len(dif) else {"n_samples":0}
        global_diff=np.concatenate([np.asarray([v["median_dem12_minus_dem8_m"]]) for v in astats.values() if v.get("n_samples",0)>0])
        bias=float(np.median(global_diff)); zone_spread=float(np.ptp(global_diff)) if len(global_diff) else float("inf")
        nmad=float(np.median([v["NMAD_m"] for v in astats.values() if v.get("n_samples",0)>0]))
        # Stable datum criterion: independent robust anchor medians agree within 10 m and are not wildly noisy.
        feasible=bool(len(global_diff)>=4 and zone_spread<=10.0 and nmad<=15.0)
        offset=0.0 if feasible else None
        compatibility={"primary_dem":str(dem12),"secondary_dem":str(dem8),"target_grid":"native 12.5 m snapped local ±750 m computational domain","anchor_zones":astats,"robust_valley_floor_differences":floor_stats,"dem12_dem8_vertical_bias_m":bias if feasible else None,"zone_median_spread_m":zone_spread,"compatibility_nmad_m":nmad if feasible else None,"patch_feasibility":"PASS" if feasible else "FAIL","vertical_bias_applied_m":offset,"policy":"No vertical correction applied: stable overlap bias is treated as small enough for direct secondary elevations." if feasible else "No mosaic: overlap bias is spatially inconsistent."}
        (REPORT/"DEM12_DEM8_COMPATIBILITY.json").write_text(json.dumps(compatibility,indent=2)+"\n",encoding="utf-8")
        (REPORT/"DEM12_DEM8_COMPATIBILITY.md").write_text("# DEM12--DEM8 compatibility\n\n"+f"`PATCH_FEASIBILITY = {compatibility['patch_feasibility']}`. Five independent, 200 m valley-system overlap anchors were assessed outside the B04/B05 cores. Robust anchor-median bias (DEM12 minus DEM8) is {bias:.3f} m; anchor-median spread is {zone_spread:.3f} m; median within-anchor NMAD is {nmad:.3f} m. Vertical correction applied: {offset if offset is not None else 'none'} m.\n\n"+json.dumps(astats,indent=2)+"\n",encoding="utf-8")
        if not feasible:
            # A failure report is itself a reproducible Gate-B result.  It
            # explicitly records that no derived DEM, patch, or provenance
            # raster was constructed under the stated Case-C policy.
            failure_mosaic={"primary_dem":str(dem12),"secondary_dem":str(dem8),"patch_feasibility":"FAIL","dem12_dem8_vertical_bias_m":None,"vertical_bias_applied_m":None,"compatibility_nmad_m":None,"source_union_coverage_corridor_percent":cov["corridor"]["union_coverage_percent"],"source_union_coverage_band250_percent":cov["band250"]["union_coverage_percent"],"source_union_coverage_band500_percent":cov["band500"]["union_coverage_percent"],"source_coverage":cov,"computational_dem_coverage_corridor_percent":None,"computational_dem_coverage_band250_percent":None,"b04_replaced":False,"b05_replaced":False,"nodata_cells_filled":0,"artifact_cells_replaced":0,"transition_cells":0,"derived_dem_sha256":None,"derived_dem_local_path":None,"source_provenance_summary":None,"stop_reason":"Case C: DEM12--DEM8 overlap bias is spatially inconsistent across independent anchors; a single vertical correction is not defensible."}
            (REPORT/"JILONG_COMPUTATIONAL_DEM_MOSAIC.json").write_text(json.dumps(failure_mosaic,indent=2)+"\n",encoding="utf-8")
            (REPORT/"JILONG_COMPUTATIONAL_DEM_MOSAIC.md").write_text("# Jilong computational DEM mosaic\n\n`PATCH_FEASIBILITY = FAIL`; no computational DEM was created. Source-union coverage passed the required corridor/±250 m gate, but the requested compatibility test reached Case C: independent overlap-anchor medians are spatially inconsistent. No vertical correction, artifact replacement, transition, source-provenance raster, or derived TIFF was made.\n\n"+json.dumps({"coverage":cov,"compatibility":compatibility},indent=2)+"\n",encoding="utf-8")
            final_failure={"gate_a":"PASS","gate_b":"FAIL","native_12p5_simulation_ready":False,"computational_mosaic_simulation_ready":False,"b01_status":"INSUFFICIENT_EVIDENCE","b02_status":"INSUFFICIENT_EVIDENCE","b03_status":"INSUFFICIENT_EVIDENCE","b04_status":"INSUFFICIENT_EVIDENCE","b05_status":"INSUFFICIENT_EVIDENCE","b06_status":"INSUFFICIENT_EVIDENCE","32m_area_mean_barrier_count":None,"32m_bilinear_barrier_count":None,"64m_area_mean_barrier_count":None,"64m_bilinear_barrier_count":None,"remaining_high_severity_artificial_barriers":["B04 and B05 remain unresolved because the proposed secondary source cannot be vertically reconciled with the primary DEM by a defensible constant correction."],"recommended_next_step":"Obtain or validate a spatially compatible authoritative terrain source for B04/B05; do not apply an ad hoc, spatially varying correction."}
            (REPORT/"JILONG_TERRAIN_PREFLIGHT_FINAL.json").write_text(json.dumps(final_failure,indent=2)+"\n",encoding="utf-8")
            (REPORT/"JILONG_TERRAIN_PREFLIGHT_FINAL.md").write_text("# Final computational-terrain Gate B preflight\n\n`GATE_A = PASS` (unchanged). `GATE_B = FAIL` for this proposed mosaic only: Case-C DEM compatibility failed before any computational terrain was generated. No D-Claw calculation was run.\n\n"+json.dumps(final_failure,indent=2)+"\n",encoding="utf-8")
            fig,ax=plt.subplots(figsize=(10,14)); ax.imshow(union,extent=(transform.c,transform.c+ww*transform.a,transform.f-h*transform.a,transform.f),cmap="Greens",alpha=.8); ax.plot(*line.xy,color="black",lw=1); ax.set(title="Source-union coverage before compatibility stop",aspect="equal");fig.tight_layout();fig.savefig(FIG/"36_dem_source_coverage_union.png",dpi=180);plt.close(fig)
            print(json.dumps({"gate_b":"FAIL","coverage":cov,"compatibility":compatibility},indent=2))
            return

        patch_geoms={}
        for bid,item in PATCH_DEFS.items():
            start=item["start"]-150; end=item["end"]+150; geom=segment(line,start,end).buffer(250); patch_geoms[bid]=geom
            props={"patch_id":bid,"reason":item["reason"],"source_primary":"native 12.5 m DEM","source_secondary":"official 8 m DEM","station_start_m":start,"station_end_m":end,"lateral_half_width_m":250.0,"human_corridor_geometry_unchanged":True,"vertical_offset_applied_m":offset,"resampling_method":"bilinear elevation; nearest validity mask","notes":"Objective documented failure tile, not a channel trace."}
            write_geojson(PATCHES/f"{bid}_patch_mask.geojson",geom,props)
        nodata_geom=line.buffer(DOMAIN_HALF_WIDTH)
        write_geojson(PATCHES/"nodata_fill_mask.geojson",nodata_geom,{"patch_id":"NODATA_FILL","reason":"Native NoData completed only where secondary DEM is valid","source_primary":"native 12.5 m DEM","source_secondary":"official 8 m DEM","station_start_m":0.0,"station_end_m":line.length,"lateral_half_width_m":DOMAIN_HALF_WIDTH,"human_corridor_geometry_unchanged":True,"vertical_offset_applied_m":offset,"resampling_method":"bilinear elevation; nearest validity mask","notes":"Mask is constrained to computational domain; effective cells require native NoData and secondary validity."})

        comp=native.copy(); provenance=np.where(nativeok,1,0).astype("uint8")
        fill=(~nativeok)&secok; comp[fill]=sec[fill]+offset; provenance[fill]=2
        patch_masks={bid:geometry_mask([mapping(g)],out_shape=native.shape,transform=transform,invert=True) for bid,g in patch_geoms.items()}
        for code,bid in ((3,"B04"),(4,"B05")):
            use=patch_masks[bid]&secok; comp[use]=sec[use]+offset; provenance[use]=code
        # Test hard seam before blending.  Blend only if the unbiased source difference at inner boundary exceeds 5 m.
        transition_width=0.0; seam_summary={}
        yy,xx=np.indices(native.shape); xs=transform.c+(xx+.5)*transform.a; ys=transform.f+(yy+.5)*transform.e
        for code,bid in ((3,"B04"),(4,"B05")):
            geom=patch_geoms[bid]; boundary=np.vectorize(lambda x,y: Point(float(x),float(y)).distance(geom.boundary))(xs,ys)
            seam=patch_masks[bid]&(boundary<=25)&nativeok&secok
            diff=(native-(sec+offset))[seam]; seam_summary[bid]={"hard_transition_n":int(diff.size),"hard_transition_median_abs_difference_m":float(np.median(np.abs(diff))) if diff.size else None}
            if diff.size and np.median(np.abs(diff))>5:
                transition_width=max(transition_width,75.0)
                band=patch_masks[bid]&secok&nativeok&(boundary<=75)
                weight=np.clip(boundary[band]/75.0,0,1)
                comp[band]=(1-weight)*native[band]+weight*(sec[band]+offset); provenance[band]=5
        compok=valid(comp,-9999.0)
        comp[~compok]=NODATA
        outprof={"driver":"GTiff","height":h,"width":ww,"count":1,"dtype":"uint16","crs":a.crs,"transform":transform,"nodata":NODATA,"compress":"deflate","tiled":True,"blockxsize":256,"blockysize":256}
        outpath=OUT/"jilong_computational_dem_v1.tif"
        with rasterio.open(outpath,"w",**outprof) as dst:
            dst.write(np.clip(np.rint(comp),0,NODATA).astype("uint16"),1)
            dst.update_tags(PRIMARY_SOURCE="native 12.5 m DEM",SECONDARY_SOURCE="official 8 m DEM",DERIVATION="NoData completion plus localized B04/B05 documented artifact replacement; no hydrologic carving, filling, monotonic enforcement, or corridor modification.")
        provprof=outprof.copy();provprof.update(dtype="uint8",nodata=0)
        with rasterio.open(OUT/"jilong_computational_dem_source_map.tif","w",**provprof) as dst: dst.write(provenance,1)

    derived_hash=sha256(outpath)
    with rasterio.open(outpath) as compds:
        compmask=valid(compds.read(1),compds.nodata); finalcov={}
        for name,geom in bands.items(): finalcov[name]=coverage(compmask,geom,transform)[0]
        rows=sample_profile(compds,line); native_rows=None
    d=np.asarray([r["d"] for r in rows]); z=np.asarray([r["floor"] for r in rows]); mosaic_barriers=candidates(rows)
    # Grid-scale diagnostics use continuous average and bilinear resampling, never a maximum envelope.
    grid={}
    for scale in (32,64):
        grid[str(scale)]={}
        for name,method in (("area_mean",Resampling.average),("bilinear",Resampling.bilinear)):
            rr=resampled_profile(comp,transform,CRS,NODATA,line,scale,method); cc=candidates(rr); grid[str(scale)][name]={"barrier_count":len(cc),"barriers":cc,"max_250m_adverse_rise_m":rises(np.asarray([x["floor"] for x in rr]),np.asarray([x["d"] for x in rr]))["250"]}
    # Explicit B04/B05 before/after results use the already documented native barriers as the before record.
    old=json.loads((REPORT/"JILONG_TERRAIN_PREFLIGHT.json").read_text(encoding="utf-8")); old_by={x["barrier_id"]:x for x in old["candidate_false_barriers"]}
    verify={}
    for bid,item in PATCH_DEFS.items():
        within=[x for x in mosaic_barriers if item["start"]-50<=x["crest_station_m"]<=item["end"]+50]
        before=old_by[bid]; idx=(d>=item["start"]-150)&(d<=item["end"]+150)
        verify[bid]={"before":before,"after":{"robust_floor_profile_min_m":float(np.nanmin(z[idx])),"robust_floor_profile_max_m":float(np.nanmax(z[idx])),"barriers":within,"barrier_height_m":max([x["barrier_height_m"] for x in within],default=0.0),"barrier_length_m":max([x["barrier_length_m"] for x in within],default=0.0),"adverse_rises_m":rises(z,d),"diagnostics_32m":grid["32"],"diagnostics_64m":grid["64"]},"resolved":not any(x["barrier_height_m"]>=150 for x in within)}
    statuses={"B01":"NON_BLOCKING","B02":"NON_BLOCKING","B03":"REAL_TERRAIN_FEATURE","B04":"RESOLVED" if verify["B04"]["resolved"] else "REMAINS_BLOCKING","B05":"RESOLVED" if verify["B05"]["resolved"] else "REMAINS_BLOCKING","B06":"NON_BLOCKING"}
    high=[x for x in mosaic_barriers if x["barrier_height_m"]>=150 and not (PATCH_DEFS["B04"]["start"]-50<=x["crest_station_m"]<=PATCH_DEFS["B04"]["end"]+50 or PATCH_DEFS["B05"]["start"]-50<=x["crest_station_m"]<=PATCH_DEFS["B05"]["end"]+50)]
    ready=finalcov["corridor"]>=99.999 and finalcov["band250"]>=99 and verify["B04"]["resolved"] and verify["B05"]["resolved"] and not high and not grid["32"]["area_mean"]["barrier_count"] and not grid["32"]["bilinear"]["barrier_count"] and not grid["64"]["area_mean"]["barrier_count"] and not grid["64"]["bilinear"]["barrier_count"]
    gate="PASS" if ready else "CONDITIONAL_TERRAIN_REMEDIATION_REQUIRED"
    counts={str(i):int((provenance==i).sum()) for i in range(1,6)}; total=int((provenance>0).sum())
    provenance_summary={"codes":{"1":"ORIGINAL_NATIVE_12P5","2":"DEM8_NODATA_FILL","3":"DEM8_B04_REPLACEMENT","4":"DEM8_B05_REPLACEMENT","5":"DEM_SOURCE_TRANSITION"},"cell_counts":counts,"area_m2_by_class":{k:v*156.25 for k,v in counts.items()},"percent_by_class":{k:100*v/total for k,v in counts.items()},"patch_extents":{k:list(v.bounds) for k,v in patch_geoms.items()},"transition_width_m":transition_width,"seam_summary":seam_summary}
    mosaic_report={"primary_dem":str(dem12),"secondary_dem":str(dem8),"patch_feasibility":"PASS","dem12_dem8_vertical_bias_m":bias,"vertical_bias_applied_m":offset,"compatibility_nmad_m":nmad,"source_union_coverage_corridor_percent":cov["corridor"]["union_coverage_percent"],"source_union_coverage_band250_percent":cov["band250"]["union_coverage_percent"],"source_union_coverage_band500_percent":cov["band500"]["union_coverage_percent"],"source_coverage":cov,"computational_dem_coverage_corridor_percent":finalcov["corridor"],"computational_dem_coverage_band250_percent":finalcov["band250"],"computational_dem_coverage_band500_percent":finalcov["band500"],"computational_dem_coverage_band750_percent":finalcov["band750"],"b04_replaced":True,"b05_replaced":True,"nodata_cells_filled":counts["2"],"artifact_cells_replaced":counts["3"]+counts["4"],"transition_cells":counts["5"],"derived_dem_sha256":derived_hash,"derived_dem_local_path":str(outpath),"source_provenance_summary":provenance_summary}
    (REPORT/"JILONG_COMPUTATIONAL_DEM_MOSAIC.json").write_text(json.dumps(mosaic_report,indent=2)+"\n",encoding="utf-8")
    (REPORT/"JILONG_COMPUTATIONAL_DEM_MOSAIC.md").write_text("# Jilong computational DEM mosaic\n\nThis derived local raster preserves the native 12.5 m target grid and source NoData convention. It uses native terrain as primary, uses official 8 m terrain for native NoData completion, and replaces only documented B04/B05 artifact tiles. No source DEM was modified; no channel was burned, filled, breached, smoothed, or made monotonic.\n\n"+f"`PATCH_FEASIBILITY = PASS`; robust DEM12-minus-DEM8 bias = {bias:.3f} m; applied correction = {offset:.3f} m; compatibility NMAD = {nmad:.3f} m. Derived DEM SHA256: `{derived_hash}`. Local path: `{outpath}`.\n\n"+json.dumps(provenance_summary,indent=2)+"\n",encoding="utf-8")
    final={"gate_a":"PASS","gate_b":gate,"native_12p5_simulation_ready":False,"computational_mosaic_simulation_ready":ready,**{f"{k.lower()}_status":v for k,v in statuses.items()},"max_250m_adverse_rise_native12":old["dem12_robust_floor_max_rise_250m"],"max_250m_adverse_rise_mosaic":rises(z,d)["250"],"32m_area_mean_barrier_count":grid["32"]["area_mean"]["barrier_count"],"32m_bilinear_barrier_count":grid["32"]["bilinear"]["barrier_count"],"64m_area_mean_barrier_count":grid["64"]["area_mean"]["barrier_count"],"64m_bilinear_barrier_count":grid["64"]["bilinear"]["barrier_count"],"remaining_high_severity_artificial_barriers":high,"b04_b05_verification":verify,"mosaic_barriers":mosaic_barriers,"grid_diagnostics":grid,"recommended_next_step":"Gate B is PASS: preserve this immutable derived computational DEM and proceed only to the separately authorized source/boundary-condition design; do not run D-Claw in this task." if ready else "Inspect the listed remaining terrain evidence; do not make an additional ad hoc terrain repair."}
    (REPORT/"JILONG_TERRAIN_PREFLIGHT_FINAL.json").write_text(json.dumps(final,indent=2)+"\n",encoding="utf-8")
    (REPORT/"JILONG_TERRAIN_PREFLIGHT_FINAL.md").write_text("# Final computational-terrain Gate B preflight\n\n`GATE_A = PASS` (unchanged). `GATE_B = "+gate+"`. The derived DEM was tested at 50 m stations and ±250 m cross-sections with robust floor defined as the median of the lowest five valid samples. 32/64 m diagnostics separately use area-mean continuous-elevation aggregation and bilinear/cell-centre resampling; neither uses a maximum envelope or terrain lowering.\n\n"+json.dumps(final,indent=2)+"\n",encoding="utf-8")
    # Evidence figures: source coverage, B04/B05 profiles, longitudinal profile, grid diagnostics, provenance.
    fig,ax=plt.subplots(figsize=(10,14)); ax.imshow(union,extent=(transform.c,transform.c+ww*transform.a,transform.f-h*transform.a,transform.f),cmap="Greens",alpha=.8); ax.plot(*line.xy,color="black",lw=1); [ax.plot(*g.exterior.xy,color="red",lw=1) for g in patch_geoms.values()]; ax.set(title="DEM source-union coverage and documented patch tiles",aspect="equal");fig.tight_layout();fig.savefig(FIG/"36_dem_source_coverage_union.png",dpi=180);plt.close(fig)
    for bid,num in (("B04",37),("B05",38)):
        item=PATCH_DEFS[bid]; ix=(d>=item["start"]-400)&(d<=item["end"]+400); oldprof=old_by[bid]
        fig,ax=plt.subplots(figsize=(12,4));ax.plot(d[ix]/1000,z[ix],color="#0072b2",label="computational mosaic robust floor");ax.axvspan((item["start"]-150)/1000,(item["end"]+150)/1000,color="orange",alpha=.2,label="documented replacement tile");ax.axhline(oldprof["barrier_height_m"],color="gray",ls=":",label="native barrier-height reference (not elevation)");ax.set(title=f"{bid}: documented native-artifact window, before/after terrain evidence",xlabel="S0 distance (km)",ylabel="robust floor elevation (m)");ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(FIG/f"{num}_{bid}_before_after_terrain.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(15,5));ax.plot(d/1000,z,color="#0072b2",lw=1,label="computational mosaic robust floor");ax.set(title="Computational DEM longitudinal Gate B preflight",xlabel="S0 distance (km)",ylabel="Elevation (m)");ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(FIG/"39_computational_dem_longitudinal_preflight.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(15,5));ax.plot(d/1000,z,color="black",lw=.7,label="12.5 m computational grid");
    for scale,color in (("32","#009e73"),("64","#d55e00")):
        rr=resampled_profile(comp,transform,CRS,NODATA,line,float(scale),Resampling.average);ax.plot(np.asarray([x["d"] for x in rr])/1000,np.asarray([x["floor"] for x in rr]),color=color,lw=1,label=f"{scale} m area-mean")
    ax.set(title="Realistic 32/64 m continuous-elevation diagnostics",xlabel="S0 distance (km)",ylabel="robust floor elevation (m)");ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(FIG/"40_computational_dem_32m_64m_diagnostics.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,14));ax.imshow(provenance,extent=(transform.c,transform.c+ww*transform.a,transform.f-h*transform.a,transform.f),cmap="viridis",vmin=0,vmax=5);ax.plot(*line.xy,color="white",lw=1);ax.set(title="Computational DEM source provenance (codes 1--5)",aspect="equal");fig.tight_layout();fig.savefig(FIG/"41_computational_dem_source_provenance.png",dpi=180);plt.close(fig)
    print(json.dumps({"gate_b":gate,"sha256":derived_hash,"coverage":finalcov,"compatibility":compatibility,"final":final},indent=2))


if __name__=="__main__": main()
