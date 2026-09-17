"""Single-source Copernicus GLO-30 computational terrain and final Gate-B QA.

Only the specified GLO-30 raster contributes elevations.  This script opens
the canonical corridor read-only; it neither conditions terrain nor runs a
flow solver.
"""
from __future__ import annotations

import csv
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
from shapely.geometry import LineString, Point, mapping

ROOT=Path(r"E:\Alfred\Jilong_GateA")
SOURCE=ROOT/"copernicus_glo30"/"Copernicus_DSM_COG_10_N28_00_E085_00_DEM.tif"
CORRIDOR=ROOT/"corridor"; REPORT=ROOT/"reports"; FINAL=ROOT/"terrain_final"; FIG=ROOT/"figures"; TERRAIN=ROOT/"computational_terrain"
CRS="EPSG:32645"; NODATA=-9999.0; BUFFER=2000.0; MARGIN=100.0; STATION=50.0; HALF=250.0
S0=Point(334051.79369854234,3143392.492568097); S4=Point(340571.8257777202,3129640.775894154)

def valid(v,nodata=NODATA): return np.isfinite(v)&(v!=nodata)

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def dists(line):
    d=np.arange(0,line.length+STATION,STATION); d[-1]=line.length; return d

def xsection(line,d,half=HALF):
    p=line.interpolate(float(d)); a=line.interpolate(max(0,d-25)); b=line.interpolate(min(line.length,d+25)); dx,dy=b.x-a.x,b.y-a.y; q=math.hypot(dx,dy); nx,ny=-dy/q,dx/q
    return LineString([(p.x-nx*half,p.y-ny*half),(p.x+nx*half,p.y+ny*half)]),p

def profile(ds,line):
    out=[]
    for d in dists(line):
        sec,p=xsection(line,d); offs=np.arange(-HALF,HALF+ds.res[0]*.51,ds.res[0]); pts=[sec.interpolate((o+HALF)/(2*HALF),normalized=True).coords[0] for o in offs]
        z=np.asarray([r[0] for r in ds.sample(pts)],dtype=float); good=valid(z,ds.nodata); vals=z[good]; goodoffs=offs[good]
        lowest=np.argsort(vals)[:5] if len(vals) else []
        robust=float(np.median(vals[lowest])) if len(lowest)>=5 else np.nan
        ci=int(np.argmin(abs(offs))); centre=float(z[ci]) if good[ci] else np.nan
        out.append({"station_m":float(d),"x":p.x,"y":p.y,"centre":centre,"raw_min":float(vals.min()) if len(vals) else np.nan,"robust":robust,"median":float(np.median(vals)) if len(vals) else np.nan,"relief":float(vals.max()-vals.min()) if len(vals) else np.nan,"valid_fraction":float(good.mean()),"robust_offset":float(np.median(goodoffs[lowest])) if len(lowest)>=5 else np.nan})
    return out

def rise(z,d,w):
    a=[]
    for i,x in enumerate(d):
        j=int(np.searchsorted(d,x+w,side="left"));a.append(z[j]-z[i] if j<len(z) and np.isfinite(z[i]) and np.isfinite(z[j]) else np.nan)
    return None if not np.isfinite(a).any() else float(np.nanmax(a))

def rise_set(rows):
    z=np.asarray([x["robust"] for x in rows]);d=np.asarray([x["station_m"] for x in rows]);return {str(w):rise(z,d,w) for w in (50,100,250,500,1000)}

def barriers(rows):
    d=np.asarray([x["station_m"] for x in rows]);z=np.asarray([x["robust"] for x in rows]);out=[]; flank=10
    for i in range(flank,len(z)-flank):
        if not np.isfinite(z[i]):continue
        up,down=z[i-flank:i+1],z[i:i+flank+1]
        if not(np.isfinite(up).any() and np.isfinite(down).any()):continue
        height=min(z[i]-np.nanmin(up),z[i]-np.nanmin(down))
        if height<60:continue
        th=z[i]-height*.5;left=right=i
        while left and np.isfinite(z[left-1]) and z[left-1]>=th:left-=1
        while right+1<len(z) and np.isfinite(z[right+1]) and z[right+1]>=th:right+=1
        if d[right]-d[left]<150:continue
        if any(abs(d[i]-q["crest_station_m"])<300 for q in out):continue
        sev="HIGH" if height>=150 else "MODERATE" if height>=90 else "LOW"
        out.append({"start_station_m":float(d[left]),"crest_station_m":float(d[i]),"end_station_m":float(d[right]),"barrier_height_m":float(height),"barrier_length_m":float(d[right]-d[left]),"severity":sev,"x":rows[i]["x"],"y":rows[i]["y"]})
    return out

def cov(mask,geom,transform):
    inside=geometry_mask([mapping(geom)],out_shape=mask.shape,transform=transform,invert=True);return 100.0*(mask&inside).sum()/inside.sum()

def write_raster(path,arr,transform,crs,res,nodata=NODATA):
    p={"driver":"GTiff","height":arr.shape[0],"width":arr.shape[1],"count":1,"dtype":"float32","crs":crs,"transform":transform,"nodata":nodata,"compress":"deflate","tiled":True,"blockxsize":256,"blockysize":256}
    with rasterio.open(path,"w",**p) as ds:
        ds.write(arr.astype("float32"),1);ds.update_tags(TERRAIN_SOURCE="Copernicus GLO-30 only",DERIVATION=f"Standard bilinear projection or standard average resampling at {res} m; no terrain conditioning.")
    return p

def resample(base,transform,resolution,method):
    h=max(1,math.ceil(base.shape[0]*transform.a/resolution));w=max(1,math.ceil(base.shape[1]*transform.a/resolution));tr=Affine(resolution,0,transform.c,0,-resolution,transform.f);dst=np.full((h,w),NODATA,dtype="float32")
    reproject(base,dst,src_transform=transform,src_crs=CRS,src_nodata=NODATA,dst_transform=tr,dst_crs=CRS,dst_nodata=NODATA,resampling=method);return dst,tr

def mem_profile(arr,transform,line):
    p={"driver":"GTiff","height":arr.shape[0],"width":arr.shape[1],"count":1,"dtype":"float32","crs":CRS,"transform":transform,"nodata":NODATA}
    with MemoryFile() as m:
        with m.open(**p) as ds:ds.write(arr,1);return profile(ds,line)

def hill(a):
    z=a.astype(float); z[~valid(z)]=np.nanmedian(z[valid(z)]);gy,gx=np.gradient(z);slope=np.pi/2-np.arctan(np.hypot(gx,gy));aspect=np.arctan2(-gx,gy);az=np.deg2rad(315);alt=np.deg2rad(45);return np.clip(255*(np.sin(alt)*np.sin(slope)+np.cos(alt)*np.cos(slope)*np.cos(az-aspect)),0,255)

def main():
    if not SOURCE.is_file(): raise FileNotFoundError("FILE_NOT_FOUND: "+str(SOURCE))
    for p in (REPORT,FINAL,FIG,TERRAIN):p.mkdir(parents=True,exist_ok=True)
    line=gpd.read_file(CORRIDOR/"jilong_canonical_corridor.geojson").to_crs(CRS).geometry.iloc[0]
    if Point(line.coords[0]).distance(S0)>.01 or Point(line.coords[-1]).distance(S4)>.01:raise RuntimeError("Canonical corridor endpoint mismatch")
    domain=line.buffer(BUFFER); write_domain=FINAL/"jilong_dclaw_domain.geojson"; write_domain.write_text(json.dumps({"type":"FeatureCollection","features":[{"type":"Feature","properties":{"domain_buffer_m":BUFFER,"canonical_corridor_geometry_unchanged":True,"terrain_source":"Copernicus GLO-30 only"},"geometry":mapping(domain)}]},indent=2)+"\n",encoding="utf-8")
    with rasterio.open(SOURCE) as src:
        source_data=src.read(1);srcvalid=valid(source_data,src.nodata)
        meta={"source_path":str(SOURCE),"crs":str(src.crs),"width":src.width,"height":src.height,"size":[src.width,src.height],"resolution":list(src.res),"bounds":list(src.bounds),"nodata":src.nodata,"dtype":src.dtypes[0],"valid_data_fraction":float(srcvalid.mean()),"transform":list(src.transform),"tags":src.tags(),"vertical_metadata":{k:v for k,v in src.tags().items() if any(x in k.lower() for x in ("vertical","datum","height","elev"))} or None}
        (REPORT/"COPERNICUS_GLO30_SOURCE_METADATA.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
        minx,miny,maxx,maxy=domain.bounds;left=math.floor((minx-MARGIN)/30)*30;right=math.ceil((maxx+MARGIN)/30)*30;bottom=math.floor((miny-MARGIN)/30)*30;top=math.ceil((maxy+MARGIN)/30)*30;tr=Affine(30,0,left,0,-30,top);width=int(round((right-left)/30));height=int(round((top-bottom)/30));base=np.full((height,width),NODATA,dtype="float32")
        reproject(rasterio.band(src,1),base,src_transform=src.transform,src_crs=src.crs,src_nodata=src.nodata,dst_transform=tr,dst_crs=CRS,dst_nodata=NODATA,resampling=Resampling.bilinear)
    basepath=TERRAIN/"jilong_copernicus_glo30_utm45_v1.tif";write_raster(basepath,base,tr,CRS,30)
    arr32,tr32=resample(base,tr,32,Resampling.average);arr64,tr64=resample(base,tr,64,Resampling.average);p32=TERRAIN/"jilong_copernicus_32m_v1.tif";p64=TERRAIN/"jilong_copernicus_64m_v1.tif";write_raster(p32,arr32,tr32,CRS,32);write_raster(p64,arr64,tr64,CRS,64)
    mask=valid(base); cover={"corridor":cov(mask,line,tr),"band250":cov(mask,line.buffer(250),tr),"band500":cov(mask,line.buffer(500),tr),"band1000":cov(mask,line.buffer(1000),tr),"domain2000":cov(mask,domain,tr)}
    # Coverage stop applies before downstream terrain interpretations.
    if cover["corridor"]<100 or cover["band250"]<99.999:
        gate="FAIL";ready=False; native_rows=[];b_native=[];qa={};high=[]
    else:
        with rasterio.open(basepath) as ds:native_rows=profile(ds,line)
        rows32=mem_profile(arr32,tr32,line);rows64=mem_profile(arr64,tr64,line);bil32=mem_profile(resample(base,tr,32,Resampling.bilinear)[0],resample(base,tr,32,Resampling.bilinear)[1],line);bil64=mem_profile(resample(base,tr,64,Resampling.bilinear)[0],resample(base,tr,64,Resampling.bilinear)[1],line)
        b_native=barriers(native_rows);b32=barriers(rows32);b64=barriers(rows64);bb32=barriers(bil32);bb64=barriers(bil64)
        def support(q,groups):return any(abs(q["crest_station_m"]-x["crest_station_m"])<=250 and x["severity"]=="HIGH" for g in groups for x in g)
        high=[]
        for q in b_native:
            if q["severity"]=="HIGH" and support(q,(b32,bb32)) and support(q,(b64,bb64)):high.append(q)
        qa={"native_rises":rise_set(native_rows),"area32_rises":rise_set(rows32),"area64_rises":rise_set(rows64),"bil32_rises":rise_set(bil32),"bil64_rises":rise_set(bil64),"native_barriers":b_native,"barriers32_area_mean":b32,"barriers64_area_mean":b64,"barriers32_bilinear":bb32,"barriers64_bilinear":bb64}
        gate="PASS" if not high else "FAIL";ready=gate=="PASS"
    # Required CSV is produced even on a coverage failure (then it has only a header).
    fields=["station_m","x","y","centreline_elevation_m","raw_min_elevation_m","robust_valley_floor_elevation_m","median_elevation_m","relief_m","valid_fraction","robust_low_point_lateral_offset_m"]
    with (FINAL/"jilong_glo30_valley_profile.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in native_rows:w.writerow({"station_m":f"{r['station_m']:.3f}","x":f"{r['x']:.3f}","y":f"{r['y']:.3f}",**{out:"" if not np.isfinite(r[inn]) else f"{r[inn]:.3f}" for out,inn in (("centreline_elevation_m","centre"),("raw_min_elevation_m","raw_min"),("robust_valley_floor_elevation_m","robust"),("median_elevation_m","median"),("relief_m","relief"),("valid_fraction","valid_fraction"),("robust_low_point_lateral_offset_m","robust_offset"))}})
    hashes={"projected_base_sha256":sha(basepath),"terrain_32m_sha256":sha(p32),"terrain_64m_sha256":sha(p64)}
    if native_rows:
        z=np.asarray([r["robust"] for r in native_rows]);s0=float(z[0]);s4=float(z[-1]);drop=s0-s4;slope=(s4-s0)/line.length
    else:s0=s4=drop=slope=None
    final={"gate_a":"PASS","gate_b":gate,"terrain_source":"Copernicus GLO-30","single_source_terrain":True,"analysis_crs":CRS,"canonical_corridor_length_km":line.length/1000,"domain_buffer_m":BUFFER,"source_resolution":meta["resolution"],"projected_base_path":str(basepath),"terrain_32m_path":str(p32),"terrain_64m_path":str(p64),**hashes,"coverage_corridor_percent":cover["corridor"],"coverage_band250_percent":cover["band250"],"coverage_band500_percent":cover["band500"],"coverage_band1000_percent":cover["band1000"],"coverage_domain2000_percent":cover["domain2000"],"s0_robust_floor_elevation_m":s0,"s4_robust_floor_elevation_m":s4,"total_floor_drop_m":drop,"mean_robust_floor_slope":slope,"max_rise_250m_native_glo30":qa.get("native_rises",{}).get("250"),"max_rise_500m_native_glo30":qa.get("native_rises",{}).get("500"),"candidate_barrier_count":len(b_native),"high_barrier_count":sum(x["severity"]=="HIGH" for x in b_native),"32m_area_mean_barrier_count":len(qa.get("barriers32_area_mean",[])),"32m_bilinear_barrier_count":len(qa.get("barriers32_bilinear",[])),"64m_area_mean_barrier_count":len(qa.get("barriers64_area_mean",[])),"64m_bilinear_barrier_count":len(qa.get("barriers64_bilinear",[])),"remaining_high_blocking_barriers":high,"computational_terrain_simulation_ready":ready,"gate_a_reopened":False,"methods":{"base_projection":"bilinear reprojection of specified source only","32m_64m_products":"area-weighted/average resampling","independent_diagnostics":"bilinear/cell-centre profiles","terrain_manipulation":"none","robust_floor":"median of lowest five valid cross-section samples at approximately native projected-grid spacing"},"qa":qa,"recommended_next_step":"Proceed to D-Claw source/boundary-condition design and six initial 64 m historical-reconstruction runs. Do not modify the accepted terrain unless a solver-level defect is demonstrated." if ready else "GLO-30 contains the listed unresolved high blocking barrier; do not patch it or switch terrain source."}
    (REPORT/"JILONG_GLO30_GATE_B_FINAL.json").write_text(json.dumps(final,indent=2)+"\n",encoding="utf-8")
    (REPORT/"JILONG_GLO30_GATE_B_FINAL.md").write_text("# Jilong single-source Copernicus GLO-30 Gate B\n\n`GATE_A = PASS` unchanged; `GATE_B = "+gate+"`. The final terrain candidate is derived from the specified Copernicus GLO-30 source alone by standard bilinear projection, then standard area-average 32/64 m products. No older DEM contributed elevations; no terrain was carved, filled, smoothed, breached, or made monotonic. No D-Claw computation was run.\n\n"+json.dumps(final,indent=2)+"\n",encoding="utf-8")
    # Figures are QA evidence of the homogeneous computational product.
    extent=(tr.c,tr.c+base.shape[1]*tr.a,tr.f-base.shape[0]*tr.a,tr.f); controls=gpd.read_file(CORRIDOR/"jilong_control_sections.geojson").to_crs(CRS)
    fig,ax=plt.subplots(figsize=(10,14));ax.imshow(mask,extent=extent,cmap="Greens");ax.plot(*domain.exterior.xy,color="white",lw=1.2,label="2000 m domain");ax.plot(*line.xy,color="black",lw=1.3,label="canonical corridor");ax.scatter([S0.x,S4.x],[S0.y,S4.y],c=["red","lime"],edgecolors="black",zorder=3);ax.legend(fontsize=8);ax.set(title="GLO-30 final domain and valid-terrain coverage",aspect="equal");fig.tight_layout();fig.savefig(FIG/"42_glo30_final_domain.png",dpi=180);plt.close(fig)
    if native_rows:
        d=np.asarray([r["station_m"] for r in native_rows]);centre=np.asarray([r["centre"] for r in native_rows]);floor=np.asarray([r["robust"] for r in native_rows])
        fig,ax=plt.subplots(figsize=(15,5));ax.plot(d/1000,centre,color="gray",lw=.7,label="centreline");ax.plot(d/1000,floor,color="#0072b2",lw=1.2,label="robust valley floor");
        for _,r in controls.iterrows():
            geom=r.geometry; hit=line.intersection(geom)
            p=hit if hit.geom_type=="Point" else geom.centroid
            near=line.project(p);ax.axvline(near/1000,color="black",lw=.5,alpha=.4);ax.text(near/1000,ax.get_ylim()[1],str(r.get("section_id","control")),fontsize=7,ha="center",va="top")
        ax.legend();ax.grid(alpha=.25);ax.set(title="Copernicus GLO-30 longitudinal terrain QA",xlabel="S0 distance (km)",ylabel="Elevation (m)");fig.tight_layout();fig.savefig(FIG/"43_glo30_final_longitudinal_profile.png",dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(15,5));ax.plot(d/1000,floor,color="black",lw=.8,label="projected GLO-30");
        for arrx,trx,label,color in ((arr32,tr32,"32 m area mean","#009e73"),(arr64,tr64,"64 m area mean","#d55e00")):
            rr=mem_profile(arrx,trx,line);ax.plot(np.asarray([x["station_m"] for x in rr])/1000,np.asarray([x["robust"] for x in rr]),color=color,lw=1,label=label)
        ax.legend();ax.grid(alpha=.25);ax.set(title="Single-source 32/64 m terrain comparison",xlabel="S0 distance (km)",ylabel="robust floor elevation (m)");fig.tight_layout();fig.savefig(FIG/"44_glo30_32m_64m_comparison.png",dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(15,5));ax.plot(d/1000,floor,color="black",lw=.8,label="robust floor");
        for q in b_native:ax.scatter(q["crest_station_m"]/1000,floor[np.argmin(abs(d-q["crest_station_m"]))],color="red" if q["severity"]=="HIGH" else "orange",zorder=3);ax.annotate(q["severity"],(q["crest_station_m"]/1000,floor[np.argmin(abs(d-q["crest_station_m"]))]),fontsize=7)
        ax.grid(alpha=.25);ax.set(title="GLO-30 persistent-barrier screening",xlabel="S0 distance (km)",ylabel="robust floor elevation (m)");fig.tight_layout();fig.savefig(FIG/"45_glo30_final_barrier_screening.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,14));ax.imshow(hill(base),extent=extent,cmap="gray");ax.plot(*domain.exterior.xy,color="#00ffff",lw=1);ax.plot(*line.xy,color="#ffeb00",lw=1.5);ax.set(title="Final Copernicus GLO-30 computational terrain context",aspect="equal");fig.tight_layout();fig.savefig(FIG/"46_glo30_final_terrain_map.png",dpi=180);plt.close(fig)
    if ready:(FINAL/"FINAL_COMPUTATIONAL_TERRAIN.txt").write_text("Authoritative terrain source: Copernicus GLO-30 (specified single file)\nBase: "+str(basepath)+"\n32 m: "+str(p32)+"\n64 m: "+str(p64)+"\nCRS: "+CRS+"\nBase SHA256: "+hashes["projected_base_sha256"]+"\n32 m SHA256: "+hashes["terrain_32m_sha256"]+"\n64 m SHA256: "+hashes["terrain_64m_sha256"]+"\nGate A: PASS\nGate B: PASS\n",encoding="utf-8")
    print(json.dumps(final,indent=2))

if __name__=="__main__":main()
