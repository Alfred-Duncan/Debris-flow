"""Build the final Gate A corridor from the approved human image interpretation.

This is a spatial-evidence task only.  It never alters a DEM or invokes
D-Claw.  The locally digitized segment is an engineering corridor axis, not a
surveyed thalweg or asserted historical centreline.
"""
from __future__ import annotations

import csv
import json
import math
import sys

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.ops import substring

sys.path.insert(0, r"E:\Alfred\Jilong_GateA\scripts")
from finalize_gate_a import (ROOT, RAW, CORRIDOR, REPORT, FIG, REVIEW, DOWNLOADS,
                             CRS, PORT, OLD_ENTRY, read_dem, hillshade, show_image,
                             section_at, decorate)

S0 = Point(334051.79369854234, 3143392.492568097)
S4 = Point(340571.8257777202, 3129640.775894154)
P2 = Point(339430.4921983012, 3130169.9150038627)
CGS_URL = "https://www.cgs.gov.cn/ywdt/ddyw/202608/t20260828_867531.html"


def parts(frame):
    return [p for geom in frame.geometry for p in (geom.geoms if geom.geom_type == "MultiLineString" else [geom])]


def concat(lines):
    coords = []
    for line in lines:
        current = list(line.coords)
        if coords and Point(coords[-1]).distance(Point(current[0])) > 0.05:
            raise ValueError("Non-continuous final corridor")
        coords.extend(current if not coords else current[1:])
    return LineString(coords)


def tangent(line, distance, half=20.0):
    return line.interpolate(max(0, distance-half)), line.interpolate(min(line.length, distance+half))


def profile_values(line, dem12, dem8, spacing=12.5):
    distances = np.arange(0.0, line.length + spacing, spacing)
    distances[-1] = line.length
    points = [line.interpolate(float(d)) for d in distances]
    with rasterio.open(dem12) as ds:
        z12 = np.asarray([v[0] for v in ds.sample([(p.x, p.y) for p in points])], dtype=float)
        raw_valid = np.isfinite(z12) & (z12 != ds.nodata)
    z12[~raw_valid] = np.nan
    with rasterio.open(dem8) as ds:
        transform = Transformer.from_crs(CRS, ds.crs, always_xy=True)
        z8 = np.asarray([v[0] for v in ds.sample([transform.transform(p.x, p.y) for p in points])], dtype=float)
        # Rasterio returns 0 outside the valid 8 m mosaic even though the file's
        # declared nodata is a float sentinel; 0 m is physically impossible here.
        valid8 = np.isfinite(z8) & (z8 != ds.nodata) & (z8 > 0)
    z8[~valid8] = np.nan
    # This is a QA mask, not a raster alteration.  In this shadowed trunk reach
    # a raw 12.5 m cell that differs by >120 m from the complete 8 m DEM is
    # retained in CSV but excluded from slope interpretation.
    qa_valid = raw_valid & valid8 & (np.abs(z12-z8) <= 120.0)
    zqa = z12.copy(); zqa[~qa_valid] = np.nan
    return distances, points, z12, z8, raw_valid, valid8, qa_valid, zqa


def stats(z, spacing=12.5):
    output = {}
    for metres in (12.5, 50, 100, 250):
        step = max(1, int(round(metres/spacing)))
        rises = z[step:] - z[:-step]
        output[f"max_adverse_rise_{int(metres) if metres != 12.5 else 'local'}m"] = float(np.nanmax(rises))
    finite = z[np.isfinite(z)]
    output["start_elevation_m"] = float(finite[0])
    output["end_elevation_m"] = float(finite[-1])
    output["total_elevation_change_m"] = output["end_elevation_m"] - output["start_elevation_m"]
    return output


def segment_label(distance, up_end, gap_end):
    if distance <= up_end + 1e-6: return "official_upstream_image_consistent"
    if distance <= gap_end + 1e-6: return "imagery_supported_607p99m_gap_trace"
    return "human_review_approved_local_correction"


def main():
    for directory in (CORRIDOR, REPORT, FIG, REVIEW): directory.mkdir(parents=True, exist_ok=True)
    dem12 = next((RAW / "dem_12p5m").rglob("*.tif"))
    dem8 = next((RAW / "dem_8m").rglob("*.tif"))
    pre = next(DOWNLOADS.rglob("05m.tif"))
    river_options = []
    for path in (RAW / "basic_geography").rglob("*.shp"):
        frame = gpd.read_file(path).to_crs(CRS)
        if set(frame.geom_type).issubset({"LineString", "MultiLineString"}):
            river_options.append((frame.geometry.union_all().distance(PORT), frame))
    river = min(river_options, key=lambda item: item[0])[1]
    upstream, downstream = sorted(parts(river), key=lambda line: line.distance(OLD_ENTRY))
    p0 = Point(downstream.coords[0])
    bridge = gpd.read_file(CORRIDOR / "candidate_gap_bridge.geojson").to_crs(CRS).geometry.iloc[0]
    upstream_segment = substring(upstream, upstream.project(S0), upstream.length)
    wrong = substring(downstream, 0, downstream.project(P2))

    # Human-approved engineering centreline: P0 south through the narrow valley,
    # east in the broad trunk, then southeast on the visible S4-side continuation.
    corrected = LineString([
        p0.coords[0], (337990,3131200), (337965,3131120), (337930,3131040),
        (337930,3130950), (337970,3130870), (337930,3130780), (337950,3130700),
        (337970,3130620), (337950,3130540), (337990,3130470), (338070,3130400),
        (338130,3130310), (338210,3130230), (338330,3130180), (338450,3130130),
        (338650,3130100), (338850,3130070), (339050,3130050), (339250,3130040),
        (339450,3130025), (339650,3129980), (339800,3129940), (339950,3129900),
        (340100,3129870), (340250,3129830), (340350,3129750), (340450,3129680), S4.coords[0]
    ])
    if corrected.coords[0] != p0.coords[0] or Point(corrected.coords[-1]).distance(S4) > .01:
        raise ValueError("Corrected segment endpoints not anchored")
    correction_attrs = {
        "segment_id":["DOWNSTREAM_ANOMALY_HUMAN_APPROVED_V1"],
        "status":["HUMAN_REVIEW_APPROVED_CORRECTION"], "human_review":["APPROVED"],
        "primary_source":["pre-event 0.5 m DOM"], "support_dem12p5":["PARTIAL_QA_MASKED"],
        "support_dem8":[True], "support_post_event":["partial_cloud_obscured"],
        "official_P0_P2_rejected":[True],
        "downstream_tie_in_basis":["No earlier S4-side official line was image/DEM-consistent; human-approved image/DEM-supported corridor continues to S4."],
        "confidence":["MODERATE"],
        "notes":["Engineering corridor axis from approved human review; follows visible narrow valley, broad trunk valley and S4-side continuation. Not a surveyed thalweg."]
    }
    gpd.GeoDataFrame(correction_attrs, geometry=[corrected], crs=CRS).to_file(CORRIDOR / "downstream_anomaly_channel_corrected.geojson", driver="GeoJSON")

    corridor = concat([upstream_segment, bridge, corrected])
    up_end, gap_end = upstream_segment.length, upstream_segment.length + bridge.length
    d, pts, zraw, z8, raw_valid, valid8, qa_valid, zqa = profile_values(corridor, dem12, dem8)
    correction_start_index = int(np.searchsorted(d, gap_end))
    wd, wp, wzraw, wz8, wraw_valid, wvalid8, wqa_valid, wzqa = profile_values(wrong, dem12, dem8)
    cd, cp, czraw, cz8, craw_valid, cvalid8, cqa_valid, czqa = profile_values(corrected, dem12, dem8)
    official_stats = stats(wzraw)
    corrected_stats_qa, corrected_stats_8 = stats(czqa), stats(cz8)
    final_stats = stats(zqa)
    final_rises = zqa[20:] - zqa[:-20]
    upstream_rise_index = int(np.nanargmax(final_rises))
    upstream_rise_start = pts[upstream_rise_index]
    upstream_rise_end = pts[upstream_rise_index + 20]
    lateral = np.asarray([wrong.distance(point) for point in cp])
    tie_in = S4
    qa = {
        "human_review_status":"APPROVED_FOR_LOCAL_CHANNEL_CORRECTION",
        "correction_accepted":True,
        "official_wrong_segment_length_m":wrong.length,
        "corrected_segment_length_m":corrected.length,
        "official_max_adverse_rise_250m_m":official_stats["max_adverse_rise_250m"],
        "corrected_max_adverse_rise_250m_m":corrected_stats_qa["max_adverse_rise_250m"],
        "corrected_max_adverse_rise_250m_8m_dem_m":corrected_stats_8["max_adverse_rise_250m"],
        "official_max_local_adverse_rise_m":official_stats["max_adverse_rise_localm"],
        "corrected_max_local_adverse_rise_m":corrected_stats_qa["max_adverse_rise_localm"],
        "maximum_lateral_offset_m":float(lateral.max()), "mean_lateral_offset_m":float(lateral.mean()),
        "downstream_tie_in_coordinate":[tie_in.x,tie_in.y], "distance_from_P0_m":corrected.length,
        "distance_to_S4_m":0.0, "official_geometry_offset_at_tie_in_m":0.0,
        "tie_in_confidence":"MODERATE",
        "tie_in_basis":"Human-approved pre-event imagery and both DEM products support continuing in the physical trunk valley to S4; P2 was not retained as a tie-in.",
        "dem12_raw_valid_fraction":float(craw_valid.mean()), "dem12_qa_valid_fraction":float(cqa_valid.mean()),
        "dem8_valid_fraction":float(cvalid8.mean()),
        "dem12_outlier_rule":"Raw 12.5 m values differing by >120 m from colocated valid 8 m elevation retained in profile but excluded from slope QA; no DEM values changed.",
        "official_stats":official_stats, "corrected_stats_dem12_qa":corrected_stats_qa,
        "corrected_stats_dem8":corrected_stats_8
    }
    (REPORT / "DOWNSTREAM_GEOMETRY_CORRECTION_QA.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    (REPORT / "DOWNSTREAM_GEOMETRY_CORRECTION_QA.md").write_text(
        f"# Downstream geometry correction QA\n\n`HUMAN_IMAGE_REVIEW = APPROVED_FOR_LOCAL_CHANNEL_CORRECTION`.\n\n"
        f"The rejected P0--P2 official line is **{wrong.length:.2f} m**; its 12.5 m maximum adverse rise over 250 m is **{official_stats['max_adverse_rise_250m']:.1f} m**. The approved physical-channel correction is **{corrected.length:.2f} m** and terminates at S4 rather than returning to P2. Its QA-masked native-12.5-m maximum 250 m rise is **{corrected_stats_qa['max_adverse_rise_250m']:.1f} m**; independent 8 m DEM gives **{corrected_stats_8['max_adverse_rise_250m']:.1f} m**.\n\n"
        f"The maximum/mean lateral separation from the rejected line is {lateral.max():.1f}/{lateral.mean():.1f} m. Tie-in is S4 `({tie_in.x:.3f}, {tie_in.y:.3f})`; no earlier S4-side official geometry was retained because it was not image/DEM consistent.\n\n"
        "The shadowed trunk has local disagreement between native 12.5 m and 8 m data. Raw 12.5 m values are preserved in the final CSV. Values differing by more than 120 m from a colocated valid 8 m sample are marked `dem_valid=false` for slope QA only; this is an evidence flag, not a DEM edit.\n",
        encoding="utf-8")

    # The full QA is deliberately run before publishing a canonical corridor.
    # It finds an independent, direct contradiction on the supposedly accepted
    # upstream official part; do not conceal it behind the approved downstream fix.
    if d[upstream_rise_index] < up_end and final_rises[upstream_rise_index] >= 150.0:
        ub = (upstream_rise_start.x-800, upstream_rise_start.y-800, upstream_rise_start.x+800, upstream_rise_start.y+800)
        z, db = read_dem(dem12, ub)
        fig, axes = plt.subplots(1,2,figsize=(16,8))
        show_image(axes[0], pre, ub, "New upstream geometry QA failure — pre-event 0.5 m DOM")
        for geom in river.geometry:
            for line in (geom.geoms if geom.geom_type == "MultiLineString" else [geom]): axes[0].plot(*line.xy,color="#00cfe8",lw=1.2)
        axes[0].plot([upstream_rise_start.x,upstream_rise_end.x],[upstream_rise_start.y,upstream_rise_end.y],color="red",lw=2.4)
        axes[0].scatter([upstream_rise_start.x],[upstream_rise_start.y],color="red",s=60);decorate(axes[0],ub)
        axes[1].imshow(hillshade(z),extent=(db[0],db[2],db[1],db[3]),cmap="gray")
        for geom in river.geometry:
            for line in (geom.geoms if geom.geom_type == "MultiLineString" else [geom]): axes[1].plot(*line.xy,color="#00cfe8",lw=1.2)
        axes[1].plot([upstream_rise_start.x,upstream_rise_end.x],[upstream_rise_start.y,upstream_rise_end.y],color="red",lw=2.4)
        axes[1].set_title("Native 12.5 m DEM hillshade");decorate(axes[1],ub)
        fig.tight_layout();fig.savefig(FIG / "21_upstream_geometry_QA_failure.png",dpi=220);plt.close(fig)
        upstream_qa={"status":"HUMAN_IMAGE_REVIEW_REQUIRED","coordinate":[upstream_rise_start.x,upstream_rise_start.y],
          "distance_from_S0_m":float(d[upstream_rise_index]),"adverse_rise_250m_m":float(final_rises[upstream_rise_index]),
          "native_12p5_elevation_start_m":float(zqa[upstream_rise_index]),"native_12p5_elevation_end_m":float(zqa[upstream_rise_index+20]),
          "independent_8m_elevation_start_m":float(z8[upstream_rise_index]),"independent_8m_elevation_end_m":float(z8[upstream_rise_index+20]),
          "image_result":"Official upstream line crosses slope/bare terrain rather than the visible channel in Figure 21.",
          "consequence":"Approved downstream correction retained, but final S0-S4 canonical corridor withheld."}
        (REPORT / "UPSTREAM_GEOMETRY_QA.json").write_text(json.dumps(upstream_qa,indent=2)+"\n",encoding="utf-8")
        (REPORT / "UPSTREAM_GEOMETRY_QA.md").write_text(f"# Upstream geometry QA\n\nA direct contradiction was found at `({upstream_rise_start.x:.3f}, {upstream_rise_start.y:.3f})` EPSG:32645, {d[upstream_rise_index]/1000:.3f} km from S0. The existing official line has a {final_rises[upstream_rise_index]:.1f} m rise over 250 m ({zqa[upstream_rise_index]:.0f} to {zqa[upstream_rise_index+20]:.0f} m in QA-valid 12.5 m samples; {z8[upstream_rise_index]:.0f} to {z8[upstream_rise_index+20]:.0f} m in 8 m DEM). Figure 21 shows it crosses slope/bare terrain instead of the visible channel.\n",encoding="utf-8")
        final={"gate_a":"HUMAN_IMAGE_REVIEW_REQUIRED","analysis_crs":CRS,"human_image_review":"APPROVED_FOR_LOCAL_CHANNEL_CORRECTION",
          "s0_coordinate":[S0.x,S0.y],"s0_confidence":"MODERATE","s4_coordinate":[S4.x,S4.y],"s4_port_offset_m":PORT.distance(S4),
          "old_vector_gap_straight_m":525.5297039262372,"old_gap_trace_length_m":607.9923225566369,"official_P0_P2_rejected":True,
          "corrected_anomaly_segment_accepted":True,"corrected_anomaly_segment_length_m":corrected.length,"downstream_tie_in_coordinate":[tie_in.x,tie_in.y],
          "official_max_adverse_rise_250m_m":official_stats["max_adverse_rise_250m"],"corrected_max_adverse_rise_250m_m":corrected_stats_qa["max_adverse_rise_250m"],
          "canonical_corridor_established":False,"canonical_corridor_length_km":None,"s1_s3_created":False,"profile_created":False,
          "final_max_adverse_rise_250m_m":float(final_rises[upstream_rise_index]),"human_image_review_required":True,
          "primary_blocker":"NEW_UPSTREAM_GEOMETRY_CONTRADICTION: existing official upstream line crosses slope and rises 161 m over 250 m; it needs separate human image review before a final corridor can be published.","cgs_documentary_url":CGS_URL}
        (REPORT / "JILONG_GATE_A_FINAL.json").write_text(json.dumps(final,indent=2)+"\n",encoding="utf-8")
        (REPORT / "JILONG_GATE_A_FINAL.md").write_text("# Final Jilong/Gyirong spatial Gate A\n\n`GATE_A = HUMAN_IMAGE_REVIEW_REQUIRED`. The human-approved downstream anomaly correction is retained in `corridor/downstream_anomaly_channel_corrected.geojson`, but full-corridor QA found a separate upstream vector contradiction at " + f"`({upstream_rise_start.x:.3f}, {upstream_rise_start.y:.3f})` EPSG:32645. The official line crosses slope/bare terrain and rises {final_rises[upstream_rise_index]:.1f} m over 250 m in both DEM checks. No canonical S0--S4 corridor, control sections, or final profile is published.\n",encoding="utf-8")
        (REVIEW / "README.txt").write_text("Current status: HUMAN_IMAGE_REVIEW_REQUIRED. The downstream correction was approved; Figure 21 documents a separate upstream geometry contradiction that must be resolved before publishing a canonical corridor.\n",encoding="utf-8")
        print(json.dumps(final,indent=2)); return

    gpd.GeoDataFrame({"status":["FINAL_EVENT_CONSTRAINED"],
                      "evidence_basis":["official image-consistent upstream geometry; 607.99 m imagery gap trace; approved human-review local correction"],
                      "confidence":["MODERATE"],
                      "notes":["Effective downstream-stage corridor ending at S4; not a surveyed historical centreline."]},
                     geometry=[corridor], crs=CRS).to_file(CORRIDOR / "jilong_canonical_corridor.geojson", driver="GeoJSON")

    # Control sections, retaining existing accepted S0/S4 section geometries.
    s0_section = gpd.read_file(CORRIDOR / "jilong_S0_section.geojson").to_crs(CRS).geometry.iloc[0]
    s4_section = gpd.read_file(CORRIDOR / "jilong_S4_port_section.geojson").to_crs(CRS).geometry.iloc[0]
    rows, geoms = [], []
    for sid, fraction in (("S0",0), ("S1",.25), ("S2",.5), ("S3",.75), ("S4",1)):
        distance = corridor.length * fraction
        point = corridor.interpolate(distance)
        if sid == "S0": geom, orient = s0_section, float(gpd.read_file(CORRIDOR / "jilong_S0_section.geojson").iloc[0]["orientation_deg"])
        elif sid == "S4": geom, orient = s4_section, float(gpd.read_file(CORRIDOR / "jilong_S4_port_section.geojson").iloc[0]["orientation_deg"])
        else:
            a,b = tangent(corridor, distance); geom, orient = section_at(point,a,b,width=160.0)
        rows.append({"section_id":sid,"distance_from_S0_m":distance,"fraction":fraction,"orientation_deg":orient,
                     "inspection_width_m":160.0,"confidence":"MODERATE","geometry_basis":"local corridor tangent; inspection span only",
                     "notes":"Spatially defensible hydraulic inspection section; not a surveyed hydraulic width."})
        geoms.append(geom)
    gpd.GeoDataFrame(rows,geometry=geoms,crs=CRS).to_file(CORRIDOR / "jilong_control_sections.geojson",driver="GeoJSON")

    # Profile CSV stores raw native elevation and flags QA-inconsistent samples.
    profile_rows=[]
    def slope_at(values, index, window):
        step=max(1,int(round(window/12.5))); a=max(0,index-step); b=min(len(values)-1,index+step)
        return None if not(np.isfinite(values[a]) and np.isfinite(values[b])) else float((values[b]-values[a])/(d[b]-d[a]))
    for i,(dist,pt) in enumerate(zip(d,pts)):
        local=None if i==0 or not(np.isfinite(zqa[i]) and np.isfinite(zqa[i-1])) else float((zqa[i]-zqa[i-1])/(d[i]-d[i-1]))
        profile_rows.append({"distance_m":dist,"x":pt.x,"y":pt.y,"elevation_m":None if not np.isfinite(zraw[i]) else zraw[i],
          "local_slope":local,"smoothed_slope_50m":slope_at(zqa,i,50),"smoothed_slope_100m":slope_at(zqa,i,100),"smoothed_slope_250m":slope_at(zqa,i,250),
          "dem_valid":bool(qa_valid[i]),"segment_source":segment_label(dist,up_end,gap_end),"confidence":"MODERATE" if dist>=gap_end else "MODERATE"})
    with (CORRIDOR / "jilong_canonical_corridor_profile.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=profile_rows[0].keys());writer.writeheader();writer.writerows(profile_rows)

    fig,ax=plt.subplots(figsize=(12,5));ax.plot(d/1000,zraw,color="#b0b0b0",lw=.8,label="raw native 12.5 m DEM")
    ax.plot(d/1000,zqa,color="#16697a",lw=1.6,label="DEM-consistent QA samples")
    ax.axvline(gap_end/1000,color="#ffb000",lw=1.3,label="correction begins")
    ax.set(xlabel="Distance from S0 (km)",ylabel="Elevation (m)",title="Final event-constrained corridor longitudinal profile")
    ax.grid(alpha=.3);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(FIG / "19_final_corridor_longitudinal_profile.png",dpi=200);plt.close(fig)

    # Final map with anomaly inset.
    bounds=(min(S0.x,S4.x)-2000,min(S0.y,S4.y)-2000,max(S0.x,S4.x)+2000,max(S0.y,S4.y)+2000)
    z,db=read_dem(dem12,bounds);fig,ax=plt.subplots(figsize=(11,11));ax.imshow(hillshade(z),extent=(db[0],db[2],db[1],db[3]),cmap="gray")
    for geom in river.geometry:
        for part in (geom.geoms if geom.geom_type=="MultiLineString" else [geom]):ax.plot(*part.xy,color="#5dade2",lw=.7,alpha=.7)
    ax.plot(*wrong.xy,color="#e31a1c",lw=1.1,ls="--",label="rejected P0-P2 official segment")
    ax.plot(*bridge.xy,color="#ffb000",lw=2,label="607.99 m imagery gap trace")
    ax.plot(*corrected.xy,color="#f1c40f",lw=1.8,label="human-review-approved correction")
    ax.plot(*corridor.xy,color="#00c853",lw=1.1,label="final event-constrained corridor")
    ax.scatter([S0.x,S4.x,PORT.x,OLD_ENTRY.x],[S0.y,S4.y,PORT.y,OLD_ENTRY.y],color=["red","lime","red","white"],marker="o",s=[55,55,65,40])
    for row,geom in zip(rows,geoms):
        ax.plot(*geom.xy,color="white",lw=.9);p=corridor.interpolate(row['distance_from_S0_m']);ax.text(p.x,p.y,row['section_id'],color='white',weight='bold')
    ax.legend(loc="upper left",fontsize=7);decorate(ax,bounds)
    inset=ax.inset_axes([.55,.04,.41,.31]);ib=(337800,3129400,340700,3131500);show_image(inset,pre,ib,"approved anomaly correction")
    inset.plot(*wrong.xy,color="#e31a1c",lw=1.2,ls="--");inset.plot(*corrected.xy,color="#f1c40f",lw=1.8);inset.scatter([tie_in.x],[tie_in.y],color="lime",s=25);inset.set_xticks([]);inset.set_yticks([])
    fig.tight_layout();fig.savefig(FIG / "20_final_event_constrained_corridor.png",dpi=200);plt.close(fig)

    length_km=corridor.length/1000; difference=length_km-15; percent=100*difference/15
    final={"gate_a":"PASS","analysis_crs":CRS,"human_image_review":"APPROVED_FOR_LOCAL_CHANNEL_CORRECTION",
      "s0_coordinate":[S0.x,S0.y],"s0_confidence":"MODERATE","s4_coordinate":[S4.x,S4.y],"s4_port_offset_m":PORT.distance(S4),
      "old_vector_gap_straight_m":525.5297039262372,"old_gap_trace_length_m":607.9923225566369,"official_P0_P2_rejected":True,
      "corrected_anomaly_segment_accepted":True,"corrected_anomaly_segment_length_m":corrected.length,"downstream_tie_in_coordinate":[tie_in.x,tie_in.y],
      "official_max_adverse_rise_250m_m":official_stats["max_adverse_rise_250m"],"corrected_max_adverse_rise_250m_m":corrected_stats_qa["max_adverse_rise_250m"],
      "corrected_max_adverse_rise_250m_8m_dem_m":corrected_stats_8["max_adverse_rise_250m"],"canonical_corridor_established":True,
      "canonical_corridor_length_m":corridor.length,"canonical_corridor_length_km":length_km,"difference_from_15km_km":difference,"relative_difference_percent":percent,
      "s1_s3_created":True,"profile_created":True,"final_max_adverse_rise_250m_m":final_stats["max_adverse_rise_250m"],
      "final_max_adverse_rise_250m_8m_dem_m":stats(z8)["max_adverse_rise_250m"],"human_image_review_required":False,"primary_blocker":None,"cgs_documentary_url":CGS_URL}
    (REPORT / "JILONG_GATE_A_FINAL.json").write_text(json.dumps(final,indent=2)+"\n",encoding="utf-8")
    (REPORT / "JILONG_GATE_A_FINAL.md").write_text(
      "# Final Jilong/Gyirong spatial Gate A\n\n## Decision\n\n`GATE_A = PASS`.\n\n"
      f"The final event-constrained downstream corridor runs from S0 `({S0.x:.3f}, {S0.y:.3f})` to S4 `({S4.x:.3f}, {S4.y:.3f})` EPSG:32645. It is **{length_km:.3f} km**, {abs(difference):.3f} km ({abs(percent):.1f}%) longer than the historical approximate 15 km description; geometry was not adjusted for that comparison.\n\n"
      "## Approved local correction\n\n"
      f"The P0--P2 official segment is rejected. Under approved human image review, the replacement follows the visible P0 narrow valley, broad eastward trunk valley and S4-side continuation. It is {corrected.length:.2f} m long and ties in at S4 rather than P2. The raw 12.5 m DEM is retained; shadowed-trunk samples that disagree by >120 m with colocated valid 8 m elevation are flagged for slope QA rather than altered. QA-masked 12.5 m/independent 8 m maximum corrected 250 m rises are {corrected_stats_qa['max_adverse_rise_250m']:.1f}/{corrected_stats_8['max_adverse_rise_250m']:.1f} m, versus {official_stats['max_adverse_rise_250m']:.1f} m on the rejected official line.\n\n"
      "S1--S3 are spatially defensible hydraulic inspection sections, not surveyed hydraulic widths. No DEM was modified and no D-Claw simulation was run. The corrected CGS documentary URL is " + CGS_URL + ".\n",encoding="utf-8")
    (REVIEW / "README.txt").write_text("Current authoritative status: Gate A PASS. Historical HUMAN_IMAGE_REVIEW_REQUIRED materials remain as provenance; the approved correction, final profile and final map are in the figures/corridor/reports folders.\n",encoding="utf-8")
    print(json.dumps(final,indent=2))


if __name__=="__main__":main()
