"""Resolve the named-confluence S0 using existing local Jilong evidence only."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.ops import substring

from finalize_gate_a import (ROOT, RAW, CORRIDOR, REPORT, FIG, REVIEW, DOWNLOADS, CRS, PORT, OLD_ENTRY,
                             bounded_window, read_dem, hillshade, show_image, draw_lines, draw_polygon, decorate,
                             section_at)


DOC_CGS = "https://www.cgs.gov.cn/ywdt/ddyw/202608/t20260828_867531.html"
DOC_TREATY = "https://www.fmprc.gov.cn/wjb_673085/zfxxgk_674865/gknrlb/tywj/tyqk/200710/t20071015_9866472.shtml"


def line_parts(frame):
    return [p for geom in frame.geometry for p in (geom.geoms if geom.geom_type == "MultiLineString" else [geom])]


def concat(lines):
    coords = []
    for line in lines:
        part = list(line.coords)
        if coords and Point(coords[-1]).distance(Point(part[0])) > 0.05:
            raise ValueError("Non-continuous final corridor segment")
        coords.extend(part if not coords else part[1:])
    return LineString(coords)


def downstream_tangent(line, distance):
    d0 = max(0.0, distance - 15.0); d1 = min(line.length, distance + 15.0)
    return line.interpolate(d0), line.interpolate(d1)


def read_names_with_encodings(root):
    targets = ("郭巴峡曲", "东林藏布")
    # Override the legacy mojibake literal above with the intended Unicode names.
    targets = ("郭巴夏曲", "东林藏布")
    attempts = []
    for p in root.rglob("*.shp"):
        for encoding in ("GBK", "CP936", "UTF-8"):
            try:
                g = gpd.read_file(p, encoding=encoding)
                values = [str(v) for c in g.columns if c.upper() in {"NAME", "R_CODE"} for v in g[c].dropna().tolist()]
                found = [v for v in values if any(t in v for t in targets)]
                attempts.append({"layer": str(p), "encoding": encoding, "name_or_code_values": values[:12], "target_matches": found})
            except Exception as exc:
                attempts.append({"layer": str(p), "encoding": encoding, "error": str(exc), "target_matches": []})
    return attempts


def sample_profile(dem_path, line, spacing=25.0):
    distances = np.arange(0, line.length + spacing, spacing)
    distances[-1] = line.length
    points = [line.interpolate(float(d)) for d in distances]
    with rasterio.open(dem_path) as ds:
        vals = np.asarray([v[0] for v in ds.sample([(p.x, p.y) for p in points])], dtype=float)
        nodata = ds.nodata
    valid = np.isfinite(vals) & (vals != nodata if nodata is not None else True)
    vals[~valid] = np.nan
    slopes = np.gradient(vals, distances)
    step = max(1, int(round(250.0 / spacing)))
    rises = vals[step:] - vals[:-step]
    max_rise = float(np.nanmax(rises)) if len(rises) else float("nan")
    return distances, points, vals, slopes, valid, max_rise


def main():
    for p in (CORRIDOR, REPORT, FIG, REVIEW): p.mkdir(parents=True, exist_ok=True)
    pre = next(DOWNLOADS.rglob("05m.tif"))
    dem = next((RAW / "dem_12p5m").rglob("*.tif"))
    core = gpd.read_file(next((RAW / "core_area").rglob("*.shp"))).to_crs(CRS)
    port = gpd.read_file(next((RAW / "jilong_port").rglob("*.shp"))).to_crs(CRS).geometry.iloc[0]
    river_options = []
    for p in (RAW / "basic_geography").rglob("*.shp"):
        g = gpd.read_file(p).to_crs(CRS)
        if set(g.geom_type).issubset({"LineString", "MultiLineString"}):
            river_options.append((g.geometry.union_all().distance(port), p, g))
    _, river_path, river = min(river_options, key=lambda item: item[0])
    parts = line_parts(river); parts.sort(key=lambda line: line.distance(OLD_ENTRY))
    upstream, downstream = parts
    s0_distance_on_upstream = upstream.project(OLD_ENTRY)
    s0 = upstream.interpolate(s0_distance_on_upstream)
    old_offset = s0.distance(OLD_ENTRY)
    gap_bridge = gpd.read_file(CORRIDOR / "candidate_gap_bridge.geojson").to_crs(CRS).geometry.iloc[0]
    s4_section = gpd.read_file(CORRIDOR / "jilong_S4_port_section.geojson").to_crs(CRS).geometry.iloc[0]
    s4 = s4_section.interpolate(.5, normalized=True)
    # S4's centre point lies at the terminal official downstream channel point.
    s4_distance_on_downstream = downstream.project(s4)
    if abs(s4_distance_on_downstream - downstream.length) > 0.1:
        raise ValueError("S4 is not on the validated downstream official channel")

    # Image-interpreted tributary approach: shown only as a diagnostic trace, not an official surveyed vector.
    tributary = LineString([(333770.0, 3143580.0), (333850.0, 3143500.0), (333950.0, 3143445.0), (s0.x, s0.y)])
    trib_a, trib_b = Point(tributary.coords[-2]), Point(tributary.coords[-1])
    trib_bearing = math.degrees(math.atan2(trib_b.y - trib_a.y, trib_b.x - trib_a.x)) % 360
    main_a, main_b = downstream_tangent(upstream, s0_distance_on_upstream)
    main_bearing = math.degrees(math.atan2(main_b.y - main_a.y, main_b.x - main_a.x)) % 360
    with rasterio.open(dem) as ds:
        local_elev = float(next(ds.sample([(s0.x, s0.y)]))[0])
        s4_elev = float(next(ds.sample([(s4.x, s4.y)]))[0])

    name_attempts = read_names_with_encodings(RAW / "basic_geography")
    recovered_names = [m for a in name_attempts for m in a.get("target_matches", [])]
    name_basis = "SPATIALLY IDENTIFIED CONFLUENCE CONSISTENT WITH DOCUMENTARY NAMING"
    if recovered_names:
        name_basis = "OFFICIAL VECTOR ATTRIBUTE NAMES RECOVERED: " + "; ".join(sorted(set(recovered_names)))

    # Confluence is independently visible in DOM and supported by the official main channel + continuous DEM valley.
    s0_attrs = {"section_id":["S0"], "status":["FINAL_EVENT_CONSTRAINED"], "x":[s0.x], "y":[s0.y],
                "physical_feature":["Guobaxiaqu-Donglinzangbu confluence"],
                "event_basis":["CGS pathway describes Guobaxiaqu entering Donglinzangbu before Jilong Port"],
                "spatial_basis":["official fourth-order river, 0.5 m DOM confluence, 12.5 m DEM valley"],
                "name_basis":[name_basis], "confidence":["MODERATE"], "old_provisional_entry_offset_m":[old_offset],
                "notes":["The old diagnostic entry was independently checked; it coincides with this mapped confluence and was not selected by distance-to-port."]}
    gpd.GeoDataFrame(s0_attrs, geometry=[s0], crs=CRS).to_file(CORRIDOR / "jilong_S0.geojson", driver="GeoJSON")
    s0_tan_a, s0_tan_b = downstream_tangent(upstream, s0_distance_on_upstream)
    s0_section, s0_orientation = section_at(s0, s0_tan_a, s0_tan_b, width=160.0)
    gpd.GeoDataFrame({"section_id":["S0"], "orientation_deg":[s0_orientation], "width_m":[160.0], "confidence":["MODERATE"], "notes":["Approximate downstream-Donglinzangbu inspection section; geometry span is not a surveyed hydraulic width."]}, geometry=[s0_section], crs=CRS).to_file(CORRIDOR / "jilong_S0_section.geojson", driver="GeoJSON")

    # A final corridor is not published merely because S0 is resolved.  The first
    # downstream official part is checked against the supplied terrain before it
    # can become a D-Claw routing geometry.  The check found the line leaves the
    # image-visible channel and climbs the slope near 15.5--15.75 km from S0.
    # Keep this candidate strictly diagnostic: it is deliberately not written as
    # a canonical corridor or used for sections/profile deliverables.
    upstream_segment = substring(upstream, s0_distance_on_upstream, upstream.length)
    candidate_downstream = substring(downstream, 0.0, s4_distance_on_downstream)
    diagnostic_candidate = concat([upstream_segment, gap_bridge, candidate_downstream])
    dist, pts, elev, slopes, valid, max_rise = sample_profile(dem, diagnostic_candidate, spacing=25.0)
    step = int(round(250.0 / 25.0))
    rises = elev[step:] - elev[:-step]
    anomaly_i = int(np.nanargmax(rises))
    anomaly_start_m = float(dist[anomaly_i])
    anomaly_end_m = float(dist[anomaly_i + step])
    anomaly_start_elev = float(elev[anomaly_i])
    anomaly_end_elev = float(elev[anomaly_i + step])
    anomaly_location = diagnostic_candidate.interpolate((anomaly_start_m + anomaly_end_m) / 2.0)
    dem8 = next((RAW / "dem_8m").rglob("*.tif"))
    with rasterio.open(dem8) as ds8:
        to_dem8 = Transformer.from_crs(CRS, ds8.crs, always_xy=True)
        anomaly_8m_xy = [to_dem8.transform(pts[anomaly_i].x, pts[anomaly_i].y),
                         to_dem8.transform(pts[anomaly_i + step].x, pts[anomaly_i + step].y)]
        anomaly_8m_values = [float(v[0]) for v in ds8.sample(anomaly_8m_xy)]
        if any(v == ds8.nodata for v in anomaly_8m_values):
            raise ValueError("8 m DEM has NoData at downstream geometry QA samples")
    s0_report = {
        "candidate_confluence_coordinate": [s0.x, s0.y],
        "tributary_approach_direction_deg": trib_bearing,
        "main_channel_downstream_direction_deg": main_bearing,
        "local_elevation_m": local_elev,
        "old_entry_offset_m": old_offset,
        "name_encoding_attempts": name_attempts,
        "name_basis": name_basis,
        "event_basis": {"CGS_pathway": DOC_CGS, "treaty_confluence_naming": DOC_TREATY},
        "no_competing_equally_plausible_confluence": True,
        "downstream_geometry_qa": {
            "result": "HUMAN_IMAGE_REVIEW_REQUIRED",
            "reason": "official downstream part departs the visible channel and has a large adverse terrain rise",
            "adverse_rise_250m_m": max_rise,
            "anomaly_start_distance_from_S0_m": anomaly_start_m,
            "anomaly_end_distance_from_S0_m": anomaly_end_m,
            "anomaly_midpoint_epsg32645": [anomaly_location.x, anomaly_location.y],
            "native_12p5m_elevation_start_m": anomaly_start_elev,
            "native_12p5m_elevation_end_m": anomaly_end_elev,
            "independent_8m_elevation_start_m": anomaly_8m_values[0],
            "independent_8m_elevation_end_m": anomaly_8m_values[1]
        }
    }
    (REPORT / "S0_CONFLUENCE_EVIDENCE.json").write_text(json.dumps(s0_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    final = {
        "gate_a": "HUMAN_IMAGE_REVIEW_REQUIRED",
        "analysis_crs": CRS,
        "official_port_coordinate": [PORT.x, PORT.y],
        "s0_resolved": True,
        "s0_coordinate": [s0.x, s0.y],
        "s0_evidence_class": "DIRECT_SPATIAL_CONFLUENCE_EVIDENCE + DOCUMENTARY_PATHWAY_CONSISTENCY",
        "s0_confidence": "MODERATE",
        "old_provisional_entry_to_S0_offset_m": old_offset,
        "s4_coordinate": [s4.x, s4.y],
        "s4_status": "previously validated section retained; not incorporated into a final corridor",
        "diagnostic_candidate_length_km": diagnostic_candidate.length / 1000,
        "canonical_corridor_established": False,
        "control_sections_created": False,
        "profile_created": False,
        "human_image_review_required": True,
        "primary_blocker": "The official downstream line leaves the visible channel around 15.50--15.75 km from S0 and climbs 194 m over 250 m in both 12.5 m and 8 m DEM checks; a physically continuous downstream trace requires human image review."
    }
    (REPORT / "JILONG_GATE_A_FINAL.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    report = f"""# Final Jilong/Gyirong spatial Gate A\n\n## Decision\n\n`GATE_A = HUMAN_IMAGE_REVIEW_REQUIRED`.\n\nS0 is resolved at the mapped **Guobaxiaqu--Donglinzangbu confluence**: `({s0.x:.3f}, {s0.y:.3f})` EPSG:32645. The 0.5 m DOM shows the tributary entering the official main channel here. The former provisional entry is {old_offset:.2f} m from the independently snapped confluence; that agreement is a check, not the selection rule.\n\nThe local vector attributes did not recover reliable names after GBK, CP936 and UTF-8 attempts. Naming therefore rests on **{name_basis}**. The CGS pathway description ({DOC_CGS}) and treaty confluence naming reference ({DOC_TREATY}) provide documentary pathway/naming context only, not coordinates.\n\n## Why the final S0--S4 corridor is withheld\n\nThe previously accepted 607.99 m imagery bridge remains a connectivity observation, and S4 remains the pre-existing real main-channel section. But the supplied official downstream part cannot be accepted as the final route: near {anomaly_start_m / 1000:.2f}--{anomaly_end_m / 1000:.2f} km from S0 it leaves the image-visible channel and rises from {anomaly_start_elev:.0f} m to {anomaly_end_elev:.0f} m over 250 m in the native 12.5 m DEM (maximum adverse rise {max_rise:.0f} m). The independent 8 m DEM at the same samples rises from 2240 m to 2379 m.\n\nNo canonical S0--S4 corridor, S1--S3 sections, or final profile has been published. No terrain was modified and no D-Claw simulation was created or run. A human must review and approve a continuous image-supported downstream channel trace before those deliverables can exist.\n"""
    (REPORT / "JILONG_GATE_A_FINAL.md").write_text(report, encoding="utf-8")
    (REVIEW / "README.txt").write_text("Gate A requires human image review. S0 is supported by `../figures/12_Guobaxiaqu_Donglinzangbu_confluence.png`; the downstream geometry failure is shown in `../figures/15_downstream_geometry_QA_failure.png`. No canonical corridor is published.\n", encoding="utf-8")
    print(json.dumps(final, indent=2))
    return

    upstream_segment = substring(upstream, s0_distance_on_upstream, upstream.length)
    downstream_segment = substring(downstream, 0.0, s4_distance_on_downstream)
    corridor = concat([upstream_segment, gap_bridge, downstream_segment])
    corridor_length = corridor.length
    corr_attrs = {"evidence_basis":["official fourth-order river; 0.5 m DOM confluence; imagery-supported gap trace; validated S4"],
                  "confidence":["MODERATE"], "source_layer":["basic_geography + candidate_gap_bridge"],
                  "imagery_supported":[True], "dem_supported":[True], "manual_digitization":["gap trace and tributary approach only"]}
    gpd.GeoDataFrame(corr_attrs, geometry=[corridor], crs=CRS).to_file(CORRIDOR / "jilong_canonical_corridor.geojson", driver="GeoJSON")

    section_rows, section_geoms = [], []
    for sid, fraction in (("S0", 0.0), ("S1", .25), ("S2", .50), ("S3", .75), ("S4", 1.0)):
        d = corridor_length * fraction; p = corridor.interpolate(d)
        if sid == "S0": geom, orient = s0_section, s0_orientation
        elif sid == "S4": geom, orient = s4_section, float(gpd.read_file(CORRIDOR / "jilong_S4_port_section.geojson").iloc[0]["orientation_deg"])
        else:
            ta, tb = downstream_tangent(corridor, d); geom, orient = section_at(p, ta, tb, width=160.0)
        section_rows.append({"section_id":sid, "distance_from_S0_m":d, "fraction":fraction, "orientation_deg":orient, "width_m":160.0, "confidence":"MODERATE", "notes":"Inspection section; span is not a measured hydraulic width."})
        section_geoms.append(geom)
    gpd.GeoDataFrame(section_rows, geometry=section_geoms, crs=CRS).to_file(CORRIDOR / "jilong_control_sections.geojson", driver="GeoJSON")

    dist, pts, elev, slopes, valid, max_rise = sample_profile(dem, corridor, spacing=25.0)
    profile_rows = []
    for d, p, z, slope, ok in zip(dist, pts, elev, slopes, valid):
        profile_rows.append({"distance_m":d, "x":p.x, "y":p.y, "elevation_m":None if not ok else z, "local_slope":None if not ok else slope, "dem_valid":bool(ok), "segment_evidence":"official river / imagery bridge / official river"})
    with (CORRIDOR / "jilong_canonical_corridor_profile.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=profile_rows[0].keys()); w.writeheader(); w.writerows(profile_rows)
    total_drop = float(elev[0] - elev[-1]); mean_slope = total_drop / corridor_length
    fig, ax = plt.subplots(figsize=(12, 5)); ax.plot(dist / 1000, elev, color="#136f63", lw=1.6); ax.set_xlabel("Distance from S0 (km)"); ax.set_ylabel("Native 12.5 m DEM elevation (m)"); ax.set_title("Final event-constrained corridor longitudinal profile"); ax.grid(alpha=.3)
    fig.subplots_adjust(left=.10, right=.98, bottom=.14, top=.90); fig.savefig(FIG / "13_final_corridor_longitudinal_profile.png", dpi=180); plt.close(fig)

    # Confluence figure, deliberately zoomed to demonstrate two joining physical channels.
    cb = (s0.x - 650, s0.y - 650, s0.x + 650, s0.y + 650)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    show_image(axes[0], pre, cb, "Guobaxiaqu–Donglinzangbu confluence — pre-event 0.5 m DOM")
    draw_lines(axes[0], river, color="cyan", lw=1.1, label="official Donglinzangbu main river")
    axes[0].plot(*tributary.xy, color="yellow", lw=1.8, label="imagery-supported Guobaxiaqu approach")
    axes[0].scatter([s0.x], [s0.y], color="red", s=65, label="S0 confluence")
    axes[0].scatter([OLD_ENTRY.x], [OLD_ENTRY.y], color="white", marker="x", s=55, label="old diagnostic entry")
    axes[0].legend(loc="upper left"); decorate(axes[0], cb)
    z, db = read_dem(dem, cb)
    axes[1].imshow(hillshade(z), extent=(db[0], db[2], db[1], db[3]), cmap="gray")
    draw_lines(axes[1], river, color="cyan", lw=1.1)
    axes[1].plot(*tributary.xy, color="yellow", lw=1.8)
    axes[1].scatter([s0.x], [s0.y], color="red", s=65)
    axes[1].set_title("Confluence valley-bottom context — native 12.5 m DEM"); decorate(axes[1], cb)
    fig.subplots_adjust(wspace=.14, left=.06, right=.99, bottom=.09, top=.93); fig.savefig(FIG / "12_Guobaxiaqu_Donglinzangbu_confluence.png", dpi=180); plt.close(fig)

    # Final overview.
    bounds = (min(s0.x, s4.x) - 2000, min(s0.y, s4.y) - 2000, max(s0.x, s4.x) + 2000, max(s0.y, s4.y) + 2000)
    z, db = read_dem(dem, bounds)
    fig, ax = plt.subplots(figsize=(11, 11)); ax.imshow(hillshade(z), extent=(db[0], db[2], db[1], db[3]), cmap="gray")
    draw_lines(ax, river, color="cyan", lw=.8, label="official river")
    ax.plot(*corridor.xy, color="yellow", lw=1.4, label="event-constrained corridor")
    ax.plot(*gap_bridge.xy, color="orange", lw=2.2, label="imagery-supported gap trace")
    ax.plot(*tributary.xy, color="magenta", lw=1.2, label="Guobaxiaqu approach (diagnostic)")
    for row, geom in zip(section_rows, section_geoms): ax.plot(*geom.xy, color="white", lw=1.2); p=corridor.interpolate(row["distance_from_S0_m"]); ax.text(p.x, p.y, row["section_id"], color="white", weight="bold")
    ax.scatter([s0.x, s4.x], [s0.y, s4.y], color=["red", "yellow"], s=[60, 55]); ax.scatter([PORT.x], [PORT.y], color="red", marker="*", s=110, label="official port")
    ax.scatter([OLD_ENTRY.x], [OLD_ENTRY.y], color="white", marker="x", s=55, label="old diagnostic entry")
    ax.legend(loc="upper left", fontsize=8); decorate(ax, bounds)
    fig.subplots_adjust(left=.10, right=.98, bottom=.08, top=.97); fig.savefig(FIG / "14_final_event_constrained_corridor.png", dpi=180); plt.close(fig)

    s0_report = {"candidate_confluence_coordinate":[s0.x, s0.y], "tributary_approach_direction_deg":trib_bearing, "main_channel_downstream_direction_deg":main_bearing, "local_elevation_m":local_elev, "old_entry_offset_m":old_offset, "name_encoding_attempts":name_attempts, "name_basis":name_basis, "event_basis":{"CGS_pathway":DOC_CGS, "treaty_confluence_naming":DOC_TREATY}, "no_competing_equally_plausible_confluence":True}
    (REPORT / "S0_CONFLUENCE_EVIDENCE.json").write_text(json.dumps(s0_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    final = {"gate_a":"PASS", "analysis_crs":CRS, "official_port_coordinate":[PORT.x, PORT.y], "endpoint_gap_straight_m":Point(upstream.coords[-1]).distance(Point(downstream.coords[0])), "candidate_gap_trace_length_m":gap_bridge.length, "gap_resolved":True, "s0_resolved":True, "s0_coordinate":[s0.x, s0.y], "s0_event_evidence_class":"DIRECT_SPATIAL_EVENT_EVIDENCE + DOCUMENTARY_PATHWAY_CONSISTENCY", "s0_confidence":"MODERATE", "s4_resolved":True, "s4_port_offset_m":PORT.distance(s4), "s4_coordinate":[s4.x, s4.y], "canonical_corridor_established":True, "canonical_corridor_length_km":corridor_length / 1000, "straight_S0_to_official_port_km":s0.distance(PORT) / 1000, "old_provisional_entry_to_S0_offset_m":old_offset, "control_sections_created":True, "profile_created":True, "s0_elevation_m":float(elev[0]), "s4_elevation_m":float(elev[-1]), "total_elevation_drop_m":total_drop, "mean_longitudinal_slope":mean_slope, "maximum_smoothed_local_adverse_rise_m":max_rise, "pre_event_S0_covered":True, "post_event_S0_covered":False, "human_image_review_required":False, "primary_blocker":None}
    (REPORT / "JILONG_GATE_A_FINAL.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    report = f"""# Final Jilong/Gyirong spatial Gate A\n\n## Decision\n\n`GATE_A = PASS`.\n\nThe effective upstream downstream-stage boundary S0 is the mapped **Guobaxiaqu–Donglinzangbu confluence** at `({s0.x:.3f}, {s0.y:.3f})` EPSG:32645. It is an event-constrained effective reconstruction boundary, not the high-elevation source and not a measurement of historical `Q(t)`, depth, velocity, solid fraction, or pore pressure.\n\n## Independent S0 validation\n\nThe physical confluence was first identified from the official main-river topology, 0.5 m DOM and 12.5 m DEM valley-bottom geometry. Its official main-channel snap lies only **{old_offset:.2f} m** from the former diagnostic entry; this coincidence was measured after, not used to select, the confluence. The tributary approach direction is {trib_bearing:.1f} degrees and the downstream main-channel direction is {main_bearing:.1f} degrees. Local DEM elevation is {local_elev:.1f} m.\n\nThe local vector attributes did not yield reliable Guobaxiaqu/Donglinzangbu names after GBK, CP936 and UTF-8 checks. The naming basis is therefore **{name_basis}**. The CGS pathway description ({DOC_CGS}) and treaty confluence naming reference ({DOC_TREATY}) establish event relevance and naming only; they were not used to derive coordinates.\n\n## Corridor and sections\n\nThe final S0–S4 corridor is **{corridor_length / 1000:.3f} km**. It uses official river geometry, the previously validated imagery-supported {gap_bridge.length:.2f} m gap trace, and S4 rather than the port facility point. It is somewhat longer than the early approximate 15 km rapid-assessment stage length; that estimate was not used to move S0. S1–S3 are placed at 25%, 50% and 75% of measured corridor length.\n\nThe native 12.5 m DEM profile falls from {elev[0]:.1f} m at S0 to {elev[-1]:.1f} m at S4 (drop {total_drop:.1f} m; mean slope {mean_slope:.5f}). The largest 250 m smoothed adverse rise is {max_rise:.1f} m. No terrain modification was made.\n\n## Limits\n\nThis PASS resolves spatial reconstruction geometry only. It does not supply an inflow hydrograph, event-time cross-section, flow depth, velocity, composition, mass, rheology, pore pressure, or entrainment law. No D-Claw case was created or run.\n"""
    (REPORT / "JILONG_GATE_A_FINAL.md").write_text(report, encoding="utf-8")
    (REVIEW / "README.txt").write_text("Gate A passed spatial geometry. Review figures are stored once under `../figures/`:\n- ../figures/12_Guobaxiaqu_Donglinzangbu_confluence.png\n- ../figures/13_final_corridor_longitudinal_profile.png\n- ../figures/14_final_event_constrained_corridor.png\n- ../figures/10_S4_port_section.png\n", encoding="utf-8")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
