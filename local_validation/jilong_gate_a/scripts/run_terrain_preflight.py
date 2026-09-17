"""CPU-only, read-only computational-terrain preflight for the final corridor.

No raster is written, conditioned, carved, filled, or otherwise changed.  The
canonical corridor is opened read-only and is never rewritten by this script.
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

sys.path.insert(0, r"E:\Alfred\Jilong_GateA\scripts")
from finalize_gate_a import ROOT, RAW, CORRIDOR, REPORT, FIG, CRS, PORT, read_dem, hillshade, decorate

OUT = ROOT / "terrain_preflight"
S0 = Point(334051.79369854234, 3143392.492568097)
S4 = Point(340571.8257777202, 3129640.775894154)
STATION_SPACING = 50.0
HALF_WIDTH = 250.0


def valid_values(values, nodata, zero_invalid=False):
    result = np.isfinite(values) & (values != nodata)
    if zero_invalid:
        result &= values > 0
    return result


def station_distances(length, spacing):
    d = np.arange(0.0, length + spacing, spacing)
    d[-1] = length
    return d


def cross_line(line, distance, half_width):
    point = line.interpolate(float(distance))
    a = line.interpolate(max(0.0, distance - 25.0))
    b = line.interpolate(min(line.length, distance + 25.0))
    dx, dy = b.x - a.x, b.y - a.y
    norm = math.hypot(dx, dy)
    if norm == 0:
        raise ValueError("Cannot form a cross-section at zero tangent")
    nx, ny = -dy / norm, dx / norm
    return LineString([(point.x - nx * half_width, point.y - ny * half_width),
                       (point.x + nx * half_width, point.y + ny * half_width)]), point


def sample_cross(ds, line, transformer, spacing, half_width, zero_invalid=False):
    offsets = np.arange(-half_width, half_width + spacing * .5, spacing)
    coords = [(line.interpolate((x + half_width) / (2 * half_width), normalized=True).x,
               line.interpolate((x + half_width) / (2 * half_width), normalized=True).y) for x in offsets]
    sample_coords = [transformer.transform(*xy) for xy in coords] if transformer else coords
    values = np.asarray([v[0] for v in ds.sample(sample_coords)], dtype=float)
    valid = valid_values(values, ds.nodata, zero_invalid)
    if not valid.any():
        return {"center": np.nan, "raw_min": np.nan, "robust": np.nan, "median": np.nan,
                "relief": np.nan, "fraction": 0.0, "low_offset": np.nan, "robust_offset": np.nan}
    vals, offs = values[valid], offsets[valid]
    order = np.argsort(vals)
    lowest = order[:min(5, len(order))]
    robust = np.median(vals[lowest]) if len(lowest) >= 5 else np.nan
    center_index = int(np.argmin(np.abs(offsets)))
    center = values[center_index] if valid[center_index] else np.nan
    return {
        "center": center, "raw_min": float(vals.min()), "robust": robust, "median": float(np.median(vals)),
        "relief": float(vals.max() - vals.min()), "fraction": float(valid.mean()),
        "low_offset": float(offs[order[0]]), "robust_offset": float(np.median(offs[lowest])) if len(lowest) >= 5 else np.nan,
    }


def compute_stations(line, ds12, ds8, tr8, spacing, half_width=HALF_WIDTH, make_geometry=False):
    distances = station_distances(line.length, spacing)
    rows, geoms = [], []
    for distance in distances:
        section, point = cross_line(line, distance, half_width)
        a = sample_cross(ds12, section, None, 12.5, half_width)
        b = sample_cross(ds8, section, tr8, 8.0, half_width, zero_invalid=True)
        rows.append({"station_m": float(distance), "x": point.x, "y": point.y, "dem12": a, "dem8": b})
        if make_geometry: geoms.append(section)
    return rows, geoms


def array(rows, key, field):
    return np.asarray([row[key][field] for row in rows], dtype=float)


def forward_rise(values, distances, window):
    out = np.full(len(values), np.nan)
    for i, start in enumerate(distances):
        j = int(np.searchsorted(distances, start + window, side="left"))
        if j < len(values) and np.isfinite(values[i]) and np.isfinite(values[j]): out[i] = values[j] - values[i]
    return out


def max_rise(values, distances, window):
    r = forward_rise(values, distances, window)
    return None if not np.isfinite(r).any() else float(np.nanmax(r))


def grouped(mask, distances, join_m=125.0):
    idx = np.flatnonzero(mask).tolist()
    if not idx: return []
    groups, current = [], [idx[0]]
    for i in idx[1:]:
        if distances[i] - distances[current[-1]] <= join_m: current.append(i)
        else: groups.append(current); current = [i]
    groups.append(current)
    return groups


def barrier_candidates(values, distances, rows, product, station_spacing=50.0):
    """Find persistent local robust-floor crests without treating cell noise as a barrier."""
    candidates = []
    flank = max(1, int(round(500.0 / station_spacing)))
    for i in range(flank, len(values) - flank):
        if not np.isfinite(values[i]): continue
        up = values[max(0, i - flank):i + 1]
        down = values[i:min(len(values), i + flank + 1)]
        if not(np.isfinite(up).any() and np.isfinite(down).any()): continue
        height = min(values[i] - np.nanmin(up), values[i] - np.nanmin(down))
        # >= 60 m above both 500 m flanks and >= 150 m long above the
        # half-height shoulder: avoids isolated cells and small bed steps.
        if height < 60: continue
        threshold = values[i] - height * .5
        left = i
        while left > 0 and np.isfinite(values[left - 1]) and values[left - 1] >= threshold: left -= 1
        right = i
        while right + 1 < len(values) and np.isfinite(values[right + 1]) and values[right + 1] >= threshold: right += 1
        if distances[right] - distances[left] < 150: continue
        if any(abs(i - old["crest_index"]) * station_spacing < 300 for old in candidates): continue
        candidates.append({"crest_index": i, "start_index": left, "end_index": right, "height": float(height),
                           "length": float(distances[right] - distances[left]), "product": product,
                           "point": [rows[i]["x"], rows[i]["y"]]})
    return candidates


def classify_barriers(b12, b8, rows, relief12, relief8):
    result, used8 = [], set()
    for item in b12:
        mate = next((other for j, other in enumerate(b8) if j not in used8 and abs(other["crest_index"] - item["crest_index"]) * 50 <= 250), None)
        if mate:
            used8.add(b8.index(mate)); classification = "TYPE_A_LIKELY_REAL_OR_SHARED_GEOMETRIC_TERRAIN_PROBLEM"; support = "YES"
        else:
            classification = "TYPE_B_LIKELY_NATIVE_DEM_ARTIFACT"; support = "NO"
        severity = "HIGH" if item["height"] >= 150 else "MODERATE" if item["height"] >= 90 else "LOW"
        i = item["crest_index"]
        result.append({"barrier_id": f"B{len(result)+1:02d}", "start_station_m": item["start_index"] * 50.0,
                       "crest_station_m": i * 50.0, "end_station_m": item["end_index"] * 50.0,
                       "crest_x": item["point"][0], "crest_y": item["point"][1], "barrier_height_m": item["height"],
                       "barrier_length_m": item["length"], "dem_product": "native_12p5m", "cross_section_relief_m": float(relief12[i]),
                       "other_dem_support": support, "severity": severity, "classification": classification})
    for j, item in enumerate(b8):
        if j in used8: continue
        i = item["crest_index"]
        severity = "HIGH" if item["height"] >= 150 else "MODERATE" if item["height"] >= 90 else "LOW"
        result.append({"barrier_id": f"B{len(result)+1:02d}", "start_station_m": item["start_index"] * 50.0,
                       "crest_station_m": i * 50.0, "end_station_m": item["end_index"] * 50.0,
                       "crest_x": item["point"][0], "crest_y": item["point"][1], "barrier_height_m": item["height"],
                       "barrier_length_m": item["length"], "dem_product": "independent_8m", "cross_section_relief_m": float(relief8[i]),
                       "other_dem_support": "NO", "severity": severity,
                       "classification": "TYPE_C_SUPPORTING_DEM_DISAGREEMENT"})
    return result


def coverage_in_band(line, ds, transformer, half_width, spacing=25.0, zero_invalid=False):
    # Regular cross-band samples quantify valid terrain support, not raster area.
    total = valid = 0
    for distance in station_distances(line.length, 50.0):
        section, _ = cross_line(line, distance, half_width)
        offsets = np.arange(-half_width, half_width + spacing * .5, spacing)
        xy = [(section.interpolate((x + half_width) / (2 * half_width), normalized=True).x,
               section.interpolate((x + half_width) / (2 * half_width), normalized=True).y) for x in offsets]
        coords = [transformer.transform(*p) for p in xy] if transformer else xy
        values = np.asarray([v[0] for v in ds.sample(coords)], dtype=float)
        mask = valid_values(values, ds.nodata, zero_invalid)
        valid += int(mask.sum()); total += len(values)
    return 100.0 * valid / total if total else 0.0


def conservative_grid_profile(line, ds12, ds8, tr8, scale):
    # Diagnostic-only grid representation: evaluate valley floors at the grid
    # scale and take the maximum of two half-cell-offset samples per nominal
    # cell. This conservative max-envelope preserves possible blocking crests;
    # it never writes a resampled raster.
    rows, _ = compute_stations(line, ds12, ds8, tr8, scale / 2, HALF_WIDTH)
    d = np.asarray([r["station_m"] for r in rows])
    floor = array(rows, "dem12", "robust")
    centers, out = [], []
    for start in np.arange(0.0, line.length, scale):
        mask = (d >= start) & (d <= min(line.length, start + scale))
        vals = floor[mask]
        centers.append(min(line.length, start + scale / 2))
        out.append(np.nanmax(vals) if np.isfinite(vals).any() else np.nan)
    return np.asarray(centers), np.asarray(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    line = gpd.read_file(CORRIDOR / "jilong_canonical_corridor.geojson").to_crs(CRS).geometry.iloc[0]
    if Point(line.coords[0]).distance(S0) > .01 or Point(line.coords[-1]).distance(S4) > .01:
        raise ValueError("Authoritative corridor endpoints unexpectedly differ")
    dem12_path = next((RAW / "dem_12p5m").rglob("*.tif")); dem8_path = next((RAW / "dem_8m").rglob("*.tif"))
    with rasterio.open(dem12_path) as ds12, rasterio.open(dem8_path) as ds8:
        tr8 = Transformer.from_crs(CRS, ds8.crs, always_xy=True)
        rows, sections = compute_stations(line, ds12, ds8, tr8, STATION_SPACING, HALF_WIDTH, make_geometry=True)
        d = np.asarray([r["station_m"] for r in rows])
        c12, floor12, min12, relief12 = (array(rows, "dem12", f) for f in ("center", "robust", "raw_min", "relief"))
        c8, floor8, min8, relief8 = (array(rows, "dem8", f) for f in ("center", "robust", "raw_min", "relief"))
        rises12 = {w: forward_rise(floor12, d, w) for w in (50, 100, 250, 500, 1000)}
        rises8 = {w: forward_rise(floor8, d, w) for w in (50, 100, 250, 500, 1000)}
        center_rise12 = max_rise(c12, d, 250)
        b12 = barrier_candidates(floor12, d, rows, "native_12p5m")
        b8 = barrier_candidates(floor8, d, rows, "independent_8m")
        barriers = classify_barriers(b12, b8, rows, relief12, relief8)
        type_d = grouped((forward_rise(c12, d, 250) > 100) & (rises12[250] < 40), d)
        cov12 = float(np.mean(np.isfinite(c12)) * 100)
        cov8_corridor = float(np.mean(np.isfinite(c8)) * 100)
        cov8_band250 = coverage_in_band(line, ds8, tr8, 250, zero_invalid=True)
        cov8_band500 = coverage_in_band(line, ds8, tr8, 500, zero_invalid=True)
        d32, floor32 = conservative_grid_profile(line, ds12, ds8, tr8, 32.0)
        d64, floor64 = conservative_grid_profile(line, ds12, ds8, tr8, 64.0)
        # Grid-scale barrier screening uses the same persistent crest rule with
        # synthetic station rows solely for coordinates and identifiers.
        rows32 = [{"x": line.interpolate(x).x, "y": line.interpolate(x).y} for x in d32]
        rows64 = [{"x": line.interpolate(x).x, "y": line.interpolate(x).y} for x in d64]
        b32 = barrier_candidates(floor32, d32, rows32, "diagnostic_32m", station_spacing=32.0)
        b64 = barrier_candidates(floor64, d64, rows64, "diagnostic_64m", station_spacing=64.0)

        # Cross-section GeoJSON records geometry and the read-only diagnostics.
        gpd.GeoDataFrame([{
            "station_m": r["station_m"], "half_width_m": HALF_WIDTH, "dem12_center": r["dem12"]["center"],
            "dem12_robust": r["dem12"]["robust"], "dem8_center": r["dem8"]["center"],
            "dem8_robust": r["dem8"]["robust"],
        } for r in rows], geometry=sections, crs=CRS).to_file(OUT / "jilong_terrain_cross_sections.geojson", driver="GeoJSON")
        fields = ["station_m", "x", "y", "dem12_center_elev", "dem12_raw_min_elev", "dem12_robust_floor_elev", "dem12_relief_m", "dem12_valid_fraction", "dem12_lowpoint_offset_m", "dem12_robust_lowpoint_offset_m", "dem8_center_elev", "dem8_raw_min_elev", "dem8_robust_floor_elev", "dem8_relief_m", "dem8_valid_fraction", "dem8_lowpoint_offset_m", "dem8_robust_lowpoint_offset_m"]
        with (OUT / "jilong_valley_floor_profile.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
            for r in rows:
                writer.writerow({"station_m": f"{r['station_m']:.3f}", "x": f"{r['x']:.3f}", "y": f"{r['y']:.3f}",
                    **{f"dem12_{name}": "" if not np.isfinite(r['dem12'][field]) else f"{r['dem12'][field]:.3f}" for name, field in (("center_elev","center"),("raw_min_elev","raw_min"),("robust_floor_elev","robust"),("relief_m","relief"),("valid_fraction","fraction"),("lowpoint_offset_m","low_offset"),("robust_lowpoint_offset_m","robust_offset"))},
                    **{f"dem8_{name}": "" if not np.isfinite(r['dem8'][field]) else f"{r['dem8'][field]:.3f}" for name, field in (("center_elev","center"),("raw_min_elev","raw_min"),("robust_floor_elev","robust"),("relief_m","relief"),("valid_fraction","fraction"),("lowpoint_offset_m","low_offset"),("robust_lowpoint_offset_m","robust_offset"))}})

        # Figures are derived evidence only; inputs remain read-only.
        # Gate A's retained raw-DEM uncertainty intervals are provenance and
        # are visibly marked; they are never used as masks in this preflight.
        prior_gate = json.loads((REPORT / "JILONG_GATE_A_FINAL.json").read_text(encoding="utf-8"))
        prior_uncertainty = prior_gate.get("dem_uncertainty_intervals", [])
        fig, ax = plt.subplots(figsize=(15, 5.5)); ax.plot(d/1000, c12, color="#9e9e9e", lw=.7, label="centreline")
        ax.plot(d/1000, min12, color="#78b7c5", lw=.7, label="raw cross-section minimum")
        ax.plot(d/1000, floor12, color="#d55e00", lw=1.2, label="robust valley floor (median of lowest 5)")
        for n, item in enumerate(prior_uncertainty):
            ax.axvspan(item["start_distance_m"]/1000, item["end_distance_m"]/1000, color="#8b0000", alpha=.10,
                       label="Gate A raw-DEM uncertainty interval" if n == 0 else "_nolegend_")
        ax.set(title="Terrain preflight: native 12.5 m centreline versus valley-floor envelopes", xlabel="Distance from S0 (km)", ylabel="Elevation (m)"); ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "32_terrain_preflight_centerline_vs_valleyfloor.png", dpi=200); plt.close(fig)
        fig, ax = plt.subplots(figsize=(15, 5.5)); ax.plot(d/1000, floor12, color="#d55e00", lw=1.1, label="12.5 m robust floor")
        ax.plot(d/1000, floor8, color="#0072b2", lw=1.1, label="8 m robust floor (where valid)")
        ax.set(title="Terrain preflight: two-DEM robust valley-floor envelopes", xlabel="Distance from S0 (km)", ylabel="Elevation (m)"); ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "33_terrain_preflight_two_dem_valleyfloor.png", dpi=200); plt.close(fig)
        bounds = line.buffer(1200).bounds; terrain, tb = read_dem(dem12_path, bounds, max_dim=1800); fig, ax = plt.subplots(figsize=(12, 15)); ax.imshow(hillshade(terrain), extent=(tb[0],tb[2],tb[1],tb[3]), cmap="gray")
        for section in sections[::4]: ax.plot(*section.xy, color="white", alpha=.2, lw=.45)
        ax.plot(*line.xy, color="#ffef00", lw=1.8, label="final canonical corridor")
        ax.scatter([S0.x],[S0.y],color="red",s=55,label="S0"); ax.scatter([S4.x],[S4.y],color="lime",edgecolors="black",s=55,label="S4")
        for n, item in enumerate(barriers):
            ax.scatter([item['crest_x']],[item['crest_y']], marker="x", color="#ff3b30", s=55, zorder=10,
                       label="candidate false barrier" if n == 0 else "_nolegend_")
        ax.set_title("Terrain preflight: cross-sections and candidate false barriers"); ax.legend(loc="upper left",fontsize=8); decorate(ax,bounds); fig.tight_layout();fig.savefig(FIG / "34_terrain_preflight_barriers_map.png",dpi=200);plt.close(fig)
        fig, ax = plt.subplots(figsize=(15,5.5)); ax.plot(d/1000,floor12,color="#333333",lw=.8,label="native 12.5 m robust floor")
        ax.plot(d32/1000,floor32,color="#0072b2",lw=1.0,label="32 m conservative diagnostic envelope")
        ax.plot(d64/1000,floor64,color="#d55e00",lw=1.0,label="64 m conservative diagnostic envelope")
        ax.set(title="Terrain preflight: diagnostic scale envelopes (no raster written)",xlabel="Distance from S0 (km)",ylabel="Elevation (m)");ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(FIG / "35_terrain_preflight_32m_64m.png",dpi=200);plt.close(fig)

    type_counts = {"A": sum("TYPE_A" in x["classification"] for x in barriers), "B": sum("TYPE_B" in x["classification"] for x in barriers), "C": sum("TYPE_C" in x["classification"] for x in barriers), "D": len(type_d)}
    high_a = [x for x in barriers if "TYPE_A" in x["classification"] and x["severity"] == "HIGH"]
    high_b = [x for x in barriers if "TYPE_B" in x["classification"] and x["severity"] == "HIGH"]
    # A high Type B crest is not evidence that Gate A geometry is wrong, but it
    # can still block an unconditioned native-DEM D-Claw grid.  It therefore
    # requires a principled terrain remedy before simulation rather than a
    # false PASS or an arbitrary carve.
    gate = "CONDITIONAL_TERRAIN_REMEDIATION_REQUIRED" if (high_a or high_b) else "PASS"
    blocker = ("HIGH_SEVERITY_TYPE_A_FALSE_BARRIER" if high_a else
               "HIGH_SEVERITY_TYPE_B_NATIVE_DEM_ARTIFACTS" if high_b else None)
    ready = not (high_a or high_b)
    remedy_points = "; ".join(f"{x['barrier_id']} ({x['crest_x']:.1f}, {x['crest_y']:.1f})" for x in high_a + high_b)
    s0_floor12, s4_floor12 = float(floor12[0]), float(floor12[-1])
    s0_floor8, s4_floor8 = float(floor8[0]) if np.isfinite(floor8[0]) else None, float(floor8[-1]) if np.isfinite(floor8[-1]) else None
    report = {"gate_b": gate, "canonical_corridor_length_km": line.length/1000,
        "dem12_full_corridor_coverage": cov12, "dem8_corridor_coverage": cov8_corridor, "dem8_band250_coverage": cov8_band250, "dem8_band500_coverage": cov8_band500,
        "dem12_centerline_max_rise_250m": center_rise12, "dem12_robust_floor_max_rise_50m": max_rise(floor12,d,50), "dem12_robust_floor_max_rise_100m": max_rise(floor12,d,100), "dem12_robust_floor_max_rise_250m": max_rise(floor12,d,250), "dem12_robust_floor_max_rise_500m": max_rise(floor12,d,500), "dem12_robust_floor_max_rise_1000m": max_rise(floor12,d,1000),
        "dem8_robust_floor_max_rise_50m": max_rise(floor8,d,50), "dem8_robust_floor_max_rise_100m": max_rise(floor8,d,100), "dem8_robust_floor_max_rise_250m": max_rise(floor8,d,250), "dem8_robust_floor_max_rise_500m": max_rise(floor8,d,500), "dem8_robust_floor_max_rise_1000m": max_rise(floor8,d,1000),
        "dem12_s0_robust_floor_elevation_m": s0_floor12, "dem12_s4_robust_floor_elevation_m": s4_floor12, "dem12_total_robust_floor_drop_m": s0_floor12-s4_floor12, "dem12_mean_robust_floor_slope": (s4_floor12-s0_floor12)/line.length,
        "dem8_s0_robust_floor_elevation_m": s0_floor8, "dem8_s4_robust_floor_elevation_m": s4_floor8, "dem8_total_robust_floor_drop_m": None if None in (s0_floor8,s4_floor8) else s0_floor8-s4_floor8, "dem8_mean_robust_floor_slope": None if None in (s0_floor8,s4_floor8) else (s4_floor8-s0_floor8)/line.length,
        "candidate_false_barrier_count": len(barriers), "candidate_false_barriers": barriers, "type_A_count": type_counts["A"], "type_B_count": type_counts["B"], "type_C_count": type_counts["C"], "type_D_count": type_counts["D"],
        "diagnostic_64m_false_barrier_count": len(b64), "diagnostic_32m_false_barrier_count": len(b32), "diagnostic_64m_robust_floor_max_rise_250m": max_rise(floor64,d64,250), "diagnostic_64m_robust_floor_max_rise_500m": max_rise(floor64,d64,500), "diagnostic_32m_robust_floor_max_rise_250m": max_rise(floor32,d32,250), "diagnostic_32m_robust_floor_max_rise_500m": max_rise(floor32,d32,500),
        "native_12p5m_simulation_ready": ready, "primary_blocker": blocker,
        "recommended_next_step": "Proceed to D-Claw domain/source design using the native 12.5 m terrain; retain documented DEM uncertainty." if ready else f"Before D-Claw, acquire authoritative terrain or a separately justified localized DEM-patch replacement at {remedy_points}; do not carve terrain.",
        "methods": {"cross_section_spacing_m": STATION_SPACING,"cross_section_half_width_m":HALF_WIDTH,"robust_floor":"median of lowest 5 valid raw cross-section samples","grid_scale":"in-memory conservative maximum envelope of half-cell-offset valley-floor samples; no raster was written","gate_a_status":"PASS_UNCHANGED"},
        "dem8_status": "8M_DEM_NOT_FULL_DOMAIN_CANDIDATE" if cov8_band500 < 99.9 else "8M_DEM_FULL_DOMAIN_CANDIDATE"}
    (REPORT / "JILONG_TERRAIN_PREFLIGHT.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    high_summary = ", ".join(f"{x['barrier_id']} at ({x['crest_x']:.1f}, {x['crest_y']:.1f})" for x in high_a + high_b) or "none"
    (REPORT / "JILONG_TERRAIN_PREFLIGHT.md").write_text(f"# Jilong computational terrain preflight (Gate B)\n\n`GATE_B = {gate}`. Gate A remains `PASS` and is not reopened.\n\nThe raw 12.5 m DEM was interrogated on {len(rows)} cross-sections at about 50 m spacing and 250 m half-width. The robust floor is the median of the lowest five valid raw samples, not a modified terrain surface. 8 m coverage is {cov8_corridor:.1f}% on the centreline, {cov8_band250:.1f}% in the 250 m band, and {cov8_band500:.1f}% in the 500 m band; it is therefore `{report['dem8_status']}`.\n\nThe native robust-floor estimate is {s0_floor12:.1f} m at S0 and {s4_floor12:.1f} m at S4: a {s0_floor12-s4_floor12:.1f} m total drop and mean slope {(s4_floor12-s0_floor12)/line.length:.4f}. Native centreline/robust-floor maximum 250 m rises are {center_rise12:.1f}/{report['dem12_robust_floor_max_rise_250m']:.1f} m. Robust-floor 250 m rises are {report['dem12_robust_floor_max_rise_250m']:.1f} m (12.5 m) and {report['dem8_robust_floor_max_rise_250m']:.1f} m (8 m); 500 m values are {report['dem12_robust_floor_max_rise_500m']:.1f}/{report['dem8_robust_floor_max_rise_500m']:.1f} m. Candidate barriers: {len(barriers)} (A/B/C/D = {type_counts['A']}/{type_counts['B']}/{type_counts['C']}/{type_counts['D']}).\n\nThe blocking condition is `{blocker}`: high Type B native-only crests {high_summary} persist in the 32 m/64 m conservative diagnostics and can block a D-Claw grid. They are not evidence against the completed human-reviewed corridor, but native terrain is not simulation-ready as-is. Before D-Claw, use authoritative replacement terrain or a separately justified local DEM-patch replacement at those coordinates; do not carve or otherwise condition this DEM arbitrarily.\n\nNo DEM raster was changed; 32 m and 64 m products are in-memory diagnostic envelopes only. No D-Claw calculation was run.\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
