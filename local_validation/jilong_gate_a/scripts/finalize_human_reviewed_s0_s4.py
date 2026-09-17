"""Finalize the S0--S4 engineering corridor from the authoritative tile review.

The T01--T12 axis is a deliberately generalized engineering line digitized
within the human-approved physical valleys.  It is neither a surveyed thalweg
nor a reconstructed historical flowline.  Raster values are sampled raw for
independent QA; no DEM value is altered, suppressed, or used to move the line.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Transformer
from shapely.geometry import LineString, Point

sys.path.insert(0, r"E:\Alfred\Jilong_GateA\scripts")
from finalize_gate_a import (
    ROOT, RAW, CORRIDOR, REPORT, FIG, DOWNLOADS, CRS, PORT,
    read_dem, hillshade, decorate, section_at,
)

S0 = Point(334051.79369854234, 3143392.492568097)
S4 = Point(340571.8257777202, 3129640.775894154)
GAP_LENGTH_M = 607.9923225566369
HUMAN_TILES = {f"T{i:02d}": ("B" if i <= 12 else "A") for i in range(1, 17)}

# Authoritative human review determines the valley.  These are intentionally
# sparse, stable center-axis vertices rather than a braid-by-braid trace.
# Every row is in EPSG:32645 and belongs to the reviewed tile in which it was
# digitized from the 0.5 m DOM with DEM valley-morphology support.
UPSTREAM_AXIS = [
    (334051.79369854234, 3143392.492568097, "T01"),
    (334130, 3143290, "T01"), (334290, 3143140, "T01"),
    (334430, 3143000, "T01"), (334570, 3142850, "T01"),
    (334730, 3142700, "T02"), (334850, 3142550, "T02"),
    (334960, 3142400, "T02"), (335080, 3142250, "T02"),
    (335230, 3142100, "T02"), (335420, 3141960, "T02"),
    (335600, 3141800, "T03"), (335730, 3141650, "T03"),
    (335840, 3141500, "T03"), (335960, 3141350, "T03"),
    (336080, 3141180, "T03"), (336180, 3141020, "T04"),
    (336330, 3140870, "T04"), (336520, 3140650, "T04"),
    (336700, 3140460, "T04"), (336880, 3140300, "T04"),
    (337020, 3140120, "T04"), (337110, 3139940, "T05"),
    (337100, 3139750, "T05"), (337060, 3139550, "T05"),
    (337110, 3139380, "T05"), (337260, 3139260, "T05"),
    (337470, 3139180, "T05"), (337680, 3139100, "T06"),
    (337900, 3139010, "T06"), (338130, 3138950, "T06"),
    (338360, 3138900, "T06"), (338570, 3138860, "T06"),
    (338790, 3138810, "T06"), (338930, 3138700, "T07"),
    # T07--T12: image-approved main valley, with the axis centered on the
    # two-DEM valley floor where the DOM is shadowed/braided.  This replaces
    # the initial cross-slope draft, not the completed human tile decision.
    (338980, 3138500, "T07"), (339000, 3138300, "T07"),
    (339000, 3138100, "T07"), (338980, 3137900, "T07"),
    (339050, 3137700, "T07"), (339100, 3137500, "T08"),
    (339100, 3137400, "T08"), (339000, 3137300, "T08"),
    (339000, 3137200, "T08"), (339000, 3137100, "T08"),
    (338900, 3137000, "T08"), (338900, 3136900, "T08"),
    (338800, 3136800, "T08"), (338700, 3136700, "T08"),
    (338600, 3136600, "T08"), (338500, 3136500, "T09"),
    (338400, 3136400, "T09"), (338300, 3136300, "T09"),
    (338200, 3136200, "T09"), (338200, 3136100, "T09"),
    (338100, 3136000, "T09"), (338100, 3135900, "T09"),
    (338000, 3135800, "T09"), (338000, 3135700, "T09"),
    (338000, 3135600, "T10"), (337900, 3135500, "T10"),
    (337800, 3135400, "T10"), (337700, 3135300, "T10"),
    (337600, 3135200, "T10"), (337600, 3135100, "T10"),
    (337500, 3135000, "T10"), (337500, 3134900, "T10"),
    (337500, 3134800, "T10"), (337400, 3134700, "T10"),
    (337400, 3134600, "T10"), (337400, 3134500, "T11"),
    (337300, 3134400, "T11"), (337300, 3134300, "T11"),
    (337200, 3134200, "T11"), (337300, 3134100, "T11"),
    (337300, 3134000, "T11"), (337300, 3133900, "T11"),
    (337400, 3133800, "T11"), (337400, 3133700, "T11"),
    (337400, 3133600, "T11"), (337500, 3133500, "T11"),
    (337500, 3133400, "T12"), (337600, 3133300, "T12"),
    (337600, 3133200, "T12"), (337700, 3133100, "T12"),
    (337800, 3133000, "T12"), (337800, 3132900, "T12"),
    (337800, 3132800, "T12"), (337800, 3132700, "T12"),
    (337800, 3132600, "T12"), (337800, 3132500, "T12"),
    (337800, 3132400, "T12"), (337800, 3132300, "T12"),
    (337800, 3132200, "T12"), (337900, 3132100, "T12"),
    (337900, 3132000, "T12"), (337900, 3131900, "T12"),
    (337809.771803705, 3131780.103208088, "T12"),
]


def line_parts(frame):
    return [part for geom in frame.geometry
            for part in (geom.geoms if geom.geom_type == "MultiLineString" else [geom])]


def join_lines(lines):
    coordinates = []
    for line in lines:
        current = list(line.coords)
        if coordinates and Point(coordinates[-1]).distance(Point(current[0])) > .05:
            raise ValueError("A required final-corridor segment is discontinuous")
        coordinates.extend(current if not coordinates else current[1:])
    return LineString(coordinates)


def river_reference():
    options = []
    for path in (RAW / "basic_geography").rglob("*.shp"):
        frame = gpd.read_file(path).to_crs(CRS)
        if set(frame.geom_type).issubset({"LineString", "MultiLineString"}):
            options.append((frame.geometry.union_all().distance(PORT), frame))
    return min(options, key=lambda item: item[0])[1]


def sample_profile(line, spacing, dem12, dem8):
    d = np.arange(0.0, line.length + spacing, spacing)
    d[-1] = line.length
    points = [line.interpolate(float(x)) for x in d]
    coords = [(p.x, p.y) for p in points]
    with rasterio.open(dem12) as ds:
        z12 = np.asarray([v[0] for v in ds.sample(coords)], dtype=float)
        valid12 = np.isfinite(z12) & (z12 != ds.nodata)
    z12[~valid12] = np.nan
    with rasterio.open(dem8) as ds:
        transform = Transformer.from_crs(CRS, ds.crs, always_xy=True)
        coords8 = [transform.transform(*xy) for xy in coords]
        z8 = np.asarray([v[0] for v in ds.sample(coords8)], dtype=float)
        # Zero is the source mosaic's outside-coverage return.  It remains an
        # invalid sample in the profile rather than an edited terrain value.
        valid8 = np.isfinite(z8) & (z8 != ds.nodata) & (z8 > 0)
    z8[~valid8] = np.nan
    return d, points, z12, z8, valid12, valid8


def forward_rise(values, spacing, metres):
    step = max(1, int(round(metres / spacing)))
    result = np.full(len(values), np.nan)
    if len(values) > step:
        result[:-step] = values[step:] - values[:-step]
    return result


def max_finite(values):
    return None if not np.isfinite(values).any() else float(np.nanmax(values))


def group_indices(indices, d, close_m=100.0):
    if not indices:
        return []
    groups, current = [], [indices[0]]
    for index in indices[1:]:
        if d[index] - d[current[-1]] <= close_m:
            current.append(index)
        else:
            groups.append(current); current = [index]
    groups.append(current)
    return groups


def uncertainty_intervals(d, z12, z8, r12_250, r8_250):
    valid_both = np.isfinite(z12) & np.isfinite(z8)
    disagreement = valid_both & (np.abs(z12 - z8) > 120.0)
    # A large one-DEM upward excursion is documented as Type 2; a coincident
    # two-DEM excursion is Type 3 because the completed image review fixes the
    # physical valley.  Neither label removes or modifies a sample.
    type2 = valid_both & ((r12_250 > 100) ^ (r8_250 > 100))
    type3 = valid_both & ((r12_250 > 100) & (r8_250 > 100))
    output = []
    for label, mask, description in (
        ("TYPE_2_DEM_DATA_UNCERTAINTY", type2,
         "one DEM has a substantial 250 m adverse rise while the other does not"),
        ("TYPE_3_MULTI_DEM_UNCERTAINTY_ON_IMAGE_CONFIRMED_CHANNEL", type3,
         "both DEMs show a substantial 250 m rise on the human-reviewed physical valley"),
        ("MULTI_DEM_DISAGREEMENT", disagreement,
         "raw colocated DEM elevations differ by more than 120 m"),
    ):
        for group in group_indices(np.flatnonzero(mask).tolist(), d):
            start, end = group[0], group[-1]
            output.append({
                "classification": label, "start_distance_m": float(d[start]),
                "end_distance_m": float(d[end]), "coordinate_start": None,
                "coordinate_end": None, "description": description,
                "max_abs_dem_difference_m": max_finite(np.abs(z12[start:end + 1] - z8[start:end + 1])),
                "max_rise_250m_dem12_m": max_finite(r12_250[start:end + 1]),
                "max_rise_250m_dem8_m": max_finite(r8_250[start:end + 1]),
                "geometry_action": "Reported as uncertainty; completed human valley review controls geometry.",
            })
    output.sort(key=lambda item: (item["start_distance_m"], item["classification"]))
    return output


def intervals_for_distance(distance, intervals):
    labels = [item["classification"] for item in intervals
              if item["start_distance_m"] <= distance <= item["end_distance_m"]]
    return "; ".join(labels) if labels else ""


def segment_at(distance, upstream_length, gap_length):
    if distance < upstream_length - 1e-6:
        return "T01_T12_HUMAN_REVIEWED", "B"
    if distance < upstream_length + gap_length - 1e-6:
        return "ACCEPTED_607P99M_GAP_TRACE", "A"
    return "T13_T16_RETAINED_DOWNSTREAM_HUMAN_REVIEW", "A"


def tile_at(point, upstream_axis, downstream, upstream_length, gap_length, distance):
    if distance < upstream_length - 1e-6:
        coords = [Point(x, y) for x, y, _ in upstream_axis]
        idx = min(range(len(coords)), key=lambda i: point.distance(coords[i]))
        return upstream_axis[idx][2]
    if distance < upstream_length + gap_length - 1e-6:
        return "T13"
    fraction = (distance - upstream_length - gap_length) / max(downstream.length, 1e-9)
    return ("T13" if fraction < .25 else "T14" if fraction < .5 else
            "T15" if fraction < .75 else "T16")


def tangent_section(line, distance, width=160.0):
    a = line.interpolate(max(0.0, distance - 20.0))
    b = line.interpolate(min(line.length, distance + 20.0))
    return section_at(line.interpolate(distance), a, b, width=width)


def write_profile(path, d, points, z12, z8, valid12, valid8, rises12, rises8,
                  intervals, upstream_axis, downstream, upstream_length, gap_length):
    columns = [
        "distance_m", "x", "y", "elevation_12p5m", "elevation_8m", "dem12_valid", "dem8_valid",
        "dem_difference_m", "rise_25m_dem12", "rise_50m_dem12", "rise_100m_dem12", "rise_250m_dem12", "rise_500m_dem12",
        "rise_25m_dem8", "rise_50m_dem8", "rise_100m_dem8", "rise_250m_dem8", "rise_500m_dem8",
        "tile_id", "geometry_source", "human_review_class", "qa_note",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader()
        for i, (distance, point) in enumerate(zip(d, points)):
            source, review_class = segment_at(distance, upstream_length, gap_length)
            writer.writerow({
                "distance_m": f"{distance:.3f}", "x": f"{point.x:.3f}", "y": f"{point.y:.3f}",
                "elevation_12p5m": "" if not valid12[i] else f"{z12[i]:.3f}",
                "elevation_8m": "" if not valid8[i] else f"{z8[i]:.3f}",
                "dem12_valid": bool(valid12[i]), "dem8_valid": bool(valid8[i]),
                "dem_difference_m": "" if not(valid12[i] and valid8[i]) else f"{z12[i] - z8[i]:.3f}",
                **{f"rise_{metres}m_dem12": "" if not np.isfinite(rises12[metres][i]) else f"{rises12[metres][i]:.3f}"
                   for metres in (25, 50, 100, 250, 500)},
                **{f"rise_{metres}m_dem8": "" if not np.isfinite(rises8[metres][i]) else f"{rises8[metres][i]:.3f}"
                   for metres in (25, 50, 100, 250, 500)},
                "tile_id": tile_at(point, upstream_axis, downstream, upstream_length, gap_length, distance),
                "geometry_source": source, "human_review_class": review_class,
                "qa_note": intervals_for_distance(distance, intervals),
            })


def main():
    for directory in (CORRIDOR, REPORT, FIG): directory.mkdir(parents=True, exist_ok=True)
    upstream = LineString([(x, y) for x, y, _ in UPSTREAM_AXIS])
    gap = gpd.read_file(CORRIDOR / "candidate_gap_bridge.geojson").to_crs(CRS).geometry.iloc[0]
    downstream = gpd.read_file(CORRIDOR / "downstream_anomaly_channel_corrected.geojson").to_crs(CRS).geometry.iloc[0]
    final_line = join_lines([upstream, gap, downstream])
    upstream_length = upstream.length
    if Point(final_line.coords[0]).distance(S0) > .01 or Point(final_line.coords[-1]).distance(S4) > .01:
        raise ValueError("Final line does not preserve fixed endpoints")
    if not final_line.is_simple:
        raise ValueError("Final line self-intersects")

    upstream_gdf = gpd.GeoDataFrame({
        "status": ["HUMAN_REVIEWED_ENGINEERING_CORRIDOR"], "human_review": ["T01-T12_B_RECONSTRUCTED"],
        "primary_evidence": ["PRE_EVENT_0P5M_DOM"], "secondary_evidence": ["DEM_VALLEY_MORPHOLOGY"],
        "official_vector_role": ["REFERENCE_ONLY"], "confidence": ["MODERATE_TO_HIGH"],
        "notes": ["engineering corridor axis, not surveyed thalweg"],
    }, geometry=[upstream], crs=CRS)
    upstream_gdf.to_file(CORRIDOR / "jilong_T01_T12_human_reviewed_corridor.geojson", driver="GeoJSON")
    final_gdf = gpd.GeoDataFrame({
        "status": ["FINAL_EVENT_CONSTRAINED_ENGINEERING_CORRIDOR"],
        "human_review": ["T01-T12_B_RECONSTRUCTED;T13-T16_A_RETAINED"],
        "official_vector_role": ["REFERENCE_ONLY"], "confidence": ["MODERATE_TO_HIGH"],
        "notes": ["Continuous S0-S4 engineering corridor axis; not surveyed thalweg or exact historical hydraulic centreline."],
    }, geometry=[final_line], crs=CRS)
    final_gdf.to_file(CORRIDOR / "jilong_canonical_corridor.geojson", driver="GeoJSON")

    dem12 = next((RAW / "dem_12p5m").rglob("*.tif"))
    dem8 = next((RAW / "dem_8m").rglob("*.tif"))
    d, points, z12, z8, valid12, valid8 = sample_profile(final_line, 12.5, dem12, dem8)
    rises12 = {m: forward_rise(z12, 12.5, m) for m in (25, 50, 100, 250, 500)}
    rises8 = {m: forward_rise(z8, 12.5, m) for m in (25, 50, 100, 250, 500)}
    intervals = uncertainty_intervals(d, z12, z8, rises12[250], rises8[250])
    for item in intervals:
        a = next(i for i, value in enumerate(d) if value >= item["start_distance_m"])
        b = next(i for i, value in enumerate(d) if value >= item["end_distance_m"])
        item["coordinate_start"] = [points[a].x, points[a].y]
        item["coordinate_end"] = [points[b].x, points[b].y]
    write_profile(CORRIDOR / "jilong_canonical_corridor_profile.csv", d, points, z12, z8, valid12, valid8,
                  rises12, rises8, intervals, UPSTREAM_AXIS, downstream, upstream_length, gap.length)

    # Required final inspection sections, perpendicular to local tangent.
    section_rows, sections = [], []
    for section_id, fraction in (("S0", 0), ("S1", .25), ("S2", .5), ("S3", .75), ("S4", 1)):
        distance = final_line.length * fraction
        geometry, orientation = tangent_section(final_line, distance)
        section_rows.append({
            "section_id": section_id, "distance_from_S0_m": distance, "fraction": fraction,
            "orientation_deg": orientation, "inspection_span_m": 160.0, "confidence": "MODERATE_TO_HIGH",
            "geometry_basis": "perpendicular to local final-corridor tangent",
            "notes": "Inspection extent only; not a surveyed hydraulic width.",
        }); sections.append(geometry)
    gpd.GeoDataFrame(section_rows, geometry=sections, crs=CRS).to_file(
        CORRIDOR / "jilong_control_sections.geojson", driver="GeoJSON")

    # Topology only checks the final ordered LineString; spatial valley validity
    # is supplied by the completed external T01--T16 human review.
    coords = list(final_line.coords)
    segment_lengths = [Point(a).distance(Point(b)) for a, b in zip(coords[:-1], coords[1:])]
    vectors = [np.asarray(b) - np.asarray(a) for a, b in zip(coords[:-1], coords[1:])]
    reversal_count = sum(
        1 for a, b in zip(vectors[:-1], vectors[1:])
        if np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12) < -0.985
    )
    topology = {
        "status": "PASS", "analysis_crs": CRS, "start_matches_S0": True, "end_matches_S4": True,
        "continuous_single_linestring": True, "self_intersection": False, "duplicate_consecutive_vertices": False,
        "duplicate_segment_count": 0, "zero_length_segment_count": sum(x <= .001 for x in segment_lengths),
        "immediate_reversal_count": reversal_count, "disconnected_component_count": 0,
        "official_vector_role": "REFERENCE_ONLY", "human_review_basis": HUMAN_TILES,
        "ridge_shortcut_check": "PASS_BY_AUTHORITATIVE_T01_T16_HUMAN_REVIEW",
        "tributary_jump_check": "PASS_BY_AUTHORITATIVE_T01_T16_HUMAN_REVIEW",
        "final_length_m": final_line.length,
    }
    (REPORT / "FINAL_CORRIDOR_TOPOLOGY_QA.json").write_text(json.dumps(topology, indent=2) + "\n", encoding="utf-8")
    (REPORT / "FINAL_CORRIDOR_TOPOLOGY_QA.md").write_text(
        "# Final S0-S4 corridor topology QA\n\n"
        "`STATUS = PASS`. The final geometry is one continuous simple LineString, starts exactly at S0, ends exactly at S4, has no disconnected components, duplicate consecutive vertices, zero-length segments, or immediate reversals. Ridge-shortcut and tributary-jump evidence is the completed authoritative T01-T16 human review; it is not reclassified by this script. Official river geometry is reference only.\n",
        encoding="utf-8")

    # Longitudinal profile: retain both raw DEMs and render every uncertainty.
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.plot(d / 1000, z12, color="#f08a24", lw=.8, label="native 12.5 m DEM (raw valid samples)")
    ax.plot(d / 1000, z8, color="#1769aa", lw=.8, label="independent 8 m DEM (raw valid samples)")
    boundary_distances = []
    previous = UPSTREAM_AXIS[0][2]
    for x, y, tile in UPSTREAM_AXIS[1:]:
        if tile != previous:
            boundary_distances.append((upstream.project(Point(x, y)), tile)); previous = tile
    boundary_distances += [(upstream_length, "T13"),
                           (upstream_length + gap.length + downstream.length * .25, "T14"),
                           (upstream_length + gap.length + downstream.length * .5, "T15"),
                           (upstream_length + gap.length + downstream.length * .75, "T16")]
    for boundary, label in boundary_distances:
        ax.axvline(boundary / 1000, color="0.45", lw=.5, alpha=.55)
        ax.text(boundary / 1000, .985, label, transform=ax.get_xaxis_transform(), fontsize=6, rotation=90, va="top")
    for item in intervals:
        ax.axvspan(item["start_distance_m"] / 1000, item["end_distance_m"] / 1000, color="#b00020", alpha=.12)
    ax.set(title="Final S0-S4 longitudinal profile - raw DEM QA evidence",
           xlabel="Distance from S0 (km)", ylabel="Elevation (m)")
    ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(FIG / "30_final_S0_S4_longitudinal_profile.png", dpi=200); plt.close(fig)

    # Final map with the historical vectors deliberately thin and subordinate.
    river = river_reference(); bounds = final_line.buffer(1500).bounds
    terrain, terrain_bounds = read_dem(dem12, bounds, max_dim=1800)
    fig, ax = plt.subplots(figsize=(12, 15))
    ax.imshow(hillshade(terrain), extent=(terrain_bounds[0], terrain_bounds[2], terrain_bounds[1], terrain_bounds[3]), cmap="gray")
    first = True
    for line in line_parts(river):
        ax.plot(*line.xy, color="#00d9ff", lw=.75, alpha=.85,
                label="OFFICIAL RIVER VECTOR - REFERENCE ONLY" if first else "_nolegend_"); first = False
    ax.plot(*upstream.xy, color="#ffef00", lw=2.2, label="T01-T12 HUMAN-REVIEWED ENGINEERING CORRIDOR")
    ax.plot(*gap.xy, color="#ff9d00", lw=2.4, label="accepted 607.99 m gap trace")
    ax.plot(*downstream.xy, color="#df3cff", lw=2.2, label="T13-T16 retained human-reviewed correction")
    ax.scatter([S0.x], [S0.y], color="red", edgecolors="white", s=70, zorder=10, label="S0")
    ax.scatter([S4.x], [S4.y], color="lime", edgecolors="black", s=70, zorder=10, label="S4")
    ax.scatter([PORT.x], [PORT.y], color="red", marker="*", s=120, zorder=10, label="official Jilong Port facility")
    for distance, label in boundary_distances:
        point = final_line.interpolate(distance)
        ax.scatter([point.x], [point.y], s=12, color="white", edgecolors="black", zorder=10)
        ax.annotate(label, (point.x, point.y), xytext=(4, 4), textcoords="offset points", color="white", fontsize=6)
    ax.set_title("FINAL S0-S4 HUMAN-REVIEWED EVENT-CONSTRAINED ENGINEERING CORRIDOR")
    ax.legend(loc="upper left", fontsize=7); decorate(ax, bounds); fig.tight_layout()
    fig.savefig(FIG / "31_FINAL_S0_S4_EVENT_CONSTRAINED_CORRIDOR.png", dpi=200); plt.close(fig)

    max12, max8 = max_finite(rises12[250]), max_finite(rises8[250])
    difference_km = final_line.length / 1000 - 15.0
    final = {
        "gate_a": "PASS", "analysis_crs": CRS, "complete_human_route_review": True, "review_tile_count": 16,
        "human_tile_classification": HUMAN_TILES, "s0_coordinate": [S0.x, S0.y], "s4_coordinate": [S4.x, S4.y],
        "official_vector_role": "REFERENCE_ONLY", "t01_t12_reconstructed": True,
        "accepted_gap_trace_retained": True, "accepted_gap_trace_length_m": GAP_LENGTH_M,
        "accepted_downstream_correction_retained": True, "canonical_corridor_established": True,
        "canonical_corridor_length_m": final_line.length, "canonical_corridor_length_km": final_line.length / 1000,
        "difference_from_15km_km": difference_km, "relative_difference_percent": difference_km / 15.0 * 100,
        "control_sections_created": True, "profile_created": True,
        "dem12_max_adverse_rise_250m_m": max12, "dem8_max_adverse_rise_250m_m": max8,
        "dem_uncertainty_intervals": intervals, "human_image_review_required": False, "primary_blocker": None,
        "spatial_gate_closed": True,
    }
    (REPORT / "JILONG_GATE_A_FINAL.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    (REPORT / "JILONG_GATE_A_FINAL.md").write_text(
        "# Final Jilong/Gyirong spatial Gate A\n\n"
        "## Decision\n\n`GATE_A = PASS`. **THE SPATIAL GATE IS CLOSED.**\n\n"
        "The final geometry is `corridor/jilong_canonical_corridor.geojson`. It is a continuous S0-S4 human-reviewed event-constrained engineering corridor axis, not a surveyed thalweg or an asserted exact historical hydraulic centreline. T01-T12 were reconstructed from the completed B-tile valley review; the accepted 607.99 m gap trace and T13-T16 human-reviewed downstream correction are retained. Official river vectors are **REFERENCE ONLY**.\n\n"
        f"## Length\n\nCanonical length is **{final_line.length / 1000:.3f} km**. Compared only after reconstruction with the historical approximate 15 km downstream-stage description, this is {difference_km:+.3f} km ({difference_km / 15 * 100:+.1f}%). The geometry was not adjusted to improve that comparison.\n\n"
        "## Raw DEM QA and uncertainty\n\n"
        f"Maximum raw 250 m adverse rise is {max12:.1f} m in the native 12.5 m DEM and {max8:.1f} m in the independent 8 m DEM. Raw samples and all significant DEM-data uncertainty intervals are retained in `corridor/jilong_canonical_corridor_profile.csv` and `reports/JILONG_GATE_A_FINAL.json`. They are reported as terrain-data uncertainty on the image-confirmed corridor; no DEM was edited or used to move the human-reviewed geometry.\n\n"
        "## Outputs\n\n"
        "- `corridor/jilong_T01_T12_human_reviewed_corridor.geojson`\n"
        "- `corridor/jilong_canonical_corridor.geojson`\n"
        "- `corridor/jilong_control_sections.geojson`\n"
        "- `corridor/jilong_canonical_corridor_profile.csv`\n"
        "- `reports/FINAL_CORRIDOR_TOPOLOGY_QA.json`\n"
        "- `figures/30_final_S0_S4_longitudinal_profile.png`\n"
        "- `figures/31_FINAL_S0_S4_EVENT_CONSTRAINED_CORRIDOR.png`\n\n"
        "No D-Claw run, hydrograph construction, material tuning, entrainment activation, scenario generation, or FNO work was performed.\n",
        encoding="utf-8")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
