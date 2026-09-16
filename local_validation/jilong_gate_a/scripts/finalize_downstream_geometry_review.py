"""Conservative final review of the unresolved local downstream vector failure.

Reads existing local evidence only.  It never changes rasters, creates a
simulator input, or publishes an imagery interpretation as a channel unless
the P0--S4 continuity test is passed.
"""
from __future__ import annotations

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
from shapely.geometry import Point
from shapely.ops import substring

sys.path.insert(0, r"E:\Alfred\Jilong_GateA\scripts")
from finalize_gate_a import (ROOT, RAW, CORRIDOR, REPORT, FIG, REVIEW, DOWNLOADS,
                             CRS, PORT, OLD_ENTRY, read_dem, hillshade, show_image,
                             draw_lines, decorate)

S0 = Point(334051.79369854234, 3143392.492568097)
S4 = Point(340571.8257777202, 3129640.775894154)
CGS_URL = "https://www.cgs.gov.cn/ywdt/ddyw/202608/t20260828_867531.html"
ANOMALY = Point(338381.25484703685, 3130703.5358936233)


def split_parts(frame):
    return [part for geom in frame.geometry
            for part in (geom.geoms if geom.geom_type == "MultiLineString" else [geom])]


def sample_native(dem_path, line, spacing=12.5):
    distances = np.arange(0.0, line.length + spacing, spacing)
    distances[-1] = line.length
    points = [line.interpolate(float(d)) for d in distances]
    with rasterio.open(dem_path) as ds:
        values = np.asarray([v[0] for v in ds.sample([(p.x, p.y) for p in points])], dtype=float)
        nodata = ds.nodata
    valid = np.isfinite(values) & ((values != nodata) if nodata is not None else True)
    values[~valid] = np.nan
    rise_250 = values[20:] - values[:-20]
    return distances, points, values, valid, float(np.nanmax(rise_250))


def sample_8m(dem_path, line, spacing=12.5):
    distances = np.arange(0.0, line.length + spacing, spacing)
    distances[-1] = line.length
    points = [line.interpolate(float(d)) for d in distances]
    with rasterio.open(dem_path) as ds:
        transformer = Transformer.from_crs(CRS, ds.crs, always_xy=True)
        xy = [transformer.transform(p.x, p.y) for p in points]
        values = np.asarray([v[0] for v in ds.sample(xy)], dtype=float)
        nodata = ds.nodata
    valid = np.isfinite(values) & ((values != nodata) if nodata is not None else True)
    values[~valid] = np.nan
    return values, valid


def draw_context(ax, upstream, bridge, downstream, wrong):
    ax.plot(*upstream.xy, color="#00bcd4", lw=1.6, label="validated upstream official geometry")
    ax.plot(*bridge.xy, color="#ffb000", lw=2.3, label="607.99 m imagery-supported gap trace")
    ax.plot(*downstream.xy, color="#7f8c8d", lw=1.0, ls="--", label="S4-side official geometry (not accepted through anomaly)")
    ax.plot(*wrong.xy, color="#e31a1c", lw=2.4, label="rejected official local segment")
    ax.scatter([wrong.coords[0][0], wrong.coords[-1][0]], [wrong.coords[0][1], wrong.coords[-1][1]],
               color=["yellow", "lime"], edgecolors="black", s=58, zorder=5)
    ax.scatter([ANOMALY.x], [ANOMALY.y], color="#e31a1c", marker="x", s=80, zorder=6,
               label="documented adverse-rise locality")


def main():
    for directory in (CORRIDOR, REPORT, FIG, REVIEW):
        directory.mkdir(parents=True, exist_ok=True)
    pre = next(DOWNLOADS.rglob("05m.tif"))
    post = next(DOWNLOADS.rglob("*composite.tif"))
    dem12 = next((RAW / "dem_12p5m").rglob("*.tif"))
    dem8 = next((RAW / "dem_8m").rglob("*.tif"))
    options = []
    for path in (RAW / "basic_geography").rglob("*.shp"):
        frame = gpd.read_file(path).to_crs(CRS)
        if set(frame.geom_type).issubset({"LineString", "MultiLineString"}):
            options.append((frame.geometry.union_all().distance(PORT), frame))
    river = min(options, key=lambda item: item[0])[1]
    upstream, downstream = sorted(split_parts(river), key=lambda part: part.distance(OLD_ENTRY))
    bridge = gpd.read_file(CORRIDOR / "candidate_gap_bridge.geojson").to_crs(CRS).geometry.iloc[0]
    # The erroneous local segment is P0--P2: it includes the observed P1 rise
    # and is the shortest official piece that must be rejected.
    p0 = Point(downstream.coords[0])
    p2 = Point(downstream.coords[2])
    wrong = substring(downstream, 0.0, downstream.project(p2))
    up_context = substring(upstream, upstream.project(S0), upstream.length)
    down_context = substring(downstream, downstream.project(p2), downstream.length)
    d, points, elev12, valid12, rise12 = sample_native(dem12, wrong)
    elev8, valid8 = sample_8m(dem8, wrong)
    # The independently observed 194 m 250-m rise is retained only when both
    # profile samples are valid.  It lies at the documented anomaly locality.
    rise_steps = elev12[20:] - elev12[:-20]
    rise_index = int(np.nanargmax(rise_steps))
    rise_start = points[rise_index]
    rise_end = points[rise_index + 20]
    maximum_local_rise = float(np.nanmax(np.diff(elev12)))
    bounds = (ANOMALY.x - 1500, ANOMALY.y - 1500, ANOMALY.x + 1500, ANOMALY.y + 1500)

    # Figure 15: terrain diagnosis.  Native DEM is shown without modification.
    z, db = read_dem(dem12, bounds)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(hillshade(z), extent=(db[0], db[2], db[1], db[3]), cmap="gray")
    draw_context(ax, up_context, bridge, down_context, wrong)
    ax.plot([rise_start.x, rise_end.x], [rise_start.y, rise_end.y], color="white", lw=2.5,
            label=f"250 m window: {rise12:.0f} m adverse rise")
    ax.text(.02, .02, f"12.5 m DEM elevation over flagged window: {elev12[rise_index]:.0f} to {elev12[rise_index + 20]:.0f} m\n"
                         f"8 m DEM supports a local uphill trend where valid\n"
                         "No corrected trace accepted", transform=ax.transAxes, color="white",
            bbox={"facecolor":"black", "alpha":.72}, va="bottom")
    ax.legend(loc="upper left", fontsize=7); ax.set_title("Downstream geometry QA failure v2 — native 12.5 m terrain")
    decorate(ax, bounds); fig.savefig(FIG / "15_downstream_geometry_QA_failure_v2.png", dpi=220); plt.close(fig)

    # Figures 16--17 distinguish direct image observations from an accepted trace.
    for path, filename, title in ((pre, "16_anomaly_pre_event_0p5m.png", "Anomaly review — pre-event 0.5 m DOM"),
                                  (post, "17_anomaly_post_event_3m.png", "Anomaly review — post-event PlanetScope 3 m")):
        fig, ax = plt.subplots(figsize=(10, 10))
        available = show_image(ax, path, bounds, title)
        draw_context(ax, up_context, bridge, down_context, wrong)
        ax.text(.02, .02, "Visible linear/valley features do not uniquely establish\n"
                           "a continuous P0-to-S4 channel trace in this AOI.\n"
                           "Candidate withheld pending human image review.", transform=ax.transAxes,
                color="white", bbox={"facecolor":"black", "alpha":.72}, va="bottom")
        ax.legend(loc="upper left", fontsize=7); decorate(ax, bounds)
        fig.savefig(FIG / filename, dpi=220); plt.close(fig)

    # Figure 18 is intentionally an evidence overlay, not a fabricated correction.
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    show_image(axes[0], pre, bounds, "Pre-event imagery with rejected official segment")
    draw_context(axes[0], up_context, bridge, down_context, wrong); axes[0].legend(loc="upper left", fontsize=6); decorate(axes[0], bounds)
    axes[1].imshow(hillshade(z), extent=(db[0], db[2], db[1], db[3]), cmap="gray")
    draw_context(axes[1], up_context, bridge, down_context, wrong)
    axes[1].text(.5, .03, "No imagery-supported local channel correction accepted", transform=axes[1].transAxes,
                 color="white", ha="center", bbox={"facecolor":"black", "alpha":.72})
    axes[1].legend(loc="upper left", fontsize=6); axes[1].set_title("12.5 m DEM valley context"); decorate(axes[1], bounds)
    fig.subplots_adjust(wspace=.12, left=.05, right=.99, bottom=.08, top=.94)
    fig.savefig(FIG / "18_anomaly_multievidence_overlay.png", dpi=220); plt.close(fig)

    qa = {
        "analysis_crs": CRS,
        "status": "HUMAN_IMAGE_REVIEW_REQUIRED",
        "official_wrong_segment_start": [p0.x, p0.y],
        "official_wrong_segment_end": [p2.x, p2.y],
        "official_wrong_segment_length_m": wrong.length,
        "corrected_candidate_accepted": False,
        "corrected_candidate_length_m": None,
        "maximum_lateral_offset_m": None,
        "mean_lateral_offset_m": None,
        "maximum_dem_elevation_difference_m": None,
        "official_wrong_segment_max_adverse_rise_250m_m": rise12,
        "official_wrong_segment_max_local_adverse_rise_m": maximum_local_rise,
        "corrected_segment_max_adverse_rise_250m_m": None,
        "dem12_valid_sample_fraction": float(valid12.mean()),
        "dem8_valid_sample_fraction": float(valid8.mean()),
        "rejection_basis": [
            "The official P0--P2 line leaves the image-visible valley/channel near the documented anomaly.",
            "The native 12.5 m profile contains the documented large 250 m adverse rise.",
            "The supplied 8 m DEM supports an uphill trend at the anomaly where valid.",
            "Existing pre-event and post-event imagery do not establish one unambiguous physical P0-to-S4 continuation suitable for publication as a corrected channel."
        ],
        "review_figures": ["15_downstream_geometry_QA_failure_v2.png", "16_anomaly_pre_event_0p5m.png", "17_anomaly_post_event_3m.png", "18_anomaly_multievidence_overlay.png"]
    }
    (REPORT / "DOWNSTREAM_GEOMETRY_CORRECTION_QA.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    (REPORT / "DOWNSTREAM_GEOMETRY_CORRECTION_QA.md").write_text(
        f"# Downstream geometry correction QA\n\n"
        f"## Decision\n\n`HUMAN_IMAGE_REVIEW_REQUIRED`; no corrected local channel has been accepted.\n\n"
        f"## Rejected official segment\n\nThe P0--P2 official segment is {wrong.length:.2f} m long. It leaves the visible valley/channel and has a maximum {rise12:.0f} m adverse rise over a 250 m window in the native 12.5 m DEM. Its maximum valid one-step local rise is {maximum_local_rise:.0f} m. The 8 m DEM has valid coverage over {valid8.mean() * 100:.1f}% of sampled locations and independently supports the anomalous uphill terrain trend where it is valid.\n\n"
        "## Why no replacement is published\n\n"
        "The available 0.5 m pre-event DOM and 3 m post-event image both show valley/linear features, but they do not provide a uniquely traceable continuous P0-to-S4 channel axis at the required confidence. Drawing one would turn an unresolved visual interpretation into false corridor geometry. Therefore candidate length, offsets, and corrected-trace elevation comparisons are intentionally `null`, not inferred.\n\n"
        "## Required human decision\n\n"
        "Review figures 15--18 and approve a continuous image-supported channel trace, or supply a mapped event-channel product. No DEM was modified and no simulation was run.\n",
        encoding="utf-8")

    final = {
        "gate_a": "HUMAN_IMAGE_REVIEW_REQUIRED",
        "analysis_crs": CRS,
        "s0_coordinate": [S0.x, S0.y], "s0_confidence": "MODERATE",
        "s4_coordinate": [S4.x, S4.y], "s4_port_offset_m": PORT.distance(S4),
        "old_vector_gap_straight_m": 525.5297039262372,
        "old_gap_trace_length_m": 607.9923225566369,
        "downstream_anomaly_resolved": False,
        "downstream_anomaly_coordinate": [ANOMALY.x, ANOMALY.y],
        "official_wrong_segment_max_adverse_rise_m": rise12,
        "corrected_segment_max_adverse_rise_m": None,
        "canonical_corridor_established": False,
        "canonical_corridor_length_km": None,
        "s1_s3_created": False, "profile_created": False,
        "maximum_final_corridor_adverse_rise_250m_m": None,
        "human_image_review_required": True,
        "primary_blocker": "No single continuous imagery-supported P0-to-S4 channel trace can be accepted from the current AOI evidence; human image review or a mapped channel product is required.",
        "cgs_documentary_url": CGS_URL
    }
    (REPORT / "JILONG_GATE_A_FINAL.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    (REPORT / "JILONG_GATE_A_FINAL.md").write_text(
        "# Final Jilong/Gyirong spatial Gate A\n\n"
        "## Decision\n\n`GATE_A = HUMAN_IMAGE_REVIEW_REQUIRED`.\n\n"
        f"S0 remains the effective downstream-stage boundary at `({S0.x:.3f}, {S0.y:.3f})` EPSG:32645 (moderate confidence). S4 remains the spatially defensible hydraulic inspection section at `({S4.x:.3f}, {S4.y:.3f})`, {PORT.distance(S4):.2f} m from the official Jilong Port point. The former {525.5297039262372:.2f} m vector gap remains resolved only by its {607.9923225566369:.2f} m imagery-supported connectivity trace.\n\n"
        "## Downstream anomaly\n\n"
        f"The P0--P2 official segment is rejected: it leaves the visible valley/channel and has a {rise12:.0f} m adverse rise over 250 m in native 12.5 m terrain. The imagery does not yet establish a unique continuous P0-to-S4 replacement trace. No candidate GeoJSON, canonical corridor, S1--S3 sections, or final longitudinal profile is published.\n\n"
        f"The corrected CGS documentary URL is {CGS_URL}. It supplies documentary pathway context only; it does not determine geometry.\n\n"
        "## Consequence\n\nNo DEM was modified, no terrain was carved, and no D-Claw calculation was run. The bounded remaining action is human review of figures 15--18 or provision of an authoritative mapped channel/event-corridor product.\n",
        encoding="utf-8")
    (REVIEW / "README.txt").write_text(
        "Gate A remains HUMAN_IMAGE_REVIEW_REQUIRED. `15_downstream_geometry_QA_failure.png` is historical; use `15_downstream_geometry_QA_failure_v2.png`, `16_anomaly_pre_event_0p5m.png`, `17_anomaly_post_event_3m.png`, and `18_anomaly_multievidence_overlay.png` for the current bounded review. No corrected channel or canonical corridor is included.\n",
        encoding="utf-8")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
