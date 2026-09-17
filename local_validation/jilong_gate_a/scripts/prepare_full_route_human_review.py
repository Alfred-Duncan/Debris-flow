"""Export evidence-only human-review tiles for the current S0--S4 reference.

This script deliberately does not digitize an upstream correction, construct a
canonical corridor, classify imagery, or alter Gate A.  The review chainage is
only an ordering device: official reference geometry plus the two previously
stored correction traces needed to span S0--S4.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import rasterio
from pyproj import Transformer
from shapely.geometry import LineString, Point, box
from shapely.ops import substring

sys.path.insert(0, r"E:\Alfred\Jilong_GateA\scripts")
from finalize_gate_a import (
    RAW, CORRIDOR, REVIEW, DOWNLOADS, CRS, PORT, OLD_ENTRY,
    bounded_window, native_bounds, read_dem, hillshade, show_image, decorate,
)

S0 = Point(334051.79369854234, 3143392.492568097)
S4 = Point(340571.8257777202, 3129640.775894154)
UPSTREAM_REVIEW = Point(337303.8418827355, 3139798.3985810857)
TILE_LENGTH_M = 1500.0
TILE_STEP_M = 1200.0
BUFFER_M = 420.0


def line_parts(frame):
    return [part for geom in frame.geometry
            for part in (geom.geoms if geom.geom_type == "MultiLineString" else [geom])]


def join_lines(lines):
    coords = []
    for line in lines:
        current = list(line.coords)
        if coords and Point(coords[-1]).distance(Point(current[0])) > 0.1:
            raise ValueError("Review reference is not continuous")
        coords.extend(current if not coords else current[1:])
    return LineString(coords)


def river_reference():
    options = []
    for path in (RAW / "basic_geography").rglob("*.shp"):
        frame = gpd.read_file(path).to_crs(CRS)
        if set(frame.geom_type).issubset({"LineString", "MultiLineString"}):
            options.append((frame.geometry.union_all().distance(PORT), frame))
    return min(options, key=lambda item: item[0])[1]


def status_for_bounds(dataset_path, bounds):
    with rasterio.open(dataset_path) as ds:
        b = native_bounds(ds, bounds)
        source = box(ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)
        target = box(*b)
        if not source.intersects(target):
            return "NO"
        return "FULL" if source.covers(target) else "PARTIAL"


def sample_profile(line, dem12, dem8, spacing=12.5):
    distances = np.arange(0.0, line.length + spacing, spacing)
    if len(distances):
        distances[-1] = line.length
    points = [line.interpolate(float(d)) for d in distances]
    xy = [(p.x, p.y) for p in points]
    with rasterio.open(dem12) as ds:
        z12 = np.asarray([v[0] for v in ds.sample(xy)], dtype=float)
        valid12 = np.isfinite(z12) & (z12 != ds.nodata)
    z12[~valid12] = np.nan
    with rasterio.open(dem8) as ds:
        transform = Transformer.from_crs(CRS, ds.crs, always_xy=True)
        xy8 = [transform.transform(*p) for p in xy]
        z8 = np.asarray([v[0] for v in ds.sample(xy8)], dtype=float)
        # The 8 m mosaic returns zeros beyond valid coverage; this is nodata,
        # not a terrain alteration or an outlier mask.
        valid8 = np.isfinite(z8) & (z8 != ds.nodata) & (z8 > 0)
    z8[~valid8] = np.nan
    return distances, z12, z8


def max_rise(values, spacing, metres):
    step = max(1, int(round(metres / spacing)))
    if len(values) <= step:
        return ""
    rise = values[step:] - values[:-step]
    return "" if not np.isfinite(rise).any() else f"{float(np.nanmax(rise)):.3f}"


def bounds_for_line(line):
    minx, miny, maxx, maxy = line.bounds
    minx, miny, maxx, maxy = (minx - BUFFER_M, miny - BUFFER_M,
                              maxx + BUFFER_M, maxy + BUFFER_M)
    # Square evidence windows avoid unreadably thin north-south valley panels.
    side = max(maxx - minx, maxy - miny)
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    return (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)


def draw_official(ax, river, label=True):
    first = True
    for line in line_parts(river):
        ax.plot(*line.xy, color="#00d9ff", lw=1.0, alpha=.9,
                label="OFFICIAL RIVER - REFERENCE ONLY" if first and label else "_nolegend_", zorder=3)
        first = False


def draw_corrections(ax, gap, downstream, label=True):
    ax.plot(*gap.xy, color="#ffb000", lw=2.4, zorder=6,
            label="PREVIOUS HUMAN REVIEW: 607.99 m gap trace" if label else "_nolegend_")
    ax.plot(*downstream.xy, color="#df3cff", lw=2.0, zorder=6,
            label="PREVIOUS HUMAN REVIEW: downstream correction" if label else "_nolegend_")


def draw_tile_marks(ax, tile_line, start, end, bounds):
    a, b = tile_line.coords[0], tile_line.coords[-1]
    ax.scatter([a[0]], [a[1]], s=38, color="#ffe400", edgecolors="black", zorder=9, label="tile entry")
    ax.scatter([b[0]], [b[1]], s=38, color="#ff6b35", edgecolors="black", zorder=9, label="tile exit")
    ax.annotate(f"ENTRY {start / 1000:.2f} km", xy=a, xytext=(6, 6), textcoords="offset points",
                color="white", fontsize=7, weight="bold", zorder=10)
    ax.annotate(f"EXIT {end / 1000:.2f} km", xy=b, xytext=(6, -12), textcoords="offset points",
                color="white", fontsize=7, weight="bold", zorder=10)
    decorate(ax, bounds)


def known_labels(ax, bounds, gap, downstream):
    area = box(*bounds)
    if area.intersects(gap):
        mid = gap.interpolate(.5, normalized=True)
        ax.scatter([mid.x], [mid.y], marker="s", s=38, color="#ffb000", edgecolors="black", zorder=9)
        ax.annotate("KNOWN REVIEW AREA\nformer vector gap", xy=(mid.x, mid.y), xytext=(7, 7),
                    textcoords="offset points", fontsize=7, color="white", weight="bold", zorder=10)
    if area.contains(UPSTREAM_REVIEW):
        ax.scatter([UPSTREAM_REVIEW.x], [UPSTREAM_REVIEW.y], marker="s", s=40, color="red", edgecolors="white", zorder=9)
        ax.annotate("KNOWN REVIEW AREA\nupstream anomaly", xy=(UPSTREAM_REVIEW.x, UPSTREAM_REVIEW.y), xytext=(42, -28),
                    textcoords="offset points", fontsize=7, color="white", weight="bold", zorder=10)
    if area.intersects(downstream):
        q = downstream.intersection(area)
        point = q.interpolate(.5, normalized=True) if q.geom_type == "LineString" and not q.is_empty else downstream.interpolate(.5, normalized=True)
        ax.annotate("PREVIOUS HUMAN REVIEW\ndownstream P0-S4 area", xy=(point.x, point.y), xytext=(7, 7),
                    textcoords="offset points", fontsize=7, color="white", weight="bold", zorder=10,
                    bbox={"facecolor": "black", "alpha": .55, "pad": 1})


def known_area_text(bounds, gap, downstream):
    area = box(*bounds)
    labels = []
    if area.intersects(gap): labels.append("former vector gap / 607.99 m trace")
    if area.contains(UPSTREAM_REVIEW): labels.append("upstream anomaly")
    if area.intersects(downstream): labels.append("previous downstream P0-S4 correction")
    return "; ".join(labels) if labels else ""


def panel_title(tile_id, panel, start, end, center):
    return (f"{tile_id} - {panel}\n"
            f"S0 chainage {start / 1000:.2f}-{end / 1000:.2f} km; "
            f"center ({center.x:.1f}, {center.y:.1f}) EPSG:32645")


def main():
    out = REVIEW / "full_route"
    out.mkdir(parents=True, exist_ok=True)
    river = river_reference()
    parts = sorted(line_parts(river), key=lambda line: line.distance(OLD_ENTRY))
    upstream = parts[0]
    # This is not a candidate corridor: it is the existing reference ordering
    # used solely to make adjacent visual-review tiles cover the reach.
    official_upstream = substring(upstream, upstream.project(S0), upstream.length)
    gap = gpd.read_file(CORRIDOR / "candidate_gap_bridge.geojson").to_crs(CRS).geometry.iloc[0]
    downstream = gpd.read_file(CORRIDOR / "downstream_anomaly_channel_corrected.geojson").to_crs(CRS).geometry.iloc[0]
    review_reference = join_lines([official_upstream, gap, downstream])
    if Point(review_reference.coords[0]).distance(S0) > 1.0 or Point(review_reference.coords[-1]).distance(S4) > 1.0:
        raise ValueError("Reference ordering does not span fixed S0-S4")

    dem12 = next((RAW / "dem_12p5m").rglob("*.tif"))
    dem8 = next((RAW / "dem_8m").rglob("*.tif"))
    pre = next(DOWNLOADS.rglob("05m.tif"))
    post = next(DOWNLOADS.rglob("*composite.tif"))
    starts = list(np.arange(0.0, review_reference.length, TILE_STEP_M))
    if starts[-1] + TILE_LENGTH_M < review_reference.length:
        starts.append(max(0.0, review_reference.length - TILE_LENGTH_M))
    # De-duplicate the tail tile while keeping ordered 20% overlaps.
    starts = sorted(set(round(float(x), 6) for x in starts))
    rows = []
    footprints = []
    for index, start in enumerate(starts, 1):
        end = min(review_reference.length, start + TILE_LENGTH_M)
        tile_line = substring(review_reference, start, end)
        bounds = bounds_for_line(tile_line)
        center = review_reference.interpolate((start + end) / 2)
        tile_id = f"T{index:02d}"
        d, z12, z8 = sample_profile(tile_line, dem12, dem8)
        row = {
            "tile_id": tile_id, "start_chainage_m": f"{start:.3f}", "end_chainage_m": f"{end:.3f}",
            "center_x": f"{center.x:.3f}", "center_y": f"{center.y:.3f}",
            "pre_event_coverage": status_for_bounds(pre, bounds), "post_event_coverage": status_for_bounds(post, bounds),
            "dem12_coverage": status_for_bounds(dem12, bounds), "dem8_coverage": status_for_bounds(dem8, bounds),
            "known_review_area": known_area_text(bounds, gap, downstream), "png_filename": f"{tile_id}_review.png",
        }
        for name, values in (("dem12", z12), ("dem8", z8)):
            for metres in (50, 100, 250): row[f"max_adverse_rise_{metres}m_{name}"] = max_rise(values, 12.5, metres)
        rows.append(row)
        footprints.append((tile_id, box(*bounds), center))

        fig, axes = plt.subplots(1, 3, figsize=(22, 7.4))
        show_image(axes[0], pre, bounds, panel_title(tile_id, "Panel A: pre-event 0.5 m DOM", start, end, center))
        draw_official(axes[0], river); draw_corrections(axes[0], gap, downstream)
        draw_tile_marks(axes[0], tile_line, start, end, bounds); known_labels(axes[0], bounds, gap, downstream)
        axes[0].legend(loc="upper left", fontsize=6)

        terrain, terrain_bounds = read_dem(dem12, bounds)
        axes[1].imshow(hillshade(terrain), extent=(terrain_bounds[0], terrain_bounds[2], terrain_bounds[1], terrain_bounds[3]), cmap="gray")
        axes[1].set_title(panel_title(tile_id, "Panel B: native 12.5 m DEM hillshade; 8 m context in CSV", start, end, center))
        draw_official(axes[1], river); draw_corrections(axes[1], gap, downstream)
        draw_tile_marks(axes[1], tile_line, start, end, bounds); known_labels(axes[1], bounds, gap, downstream)

        show_image(axes[2], pre, bounds, panel_title(tile_id, "Panel C: clean evidence overlay", start, end, center))
        draw_official(axes[2], river); draw_corrections(axes[2], gap, downstream)
        draw_tile_marks(axes[2], tile_line, start, end, bounds); known_labels(axes[2], bounds, gap, downstream)
        axes[2].text(.02, .02, "Official river network: REFERENCE ONLY\nNo new route is proposed in this package.",
                     transform=axes[2].transAxes, color="white", fontsize=7,
                     bbox={"facecolor": "black", "alpha": .7, "pad": 2})
        fig.tight_layout(); fig.savefig(out / row["png_filename"], dpi=160); plt.close(fig)

    fields = [
        "tile_id", "start_chainage_m", "end_chainage_m", "center_x", "center_y",
        "pre_event_coverage", "post_event_coverage", "dem12_coverage", "dem8_coverage",
        "known_review_area", "png_filename",
        "max_adverse_rise_50m_dem12", "max_adverse_rise_100m_dem12", "max_adverse_rise_250m_dem12",
        "max_adverse_rise_50m_dem8", "max_adverse_rise_100m_dem8", "max_adverse_rise_250m_dem8",
    ]
    with (out / "FULL_ROUTE_TILE_INDEX.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)

    # Index map: all shapes are review footprints, evidence references, or
    # previously stored corrections; no new route geometry is saved or drawn.
    extent = review_reference.buffer(1500).bounds
    terrain, terrain_bounds = read_dem(dem12, extent, max_dim=1800)
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.imshow(hillshade(terrain), extent=(terrain_bounds[0], terrain_bounds[2], terrain_bounds[1], terrain_bounds[3]), cmap="gray")
    draw_official(ax, river); draw_corrections(ax, gap, downstream)
    for tile_id, footprint, center in footprints:
        x, y = footprint.exterior.xy
        ax.plot(x, y, color="#ffffff", lw=.75, alpha=.85)
        ax.text(center.x, center.y, tile_id, color="white", fontsize=7, weight="bold", ha="center", va="center",
                bbox={"facecolor": "black", "alpha": .55, "pad": 1})
    for chain in np.arange(0.0, review_reference.length + 1, 2000.0):
        p = review_reference.interpolate(float(chain))
        ax.scatter([p.x], [p.y], color="white", edgecolors="black", s=23, zorder=9)
        ax.annotate(f"{chain / 1000:.0f} km", (p.x, p.y), xytext=(5, 5), textcoords="offset points", color="white", fontsize=7)
    ax.scatter([S0.x], [S0.y], color="red", s=55, edgecolors="white", zorder=10, label="S0")
    ax.scatter([S4.x], [S4.y], color="lime", s=55, edgecolors="black", zorder=10, label="S4")
    ax.scatter([UPSTREAM_REVIEW.x], [UPSTREAM_REVIEW.y], marker="s", color="red", edgecolors="white", s=48, zorder=10)
    ax.annotate("KNOWN REVIEW AREA\nupstream anomaly", (UPSTREAM_REVIEW.x, UPSTREAM_REVIEW.y), xytext=(6, 6),
                textcoords="offset points", color="white", fontsize=7, weight="bold")
    ax.set_title("Full S0-S4 human-review tile index\nOfficial river vectors are REFERENCE ONLY")
    ax.legend(loc="upper left", fontsize=8); decorate(ax, extent)
    fig.tight_layout(); fig.savefig(out / "00_FULL_ROUTE_TILE_INDEX.png", dpi=180); plt.close(fig)

    readme = (
        "This package is for external human visual review of the complete S0-S4 corridor.\n\n"
        "Reference ordering only: official upstream vector, the pre-existing 607.99 m imagery-supported gap trace, and the pre-existing downstream human-review correction to S4. It is not a new candidate route or a Gate A decision.\n\n"
        "The upstream anomaly is marked as a known review area. Although a human decision exists in prior materials, no upstream correction line is digitized here because this package must not invent one.\n\n"
        "Review every tile without treating this package's numerical context as a classification. For every tile, assign one of:\n"
        "A - official/reference geometry visually consistent\n"
        "B - local correction required, route visually clear\n"
        "C - ambiguous, higher-resolution/additional evidence required\n\n"
        "Tile panels share a tile extent. The CSV records imagery/DEM footprint coverage and raw numerical adverse-rise context sampled at about 12.5 m. These metrics do not approve or reject a tile.\n\n"
        f"Tiles: {len(rows)}. Chainage is a review-order coordinate from S0, not a final canonical-corridor measurement.\n"
    )
    (out / "README.txt").write_text(readme, encoding="utf-8")
    print(f"TILES={len(rows)} LENGTH_M={review_reference.length:.3f}")


if __name__ == "__main__":
    main()
