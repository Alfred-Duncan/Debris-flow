"""Prepare the approved unit-Froude S0 external-boundary D-Claw closure."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import rasterio

CASE = Path(__file__).resolve().parent
HANDOFF = Path("/root/autodl-tmp/Jilong_DClaw_Handoff")
TERRAIN = HANDOFF / "terrain/final/jilong_copernicus_64m_v1.tif"
CORRIDOR = HANDOFF / "geometry/jilong_canonical_corridor.geojson"
SECTIONS = HANDOFF / "geometry/jilong_control_sections.geojson"
DX = 64.0
G = 9.81
FR_NORMAL = 1.20
DRY = 1.0e-3
RHO_F, M0, CHI0 = 1100.0, 0.62, 0.5
CASES = {"C1": (1.0e6, 90.0), "C2": (1.0e6, 180.0),
         "C3": (2.0e6, 90.0), "C4": (2.0e6, 180.0),
         "C5": (3.0e6, 90.0), "C6": (3.0e6, 180.0)}

def point_at_distance(points: np.ndarray, distance: float) -> np.ndarray:
    remaining = distance
    for a, b in zip(points[:-1], points[1:]):
        length = float(np.linalg.norm(b - a))
        if remaining <= length:
            return a + (remaining / length) * (b - a)
        remaining -= length
    raise RuntimeError("corridor is shorter than tangent support")

def write_tt3(src: rasterio.DatasetReader, target: Path) -> None:
    z = src.read(1)
    if not np.isfinite(z).all():
        raise RuntimeError("accepted 64 m terrain contains non-finite elevations")
    if src.crs.to_epsg() != 32645 or src.res != (DX, DX):
        raise RuntimeError("unexpected accepted terrain CRS/resolution")
    target.write_text(
        f"{src.width} ncols\n{src.height} nrows\n"
        f"{src.bounds.left:.12g} xllcorner\n{src.bounds.bottom:.12g} yllcorner\n"
        f"{DX:.12g} cellsize\n-9999 nodata_value\n",
        encoding="ascii",
    )
    with target.open("a", encoding="ascii") as f:
        np.savetxt(f, z, fmt="%.9g")

def main() -> None:
    CASE.mkdir(parents=True, exist_ok=True)
    (CASE / "reports").mkdir(exist_ok=True)
    (CASE / "case_config").mkdir(exist_ok=True)
    corridor = np.asarray(json.loads(CORRIDOR.read_text())["features"][0]["geometry"]["coordinates"], float)
    s0 = corridor[0]
    tangent_endpoint = point_at_distance(corridor, 160.0)
    tangent = tangent_endpoint - s0
    tangent /= np.linalg.norm(tangent)
    candidates = {"WEST": np.array([1.0, 0.0]), "EAST": np.array([-1.0, 0.0]),
                  "SOUTH": np.array([0.0, 1.0]), "NORTH": np.array([0.0, -1.0])}
    selected, n_in = max(candidates.items(), key=lambda kv: float(np.dot(tangent, kv[1])))
    alpha = float(np.dot(tangent, n_in))
    if alpha < 0.5:
        raise RuntimeError(f"geometry incompatibility: max alpha={alpha:.6f}")
    if selected != "NORTH":
        raise RuntimeError(f"unexpected selected boundary {selected}; audited setup requires NORTH")
    with rasterio.open(TERRAIN) as src:
        y_candidates = src.bounds.top - DX * np.arange(src.height + 1)
        yupper = float(y_candidates[np.argmin(abs(y_candidates - s0[1]))])
        xlower, xupper, ylower = map(float, (src.bounds.left, src.bounds.right, src.bounds.bottom))
        mx = int(round((xupper - xlower) / DX))
        my = int(round((yupper - ylower) / DX))
        if mx <= 0 or my <= 0:
            raise RuntimeError("invalid S0-cropped domain")
        target = CASE / "jilong_copernicus_64m.tt3"
        write_tt3(src, target)
        z = src.read(1)
        z_sha = __import__("hashlib").sha256(TERRAIN.read_bytes()).hexdigest()
    b_target = 160.0 / alpha
    ncell = int(np.clip(np.rint(b_target / DX), 2, 4))
    center_col = int(np.rint((s0[0] - (xlower + DX / 2)) / DX))
    first_col, last_col = center_col - ncell // 2, center_col - ncell // 2 + ncell - 1
    if first_col < 0 or last_col >= mx:
        raise RuntimeError("inlet support falls outside cropped terrain")
    centers = xlower + DX / 2 + DX * np.arange(first_col, last_col + 1)
    b_actual, w_eff = ncell * DX, ncell * DX * alpha
    boundary_offset = float(abs(s0[1] - yupper))
    top_row = int((src.bounds.top - yupper) / DX)
    ground = [float(z[top_row, k]) for k in range(first_col, last_col + 1)]
    geom = {
        "terrain": str(TERRAIN), "terrain_sha256": z_sha, "crs": "EPSG:32645",
        "tangent_distance_m": 160.0, "s0_xy_m": s0.tolist(),
        "tangent_endpoint_xy_m": tangent_endpoint.tolist(), "t_hat": tangent.tolist(),
        "selected_boundary": selected, "n_in": n_in.tolist(), "alpha": alpha,
        "xlower": xlower, "xupper": xupper, "ylower": ylower, "yupper": yupper,
        "mx": mx, "my": my, "s0_to_boundary_offset_m": boundary_offset,
        "L_ref_m": 160.0, "B_target_m": b_target, "N": ncell,
        "cell_columns_zero_based": list(range(first_col, last_col + 1)),
        "cell_centers_x_m": centers.tolist(), "B_actual_m": b_actual, "W_eff_m": w_eff,
        "ground_elevation_m": ground, "dry_tolerance_m": DRY,
        "closure": "fixed supercritical-normal computational inflow closure",
        "Fr_normal": FR_NORMAL,
        "dry_Q_threshold_m3s": float(b_actual * FR_NORMAL * np.sqrt(G) * DRY ** 1.5),
    }
    (CASE / "inlet_geometry.json").write_text(json.dumps(geom, indent=2) + "\n")
    report = [
        "# Unit-Froude computational inflow closure", "",
        "This is a computational closure, not an observed Jilong inlet state.",
        "The 160 m S0 section is an engineering inspection support, not an observed",
        "channel width. The computational boundary support is its nearest 64 m",
        "discretization.", "",
        "## Geometry", "",
        f"- Selected external boundary: {selected}; inward normal: {n_in.tolist()}",
        f"- S0 downstream tangent from first 160 m: ({tangent[0]:.9f}, {tangent[1]:.9f})",
        f"- alpha = dot(t_hat, n_in): {alpha:.9f}",
        f"- Domain: x=[{xlower:.1f}, {xupper:.1f}], y=[{ylower:.1f}, {yupper:.1f}] m",
        f"- S0-to-boundary offset: {boundary_offset:.3f} m",
        f"- Inlet cells (0-based raster columns): {geom['cell_columns_zero_based']}",
        f"- B_actual: {b_actual:.1f} m; W_eff: {w_eff:.6f} m", "",
        "## State closure", "",
        "This is a fixed supercritical-normal computational inflow closure used",
        "to make the truncated upstream boundary fully incoming; it is not observed.",
        f"For Q(t)>0: U_n={FR_NORMAL:g} sqrt(g h) and",
        "h=[Q/(B_actual Fr_n sqrt(g))]^(2/3). Total U=U_n/alpha and velocity",
        "follows the fixed corridor tangent. Q values yielding h <= the",
        f"D-Claw dry tolerance ({DRY:g} m) are dry/no-inflow.",
        "",
        "D-Claw source relation: dclaw/src/2d/dig/qinit.f90 sets hm=m0*h when",
        "no hm raster is supplied; with init_ptype=0 and bed_normal=0 it sets",
        "pb=rho_f*grav*h. hchi is initialized only when segregation=1, so this",
        "case (segregation=0) uses hchi=0. bdif remains its initialized zero.",
    ]
    (CASE / "reports/INFLOW_CLOSURE.md").write_text("\n".join(report) + "\n")
    preflight = {"closure": geom, "gravity_ms2": G, "Fr_normal": FR_NORMAL, "cases": {}}
    for name, (volume, duration) in CASES.items():
        qpeak = np.pi * volume / (2.0 * duration)
        t = np.linspace(0.0, duration, 200001)
        q = qpeak * np.sin(np.pi * t / duration)
        h = (q / (b_actual * FR_NORMAL * np.sqrt(G))) ** (2.0 / 3.0)
        un = FR_NORMAL * np.sqrt(G * h)
        utotal = un / alpha
        ux, vy = utotal * tangent[0], utotal * tangent[1]
        normal_flux = b_actual * h * (-vy)
        integrated = float(np.trapz(normal_flux, t))
        peak_flux = float(normal_flux.max())
        entry = {
            "target_volume_m3": volume, "integrated_boundary_volume_m3": integrated,
            "relative_volume_error": (integrated - volume) / volume,
            "analytic_Q_peak_m3s": qpeak, "implemented_peak_normal_flux_m3s": peak_flux,
            "relative_peak_flux_error": (peak_flux - qpeak) / qpeak,
            "peak_h_m": float(h.max()), "peak_U_normal_ms": float(un.max()),
            "peak_U_total_ms": float(utotal.max()), "peak_u_ms": float(ux.max()),
            "peak_v_ms": float(vy.min()),
            "Fr_normal_min": float(np.min(np.divide(un, np.sqrt(G*h), out=np.zeros_like(un), where=h > 0)[h > 0])),
            "Fr_normal_max": float(np.max(np.divide(un, np.sqrt(G*h), out=np.zeros_like(un), where=h > 0)[h > 0])),
            "Q_at_0": float(q[0]), "Q_at_T": float(q[-1]), "source_after_T": 0.0,
            "finite": bool(np.isfinite(np.r_[q, h, un, utotal, ux, vy, normal_flux]).all()),
            "pass": bool(abs((integrated-volume)/volume) <= .05 and abs((peak_flux-qpeak)/qpeak) <= .05
                         and h.max() <= 50.0 and utotal.max() <= 40.0),
        }
        if not entry["pass"]:
            raise RuntimeError(f"preflight guardrail failed for {name}: {entry}")
        preflight["cases"][name] = entry
        cfg = {"case_id": name, "V_m3": volume, "T_s": duration, "Q_peak_m3s": qpeak,
               "tfinal_s": 900.0, "output_interval_s": 30.0, "closure": "fixed-normal-Froude"}
        (CASE / f"case_config/{name}.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (CASE / "reports/INFLOW_PREFLIGHT.json").write_text(json.dumps(preflight, indent=2) + "\n")
    rows = ["# Inflow preflight", "", "| Case | target V (m3) | integrated V (m3) | volume error | Qpeak (m3/s) | flux error | hpeak (m) | Un peak (m/s) | U total peak (m/s) | pass |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for name, x in preflight["cases"].items():
        rows.append(f"| {name} | {x['target_volume_m3']:.3f} | {x['integrated_boundary_volume_m3']:.3f} | {x['relative_volume_error']:.3e} | {x['analytic_Q_peak_m3s']:.3f} | {x['relative_peak_flux_error']:.3e} | {x['peak_h_m']:.3f} | {x['peak_U_normal_ms']:.3f} | {x['peak_U_total_ms']:.3f} | {x['pass']} |")
    rows += ["", "All rows use the actual N, B_actual, W_eff and alpha recorded in inlet_geometry.json.",
             "The normal-flux integration is numerical (200001 uniformly spaced samples)."]
    (CASE / "reports/INFLOW_PREFLIGHT.md").write_text("\n".join(rows) + "\n")

if __name__ == "__main__":
    main()
