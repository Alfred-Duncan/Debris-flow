"""Setrun for the fixed-S0 conservative internal source-zone cases."""
from __future__ import annotations
import json
from pathlib import Path
from clawpack.clawutil import data
from clawpack.geoclaw import fgmax_tools

CASE = Path(__file__).resolve().parent
ACTIVE = json.loads((CASE / "active_run.json").read_text())
DOMAIN = json.loads((CASE / ("model_geometry_sourcefix.json" if ACTIVE.get("sourcefix",False) else "model_geometry.json")).read_text())
V, T = float(ACTIVE["V_m3"]), float(ACTIVE["T_s"])
TFINAL, OUTINT = float(ACTIVE["tfinal_s"]), float(ACTIVE["output_interval_s"])
ENTRAINMENT = int(ACTIVE.get("entrainment",0))
KU = float(ACTIVE["momentum_factor"])
# Defaults preserve the frozen V10 configuration; only predeclared ladder
# values in active_run.json may override them.
MANNING = float(ACTIVE.get("manning", 0.025))
PHI = float(ACTIVE.get("phi_deg", 38.0))
EROSION_RATE = float(ACTIVE.get("entrainment_rate", 0.20))
ERODIBLE_FILE = str(ACTIVE.get("erodible_thickness_file", "entrainment/erodible_thickness_e4_published.tt3"))
if KU <= 0.0:
    raise ValueError("momentum_factor must be > 0")

def setrun(claw_pkg="dclaw"):
    r = data.ClawRunData("dclaw", 2)
    c = r.clawdata
    c.lower[:], c.upper[:], c.num_cells[:] = [DOMAIN["xlower"], DOMAIN["ylower"]], [DOMAIN["xupper"], DOMAIN["yupper"]], [DOMAIN["mx"], DOMAIN["my"]]
    c.num_eqn, c.num_aux, c.capa_index = 7, 10, 0
    c.t0, c.restart = 0.0, False
    c.output_style, c.num_output_times, c.tfinal, c.output_t0 = 1, int(round(TFINAL / OUTINT)), TFINAL, True
    c.output_format, c.output_q_components, c.output_aux_components, c.output_aux_onlyonce = "ascii", "all", "none", True
    # src2 is applied after the hyperbolic CFL estimate.  A finite cap prevents
    # the initially dry grid from proposing a zero-flow, multi-second step before
    # the first finite conservative source increment participates in CFL control.
    c.verbosity, c.dt_variable, c.dt_initial, c.dt_max = 1, True, 0.05, 0.5
    c.cfl_desired, c.cfl_max, c.steps_max = .45, .50, 100000
    c.order, c.dimensional_split, c.transverse_waves = 1, "unsplit", 0
    c.num_waves, c.limiter, c.use_fwaves, c.source_split = 5, [4] * 5, True, "godunov"
    # Standard boundaries: northern cropped/upstream edge is a reflective wall.
    c.num_ghost, c.bc_lower[:], c.bc_upper[:], c.checkpt_style = 2, ["extrap", "extrap"], ["extrap", "extrap"], 0
    a = r.amrdata
    a.amr_levels_max, a.refinement_ratios_x, a.refinement_ratios_y, a.refinement_ratios_t = 1, [1], [1], [1]
    a.aux_type = ["center", "center", "yleft", "center", "center", "center", "center", "center", "center", "center"]
    a.flag_richardson, a.flag2refine, a.regrid_interval, a.regrid_buffer_width, a.clustering_cutoff, a.verbosity_regrid, a.max1d = False, False, 3, 0, .7, 0, 500
    r.regiondata.regions = []
    g = r.geo_data
    g.gravity, g.coordinate_system, g.earth_radius, g.coriolis_forcing = 9.81, 1, 6367500.0, False
    g.sea_level, g.dry_tolerance, g.friction_forcing = -9999.0, 1.e-3, True
    g.manning_coefficient, g.friction_depth = MANNING, 1.e6
    r.refinement_data.variable_dt_refinement_ratios, r.refinement_data.wave_tolerance = True, .01
    r.topo_data.topofiles.append([3, "terrain/published_route_domain_64m.tt3"])
    d = r.dclaw_data
    d.rho_f, d.rho_s = 1100.0, 2700.0
    d.m_crit, d.m0, d.mref, d.kref = .64, .62, .60, 1.e-11
    d.phi, d.delta, d.mu, d.alpha_c, d.c1, d.sigma_0 = PHI, .01, .005, .05, 1.0, 1.e3
    d.src2method, d.alphamethod = 2, 1
    d.segregation, d.beta_seg, d.chi0, d.chie = 0, 0.0, .5, .5
    d.bed_normal, d.theta_input = 0, 0.0
    d.entrainment, d.entrainment_method, d.entrainment_rate, d.me = ENTRAINMENT, 0, EROSION_RATE, .62
    if ENTRAINMENT: r.auxinitdclaw_data.auxinitfiles.append([3, 7, CASE / ERODIBLE_FILE])
    r.pinitdclaw_data.init_ptype = 0
    r.flowgrades_data.flowgrades = []
    fg = fgmax_tools.FGmaxGrid()
    fg.point_style, fg.x1, fg.x2, fg.y1, fg.y2, fg.dx = 2, DOMAIN["xlower"]+32, DOMAIN["xupper"]-32, DOMAIN["ylower"]+32, DOMAIN["yupper"]-32, 64.0
    fg.tstart_max, fg.tend_max, fg.dt_check, fg.min_level_check, fg.arrival_tol, fg.interp_method = 0.0, TFINAL, 10.0, 1, .01, 0
    r.fgmax_data.num_fgmax_val = 5
    r.fgmax_data.fgmax_grids.append(fg)
    with (CASE / "conservative_source.data").open("w", encoding="ascii") as f:
        f.write(f"{V:.16g} {T:.16g} {len(DOMAIN['cells'])}\n")
        for cell in DOMAIN["cells"]:
            f.write(f"{cell['center_x']:.16g} {cell['center_y']:.16g} {cell['tangent_x']:.16g} {cell['tangent_y']:.16g}\n")
    (CASE / "source_momentum_factor.data").write_text(f"{KU:.17g}\n", encoding="ascii")
    return r

if __name__ == "__main__":
    setrun().write()
