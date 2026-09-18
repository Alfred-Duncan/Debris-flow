"""Setrun for one approved 64 m unit-Froude external-boundary case."""
from __future__ import annotations
import json, math, os
from pathlib import Path
from clawpack.clawutil import data
from clawpack.geoclaw import fgmax_tools

CASE = Path(__file__).resolve().parent
GEO = json.loads((CASE / "inlet_geometry.json").read_text())
ACTIVE = json.loads((CASE / "active_run.json").read_text())
V = float(ACTIVE["V_m3"])
T = float(ACTIVE["T_s"])
TFINAL = float(ACTIVE["tfinal_s"])
OUTINT = float(ACTIVE["output_interval_s"])
QPEAK = math.pi * V / (2.0 * T)

def setrun(claw_pkg="dclaw"):
    r = data.ClawRunData("dclaw", 2)
    c = r.clawdata
    c.lower[:], c.upper[:], c.num_cells[:] = [GEO["xlower"], GEO["ylower"]], [GEO["xupper"], GEO["yupper"]], [GEO["mx"], GEO["my"]]
    c.num_eqn, c.num_aux, c.capa_index = 7, 10, 0
    c.t0, c.restart = 0.0, False
    c.output_style, c.num_output_times, c.tfinal, c.output_t0 = 1, int(round(TFINAL / OUTINT)), TFINAL, True
    c.output_format, c.output_q_components, c.output_aux_components, c.output_aux_onlyonce = "ascii", "all", "none", True
    c.verbosity, c.dt_variable, c.dt_initial, c.dt_max = 1, True, 0.05, 1.e99
    c.cfl_desired, c.cfl_max, c.steps_max = .45, .50, 100000
    c.order, c.dimensional_split, c.transverse_waves = 1, "unsplit", 0
    c.num_waves, c.limiter, c.use_fwaves, c.source_split = 5, [4]*5, True, "godunov"
    c.num_ghost, c.bc_lower[:], c.bc_upper[:], c.checkpt_style = 2, ["extrap","extrap"], ["extrap","user"], 0
    a = r.amrdata
    a.amr_levels_max, a.refinement_ratios_x, a.refinement_ratios_y, a.refinement_ratios_t = 1, [1], [1], [1]
    a.aux_type = ["center","center","yleft","center","center","center","center","center","center","center"]
    a.flag_richardson, a.flag2refine, a.regrid_interval, a.regrid_buffer_width, a.clustering_cutoff, a.verbosity_regrid, a.max1d = False, False, 3, 0, .7, 0, 500
    r.regiondata.regions = []
    g = r.geo_data
    g.gravity, g.coordinate_system, g.earth_radius, g.coriolis_forcing = 9.81, 1, 6367500.0, False
    g.sea_level, g.dry_tolerance, g.friction_forcing = -9999.0, 1.e-3, True
    g.manning_coefficient, g.friction_depth = .025, 1.e6
    r.refinement_data.variable_dt_refinement_ratios, r.refinement_data.wave_tolerance = True, .01
    r.topo_data.topofiles.append([3, "jilong_copernicus_64m.tt3"])
    d = r.dclaw_data
    d.rho_f, d.rho_s = 1100.0, 2700.0
    d.m_crit, d.m0, d.mref, d.kref = .64, .62, .60, 1.e-11
    d.phi, d.delta, d.mu, d.alpha_c, d.c1, d.sigma_0 = 38.0, .01, .005, .05, 1.0, 1.e3
    d.src2method, d.alphamethod = 2, 1
    d.segregation, d.beta_seg, d.chi0, d.chie = 0, 0.0, .5, .5
    d.bed_normal, d.theta_input = 0, 0.0
    d.entrainment, d.entrainment_method, d.entrainment_rate, d.me = 0, 1, 0.0, .62
    r.pinitdclaw_data.init_ptype = 0
    r.flowgrades_data.flowgrades = []
    fg = fgmax_tools.FGmaxGrid()
    fg.point_style, fg.x1, fg.x2, fg.y1, fg.y2, fg.dx = 2, GEO["xlower"]+32, GEO["xupper"]-32, GEO["ylower"]+32, GEO["yupper"]-32, 64.0
    fg.tstart_max, fg.tend_max, fg.dt_check, fg.min_level_check, fg.arrival_tol, fg.interp_method = 0.0, TFINAL, 10.0, 1, .01, 0
    r.fgmax_data.num_fgmax_val = 5
    r.fgmax_data.fgmax_grids.append(fg)
    with (CASE / "inflow.data").open("w") as f:
        f.write(f"{QPEAK:.16g}\n{T:.16g}\n{GEO['t_hat'][0]:.16g}\n{GEO['t_hat'][1]:.16g}\n")
        f.write(f"{GEO['cell_centers_x_m'][0]:.16g}\n{GEO['cell_centers_x_m'][-1]:.16g}\n")
        f.write("0.62\n1100.0\n9.81\n0.001\n1.20\n")
    return r

if __name__ == "__main__":
    setrun().write()
