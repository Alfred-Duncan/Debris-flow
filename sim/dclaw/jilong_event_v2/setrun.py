"""Phase 2C: 600-s downstream D-Claw case with a case-level finite inlet pulse."""
from __future__ import annotations
import os
if "CLAW" not in os.environ:
    raise RuntimeError("Set CLAW before running setrun.py")
from clawpack.clawutil import data
from clawpack.geoclaw import fgmax_tools

# 64-m grid aligned to Phase-2B DEM.  West edge is the model-entry transect,
# 29.9 m upstream of the documented entry point; downstream extent is retained.
XL, YL, DX, MX, MY = 334_021.9, 3_128_083.2, 64.0, 141, 261
XU, YU = XL + MX * DX, YL + MY * DX
ENTRY_X, ENTRY_Y = 334_051.8, 3_143_392.5
PORT_X, PORT_Y = 340_837.9, 3_129_051.7

def setrun(claw_pkg="dclaw"):
    qpeak = float(os.environ["INLET_Q_PEAK"])
    duration = float(os.environ["INLET_DURATION"])
    entrain = int(os.environ["ENTRAINMENT_ENABLED"])
    r = data.ClawRunData("dclaw", 2)
    c = r.clawdata
    c.lower[:], c.upper[:], c.num_cells[:] = [XL,YL], [XU,YU], [MX,MY]
    c.num_eqn, c.num_aux, c.capa_index = 7, 10, 0
    c.t0, c.restart = 0.0, False
    c.output_style, c.num_output_times, c.tfinal, c.output_t0 = 1, 60, 600.0, True
    c.output_format, c.output_q_components, c.output_aux_components, c.output_aux_onlyonce = "ascii", "all", "none", True
    c.verbosity, c.dt_variable, c.dt_initial, c.dt_max = 1, True, 0.25, 1.e99
    c.cfl_desired, c.cfl_max, c.steps_max = 0.45, 0.50, 50000
    c.order, c.dimensional_split, c.transverse_waves = 1, "unsplit", 0
    c.num_waves, c.limiter, c.use_fwaves, c.source_split = 5, [4]*5, True, "godunov"
    c.num_ghost, c.bc_lower[:], c.bc_upper[:], c.checkpt_style = 2, ["user","extrap"], ["extrap","extrap"], 0
    a = r.amrdata
    a.amr_levels_max, a.refinement_ratios_x, a.refinement_ratios_y, a.refinement_ratios_t = 1, [1], [1], [1]
    a.aux_type = ["center","center","yleft","center","center","center","center","center","center","center"]
    a.flag_richardson, a.flag2refine, a.regrid_interval, a.regrid_buffer_width, a.clustering_cutoff, a.verbosity_regrid, a.max1d = False, False, 3, 0, .70, 0, 500
    r.regiondata.regions = []
    r.gaugedata.gauges = [[1, PORT_X, PORT_Y, 0.0, 600.0]]
    g = r.geo_data
    g.gravity, g.coordinate_system, g.earth_radius, g.coriolis_forcing = 9.81, 1, 6_367_500., False
    g.sea_level, g.dry_tolerance, g.friction_forcing = -9999., 1.e-3, True
    g.manning_coefficient, g.friction_depth, g.speed_limit = .025, 1.e6, 30.0
    r.refinement_data.variable_dt_refinement_ratios, r.refinement_data.wave_tolerance = True, .01
    r.topo_data.topofiles.append([3, "jilong_dem_primary_64m.tt3"])
    d = r.dclaw_data
    d.rho_f, d.rho_s = 1000., 2700.
    d.m_crit, d.m0, d.mref, d.kref = .64, .63, .60, 1.e-10
    d.phi, d.delta, d.mu, d.alpha_c, d.c1, d.sigma_0 = 32., .001, .005, .01, 1., 1.e3
    d.src2method, d.alphamethod = 2, 1
    d.segregation, d.beta_seg, d.chi0, d.chie = 0, 0., .5, .5
    d.bed_normal, d.theta_input = 0, 0.
    # D-Claw documented standard mechanism.  Values are provisional defaults,
    # not event observations; only the on/off flag varies across the control pair.
    d.entrainment, d.entrainment_method, d.entrainment_rate, d.me = entrain, 1, .2, .65
    if entrain:
        r.auxinitdclaw_data.auxinitfiles.append([3, 7, "entrainable_thickness.tt3"])
    r.pinitdclaw_data.init_ptype = 0
    r.flowgrades_data.flowgrades = []
    fg = fgmax_tools.FGmaxGrid()
    fg.point_style, fg.x1, fg.x2, fg.y1, fg.y2, fg.dx = 2, XL+DX/2, XU-DX/2, YL+DX/2, YU-DX/2, DX
    fg.tstart_max, fg.tend_max, fg.dt_check, fg.min_level_check, fg.arrival_tol, fg.interp_method = 0., 600., 10., 1, .01, 0
    r.fgmax_data.num_fgmax_val = 5
    r.fgmax_data.fgmax_grids.append(fg)
    with open("inflow.data", "w") as f:
        f.write(f"{qpeak:.12g}  =: q_peak_m3s\n{duration:.12g}  =: duration_s\n")
    return r
if __name__ == "__main__": setrun().write()
