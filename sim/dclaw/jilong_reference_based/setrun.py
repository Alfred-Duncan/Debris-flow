"""Reference-based fixed-grid D-Claw set-up for the Jilong downstream corridor."""
from __future__ import annotations
import os
if "CLAW" not in os.environ:
    raise RuntimeError("Set CLAW to the installed D-Claw checkout")
from clawpack.clawutil import data
from clawpack.geoclaw import fgmax_tools
from terrain_preprocess import XL, YL, XU, YU, DX, MX, MY

PORT_X, PORT_Y = 340571.8257777202, 3129640.775894154


def setrun(claw_pkg="dclaw"):
    kref = float(os.environ.get("DCLAW_KREF", "1e-11"))
    p_ratio = float(os.environ.get("DCLAW_PINIT_PRATIO", "1.0"))
    rundata = data.ClawRunData("dclaw", 2)
    c = rundata.clawdata
    c.lower[:], c.upper[:], c.num_cells[:] = [XL, YL], [XU, YU], [MX, MY]
    c.num_eqn, c.num_aux, c.capa_index = 7, 10, 0
    c.t0, c.restart = 0.0, False
    # Defaults are the accepted 600-s / 10-s production configuration. The
    # batch runner may request a zero-time native-qinit preflight only.
    tfinal = float(os.environ.get("DCLAW_TFINAL", "600.0"))
    nout = int(os.environ.get("DCLAW_NUM_OUTPUT", "60"))
    c.output_style, c.num_output_times, c.tfinal, c.output_t0 = 1, nout, tfinal, True
    c.output_format, c.output_q_components, c.output_aux_components, c.output_aux_onlyonce = "ascii", "all", "none", True
    c.verbosity, c.dt_variable, c.dt_initial, c.dt_max = 1, True, 0.1, 1e99
    c.cfl_desired, c.cfl_max, c.steps_max = 0.45, 0.5, 50000
    c.order, c.dimensional_split, c.transverse_waves = 1, "unsplit", 0
    c.num_waves, c.limiter, c.use_fwaves, c.source_split = 5, [4] * 5, True, "godunov"
    c.num_ghost, c.bc_lower[:], c.bc_upper[:], c.checkpt_style = 2, ["extrap", "extrap"], ["extrap", "extrap"], 0

    a = rundata.amrdata
    a.amr_levels_max, a.refinement_ratios_x, a.refinement_ratios_y, a.refinement_ratios_t = 1, [1], [1], [1]
    a.aux_type = ["center", "center", "yleft", "center", "center", "center", "center", "center", "center", "center"]
    a.flag_richardson, a.flag2refine, a.regrid_interval, a.regrid_buffer_width = False, False, 3, 0
    a.clustering_cutoff, a.verbosity_regrid, a.max1d = 0.7, 0, 500
    rundata.regiondata.regions = []
    rundata.gaugedata.gauges = [[1, PORT_X, PORT_Y, 0.0, 600.0]]

    g = rundata.geo_data
    g.gravity, g.coordinate_system, g.earth_radius, g.coriolis_forcing = 9.81, 1, 6367500.0, False
    g.sea_level, g.dry_tolerance, g.friction_forcing = -9999.0, 1e-3, True
    # Standard GeoClaw/D-Claw numerical bed-friction setting retained from the official application examples.
    g.manning_coefficient, g.friction_depth, g.speed_limit = 0.025, 1e6, 30.0
    rundata.refinement_data.variable_dt_refinement_ratios, rundata.refinement_data.wave_tolerance = True, 0.01
    rundata.topo_data.topofiles.append([3, "jilong_dem_12p5m_64m.tt3"])

    qi = rundata.qinitdclaw_data
    qi.qinitfiles.append([3, 1, "initial_thickness.tt3"])
    qi.qinitfiles.append([3, 2, "initial_u.tt3"])
    qi.qinitfiles.append([3, 3, "initial_v.tt3"])
    qi.qinitfiles.append([3, 4, "initial_solid_fraction.tt3"])

    d = rundata.dclaw_data
    # Mount Baker USGS 2025 central material configuration (kref is the middle 10^-12--10^-10 case).
    d.rho_f, d.rho_s = 1100.0, 2700.0
    d.m_crit, d.m0, d.mref, d.kref = 0.64, 0.62, 0.60, kref
    d.phi, d.delta, d.mu, d.alpha_c, d.c1, d.sigma_0 = 38.0, 0.01, 0.005, 0.05, 1.0, 1e3
    d.src2method, d.alphamethod = 2, 1
    d.segregation, d.beta_seg, d.chi0, d.chie = 0, 0.0, 0.5, 0.5
    d.bed_normal, d.theta_input = 0, 0.0
    # No erodible-thickness field is publicly constrained for this reach: no arbitrary entrainment raster/rate.
    d.entrainment, d.entrainment_method, d.entrainment_rate, d.me = 0, 1, 0.0, 0.62
    # Hydrostatic is the baseline; the only failure-branch correction uses the
    # documented current gully example's pressure-ratio initialization.
    rundata.pinitdclaw_data.init_ptype = 0 if p_ratio == 1.0 else 3
    rundata.pinitdclaw_data.init_pratio = p_ratio
    rundata.flowgrades_data.flowgrades = []

    fg = fgmax_tools.FGmaxGrid()
    fg.point_style = 2
    fg.x1, fg.x2, fg.y1, fg.y2, fg.dx = XL + DX/2, XU - DX/2, YL + DX/2, YU - DX/2, DX
    fg.tstart_max, fg.tend_max, fg.dt_check, fg.min_level_check, fg.arrival_tol, fg.interp_method = 0.0, 600.0, 10.0, 1, 0.01, 0
    rundata.fgmax_data.num_fgmax_val = 5
    rundata.fgmax_data.fgmax_grids.append(fg)
    return rundata


if __name__ == "__main__":
    setrun().write()
