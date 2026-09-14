"""Phase 2B event-constrained downstream reconstruction; source state is external."""
import os

if "CLAW" not in os.environ:
    raise RuntimeError("Set CLAW to the installed Clawpack root before running setrun.py")

from clawpack.clawutil import data
from clawpack.geoclaw import fgmax_tools

XL, YL = 331_781.9, 3_128_083.2
DX, MX, MY = 64.0, 176, 261
XU, YU = XL + MX * DX, YL + MY * DX
PORT_X, PORT_Y = 340_837.9, 3_129_051.7


def setrun(claw_pkg="dclaw"):
    if claw_pkg.lower() != "dclaw":
        raise ValueError("This case requires the D-Claw package")
    rundata = data.ClawRunData("dclaw", 2)
    clawdata = rundata.clawdata
    clawdata.lower[:] = [XL, YL]
    clawdata.upper[:] = [XU, YU]
    clawdata.num_cells[:] = [MX, MY]
    clawdata.num_eqn, clawdata.num_aux, clawdata.capa_index = 7, 10, 0
    clawdata.t0, clawdata.restart = 0.0, False
    clawdata.output_style, clawdata.num_output_times, clawdata.tfinal = 1, 30, 1800.0
    clawdata.output_t0, clawdata.output_format = True, "ascii"
    clawdata.output_q_components, clawdata.output_aux_components, clawdata.output_aux_onlyonce = "all", "none", True
    clawdata.verbosity, clawdata.dt_variable, clawdata.dt_initial, clawdata.dt_max = 1, True, 0.5, 1.0e99
    clawdata.cfl_desired, clawdata.cfl_max, clawdata.steps_max = 0.45, 0.50, 5000
    clawdata.order, clawdata.dimensional_split, clawdata.transverse_waves = 1, "unsplit", 0
    clawdata.num_waves, clawdata.limiter, clawdata.use_fwaves, clawdata.source_split = 5, [4, 4, 4, 4, 4], True, "godunov"
    clawdata.num_ghost, clawdata.bc_lower[:], clawdata.bc_upper[:], clawdata.checkpt_style = 2, ["extrap", "extrap"], ["extrap", "extrap"], 0

    amr = rundata.amrdata
    amr.amr_levels_max, amr.refinement_ratios_x, amr.refinement_ratios_y, amr.refinement_ratios_t = 1, [1], [1], [1]
    amr.aux_type = ["center", "center", "yleft", "center", "center", "center", "center", "center", "center", "center"]
    amr.flag_richardson, amr.flag2refine, amr.regrid_interval, amr.regrid_buffer_width = False, False, 3, 0
    amr.clustering_cutoff, amr.verbosity_regrid, amr.max1d = 0.70, 0, 500
    rundata.regiondata.regions = []
    rundata.gaugedata.gauges = [[1, PORT_X, PORT_Y, 0.0, 1800.0]]

    geo = rundata.geo_data
    geo.gravity, geo.coordinate_system, geo.earth_radius, geo.coriolis_forcing = 9.81, 1, 6_367_500.0, False
    geo.sea_level, geo.dry_tolerance, geo.friction_forcing = -9999.0, 1.0e-3, True
    geo.manning_coefficient, geo.friction_depth, geo.speed_limit = 0.025, 1.0e6, 30.0
    rundata.refinement_data.variable_dt_refinement_ratios, rundata.refinement_data.wave_tolerance = True, 0.01
    rundata.topo_data.topofiles.append([3, "jilong_dem_primary_64m.tt3"])
    # D-Claw-supported qinit state: depth plus velocities (not a user-modified solver).
    rundata.qinitdclaw_data.qinitfiles.extend([
        [3, 1, "initial_thickness.tt3"],
        [3, 2, "initial_velocity_u.tt3"],
        [3, 3, "initial_velocity_v.tt3"],
    ])

    dclaw = rundata.dclaw_data
    dclaw.rho_f, dclaw.rho_s = 1000.0, 2700.0
    dclaw.m_crit, dclaw.m0, dclaw.mref, dclaw.kref = 0.64, 0.63, 0.60, 1.0e-10
    dclaw.phi, dclaw.delta, dclaw.mu, dclaw.alpha_c, dclaw.c1, dclaw.sigma_0 = 32.0, 0.001, 0.005, 0.01, 1.0, 1.0e3
    dclaw.src2method, dclaw.alphamethod = 2, 1
    dclaw.segregation, dclaw.beta_seg, dclaw.chi0, dclaw.chie = 0, 0.0, 0.5, 0.5
    dclaw.bed_normal, dclaw.theta_input = 0, 0.0
    dclaw.entrainment, dclaw.entrainment_rate, dclaw.entrainment_method, dclaw.me = 0, 0.0, 1, 0.6
    rundata.pinitdclaw_data.init_ptype = 0
    rundata.flowgrades_data.flowgrades = []

    fg = fgmax_tools.FGmaxGrid()
    fg.point_style, fg.x1, fg.x2, fg.y1, fg.y2, fg.dx = 2, XL + DX / 2, XU - DX / 2, YL + DX / 2, YU - DX / 2, DX
    fg.tstart_max, fg.tend_max, fg.dt_check, fg.min_level_check, fg.arrival_tol, fg.interp_method = 0.0, 1800.0, 10.0, 1, 1.0e-2, 0
    rundata.fgmax_data.num_fgmax_val = 5
    rundata.fgmax_data.fgmax_grids.append(fg)
    return rundata


if __name__ == "__main__":
    setrun().write()
