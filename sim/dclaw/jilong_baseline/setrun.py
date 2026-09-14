"""Single, provisional D-Claw baseline for downstream Jilong propagation."""
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
    clawdata.num_eqn = 7
    clawdata.num_aux = 10
    clawdata.capa_index = 0
    clawdata.t0 = 0.0
    clawdata.restart = False
    clawdata.output_style = 1
    clawdata.num_output_times = 6
    clawdata.tfinal = 1800.0
    clawdata.output_t0 = True
    clawdata.output_format = "ascii"
    clawdata.output_q_components = "all"
    clawdata.output_aux_components = "none"
    clawdata.output_aux_onlyonce = True
    clawdata.verbosity = 1
    clawdata.dt_variable = True
    clawdata.dt_initial = 0.5
    clawdata.dt_max = 1.0e99
    clawdata.cfl_desired = 0.45
    clawdata.cfl_max = 0.50
    clawdata.steps_max = 5000
    clawdata.order = 1
    clawdata.dimensional_split = "unsplit"
    clawdata.transverse_waves = 0
    clawdata.num_waves = 5
    clawdata.limiter = [4, 4, 4, 4, 4]
    clawdata.use_fwaves = True
    clawdata.source_split = "godunov"
    clawdata.num_ghost = 2
    clawdata.bc_lower[:] = ["extrap", "extrap"]
    clawdata.bc_upper[:] = ["extrap", "extrap"]
    clawdata.checkpt_style = 0

    amr = rundata.amrdata
    amr.amr_levels_max = 1
    amr.refinement_ratios_x = [1]
    amr.refinement_ratios_y = [1]
    amr.refinement_ratios_t = [1]
    amr.aux_type = ["center", "center", "yleft", "center", "center", "center", "center", "center", "center", "center"]
    amr.flag_richardson = False
    amr.flag2refine = False
    amr.regrid_interval = 3
    amr.regrid_buffer_width = 2
    amr.clustering_cutoff = 0.70
    amr.verbosity_regrid = 0
    amr.max1d = 500
    rundata.regiondata.regions = []
    rundata.gaugedata.gauges = [[1, PORT_X, PORT_Y, 0.0, 1800.0]]

    geo = rundata.geo_data
    geo.gravity = 9.81
    geo.coordinate_system = 1
    geo.earth_radius = 6_367_500.0
    geo.coriolis_forcing = False
    geo.sea_level = -9999.0
    geo.dry_tolerance = 1.0e-3
    geo.friction_forcing = True
    geo.manning_coefficient = 0.025
    geo.friction_depth = 1.0e6
    geo.speed_limit = 30.0
    rundata.refinement_data.variable_dt_refinement_ratios = True
    rundata.refinement_data.wave_tolerance = 0.01
    rundata.topo_data.topofiles.append([3, "jilong_dem_primary_64m.tt3"])
    rundata.qinitdclaw_data.qinitfiles.append([3, 1, "initial_thickness.tt3"])

    dclaw = rundata.dclaw_data
    dclaw.rho_f = 1000.0
    dclaw.rho_s = 2700.0
    dclaw.m_crit = 0.64
    dclaw.m0 = 0.63
    dclaw.mref = 0.60
    dclaw.kref = 1.0e-10
    dclaw.phi = 32.0
    dclaw.delta = 0.001
    dclaw.mu = 0.005
    dclaw.alpha_c = 0.01
    dclaw.c1 = 1.0
    dclaw.sigma_0 = 1.0e3
    dclaw.src2method = 2
    dclaw.alphamethod = 1
    dclaw.segregation = 0
    dclaw.beta_seg = 0.0
    dclaw.chi0 = 0.5
    dclaw.chie = 0.5
    dclaw.bed_normal = 0
    dclaw.theta_input = 0.0
    dclaw.entrainment = 0
    dclaw.entrainment_rate = 0.0
    dclaw.entrainment_method = 1
    dclaw.me = 0.6
    rundata.pinitdclaw_data.init_ptype = 0
    rundata.flowgrades_data.flowgrades = []

    fg = fgmax_tools.FGmaxGrid()
    fg.point_style = 2
    fg.x1, fg.x2 = XL + DX / 2, XU - DX / 2
    fg.y1, fg.y2 = YL + DX / 2, YU - DX / 2
    fg.dx = DX
    fg.tstart_max, fg.tend_max, fg.dt_check = 0.0, 1800.0, 10.0
    fg.min_level_check, fg.arrival_tol, fg.interp_method = 1, 1.0e-2, 0
    rundata.fgmax_data.num_fgmax_val = 5
    rundata.fgmax_data.fgmax_grids.append(fg)
    return rundata


if __name__ == "__main__":
    setrun().write()
