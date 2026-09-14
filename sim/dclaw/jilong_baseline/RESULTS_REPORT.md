# First Jilong D-Claw baseline: results report

## Scope

This is one provisional, uncalibrated D-Claw run for the downstream propagation stage after debris material enters the mapped main river. It does not model the upstream ice-collapse stage and does not attempt to match published event constraints.

## Study domain and data

- CRS: WGS 84 / UTM zone 45N (EPSG:32645).
- Domain: `x=331781.9..343045.9 m`, `y=3128083.2..3144787.2 m`; 11.264 by 16.704 km.
- Mapped fourth-order river route from entry to Jilong Port: 17.624 km.
- Primary terrain: downloaded 8 m DEM. It supplies 55.53% of the topography-grid nodes in this domain.
- The downloaded 8 m DEM has a clear no-data reach near the upstream entry. The downloaded 12.5 m DEM fills only those 8 m voids; no alternative terrain version was evaluated.
- River alignment is from the downloaded fourth-order river vector; the port gauge is at the downloaded Jilong Port location.

## Numerical configuration

- Fixed grid: 64 m, 176 by 261 cells; one AMR level (no AMR).
- Simulation time: 1,800 s; output every 300 s plus the initial state.
- Source: D-Claw-native `qinit` thickness field, initialized as a stationary elliptical mass at the mapped main-river entry (`334051.5, 3143392.2 m` UTM).
- Source geometry: semi-axes 192 and 128 m; depth 3 m; grid volume 221,184 m3.
- Boundary conditions: extrapolation on all sides.
- Provisional material parameters: `rho_f=1000 kg/m3`, `rho_s=2700 kg/m3`, `m0=0.63`, `m_crit=0.64`, `mref=0.60`, `phi=32 deg`, `kref=1e-10`, `mu=0.005`, and Manning `n=0.025`.
- Entrainment and segregation are disabled. These values are copied from supported D-Claw example-style settings and are not calibrated to Jilong.

## Run outcome

- Solver exit status: 0; 1,800 s completed.
- Wall-clock runtime: 26 s using 12 threads.
- Output: 7 flow-field frames, a Jilong Port gauge time series, and an FGmax grid.
- No NaNs or solver abort occurred.
- Final maximum depth: 3.0 m; final maximum speed: 4.0261 m/s.
- Material remained associated with the mapped river corridor and made 75.5 m of downstream centroid progress (88.4 m direct centroid displacement).
- The port gauge recorded zero flow depth in this run.

## Interpretation and limitation

The baseline proves that the installed D-Claw solver accepts the real-data terrain, source representation, and case configuration and produces normal outputs. It is not an event reconstruction. Its dominant mismatch is transport: after 1,800 s it advances about 76 m, versus the roughly 15 km / 30 min published reference. No rerun, tuning, calibration, resolution comparison, or parameter sweep was performed to address that mismatch.

## Files

- `build_inputs.py`: real DEM reprojection, 8 m primary / 12.5 m void fill, and source field generation.
- `setrun.py`: D-Claw configuration.
- `jilong_dem_primary_64m.tt3` and `initial_thickness.tt3`: model inputs.
- `*.data`: generated D-Claw input files.
- `_output/fort.q0000` through `_output/fort.q0006`: field outputs.
- `_output/fgmax0001.txt`: maximum-depth/speed and arrival-related fixed-grid output.
- `_output/gauge00001.txt`: Jilong Port gauge output.
- `run_summary.json`, `run.log`, and `runtime_*.txt`: numerical and runtime record.
- `previews/depth_snapshots.png` and `previews/final_depth_speed.png`: minimal visual checks.
