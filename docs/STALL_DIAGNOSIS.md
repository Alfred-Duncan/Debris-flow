# Phase 2D: repeatable 1.97 km stall

## Diagnosis

The mapped-route position `s=1.970 km` is `(335359.419, 3141856.819)` m EPSG:32645 (`85.319505 E, 28.392731 N`). The actual RUN_D wet-front cell at both t=240 s and t=600 s is `(335269.9, 3141811.2)` m. The mapped fourth-order river direction is approximately `(0.542, -0.840)`.

The two vector components have a 525.53 m gap, but at `s≈13.91–14.42 km`, about 11.93 km downstream of the stall. It cannot explain the physical front location.

At the stall neighbourhood the old 64 m terrain profile has a 23.31 m local uphill cell-scale step. The continuously sampled original 12.5 m DEM at the same mapped reach has a maximum 5.0 m local jump and no source transition. Thus the evidence supports **C. COARSE_GRID_CHANNEL_BLOCKAGE**: the 64 m representation creates a stronger local rise than the native DEM supports. No DEM merge seam occurs there.

## Corrective action

V3 applies exactly one terrain-representation correction: a 32 m fixed grid. It uses nodata-masked observed 8 m DEM as primary and observed 12.5 m DEM as fallback; throughout the affected first 3 km the 12.5 m DEM supplies the missing 8 m coverage. Off-corridor cells absent from both observed rasters retain the prior terrain only to maintain the unchanged full model domain. No channel was carved and no cell elevation was manually edited.

## Corrected G batch

| Run | Source | Entrainment | Max / final front (km) | Clean? | raw / h>.1 / h>1 / p99 speed (m/s) |
|---|---|---|---:|---|---:|
| G1 | 30k, 60 s | off | 1.471 / 1.471 | yes | 99.08 / 99.08 / 46.25 / 30.70 |
| G2 | 30k, 60 s | on | 1.951 / 1.951 | yes | 136.32 / 136.32 / 136.32 / 34.64 |
| G3 | 30k, 90 s | on | 1.951 / 1.951 | yes | 151.07 / 151.07 / 151.07 / 37.07 |
| G4 | 30k, 120 s | on | 1.951 / 1.951 | yes | 149.32 / 149.32 / 149.32 / 38.00 |

All four solver runs completed with finite fields; maximum logged CFL was 0.764, 0.663, 0.537 and 0.587, respectively. None reached the 17.624 km hydraulic port-proxy. The 1.97 km barrier therefore did **not** disappear under this single 32 m representation correction; the best front is 1.951 km.

## Next step

Do not increase Q. The next investigation should use independently constrained channel geometry/terrain representation or evidence-based material and entrainment characterization, not automatic tuning.
