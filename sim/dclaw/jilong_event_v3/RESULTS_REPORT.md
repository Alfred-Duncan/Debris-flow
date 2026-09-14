# Phase 2D V3: 32 m terrain-representation check

V3 is the single targeted response to the V2 coarse-grid diagnostic. It uses a 32 m fixed grid, observed nodata-masked 8 m terrain with observed 12.5 m fallback, the unchanged domain/source/material setup, and no hand-carved terrain.

| Run | Max / final front (km) | Raw / h>.1 / h>1 / p99 speed (m/s) | Max CFL | Numerically clean |
|---|---:|---:|---:|---|
| G1 | 1.471 / 1.471 | 99.08 / 99.08 / 46.25 / 30.70 | 0.764 | yes |
| G2 | 1.951 / 1.951 | 136.32 / 136.32 / 136.32 / 34.64 | 0.663 | yes |
| G3 | 1.951 / 1.951 | 151.07 / 151.07 / 151.07 / 37.07 | 0.537 | yes |
| G4 | 1.951 / 1.951 | 149.32 / 149.32 / 149.32 / 38.00 | 0.587 | yes |

No candidate reaches the 17.624 km hydraulic port-proxy. The prior approximately 1.97 km stall persists after the one justified terrain-representation correction. Best front: G2/G3/G4 tied at 1.951 km. G-batch wall time: 851.24 s.
