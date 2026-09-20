# Physics output schema

Definitions are read from upstream swe/solver_sparse.py; this workspace does not alter them.

| Name | Class | Upstream semantics |
|---|---|---|
| h, hu, hv, c, ice, bed_change | cell fields in sparse frames | Wet-cell fields indexed by row-major idx; c=hc/h, ice=hi/h; bed_change=z-z0. |
| speed | cell field in sparse frames | sqrt(u squared + v squared), where u=hu/h and v=hv/h. |
| Q, Qdebris, hmax, stage, cmax, wet_width_m | transect metrics in series.csv | Flux and geometry across author-defined transect cells, prefixed by station ID. |
| total_volume_m3, debris_volume_m3, entrained_m3, ice_volume_m3, melted_m3 | scalar time diagnostics | Domain-integrated model quantities. |
| debris_front_route_km, debris_front_upper_km, flood_front_route_km | scalar time diagnostics | Front locations using stored chainage fields. |

Sparse frame orientation: idx = row * cols + col; dense fields are [C, rows, cols] using NumPy C-order.
