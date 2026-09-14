# Terrain consistency check

**Result: PASS**

The source raster was reprojected with physical rows north-to-south, then reversed only when assigning GeoClaw's increasing-y `Topography.Z`. The TT3 file was read back before comparison.

| Metric | Value |
|---|---:|
| Samples / valid | 107 / 107 |
| Median |Δz| | 2.625 m |
| 95th percentile |Δz| | 8.724 m |
| Maximum |Δz| | 19.386 m |
| Counterfactual y-mirror median |Δz| | 399.406 m |

| Nearest-observed nodata fills outside coverage | 1642 nodes |

## Required fixed physical points

| x (m) | y (m) | Native DEM (m) | Generated TT3 (m) | |Δz| (m) |
|---:|---:|---:|---:|---:|
| 334051.8 | 3143392.5 | 2814.08 | 2820.63 | 6.546 |
| 334450.0 | 3141000.0 | 3175.92 | 3179.04 | 3.126 |
| 335100.0 | 3139100.0 | 3192.58 | 3196.67 | 4.091 |
| 335850.0 | 3137000.0 | 3383.02 | 3388.96 | 5.946 |
| 337200.0 | 3134700.0 | 2217.11 | 2213.19 | 3.921 |
| 340571.8 | 3129640.8 | 1814.49 | 1822.89 | 8.407 |
| 340837.9 | 3129051.7 | 1776.39 | 1795.77 | 19.386 |

The 100 seeded random **valid native-DEM** points are included in the aggregate metrics above. Nearest-observed fills are confined to DEM nodata nodes; the required entry, route, mid-corridor and port checks must all be native-observation comparisons. A failed check is a hard stop for the solver.
