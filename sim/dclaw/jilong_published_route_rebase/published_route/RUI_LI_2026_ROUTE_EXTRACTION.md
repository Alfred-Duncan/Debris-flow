# Rui Li 2026 representative route extraction

- Source PDF: `2609.04563v2.pdf`, Figure 1A.
- CRS: EPSG:32645 (WGS84 / UTM zone 45N).
- Extraction method: direct PDF vector-path extraction, then affine coordinate recovery from the plotted UTM axis ticks.
- Vertices: 46
- Segments: 45
- Extracted planar length: 21843.425064 m
- Paper-reported planar length: 21843.424880 m
- Absolute difference: 0.000184 m
- Extracted source endpoint: (356076.514348, 3129773.452394)
- Extracted port endpoint: (340900.994467, 3129256.533688)
- Old S0 distance to published route: 13883.540 m
- Existing corridor S4 endpoint distance to published route: 505.959 m
- Official port point distance to published route endpoint/nearest route: 214.290 m

## Interpretation

The geometry is a **published representative source-to-port centreline**. It should be used as a literature-constrained propagation corridor, not described as a surveyed channel thalweg or a tracked-particle trajectory.

The previous S0 is ~13.88 km from this published route, so the previous S0/corridor geometry must not be inherited without re-evaluation.
