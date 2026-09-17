# Local Gate A spatial-evidence validation

This folder records the local, read-only GIS validation of the 2026-08-26 Jilong event evidence. It is deliberately separate from simulation inputs and outputs.

## Included

- `scripts/`: reproducible preparation and finalization scripts;
- `inventory/`: source and spatial-file inventories;
- `derived/`: official port point and explicitly diagnostic-only reference points;
- `corridor/`: an imagery-supported **candidate** bridge, resolved S0 confluence point/inspection section, an approved local downstream correction, and a defensible S4 port-channel inspection section;
- `reports/`, `figures/`, and `review_package/`: the evidence audit and review materials.

## Key result

**THE SPATIAL GATE IS CLOSED.** `GATE_A = PASS` after incorporation of the completed external human review of all 16 S0--S4 tiles. The official river vector is now **REFERENCE ONLY**.

The final geometry is [corridor/jilong_canonical_corridor.geojson](corridor/jilong_canonical_corridor.geojson): a continuous S0--S4 human-reviewed event-constrained engineering corridor axis, not a surveyed thalweg or exact historical hydraulic centreline. T01--T12 are reconstructed in [jilong_T01_T12_human_reviewed_corridor.geojson](corridor/jilong_T01_T12_human_reviewed_corridor.geojson); the accepted 607.99 m former-vector-gap trace and the T13--T16 downstream human-review correction are retained. Final inspection sections, a raw two-DEM profile, topology QA, and final maps are included. Raw DEM disagreements remain explicitly reported as uncertainty rather than being hidden or used to relocate an image-confirmed valley axis.

No D-Claw run, hydrograph construction, material calibration, entrainment activation, scenario generation, or FNO work is included. Start with [the final Gate report](reports/JILONG_GATE_A_FINAL.md).

## Deliberately excluded

No original download ZIP, DEM, orthophoto/PlanetScope raster, extracted raw data, or AOI GeoTIFF is committed. The inventories retain their original local paths so the provenance of the local audit remains explicit.
