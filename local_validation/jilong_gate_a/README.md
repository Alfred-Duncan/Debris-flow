# Local Gate A spatial-evidence validation

This folder records the local, read-only GIS validation of the 2026-08-26 Jilong event evidence. It is deliberately separate from simulation inputs and outputs.

## Included

- `scripts/`: reproducible preparation and finalization scripts;
- `inventory/`: source and spatial-file inventories;
- `derived/`: official port point and explicitly diagnostic-only reference points;
- `corridor/`: an imagery-supported **candidate** bridge, S0 candidate table, and a defensible S4 port-channel inspection section;
- `reports/`, `figures/`, and `review_package/`: the evidence audit and review materials.

## Key result

The official fourth-order river vector has a 525.53 m straight endpoint discontinuity. The imagery-supported candidate trace is 607.99 m long; it is not an official original line, hydraulic centreline, or canonical D-Claw corridor.

Gate A remains `HUMAN_IMAGE_REVIEW_REQUIRED`: the event-specific entry boundary S0 is not established. The S4 inspection section is resolved on the nearby actual official main-channel geometry, 646.33 m from the official port point. No canonical corridor, S0/S1/S2/S3 sections, profile, calibration, or simulation result is included here. Start with [the final Gate report](reports/JILONG_GATE_A_FINAL.md).

## Deliberately excluded

No original download ZIP, DEM, orthophoto/PlanetScope raster, extracted raw data, or AOI GeoTIFF is committed. The inventories retain their original local paths so the provenance of the local audit remains explicit.
