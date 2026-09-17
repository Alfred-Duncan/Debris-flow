# Final Jilong/Gyirong spatial Gate A

## Decision

`GATE_A = PASS`. **THE SPATIAL GATE IS CLOSED.**

The final geometry is `corridor/jilong_canonical_corridor.geojson`. It is a continuous S0-S4 human-reviewed event-constrained engineering corridor axis, not a surveyed thalweg or an asserted exact historical hydraulic centreline. T01-T12 were reconstructed from the completed B-tile valley review; the accepted 607.99 m gap trace and T13-T16 human-reviewed downstream correction are retained. Official river vectors are **REFERENCE ONLY**.

## Length

Canonical length is **19.525 km**. Compared only after reconstruction with the historical approximate 15 km downstream-stage description, this is +4.525 km (+30.2%). The geometry was not adjusted to improve that comparison.

## Raw DEM QA and uncertainty

Maximum raw 250 m adverse rise is 268.0 m in the native 12.5 m DEM and 115.0 m in the independent 8 m DEM. Raw samples and all significant DEM-data uncertainty intervals are retained in `corridor/jilong_canonical_corridor_profile.csv` and `reports/JILONG_GATE_A_FINAL.json`. They are reported as terrain-data uncertainty on the image-confirmed corridor; no DEM was edited or used to move the human-reviewed geometry.

## Outputs

- `corridor/jilong_T01_T12_human_reviewed_corridor.geojson`
- `corridor/jilong_canonical_corridor.geojson`
- `corridor/jilong_control_sections.geojson`
- `corridor/jilong_canonical_corridor_profile.csv`
- `reports/FINAL_CORRIDOR_TOPOLOGY_QA.json`
- `figures/30_final_S0_S4_longitudinal_profile.png`
- `figures/31_FINAL_S0_S4_EVENT_CONSTRAINED_CORRIDOR.png`

No D-Claw run, hydrograph construction, material tuning, entrainment activation, scenario generation, or FNO work was performed.
