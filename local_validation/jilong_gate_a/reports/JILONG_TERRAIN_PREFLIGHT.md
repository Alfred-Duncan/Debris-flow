# Jilong computational terrain preflight (Gate B)

`GATE_B = CONDITIONAL_TERRAIN_REMEDIATION_REQUIRED`. Gate A remains `PASS` and is not reopened.

The raw 12.5 m DEM was interrogated on 392 cross-sections at about 50 m spacing and 250 m half-width. The robust floor is the median of the lowest five valid raw samples, not a modified terrain surface. 8 m coverage is 77.8% on the centreline, 77.2% in the 250 m band, and 76.7% in the 500 m band; it is therefore `8M_DEM_NOT_FULL_DOMAIN_CANDIDATE`.

The native robust-floor estimate is 2738.0 m at S0 and 1796.0 m at S4: a 942.0 m total drop and mean slope -0.0482. Native centreline/robust-floor maximum 250 m rises are 259.0/234.0 m. Robust-floor 250 m rises are 234.0 m (12.5 m) and 92.0 m (8 m); 500 m values are 233.0/100.0 m. Candidate barriers: 6 (A/B/C/D = 1/5/0/0).

The blocking condition is `HIGH_SEVERITY_TYPE_B_NATIVE_DEM_ARTIFACTS`: high Type B native-only crests B04 at (337968.9, 3130872.3), B05 at (339671.3, 3129974.3) persist in the 32 m/64 m conservative diagnostics and can block a D-Claw grid. They are not evidence against the completed human-reviewed corridor, but native terrain is not simulation-ready as-is. Before D-Claw, use authoritative replacement terrain or a separately justified local DEM-patch replacement at those coordinates; do not carve or otherwise condition this DEM arbitrarily.

No DEM raster was changed; 32 m and 64 m products are in-memory diagnostic envelopes only. No D-Claw calculation was run.
