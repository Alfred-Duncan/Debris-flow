# Phase 2B source model

## Chosen D-Claw mechanism

This case uses D-Claw's supported `qinit_dclaw_data` state initialization, not a solver modification and not a time-dependent boundary condition.  It initializes one fixed-geometry elliptical mass at the Phase-2B model entry with:

- `h`: uniform source thickness inside the ellipse;
- `hu` and `hv`: supplied through D-Claw qinit components 2 and 3, which accept velocity components and form momentum internally.

The initial velocity vector follows the mapped river's local downstream tangent.  This represents an already-moving debris flow at the selected downstream model entry, which matches the chosen model stage better than a stationary release or an invented upstream hydrograph.

## Inverse-model parameters

Only two source quantities vary between candidates:

| Parameter | Meaning | Status |
|---|---|---|
| `source_depth_m` | Uniform initial thickness of the fixed entry ellipse; determines released grid volume. | Inverse-model parameter, not observed. |
| `initial_speed_ms` | Initial speed of that mass in the mapped downstream direction. | Inverse-model parameter, not observed and not set to 8.3 m/s. |

The entry point, ellipse geometry, domain, 64 m grid, DEM construction, 1800 s duration, material parameters, Manning coefficient, friction angle, viscosity, solid fraction defaults, and disabled entrainment/segregation are held fixed from `jilong_baseline` except for the explicitly authorised initial momentum state.

## Diagnostics

Port arrival is the first 60 s output at which interpolated depth at the official Jilong Port point exceeds `0.01 m`.  Port discharge is estimated across a fixed 384 m cross-section centered at the official port point, using six 64 m strips and positive downstream flux only.  The cross-section is a reproducible numerical diagnostic, not a surveyed section or observed hydrograph.
