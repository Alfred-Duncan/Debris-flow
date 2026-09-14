# Phase 2B — first event-constrained downstream reconstruction

## Scope and source representation

This is a first, deliberately small source-only reconstruction exercise for propagation **after entry to the main river**.  It uses the fixed Phase-2B entry, 64 m grid, 1800 s duration, DEM construction, D-Claw material parameters, Manning coefficient, friction angle, viscosity, solid-fraction defaults, and disabled entrainment/segregation from `jilong_baseline`.  The original baseline was not modified.

The sole source mechanism is D-Claw-supported `qinit_dclaw_data`: a fixed elliptical initial mass with nonzero downstream `u,v` state.  This represents already-moving material at the model entry.  `source_depth_m` and `initial_speed_ms` are **inverse-model parameters**, not observations and not claims about entry depth or velocity.  `SOURCE_MODEL.md` records the mechanism and fixed assumptions.

Arrival is the first 60 s output with interpolated depth at the official port point above 0.01 m.  Discharge is positive downstream flux integrated across a fixed 384 m, six-strip (64 m each) numerical cross-section centered at that point.  It is not inferred from the point gauge.

## Candidate runs

| run_id | source depth (m) | initial speed (m/s) | initial grid volume (m³) | reached port | arrival (s) | port peak Q (m³/s) | port peak depth (m) | port peak speed (m/s) | runtime (s) | stable |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| run_01 | 6 | 8 | 442,368 | false | not reached | 0.0 | 0.000 | 0.000 | 47.6 | true |
| run_02 | 10 | 14 | 737,280 | false | not reached | 0.0 | 0.000 | 0.000 | 51.6 | true |
| run_03 | 25 | 28 | 1,843,200 | false | not reached | 0.0 | 0.000 | 0.000 | 75.8 | false |

The first two candidates are stable but do not reach the port by 1800 s.  The third broad adaptive increase remains short of the port and produces a domain maximum speed of about 95.8 m/s, so it is excluded as numerically/physically unsuitable despite completing without NaNs.

## Selection and comparison with soft event constraints

No event-scale candidate was found.  The most informative stable case is `run_02` (`h=10 m`, `V0=14 m/s`), because it is the stronger of the stable tests; it still does not reach Jilong Port, so it has no port arrival time and zero diagnostic port discharge.

- **Arrival mismatch:** no stable candidate reached the port by the 1800 s soft downstream consistency time.
- **Peak-discharge mismatch:** all candidates have 0 m³/s at the fixed port section versus the soft 3–5 × 10⁴ m³/s estimate.
- **Stability:** `run_01` and `run_02` have finite fields and domain maximum speeds below 36 m/s. `run_03` is excluded because its maximum is about 95.8 m/s; it is not an acceptable way to force transport.

The sole diagnostic figure is [event_reconstruction_candidates.png](../../../outputs/phase2b/event_reconstruction_candidates.png).

## Interpretation and stop decision

Within this three-run, two-parameter adaptive search, simple moving-initial-mass source adjustment is **not sufficient** to create an event-consistent downstream reconstruction.  Increasing both source thickness and momentum from 6 m/8 m s⁻¹ to 25 m/28 m s⁻¹ changed transport only from sub-kilometre progression to approximately one kilometre, before unacceptable high velocities appeared.  Continuing to force this same source representation would not be scientifically justified.

Further source-representation work is needed before material-parameter calibration.  In particular, these results do **not** justify changing friction angle, viscosity, solid fraction, or entrainment to compensate.  No hydrograph has been inferred or created in this phase.
