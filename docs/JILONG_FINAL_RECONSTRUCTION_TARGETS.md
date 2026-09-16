# Event-constrained downstream reconstruction: evidence targets

## Purpose and decision rule

This document locks the public evidence before construction of a new final D-Claw case.  It does not create an inflow hydrograph or a parameter inversion.  The required product is an **event-constrained downstream reconstruction**, not an exact historical reproduction or a unique reconstruction of the unobserved boundary state.

The evidence hierarchy below is taken from `EVENT_PARAMETER_TABLE.md`, `TIMING_EVIDENCE_RECONCILIATION.md`, `MODEL_ENTRY_DEFINITION.md`, `JILONG_GEOMORPHIC_MOBILITY_DIAGNOSIS.md`, and `REFERENCE_DCLAW_BASIS.md`.  No broad external search was performed.

## A. Higher-confidence event evidence

| Constraint | Status and permitted use |
|---|---|
| Source-to-port chain is about 22 km with about 3,400 m relief | Peer-reviewed reconstruction / official ITP-CAS summary.  Whole source-to-port chain only; not the downstream model reach. |
| Whole-chain elapsed time is about 7 min | Peer-reviewed reconstruction / official ITP-CAS summary.  It includes high-elevation release, transformation and entrainment; it is not an entry-to-port arrival observation. |
| Progressive entrainment and transformation are part of the event narrative | Qualitatively supported by the peer-reviewed reconstruction.  This supports a future process hypothesis only; it supplies no erosion depth, rate, volume or spatial field. |
| About 8.4e6 m2 downstream scour/erosion area | Official remote-sensing/process evidence.  It is not an erosion-volume observation or an erodible-depth raster. |
| About 0.7 km2 affected core near Jilong Port | Official impact-area information.  It is not automatically a debris-flow wet-area, inundation, or deposition polygon. |
| Jilong Port location | Official downloaded point layer; suitable as a port reference point, subject to the difference between a point and a hydraulic cross-section. |

## B. Lower-confidence / soft engineering constraints

| Constraint | Status and permitted use |
|---|---|
| About 15 km, main-river entry to Jilong Port | NCDC/NIEER rapid-assessment estimate.  It is a soft stage-length check only because no entry cross-section is published.  It must not be used to move S0 to a target length. |
| About 30 min, main-river downstream stage | Early rapid-assessment context.  It is explicitly not a final calibration target. |
| About 8.3 m/s | Early reach-average estimate; it is not an entry velocity. |
| Port peak discharge about 3e4--5e4 m3/s | Preliminary port diagnostic only: no rating curve, timestamp, cross-section or hydrograph is public. |
| Reference D-Claw material configuration | Mature-application baseline (`rho_f=1100`, `rho_s=2700`, `m0=0.62`, `mcrit=0.64`, `phi=38 deg`, `mu=0.005`, central `kref=1e-11`).  These are implementation priors, not event measurements. |

## C. Unknown / not publicly available

The following remain unobserved for the downstream entry: a georeferenced physical main-river-entry section; a continuous surveyed event corridor from that section to the port; `Q(t)`; entry depth, velocity vector, solid fraction, pore-pressure state and mixture volume; event-time cross-sections; material/rheology measurements; an erosion volume; an entrainment law and active erosion area; timestamped S1--S4 arrivals; and a georeferenced inundation/deposition polygon whose semantics are compatible with a D-Claw wet-area metric.

Consequently, a model-derived hydrograph would have to be labelled **inferred effective reconstruction parameters**, not measured historical conditions.  This remains true even if a later geometry gate permits simulations.
