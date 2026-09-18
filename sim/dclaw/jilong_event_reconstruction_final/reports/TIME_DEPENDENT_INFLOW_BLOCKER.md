# Blocker: fixed-S0 time-dependent D-Claw inflow

## Decision

**Stop before simulation.** No smoke test or C1--C6 source-only run was
started, and no inflow preflight was produced.

The installed local D-Claw source was inspected rather than substituted with a
new unvalidated source formulation. It provides:

1. qinit for initialization at t=0, which cannot represent the required
   time-dependent source;
2. src2, which applies D-Claw physical friction, pore-pressure, and
   entrainment updates, not a configured spatial discharge hydrograph; and
3. a user physical-boundary pathway (mthbc=0) that requires a case-specific
   Fortran boundary state.

There is no local, documented configuration interface for a fixed interior S0
source carrying the required time-varying mixture mass and momentum.

## Why the existing custom boundary routine is not reusable

Older Jilong experiments include a custom bc2amr.f90. It sets ghost cells only
on the rectangular western external domain boundary. It hard-codes an entry
location, 256 m support width, 10 m flow depth, solid fraction 0.63, a
prescribed direction ratio, and an old material configuration. It also belongs
to cases that contain the prohibited speed_limit=30.0 setting.

Those values are neither supplied by the accepted S0 handoff nor among the
permitted two-dimensional source design variables:

- the six cases may vary only total mixture volume V and duration T;
- the accepted S0 section is explicitly not a surveyed hydraulic width;
- no observed entry depth or entry velocity is available; and
- importing that state would silently introduce extra source parameters.

Repositioning a rectangular computational boundary near S0 would not resolve
the missing state: a boundary discharge Q(t) still requires a depth--velocity
state and compatible D-Claw pore-pressure state.

## Conserved-state requirement

The installed D-Claw source defines seven state slots:

| Slot | Meaning |
|---|---|
| 1 | h |
| 2 | hu |
| 3 | hv |
| 4 | hm |
| 5 | pb |
| 6 | hchi |
| 7 | bdif |

Thus a new source hook would have to prescribe or evolve all applicable
mixture-state components consistently. hm can use the locked m0, and a
hydrostatic pb relation exists for an initialized state, but those facts do not
determine the missing entry depth, velocity magnitude, or validated transient
boundary/source pressure condition.

## Required next decision

Before a simulation can be scientifically auditable, provide either:

1. an approved documented D-Claw fixed-S0 source formulation that specifies
   the full transient conserved state; or
2. explicit authorization for a named computational closure for inlet depth and
   velocity/momentum magnitude, clearly added as further exploratory source
   parameters rather than represented as observations.

After that decision, the deterministic S0 support can be selected from the
accepted S0 line, 64 m terrain, and downstream corridor tangent. The prescribed
half-sine Q(t) can then be checked for required volume and peak conservation
before any D-Claw run.
