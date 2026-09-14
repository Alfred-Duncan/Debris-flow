# Phase 2C source model

V2 replaces the Phase-2B finite moving `qinit` ellipse with one **case-level finite-duration left-boundary inflow**. `bc2amr.f90` is an application-local replacement of the standard boundary wrapper only; no D-Claw constitutive routine, source routine, or Riemann solver is modified.

The inlet plane is at x=334021.9 m, EPSG:32645, 29.9 m upstream of the existing entry point (334051.8, 3143392.5 m), on the same grid-aligned river transect. The downstream terrain is the same real DEM and 64 m grid; only unused terrain west of the inlet plane is removed.

## Fixed provisional boundary state

A 256 m-wide ribbon centred at y=3143392.5 m has a triangular pulse: `Q(t)=Q_peak*(1-|2t/T_in-1|)` for 0≤t≤T_in and zero afterwards. Its imposed volume is `Q_peak*T_in/2`.

The inverse variables are only `Q_peak` and `T_in`. The fixed internal construction is depth 10 m, solid fraction 0.63, species fraction 0.5, hydrostatic basal pore pressure `rho_f*g*h`, and velocity `(u,v)=(Q/(hW), -0.6472/0.7623*u)`. These are a numerically consistent boundary-state construction, **not observed entry depth, velocity, or composition**.

For the mandatory entrainment comparison, the documented D-Claw mechanism uses `entrainment=1`, method 1, rate 0.2, entrainable solid fraction 0.65, and a uniform provisional 2 m `h_e` layer. The parameters are D-Claw-supported defaults/provisional inputs, not event measurements; after run 2 they remain fixed.
