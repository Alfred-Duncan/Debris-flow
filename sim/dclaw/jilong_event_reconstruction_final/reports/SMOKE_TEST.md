# Final C3 smoke test

Status: **FAIL -- OTHER (boundary-flux consistency)**.

The final authorized C3 smoke test used V=2.0e6 m3, T=90 s, a 120 s end time,
and outputs at 0, 30, 60, 90, and 120 s. The generated inflow.data passed a
strict local format check: ten separate finite numeric lines and no literal
backslash-n token.

The solver completed normally and all five frames parsed with finite values.
The 64 m terrain loaded unchanged; wet cells first occurred at the north S0
support, and the flow advanced in the accepted downstream corridor direction.
No artificial 30 m/s cap is present; the smoke maximum speed was 15.579 m/s.

| t (s) | max depth (m) | max speed (m/s) | P99 wet speed (m/s) | wet cells | domain h-volume (m3) | max corridor chainage (m) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 | 0 | 0 | 0.0 |
| 30 | 0.177 | 0.000 | 0.000 | 3 | 2,178 | 54.6 |
| 60 | 37.167 | 8.031 | 7.908 | 8 | 694,135 | 142.1 |
| 90 | 32.105 | 15.579 | 14.964 | 20 | 733,817 | 509.5 |
| 120 | 33.952 | 12.769 | 12.086 | 24 | 730,303 | 689.9 |

The prescribed cumulative volume is 2.0e6 m3 by t=90 s. At that time the
model contains 0.734e6 m3 (36.7 percent of the prescribed volume). The front
is only 0.509 km from S0, so this shortfall is not plausibly explained by
downstream domain exit. The actual numerical external-boundary flux is thus
not broadly consistent with the unit-Froude preflight flux, despite the
analytical ghost-state flux check.

This is not a terrain, Gate A/B, material, entrainment, or source-sign failure:
the flow direction is downstream and the solver remains finite. It is a
boundary-flux consistency issue in the Riemann solution and is classified
OTHER under the authorized smoke taxonomy. The task requires stopping here;
there is no C1--C6 full run and no physical retuning.
