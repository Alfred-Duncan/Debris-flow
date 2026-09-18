# Final supercritical-normal C3 smoke test

Status: **FAIL -- EXTERNAL_BOUNDARY_METHOD_REJECTED**.

This final permitted external-boundary test retains the accepted terrain,
S0 geometry, three 64 m inlet cells, B_actual=192 m, material state,
entrainment-off setting, pulse, and V/T case. It supersedes only the prior
total-speed Fr closure with the authorized fixed supercritical-normal
computational inflow closure, Fr_n=1.20.

The local D-Claw characteristic check passes: kappa=1 and bed_normal=0 imply
c=sqrt(g*h); prescribed v_n=-1.20*c produces -2.20*c and -0.20*c gravity
characteristics, both inward. The non-inlet north boundary is reflective.

The solver reaches t=120 s normally with finite fields and downstream flow.
Maximum depth, speed, P99 wet speed, and chainage are 34.938 m, 20.818 m/s,
20.527 m/s, and 965.622 m, respectively. No 30 m/s artificial cap, terrain
change, source-sign reversal, or numerical instability occurred.

## Required mass-budget test

| t (s) | domain h-volume (m3) | analytic target (m3) | relative discrepancy |
|---:|---:|---:|---:|
| 30 | 1,824.657 | 500,000 | -99.635% |
| 60 | 751,498.397 | 1,500,000 | -49.900% |
| 90 | 856,546.826 | 2,000,000 | -57.173% |

All discrepancies exceed the required 10 percent tolerance. The downstream
front is only 0.649 km at t=90 s, so the missing volume is not explained by
far-boundary exit. The analytical ghost-state preflight is not the numerical
Riemann boundary flux.

Per the hard decision rule, EXTERNAL_BOUNDARY_METHOD is **REJECTED**. No
additional normal Froude number, discharge multiplier, width, terrain,
friction, material, or hydrograph variant was run. The next method is
CONSERVATIVE_FIXED_S0_SOURCE_ZONE.
