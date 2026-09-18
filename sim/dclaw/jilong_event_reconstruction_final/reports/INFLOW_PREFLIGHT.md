# Inflow preflight

| Case | target V (m3) | integrated V (m3) | volume error | Qpeak (m3/s) | flux error | hpeak (m) | Upeak (m/s) | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| C1 | 1000000.000 | 1000000.000 | -2.056e-11 | 17453.293 | 0.000e+00 | 11.192 | 10.478 | True |
| C2 | 1000000.000 | 1000000.000 | -2.056e-11 | 8726.646 | 0.000e+00 | 7.051 | 8.317 | True |
| C3 | 2000000.000 | 2000000.000 | -2.056e-11 | 34906.585 | 0.000e+00 | 17.767 | 13.202 | True |
| C4 | 2000000.000 | 2000000.000 | -2.056e-11 | 17453.293 | 0.000e+00 | 11.192 | 10.478 | True |
| C5 | 3000000.000 | 3000000.000 | -2.056e-11 | 52359.878 | -1.390e-16 | 23.281 | 15.113 | True |
| C6 | 3000000.000 | 3000000.000 | -2.056e-11 | 26179.939 | -1.390e-16 | 14.666 | 11.995 | True |

All rows use the actual N, B_actual, W_eff and alpha recorded in inlet_geometry.json.
The normal-flux integration is numerical (200001 uniformly spaced samples).
