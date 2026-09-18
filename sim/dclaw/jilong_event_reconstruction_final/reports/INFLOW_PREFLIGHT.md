# Inflow preflight

| Case | target V (m3) | integrated V (m3) | volume error | Qpeak (m3/s) | flux error | hpeak (m) | Un peak (m/s) | U total peak (m/s) | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C1 | 1000000.000 | 1000000.000 | -2.056e-11 | 17453.293 | -2.084e-16 | 8.363 | 10.869 | 14.023 | True |
| C2 | 1000000.000 | 1000000.000 | -2.056e-11 | 8726.646 | 0.000e+00 | 5.268 | 8.627 | 11.130 | True |
| C3 | 2000000.000 | 2000000.000 | -2.056e-11 | 34906.585 | -2.084e-16 | 13.276 | 13.694 | 17.668 | True |
| C4 | 2000000.000 | 2000000.000 | -2.056e-11 | 17453.293 | -2.084e-16 | 8.363 | 10.869 | 14.023 | True |
| C5 | 3000000.000 | 3000000.000 | -2.056e-11 | 52359.878 | 0.000e+00 | 17.396 | 15.676 | 20.225 | True |
| C6 | 3000000.000 | 3000000.000 | -2.056e-11 | 26179.939 | -2.779e-16 | 10.959 | 12.442 | 16.053 | True |

All rows use the actual N, B_actual, W_eff and alpha recorded in inlet_geometry.json.
The normal-flux integration is numerical (200001 uniformly spaced samples).
