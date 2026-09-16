# Moving-entry source results

## Native qinit verification

The installed q mapping is `h, hu, hv, hm, pb, hchi, Delta b` = q1--q7. qinit files for q2/q3/q4 supply `u/v/m` and the native implementation multiplies by h. Hydrostatic q5 is applied by unchanged `init_ptype=0`. All t=0 native-qinit gates passed: M1 8.00/8 m/s, M2 15.00/15 m/s, M3 25.00/25 m/s, M4 8.00/8 m/s, M5 15.00/15 m/s, M6 25.00/25 m/s.

## Six engineering entry-state scenarios

| Case | Volume (m3) | Prescribed speed (m/s) | max / final front (km) | proxy arrival (s) | max h>.1 / h>1 / p99 speed (m/s) |
|---|---:|---:|---:|---:|---:|
| M1 | 1000000 | 8 | 1.672 / 1.672 | None | 30.22 / 30.22 / 20.45 |
| M2 | 1000000 | 15 | 1.672 / 1.672 | None | 32.20 / 32.20 / 22.67 |
| M3 | 1000000 | 25 | 2.324 / 2.324 | None | 30.55 / 30.55 / 25.00 |
| M4 | 3000000 | 8 | 2.572 / 2.572 | None | 31.67 / 31.67 / 22.93 |
| M5 | 3000000 | 15 | 2.572 / 2.572 | None | 31.53 / 31.53 / 24.68 |
| M6 | 3000000 | 25 | 2.612 / 2.612 | None | 32.38 / 32.38 / 26.07 |

Best propagation: **2.612 km** (M6). Port proxy reached: **False**. Arrival times: none. Entry-state outcomes meaningfully differ: **True**.

**SIMULATOR_UNBLOCKED = NO**.

Total wall time (including native-qinit gate): 407.8 s.

## Required <3-km diagnostic

All six moving-entry cases remained below 3 km. The native t=0 gates passed: M1=8.00 m/s, M2=15.00 m/s, M3=25.00 m/s, M4=8.00 m/s, M5=15.00 m/s, M6=25.00 m/s.

| Case | final s-distance (km) | terrain 128 m before / at / after final front (m) |
|---|---:|---:|
| M1 | 1.672 | 2763.9 / 2757.5 / 2748.2 |
| M2 | 1.672 | 2763.9 / 2757.5 / 2748.2 |
| M3 | 2.324 | 2708.2 / 2696.3 / 2670.1 |
| M4 | 2.572 | 2672.6 / 2672.6 / 2686.0 |
| M5 | 2.572 | 2672.6 / 2672.6 / 2686.0 |
| M6 | 2.612 | 2659.1 / 2684.2 / 2678.9 |

The common stop location despite verified 8--25 m/s initial momentum indicates a geometric/corridor-source structural limitation, not missing entry momentum.
