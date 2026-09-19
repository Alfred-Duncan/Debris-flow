# C3 conservative-source smoke test

The accepted C3 smoke is `runs/C3_conservative_debug3/_output` (0–120 s, output every 30 s). It used the frozen three-cell source zone, standard northern wall, entrainment off, and no artificial speed cap. All fields were finite and the maximum reported Courant number was 0.41516.

| time (s) | model domain volume (m3) | analytic F(t) (m3) | relative error |
|---:|---:|---:|---:|
| 30 | 500212.944 | 500000 | +0.042589% |
| 60 | 1500097.078 | 1500000 | +0.006472% |
| 90 | 2000003.334 | 2000000 | +0.000167% |
| 120 | 1999935.780 | 2000000 | -0.003211% |

The required 30/60/90 s discrepancies are each below 1%, so `CONSERVATIVE_SOURCE_GATE = PASS`.

Three implementation-debug executions followed the initial no-injection smoke: the first exposed an unintentionally stale 900 s data file; the second exposed the post-hyperbolic source/CFL start-up step; the third is the accepted run after the finite `dt_max=0.5 s` numerical scheduling guard. The guard does not change source volume, geometry, momentum closure, terrain, material, or physical speed cap.
