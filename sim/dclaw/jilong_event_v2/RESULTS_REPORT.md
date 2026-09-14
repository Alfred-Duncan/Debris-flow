# Phase 2C-fix: six-case short-pulse batch

## Corrected mechanics and diagnostic geometry

Entrainment-on candidates use the installed D-Claw old-style mechanism (`entrainment_method=0`), with the fixed provisional 2 m entrainable layer, rate 0.2 and `me=0.65`. The official port landmark is (340837.9, 3129051.7) m EPSG:32645. The hydraulic port-proxy is the downstream mapped fourth-order-river terminus at (340571.826, 3129640.776) m, 646.379 m away. The fixed model-entry-to-proxy mapped route is 17.624 km. The proxy is a diagnostic channel section, not the surveyed official port cross section.

All six runs completed through 600 s with finite fields. A front is the farthest h>0.01 m cell within a fixed 320 m corridor around the mapped river. The accumulated entrained-volume state is **unavailable**: no q component was assumed to be it without a verified installed-state definition.

| Run | Qpeak (m3/s) | T (s) | Entr. | Max/final front (km) | Max speed (m/s) | Proxy reached? |
|---|---:|---:|---|---:|---:|---|
| A | 30000 | 60 | off | 1.119 / 1.037 | 48.22 | no |
| B | 30000 | 60 | on, method 0 | 1.952 / 1.930 | 117.95 | no |
| C | 20000 | 90 | on, method 0 | 0.070 / 0.070 | 0.00 | no |
| D | 30000 | 90 | on, method 0 | 1.970 / 1.970 | 128.63 | no |
| E | 20000 | 120 | on, method 0 | 0.070 / 0.070 | 0.00 | no |
| F | 30000 | 120 | on, method 0 | 1.970 / 1.970 | 137.17 | no |

## Control comparison and interpretation

Run B minus Run A increases maximum front progress by **832.68 m** and final front progress by **892.91 m**. Neither reaches the proxy, so arrival-time change is unavailable. The method-0 entrainment mechanism therefore materially changes the early downstream progression under this provisional setup, unlike the previous method-1 test.

Runs D and F tie for maximum front distance (1.970 km); **Run D** is selected as the best front-progress candidate because it attains that distance with less imposed volume and lower maximum speed. It leaves 15.654 km to the hydraulic proxy. The Qpeak=30,000 entrainment-on cases repeatedly stall near 1.93–1.97 km by 600 s; Qpeak=20,000 cases remain near 0.070 km. No candidate reaches the proxy, so proxy arrival time, peak discharge, depth and speed are all zero/unavailable at the section.

Source-only adjustment remains inadequate. Simply increasing Q is not the justified next move because it raises domain maximum speed to 118–137 m/s while leaving more than 15.6 km unresolved. The next physical investigation should prioritize terrain/channel representation, then material and entrainment parameterization constrained by evidence, rather than further source escalation.
