# Reference-based Jilong D-Claw status

1. **Primary application:** USGS Mount Baker long-runout lahar scenarios (Gardner et al., 2025; SIR 2024-5133).
2. **Adopted components:** a projected DEM workflow, stationary D-Claw qinit mass, `m0=0.62`, `mcrit=0.64`, `rho_f=1100 kg/m3`, `rho_s=2700 kg/m3`, `mu=0.005 Pa s`, `phi=38°`, and the published permeability range.
3. **Old mismatch confirmed?** Yes: the previously observed ~3158 versus ~2809 m discrepancy at the entry was consistent with a terrain-coordinate/row-order problem. This case does not reuse that terrain product.
4. **Correction:** 8-m observed DEM where valid plus 12.5-m fallback are reprojected with physical north-to-south rows, reordered exactly once into GeoClaw increasing-y storage, written to TT3, then read back and checked.
5. **Baseline:** 1 Mm3 stationary, no-velocity qinit source; `kref=1e-11 m2`; hydrostatic pressure; no invented entrainment raster/rate.
6. **Baseline maximum propagation:** 0.420 km.
7. **Downstream proxy reached:** no.
8. **Baseline proxy arrival:** unavailable (not reached).
9. **Bulk diagnostics (h>0.1 m):** baseline maximum 6.65 m/s and P99 1.61 m/s.
10. **Scenario family generated:** no. The baseline stalled within 2 km, so the prescribed failure branch limited execution to two targeted corrections rather than a parameter sweep.
11. **Completed-run range:** front 0.420--0.420 km; no downstream-proxy arrival; proxy depth 0.000 m in every run.
12. **Structural comparison:** terrain is not the limiting difference (TT3 check PASS: median |Δz| 2.625 m, P95 8.724 m, maximum 19.386 m; y-mirror control 399.406 m). The baseline already matches the primary reference's `m0/mcrit` relation and uses its central `kref`; `kref=1e-12 m2` and `1.5×` hydrostatic pinit were the two permitted documented corrections and did not increase the front beyond the initial 0.420-km footprint.
13. **Single remaining blocker:** the Jilong main-river-entry source condition is structurally unobserved. In contrast with Mount Baker's mapped high-elevation landslide geometry, the stationary 1-Mm3 downstream qinit footprint is only a transparent placeholder; its location/geometry/saturation cannot yet initiate a mobile corridor flow. This must be resolved with defensible entry-source geometry or boundary evidence before a broader scenario matrix.
14. **Sufficient for engineering scenario generation:** **NO**. Recommended next step: one source-condition formulation correction, not calibration, FNO, RL, or dataset generation.

Planned batch wall time: 140.1 s; solver runs: 3; all three finished with finite fields.
