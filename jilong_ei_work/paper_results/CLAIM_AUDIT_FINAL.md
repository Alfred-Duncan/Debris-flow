# Final claim audit

## Supported

- RiskTemporal-v1 B10 improves change-region h in 20/20 TEST cases and reduces false-positive wet predictions in 20/20 cases versus FrozenGlobal.
- It improves front MAE in 19/20 cases, improves trajectory momentum on average, and improves final wet IoU.
- Mean wet IoU declines over the rollout.
- Temporal Refresh mitigates several SupportRisk-only closed-loop errors; B20 shows more refinement is not uniformly better.
- H0 provides field and sampled formal-front comparisons against a documented numerical reference.

## Not supported

- All metrics improve; universal superiority over RandomAll; global h improvement; mean wet-IoU improvement; consistent volume improvement; accurate operational station arrival; observational or real-world forecasting validation; or controlled D-Claw speedup.
