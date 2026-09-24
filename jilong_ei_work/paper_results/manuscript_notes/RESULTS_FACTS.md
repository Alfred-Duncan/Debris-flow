# Results facts

## Robust improvements vs FrozenGlobal
Change-region h: 20/20; change-region momentum: 16/20; false-positive wet: 20/20; front MAE: 19/20; trajectory momentum: 15/20.

## Wet-support distinction
Mean wet IoU: 0.3705 -> 0.3589 (trajectory-average decline). Final wet IoU: 0.2224 -> 0.2697 (final-state improvement).

## Trade-offs
Trajectory h: 1.0153 -> 1.1328; volume error: 0.1874 -> 0.2264. RandomAll has strong global averages (1.0035 trajectory h) while RiskTemporal is stronger on change-region h (0.7258 vs 0.7594) and front MAE (14.764 vs 15.571). SupportRisk Base has slightly lower front MAE (14.649 vs 14.764); Temporal Refresh substantially reduces its volume drift (0.3864 -> 0.2264).

B10 is a selected engineering operating point, not a mathematical optimum.

## H0 computational latency

The existing H0 rollout covers a 1440-s (24-min) simulated horizon. FrozenGlobal completed it in approximately 15.7 s; RiskTemporal-v1 B10 completed it in approximately 27.1 s. The resulting simulated-horizon/wall-clock ratio for RiskTemporal is approximately 53.2. Local refinement therefore adds overhead relative to FrozenGlobal while remaining far faster than real-time progression of the simulated event. This is not a controlled speedup comparison with a high-fidelity solver.
