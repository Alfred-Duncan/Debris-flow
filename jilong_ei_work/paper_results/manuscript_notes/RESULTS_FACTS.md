# Results facts

## Robust improvements vs FrozenGlobal
Change-region h: 20/20; change-region momentum: 16/20; false-positive wet: 20/20; front MAE: 19/20; trajectory momentum: 15/20.

## Wet-support distinction
Mean wet IoU: 0.3705 -> 0.3589 (trajectory-average decline). Final wet IoU: 0.2224 -> 0.2697 (final-state improvement).

## Trade-offs
Trajectory h: 1.0153 -> 1.1328; volume error: 0.1874 -> 0.2264. RandomAll has strong global averages (1.0035 trajectory h) while RiskTemporal is stronger on change-region h (0.7258 vs 0.7594) and front MAE (14.764 vs 15.571). SupportRisk Base has slightly lower front MAE (14.649 vs 14.764); Temporal Refresh substantially reduces its volume drift (0.3864 -> 0.2264).

B10 is a selected engineering operating point, not a mathematical optimum.
