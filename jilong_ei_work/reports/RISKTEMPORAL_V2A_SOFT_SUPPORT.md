# RiskTemporal-v2A Soft Support

**REJECT — do not proceed to v2B.**

Configuration: one-cell (3×3) front-connected dilation and `alpha_front=0.50`; Zone A uses full local correction, Zone B scales only local delta, Zone C blocks it. The selector, temporal refresh, frozen checkpoints, momentum guard, and physical projection were unchanged.

VAL and engineering stress-test both completed 20/20 cases. Blocking declined, but FP-wet and volume metrics worsened materially; the v2A acceptance gate therefore fails. Full aggregate, telemetry, and per-case comparisons are in `results/risk_temporal_v2/v2A_soft_support_r1a50/`.
