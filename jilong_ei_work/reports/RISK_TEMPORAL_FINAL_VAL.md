# RiskTemporal-v1 final VAL sweep

Risk-Aware Temporally Refreshed Local Refinement uses SupportRisk ROI, spatial diversity, one-step temporal refresh, frozen Local@4000, SupportGuard, and the existing train-derived momentum guard. DepthEnvelope is disabled. The sweep is VAL-only (20 fixed scenarios per budget); no training, TEST, H0, or holdout evaluation was run.

## Budget results

| method           |   budget_fraction |   trajectory_h_rel_l2 |   trajectory_momentum_rel_l2 |   change_region_h_rel_l2 |   change_region_momentum_rel_l2 |   mean_wet_iou |   false_positive_wet_fraction |   mixture_volume_relative_error |   debris_front_mae_km |   case_wall_runtime_seconds |
|:-----------------|------------------:|----------------------:|-----------------------------:|-------------------------:|--------------------------------:|---------------:|------------------------------:|--------------------------------:|----------------------:|----------------------------:|
| RiskTemporal_B05 |              0.05 |               1.10248 |                      3.11119 |                 0.759073 |                        0.542082 |       0.358486 |                      0.574196 |                        0.154835 |               11.4772 |                     18.8175 |
| RiskTemporal_B10 |              0.1  |               1.08568 |                      2.10897 |                 0.742471 |                        0.539814 |       0.359157 |                      0.550345 |                        0.143238 |               11.2898 |                     25.483  |
| RiskTemporal_B20 |              0.2  |               1.8969  |                      1.29848 |                 0.732521 |                        0.541145 |       0.394072 |                      0.421107 |                        0.295176 |               13.1577 |                     35.2327 |

## Freeze status

Final code SHA: `146184d1fbbf366c0c9c40066a56753b398b10f4`. Config SHA-256: `13bd00c54e3dbdb5ee2543e2f0a8b222addb6ad2ed34f305784b6eefc6f0834d`. Architecture is frozen; primary budget remains an explicit scientific choice, not an automatic winner. Temporal consecutive overlap and depth-guard activation are both zero for every budget. B10 is exactly reproducible against `SupportRisk_Temporal_B10` under the recorded comparison gate.
