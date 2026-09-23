# Paper results summary

Final method: **RiskTemporal-v1 B10**. The global operator and final method implementation are frozen. The completed evidence package contains 20-case untouched TEST comparisons against FrozenGlobal, RandomAll B10, and SupportRisk Base B10; paired statistics; exploratory mechanism and stratified analysis; H0 documented engineering results; and a rejected v2A soft-support ablation.

v2B/v2C were not developed. Further algorithm experiment required: **NO**.

Key TEST summary:

| method               |   trajectory_h_rel_l2 |   trajectory_momentum_rel_l2 |   change_region_h_rel_l2 |   change_region_momentum_rel_l2 |   false_positive_wet_fraction |   final_wet_iou |   mixture_volume_relative_error |   debris_front_mae_km |   arrival_MAE_s |   case_wall_runtime_seconds |
|:---------------------|----------------------:|-----------------------------:|-------------------------:|--------------------------------:|------------------------------:|----------------:|--------------------------------:|----------------------:|----------------:|----------------------------:|
| FrozenGlobal         |                1.0153 |                       4.3988 |                   0.7942 |                          0.5713 |                        0.5917 |          0.2224 |                          0.1874 |               15.4719 |         552     |                     13.1602 |
| RandomAll_B10        |                1.0035 |                       2.9227 |                   0.7594 |                          0.5328 |                        0.5445 |          0.2983 |                          0.2162 |               15.5708 |         567.5   |                     26.2651 |
| SupportRisk_Base_B10 |                1.2286 |                       3.9095 |                   0.7362 |                          0.5771 |                        0.5659 |          0.2575 |                          0.3864 |               14.6489 |         529.333 |                     25.9491 |
| RiskTemporal_B10     |                1.1328 |                       3.9136 |                   0.7258 |                          0.5434 |                        0.5621 |          0.2697 |                          0.2264 |               14.7645 |         540.167 |                     23.6522 |
