# EngineeringROI-v2 formal 20-case VAL

Result audit: PASS.

## Method summaries

| Method | trajectory h | volume error | front MAE km | arrival MAE s |
|---|---:|---:|---:|---:|
| FrozenGlobal | 1.008697 | 0.246521 | 11.912151 | 501.833 |
| RandomAll_B10 | 0.947769 | 0.187710 | 12.877748 | 482.500 |
| RandomSupport_B10 | 1.125538 | 0.268198 | 12.897498 | 474.667 |
| EngineeringROI_B10 | 8.396898 | 0.676275 | 11.826948 | 400.000 |
| SupportRiskOnly_B10 | 1.177500 | 0.189864 | 11.204066 | 479.000 |
| SupportRisk_Base_B10 | 1.177500 | 0.189864 | 11.204066 | 479.000 |
| SupportRisk_Temporal_B10 | 1.085683 | 0.143238 | 11.289821 | 484.500 |
| SupportRisk_DepthGuard_B10 | 1.006168 | 0.320205 | 11.450272 | 475.667 |
| EngineeringROI_v2_B05 | 1.009012 | 0.298087 | 11.510070 | 475.833 |
| EngineeringROI_v2_B10 | 1.010871 | 0.292492 | 11.443973 | 481.833 |
| EngineeringROI_v2_B20 | 1.013159 | 0.276628 | 12.933848 | 493.667 |

Findings are descriptive comparisons of the completed frozen VAL protocol; no training, TEST, H0, or holdout execution occurred.
