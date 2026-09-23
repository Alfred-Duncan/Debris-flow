# EngineeringROI-v1 failure-mechanism audit

Status: PASS. Input artifacts were read only; no model execution occurred.

## Observed facts

| Method | Unique patch fraction | Mean top-10 share | Mean consecutive reselection | Mean longest streak |
|---|---:|---:|---:|---:|
| RandomAll_B10 | 1.0000 | 0.0821 | 0.1047 | 1.9223 |
| RandomSupport_B10 | 0.7269 | 0.1099 | 0.1550 | 2.3263 |
| EngineeringROI_B05 | 0.4747 | 0.5120 | 0.5709 | 6.7448 |
| EngineeringROI_B10 | 0.6247 | 0.3294 | 0.6054 | 8.8140 |
| EngineeringROI_B20 | 0.7005 | 0.2269 | 0.7169 | 16.9786 |
| DynamicOnly_B10 | 0.5772 | 0.4493 | 0.8806 | 21.2724 |
| SupportRiskOnly_B10 | 0.5986 | 0.2936 | 0.4025 | 4.8971 |
| EngineeringROI_NoDiversity_B10 | 0.5648 | 0.3860 | 0.7059 | 13.3607 |

## Interpretation boundary

Pooled correlations combine between-method and within-method variation, so large pooled Spearman values are not within-method causal evidence. The correlations in the CSV/JSON outputs are descriptive associations, not causal proof. Raw local correction amplitude is **NOT DIRECTLY OBSERVABLE FROM EXISTING V1 ARTIFACTS**; therefore amplitude accumulation cannot be directly proven by retained V1 telemetry alone.
