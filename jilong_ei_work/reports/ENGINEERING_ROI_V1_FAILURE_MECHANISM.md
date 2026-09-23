# EngineeringROI-v1 failure-mechanism audit

Status: PASS. Input artifacts were read only; no model execution occurred.

## Observed facts

| Method | Mean top-10 share | Mean consecutive reselection | Mean revisit fraction |
|---|---:|---:|---:|
| RandomAll_B10 | 0.0821 | 0.1039 | 1.0000 |
| RandomSupport_B10 | 0.1099 | 0.1539 | 0.9861 |
| EngineeringROI_B05 | 0.5120 | 0.5669 | 0.8521 |
| EngineeringROI_B10 | 0.3294 | 0.6012 | 0.9170 |
| EngineeringROI_B20 | 0.2269 | 0.7119 | 0.9691 |
| DynamicOnly_B10 | 0.4493 | 0.8745 | 0.8539 |
| SupportRiskOnly_B10 | 0.2936 | 0.3997 | 0.9356 |
| EngineeringROI_NoDiversity_B10 | 0.3860 | 0.7010 | 0.9084 |

## Interpretation boundary

The correlations in the CSV/JSON outputs are descriptive associations, not causal proof. Raw local correction amplitude is **NOT DIRECTLY OBSERVABLE FROM EXISTING V1 ARTIFACTS**; therefore amplitude accumulation cannot be directly proven by retained V1 telemetry alone.
