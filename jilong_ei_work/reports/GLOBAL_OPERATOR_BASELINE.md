# Global Neural Operator baseline

```json
{
  "dataset": {
    "train": 160,
    "validation": 20,
    "test": 20,
    "h0_excluded": true
  },
  "architecture": {
    "width": 32,
    "modes": 24,
    "depth": 3,
    "parameters": 3543877,
    "peak_vram_mib": 1331.3896484375,
    "step_s": 0.12047423500043805
  },
  "training_steps": 12000,
  "best_validation_metric": "NOT_AVAILABLE: checkpoint selection was absent in first launcher",
  "test_mean": {
    "time_s": 660.0,
    "h_rmse": 3.3868291633469716,
    "h_mae": 0.5330871309552874,
    "h_rell2": 1.1045702985354833,
    "hu_rmse": 63.05526578085763,
    "hu_mae": 9.68440715755735,
    "hu_rell2": 2.4937929149184908,
    "hv_rmse": 71.48353817803519,
    "hv_mae": 8.98054416860853,
    "hv_rell2": 1.4906183415225573,
    "c_rmse": 0.16993178421897545,
    "c_mae": 0.05980384236733823,
    "c_rell2": 1.1526054986885617,
    "ice_rmse": 0.03065303522827365,
    "ice_mae": 0.008984111839838804,
    "ice_rell2": 1.2695795235889298,
    "wet_area_error_cells": 351839.02857142856,
    "wet_iou": 0.026156268719166664
  },
  "h0_final": {
    "time_s": 1440.0,
    "h_rmse": 4.574028968811035,
    "h_rell2": 4.108689785003662,
    "wet_iou": 0.0087047177225137
  },
  "runtime_s": {
    "median": 15.54028690000996,
    "mean": 15.603369585005566,
    "p95": 15.976492155008602
  },
  "engineering_metrics": "NOT_CURRENTLY_FIELD_DERIVABLE from retained state schema; no solver station-flux state is available",
  "front_metrics": "NOT_CURRENTLY_FIELD_DERIVABLE with retained state schema",
  "result_status": "FROZEN_CANDIDATE_EVALUATED_NOT_PUBLICATION_READY: long-rollout wet footprint and field errors are high"
}
```

This is a descriptive frozen-checkpoint evaluation. It must not be presented as an accuracy-successful surrogate or as observational truth.
