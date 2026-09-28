# K=1 inherited-momentum implementation regression

```json
{
  "status": "PASS",
  "dynamic_regression": [
    {
      "time_s": 90.0,
      "front_m": 1888.3721406971122,
      "front_ref_m": 1888.3721406971122,
      "front_abs_diff_m": 0.0,
      "max_depth_m": 15.88540438359869,
      "depth_ref_m": 15.88540438359869,
      "depth_rel_diff": 0.0,
      "max_speed_ms": 35.253090840634975,
      "speed_ref_ms": 35.253090840634975,
      "speed_rel_diff": 0.0,
      "pass": true
    },
    {
      "time_s": 180.0,
      "front_m": 3907.457947973987,
      "front_ref_m": 3907.457947973987,
      "front_abs_diff_m": 0.0,
      "max_depth_m": 17.15025204680207,
      "depth_ref_m": 17.15025204680207,
      "depth_rel_diff": 0.0,
      "max_speed_ms": 36.35365128057303,
      "speed_ref_ms": 36.35365128057303,
      "speed_rel_diff": 0.0,
      "pass": true
    }
  ],
  "mass_regression": [
    {
      "time_s": 30.0,
      "expected_source_volume_m3": 500000.0,
      "integrated_source_volume_m3": 499999.9999999999,
      "relative_error": 2.328306436538696e-16,
      "pass": true
    },
    {
      "time_s": 60.0,
      "expected_source_volume_m3": 1500000.0,
      "integrated_source_volume_m3": 1499999.9999999998,
      "relative_error": 1.5522042910257976e-16,
      "pass": true
    },
    {
      "time_s": 90.0,
      "expected_source_volume_m3": 2000000.0,
      "integrated_source_volume_m3": 2000000.0,
      "relative_error": 0.0,
      "pass": true
    }
  ],
  "thresholds": {
    "front_m": 80,
    "depth_relative": 0.05,
    "speed_relative": 0.05,
    "mass_relative": 0.01
  },
  "run": "E4_momentum_k1_regression",
  "frames": 7,
  "finite_fields": true
}
```
