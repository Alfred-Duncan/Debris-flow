# Transient nonmonotonicity diagnosis

{
  "MOMENTUM_RESPONSE_NONMONOTONIC": "YES",
  "scope": "Transient K=1.5 > K=2 historical-reach differences at 300--600 s; all K share the same 900 s historical reach and fail the sill criterion.",
  "comparisons": [
    {
      "time_s": 300.0,
      "K1p5_historical_reach_m": 7600.225458078729,
      "K2_historical_reach_m": 7380.237484001108,
      "K1p5_minus_K2_reach_m": 219.9879740776214,
      "K1p5_wet_volume_m3": 4038893.7473430233,
      "K2_wet_volume_m3": 3930434.469992292,
      "K1p5_minus_K2_wet_volume_m3": 108459.2773507312
    },
    {
      "time_s": 420.0,
      "K1p5_historical_reach_m": 11237.313119347911,
      "K2_historical_reach_m": 10957.450688263414,
      "K1p5_minus_K2_reach_m": 279.8624310844971,
      "K1p5_wet_volume_m3": 5450296.815727207,
      "K2_wet_volume_m3": 5297591.218008509,
      "K1p5_minus_K2_wet_volume_m3": 152705.59771869797
    },
    {
      "time_s": 540.0,
      "K1p5_historical_reach_m": 12996.563800161372,
      "K2_historical_reach_m": 12896.563800161231,
      "K1p5_minus_K2_reach_m": 100.00000000014006,
      "K1p5_wet_volume_m3": 6415460.635726981,
      "K2_wet_volume_m3": 6273546.719507929,
      "K1p5_minus_K2_wet_volume_m3": 141913.91621905193
    },
    {
      "time_s": 600.0,
      "K1p5_historical_reach_m": 13775.611321880913,
      "K2_historical_reach_m": 13575.61132188121,
      "K1p5_minus_K2_reach_m": 199.9999999997035,
      "K1p5_wet_volume_m3": 6633352.550981545,
      "K2_wet_volume_m3": 6532014.775498778,
      "K1p5_minus_K2_wet_volume_m3": 101337.77548276726
    }
  ],
  "numerical_checks": {
    "all_q_finite": true,
    "all_solver_exits_zero": true,
    "maximum_CFL_below_0p44": true,
    "max_speed_below_80_ms": true
  },
  "interpretation": "The K=2 wet volume is lower during every violating frame, consistent with a different spreading/entrainment/pooling partition rather than a numerical blow-up. This transient response does not alter the robust K4-not-crossing conclusion.",
  "classification_rule": "Use MOMENTUM_ONLY_INSUFFICIENT_UP_TO_K4 because K4 does not satisfy LOCAL_SILL_CROSSED; retain the nonmonotonic flag and do not infer a monotonic calibrated K."
}
