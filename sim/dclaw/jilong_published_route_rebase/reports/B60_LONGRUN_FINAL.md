# B60 long-duration continuation

{
  "continuation_method": "EXACT_B60_RERUN_FROM_ZERO",
  "physics_identical_to_RECON_B_rate060": "YES",
  "B60_900_reproduction": "PASS",
  "reproduction_detail": {
    "route_difference_m": 0.0,
    "maxdepth_relative_difference": 0.0,
    "R1_R2_R3_arrivals": {
      "R0": 30.0,
      "R1": 150.0,
      "R2": 270.0,
      "R3": 390.0,
      "R4": 1200.0
    },
    "pass": true
  },
  "R1_h010_arrival_s": 150.0,
  "R2_h010_arrival_s": 270.0,
  "R3_h010_arrival_s": 390.0,
  "R4_h0001_arrival_s": 1200.0,
  "R4_h005_arrival_s": 1200.0,
  "R4_h010_arrival_s": 1200.0,
  "R4_h020_arrival_s": 1200.0,
  "R4_h050_arrival_s": 1200.0,
  "R4_h100_arrival_s": 1230.0,
  "R4_reached_h010": "YES",
  "post_R4_extension_length_available_m": 2328.6256838944155,
  "post_R4_h010_reach_at_3600_m": 2328.6256838944155,
  "total_diagnostic_reach_at_3600_m": 22036.09159391796,
  "furthest_wet_XY_at_3600": [
    340238.0,
    3127228.0
  ],
  "furthest_distance_from_R4_at_3600_m": 2134.1299366775347,
  "domain_boundary_touched": "YES",
  "first_domain_boundary_touch_s": 1590.0,
  "boundary_detail": {
    "first_boundary_touch_s": 1590.0,
    "boundary_side": "south (y = 3127228 m center row; open extrapolation boundary)",
    "x_range": [
      340302.0,
      340302.0
    ],
    "y_range": [
      3127228.0,
      3127228.0
    ],
    "depth_median": 0.2548756216287641,
    "depth_max": 0.2548756216287641,
    "speed_median": 46.376144244650234,
    "speed_max": 46.376144244650234
  },
  "approximately_stationary_at_3600": "YES",
  "advance_last_300s_m": 0.0,
  "advance_last_600s_m": 0.0,
  "max_depth_m": 97.23032059627937,
  "max_speed_ms": 175.21791478716975,
  "P99_speed_ms": 51.688584978745425,
  "max_CFL": 0.45623,
  "numerical_warning": "YES",
  "final_classification": "DOMAIN_LIMITED_AFTER_R4",
  "recommended_next_step": "Extend/rebuild the computational domain before interpreting terminal runout.",
  "numerical_safety": {
    "frames_expected": 121,
    "frames_actual": 121,
    "finite_q": true,
    "NaN_count": 0,
    "Inf_count": 0,
    "solver_exit_code": 0,
    "max_CFL": 0.45623,
    "global_max_speed_ms": 175.21791478716975,
    "global_max_speed_time_s": 1710.0,
    "global_max_speed_x": 340302.0,
    "global_max_speed_y": 3127228.0,
    "global_max_speed_depth_m": 1.44349291066235,
    "P99_speed_ms": 51.688584978745425,
    "max_depth_m": 97.23032059627937,
    "HIGH_SPEED_WARNING": true,
    "HIGH_DEPTH_WARNING": false
  },
  "R4_arrival_state_file": "results/B60_R4_ARRIVAL_STATE.csv"
}
