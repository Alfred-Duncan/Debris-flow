# Global Operator V2 state-closure audit

**Status:** PASS_THRESHOLD_AWARE

Retained ML state: `h, hu, hv, c, ice, thresholded_dz`. `hc=h*c`, `hi=h*ice`, and `current_bed=z_initial+dz_retained` are reconstructible. `hq` is a heat/thermal-energy storage state; `erodible` is the separate remaining-erodible-depth field. Neither is retained, therefore this is not an exact Markov-complete state.

## dz representation contract

`FULL_PHYSICAL_BED_CHANGE` is float32 `final_state.bed_change`. `RETAINED_ML_BED_CHANGE` stores only `abs(dz)>0.05 m` cells as float16. Omitted below-threshold cells are expected; retained cells use rtol=2e-3, atol=0.002 m.

| Cases | threshold (m) | Result |
|---:|---:|---|
| 20 | 0.05 | PASS_THRESHOLD_AWARE |
