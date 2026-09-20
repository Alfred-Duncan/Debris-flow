# Global Operator V2 state-closure audit

**Status:** FAIL

The V2 predictor retains `h, hu, hv, c, ice, dz`. It reconstructs `hc=h*c`, `hi=h*ice`, and `z_current=z_initial+dz`. `hq` (remaining erodible/material state) is not retained, so this is not claimed as an exact Markov closure.

## dz cross-check

| Cases | tolerance (m) | maximum observed (m) | Result |
|---:|---:|---:|---|
| 20 | 2.0e-06 | 4.993e-02 | FAIL |
