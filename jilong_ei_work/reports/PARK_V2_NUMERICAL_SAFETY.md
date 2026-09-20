# PARK_V2_H0 numerical-safety and ledger check

* Completion: PASS — 1440 s reached, 6560 steps, finite final arrays (`h`, `hu`, `hv`, `hc`, `hi`, `hq`).
* Positivity: PASS — the solver applies non-negative depth/component clamps; final diagnostic fields are finite.
* Released v2 mass closure: PASS — Zenodo support reports mass residual `-3651093.817` kg, absolute residual `3734195.040` kg, or `-0.0039%` of final mass.
* Local independent full boundary-flux closure: NOT AVAILABLE — the published v2 result gives its ledger, but its original restart and revised ledger implementation are not released.  This script therefore checks final-state/output parity rather than inventing a local closure term.
* H0 final-volume parity: `-0.2904%` versus released v2.
