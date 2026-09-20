# Global Operator V2 — code-only engineering status

This update deliberately does **not** launch formal V2 training, generate the
20 final-holdout physics cases, evaluate H0, or modify any V1 artifact.

## Implemented

- V2 state `[h, hu, hv, c, ice, dz]`, two-frame history, dynamic bed
  `z_initial + dz`, physical projection, and zero-preserving transforms.
- Real static/source/exogenous maps from `upper30h.npz` and its JSON metadata.
  Route chainage and station transects are evaluation-only and are not model
  predictors.
- Train-only transform fitting, physical wet-mask loss, closed-loop K-step
  primitives, checkpoint RNG state, and engineering metric primitives.
- A single guarded launcher: `scripts/run_global_operator_v2_full.py`.

## Verified today

`--dry-run` passes. A 64-update GPU developer smoke in
`models/global_operator_v2_smoke/` and `results/global_operator_v2_smoke/`
passed forward/backward, two-step projection, K=2/4/6 developer probes, and
checkpoint save/load. The RVPI FNO complex spectral kernel does not support
BF16 or FP16 complex einsum on this CUDA build, so AMP is feature-gated to
FP32 rather than silently failing.

## Blocker recorded, not hidden

The existing sparse frame exporter writes `dz` only where `abs(dz) > 0.05 m`
and stores it as float16. In 20 cross-checks, frame-final `dz` therefore does
not exactly equal the full-precision `final_state.bed_change` (maximum missing
change is about 0.05 m). `GLOBAL_V2_STATE_CLOSURE.md/json` records this as a
storage-provenance failure. No old physics output was rewritten or treated as
zero-valued bed change. Formal V2 training must not be claimed closed until a
lossless per-frame `dz` export (or a documented threshold-aware target policy)
is approved.

## Approved-day command

```powershell
python scripts/run_global_operator_v2_full.py --formal
```

The command remains intentionally locked during this code-only task; formal
execution needs explicit approval after the `dz` representation decision.
