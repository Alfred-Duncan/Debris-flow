# Park 2026 revision lock: v2 historical physics

## Decision

`PARK_V2_H0` is a **PARK_V2_DOCUMENTED_REIMPLEMENTATION_REPRODUCED** historical reference.  It is not the unmodified author solver, and it must not be represented as such.  It reproduces the released v2 five-second upper-domain numerical summary using the archived v1 sparse solver as a read-only reference plus the v2-documented physics corrections.

## Version evidence

| Item | Locked evidence | Interpretation |
| --- | --- | --- |
| Upstream repository | `external/langtang-2026-cascade` at `6210c663a6e4527793f92c34fea72ef711e38d52` | Read-only source; `swe/solver_sparse.py` history begins at `b5efb873006a3714b95cbb9c0b784c404df01f3f` (2026-09-07). |
| v1 code archive | upstream `swe/solver_sparse.py` | Does not itself constitute v2 physics. |
| v2 public release | Zenodo record `22566624`, `README_v2.md`, 2026-09-07 | Replaces prior animations; explicitly says source enters at rest and melting conserves mass.  It does **not** release a revised rerun source package. |
| v2 manuscript | Park, EarthArXiv v2, DOI `10.31223/X5250R` | Defines the mixture density and mass-conserving melt update used below. |
| v2 support result | `upper_visual_5s/result.json` + `series.csv` | Reference numerical output for 30 m / 5 s / 1440 s. |

The public v2 input hashes match the local copies: `upper30h.npz` `4e0148…b2c4` and `upper30h.json` `6daa55…4d58`.  The upstream tree was not modified.

## Non-negotiable v2 semantics

| Topic | v1-style / invalid substitute | Locked v2 H0 treatment |
| --- | --- | --- |
| Initial source velocity | 150 m/s impact/smoke release | `release_speed = 0`; source added over 30 s at rest. |
| Ice melting | a volume-loss shortcut | Conserve mass: for melted ice thickness `δ`, `hi' = hi − δ`, `hc' = hc − δ`, `h' = h − 0.1δ`; discharge is scaled by `h'/h` so velocity is preserved. |
| Mixture density | water-plus-solid approximation | `rho_m = 1000 + 1650*c − 1750*i` kg m-3. |
| Source | arbitrary volume/water | 35.42 Mm3 over the archived UNOSAT source mask; 20% ice, 80% rock, no initial free water. |
| River initial condition | blank or an undocumented restart | 4 h pre-event spinup, then restart. |

## Released H0 constraints

The v2 `upper_visual_5s` case is the 30 m upper-domain reference for the first 1440 s, with 289 stored states.  Its release arguments are mirrored in `configs/PARK_V2_H0.json`: 30 s source duration, `c0=1`, ice fraction 0.2, `K=0.0074`, erosion threshold 6 m/s, `n_w=0.0208`, `n_d=0.0145`, `mu_s=0`, settling threshold 1.833 m/s and timescale 337 s.

Published v2 support output reports Gyirong / Rasu / Syab arrivals of 365 / 390 / 1025 s for the fine-output run.  The manuscript’s 30 s-output reference reports 390 / 390 / 1050 s; Zenodo explains that forcing 5 s output slightly changes the adaptive time-step path.  The two are therefore retained as distinct reference records, not averaged.

## Provenance boundaries

`src/park_v2/solver_sparse_v2.py` is isolated from the upstream tree.  It began as an exact copy of the v1 sparse solver and contains only the v2 density and melt-state updates documented above; its final SHA-256 is recorded in `reports/PARK_V2_REIMPLEMENTATION_STATUS.md`.  Inputs, frames, downloaded PDFs/zips, and raw restart output remain ignored.  Only compact derived reports, CSVs and PNGs are versioned.

## H0 label rule

Use label **B / PARK_V2_DOCUMENTED_REIMPLEMENTATION_REPRODUCED** only because the recreated 30 m H0 matches the released 5 s numerical aggregate and station series closely (see the scorecard).  It is still not an author-code reproduction: the v2 rerun source and original restart file are not publicly distributed.  Any future code change requires a new provenance and parity check.
