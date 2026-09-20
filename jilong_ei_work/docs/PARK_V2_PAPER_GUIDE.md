# Park v2 engineering paper guide

## Citation identity

Park, H. (2026), revised EarthArXiv manuscript, DOI `10.31223/X5250R`; supporting revised animation record: Zenodo `22566624` (v2.0.0, 7 September 2026).  Use the v2 label explicitly; older archived animations are not evidence for the revised physics.

## What the paper/support directly supplies

| Engineering item | v2 value / statement | Use in this workspace |
| --- | --- | --- |
| Release volume | 35.42 Mm3 | H0 source total. |
| Initial composition | 20% ice and 80% rock; no initial free water | H0 source state. |
| Source kinematics | inserted at rest in 30 s | Replaces all 150 m/s smoke usage. |
| Melt law | density-aware, mass-conserving ice-to-water conversion | Implemented in `src/park_v2`. |
| Upper H0 grid | 30 m, first 1440 s, 5 s frames in released visual run | H0 comparison domain. |
| Observational timing evidence | Gyirong CCTV 463 ± 90 s; Rasu 170–770 s; Syab 770–1370 s | Timing evaluation bands, not calibration targets. |
| Model arrival criterion | first sampled stage rise >0.5 m or Q > 2× pre-event + 50; Gyirong uses Q | Scorecard arrival calculation. |
| Released H0 arrivals | 365/390/1025 s (5 s support); 390/390/1050 s (manuscript 30 s reference) | Numerical parity reference. |

## What remains unavailable

The v2 README explicitly says that the existing code archive was not updated, and the support archive is not a standalone rerun package.  It lacks the revised author solver, stored 2-D frame fields, and original spinup state.  It therefore cannot support a claim of bitwise author-code identity, detailed v2 field error maps, or re-tuning.

## Citation/readers’ guide

The primary engineering evidence is the revised manuscript for equations and assumptions, the Zenodo v2 README for version provenance, and the released `upper_visual_5s` result/series for numerical parity.  The GitHub sparse solver is an implementation ancestor only.  The report intentionally keeps observed timings, manuscript model outputs and the reimplementation outputs in separate columns.
