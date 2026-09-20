# Park version provenance audit

`external/langtang-2026-cascade` is read-only at `6210c663a6e4527793f92c34fea72ef711e38d52`; tags are `v1.0.0`, `v1.0.1`, and `v1.0.2`.  `git log --follow -- swe/solver_sparse.py` returns only `b5efb873006a3714b95cbb9c0b784c404df01f3f` (2026-09-07, “Reconstruction code…”).  Thus the public sparse solver has no separately committed v2 revision.

| Quantity | v1 semantics | v2 semantics | Exact evidence | Public-code reproducible? | Confidence |
| --- | --- | --- | --- | --- | --- |
| Model code | original sparse solver | revised source not distributed | Git history; Zenodo v2 README “existing code archive … not updated” | No author-code identity | High |
| Source velocity | smoke used 150 m/s | source enters at rest | Zenodo v2 README, “Changes from version 1”; `upper_visual_5s/result.json: args.release_speed=0` | Yes, documented reimplementation | High |
| Melting | historical implementation | mass conserving | Zenodo v2 README; manuscript v2 Methods, melting equations | Yes, documented reimplementation | High |
| Upper result | historical archive | `upper_visual_5s`, 30 m, 1440 s, 5 s states | README v2 “Upper gorge”; support `runs/upper_visual_5s/*` | Scalar series only | High |
| Corridor result | historical archive | corrected 60 m, 8 h, 60 s output | README v2 “Full corridor” | Scalar series/profile only | High |
