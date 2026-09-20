# Park v2 reimplementation status

**Classification: B / PARK_V2_DOCUMENTED_REIMPLEMENTATION_REPRODUCED**

This is a new, isolated implementation derived from the v1 sparse solver and constrained by v2 public equations/results; it is not the author’s unpublished revised solver.  The final volume relative difference is `-0.2904%` and the largest final station-Q relative difference is `0.6293%` versus the Zenodo v2 5 s result.  Arrival comparison is in `results/PARK_V2_H0_SCORECARD.csv`.

* upstream reference commit: `6210c663a6e4527793f92c34fea72ef711e38d52`
* upstream solver origin commit: `b5efb873006a3714b95cbb9c0b784c404df01f3f`
* local v2 solver SHA-256: `964ee7ad758e12a0e27a3cd86c8de69874650b48800993396c73672d8037b21a`
* v2 inputs: `upper30h.npz` `4e0148c86cb42e702823cf8ae2b2c2255907764530138d3bbf68636ae728b2c4`; `upper30h.json` `6daa55faf7bbfa9d2d298a02e8310d400c0762a95206df643ad1a5fcfb7b4d58`
