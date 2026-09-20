# External model version note

| Quantity | ORIGINAL CODE RELEASE | REVISED MODEL / REVISED ANIMATIONS | Reproducible locally now | Benchmark use |
|---|---|---|---|---|
| Release state | Historical production command releases rock-ice over 30 s. | README states revised animations add source material at rest. | Original command only. | Software/data-interface smoke only. |
| Release speed | Historical command uses release-speed 150; source code initializes downslope momentum from that speed. | Revised note describes material added at rest. | 150 m/s path is reproducible; at-rest implementation is not supplied in current code. | Do not mix semantics. |
| Melting | Historical code supports heat/melt terms. | Revised archive says melting is mass-conserving. | Original code release. | No event-validation claim in this workspace. |

The 120-s run validates software, CUDA, output frames, and tensor interfaces. It is not a final event reconstruction or a validated reference simulation.
