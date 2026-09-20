# RTX 5090 migration plan

A 5090 is **not required now**: this Windows RTX 5060 Laptop completed the defined 60 m preflight, 120-s software smoke, frame adapter, and untrained operator forward/backprop test.

Move to a larger GPU only when scientifically declared scenario generation, longer physical runs, or operator training needs exceed the local development envelope.

Transfer only:

- project-owned source, scripts, configs, environment files, and reports;
- exact upstream commit identifiers (or fresh read-only clones);
- preprocessed physics inputs with SHA-256 manifest;
- explicit scenario definitions and non-large result summaries.

Do not transfer Windows absolute paths. Start on the target with the same relative workspace layout, confirm CUDA availability, rerun preflight, then rerun the smoke pipeline before any production workload.

