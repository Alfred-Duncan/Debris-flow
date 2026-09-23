# Jilong EI local development workspace

This is an isolated Windows workspace for the minimum pipeline:

physics smoke frame -> dense tensor adapter -> untrained global operator -> manual local corrector patch.

It is not a D-Claw workspace and does not claim a validated event reconstruction.

## Frozen paper release

Final method: **RiskTemporal-v1 B10**. B05/B10/B20 are budget sensitivity; SupportRisk/Temporal/DepthGuard are ablations; EngineeringROI-v1 and DynamicOnly are failure studies; Oracle/RandomPerfect are feasibility diagnostics; H0 is a historical numerical-reference application. See `FINAL_METHOD.md`, `FINAL_EXPERIMENTS.md`, `REPRODUCIBILITY.md`, and `paper_results/`.

## Setup

The user-approved Conda base environment is used. Its tested runtime is Python 3.13 with PyTorch 2.9.0+cu130. The supplied environment file records the package family, but this workspace does not recreate base.

Set the local proxy only when a download needs it:

    $env:HTTP_PROXY="http://127.0.0.1:7897"
    $env:HTTPS_PROXY="http://127.0.0.1:7897"

## Run sequence

    conda run -n base python scripts/preflight_corridor60.py --workspace .
    conda run -n base python scripts/check_physics_smoke.py --workspace .
    conda run -n base python scripts/audit_frame_orientation.py --workspace .
    conda run -n base python scripts/build_smoke_ml_dataset.py --workspace .
    conda run -n base python scripts/run_local_pipeline_demo.py --workspace .
    conda run -n base python scripts/finalize_reports.py

The pre-existing run in outputs/physics_smoke60 is a 120-s ORIGINAL-CODE SOFTWARE SMOKE ONLY. Do not interpret it as a physical event reference.

All paths are relative/pathlib-based. Read-only upstreams and large data/results are ignored by git.

