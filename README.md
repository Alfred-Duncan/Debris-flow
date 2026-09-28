# RiskTemporal

RiskTemporal is a risk-aware temporally refreshed local-refinement framework for rapid first-response forecasting of sudden debris-flow propagation.

## Overview

After a hazardous event has been identified and an initial state is available, a rapid forecast can support early situation assessment before a detailed high-fidelity analysis is available. A Global Operator provides the fast domain-wide forecast. RiskTemporal allocates a limited local-computation budget to propagation-sensitive regions and uses Temporal Refresh to avoid repeatedly correcting the same patch in consecutive steps.

This repository is not an event-detection system, a substitute for observational ground truth, an exact station-arrival prediction system, or a replacement for a high-fidelity physical solver.

## Method

The frozen inference chain is:

`Global Operator → Support-Risk ROI → spatial de-redundancy → one-step Temporal Refresh → Local Corrector → wet-support guard → momentum-state guard → physical projection → autoregressive rollout`

RiskTemporal B10 permits at most 19 of 256 logical patches at each step. The Global Operator and Local Corrector are frozen at inference time; refinement is deterministic and has no ground-truth access.

## Repository structure

```text
.
├── README.md
├── LICENSE
└── jilong_ei_work/
    ├── configs/        formal configuration inputs
    ├── environment/    conda environment specification
    ├── models/         model metadata and external checkpoint locations
    ├── paper_results/  frozen tables, statistics, and publication figures
    ├── results/        formal validation and allocation records
    ├── scripts/        training, inference, evaluation, and paper workflows
    └── src/            operators, correctors, ROI logic, guards, and H0 reference code
```

## Data

The study uses 200 high-fidelity numerical scenarios: 160 TRAIN, 20 VAL, and 20 TEST. Each scenario uses a 10 s step, 144 steps, and a 1,440 s (24 min) horizon.

The TEST split was untouched for the development, selection, and freezing of RiskTemporal B10.

H0 is an independent event-constrained numerical-reference case. It is not observational ground truth.

Large frame data are not distributed in Git. Place acquired scenario metadata and frames below `jilong_ei_work/data/`, including `data/scenario_index.csv` and the configured `data/downloads/` hierarchy. The expected paths are defined by the formal configuration files and loaders.

## Models

The Global Operator is a Fourier Neural Operator. The Local Corrector is a patch-wise residual CNN.

Checkpoint binaries are external runtime artifacts and are intentionally not committed. Place them under `jilong_ei_work/models/` at the locations expected by the formal scripts, including the Global Operator checkpoint, Local Corrector checkpoint, and normalization metadata. H0 frame inputs are expected under `jilong_ei_work/outputs/park_v2_h0/h0_global_eval_reference/frames/` when the H0 workflow is run.

## Reproduction

Run commands from `jilong_ei_work/`. They require the external data and checkpoints described above; the paper-result CSV/JSON files already included in this repository are frozen evidence and need not be recomputed to inspect the reported results.

### Environment

```bash
conda env create -f jilong_ei_work/environment/environment.yml
conda activate base
cd jilong_ei_work
```

### Data preparation

Provide the acquired frame hierarchy and `data/scenario_index.csv` at the paths described in [Data](#data). The loaders validate these inputs before training or evaluation.

### Optional model training

```bash
python scripts/run_global_operator_v2_full.py --formal
python scripts/run_local_corrector_v1_1_full.py --formal
```

### RiskTemporal inference and budget sensitivity

```bash
python scripts/evaluate_engineering_roi_v2.py --method RiskTemporal --budget 0.10 --resume
python scripts/evaluate_engineering_roi_v2.py --method RiskTemporal --budget 0.05 --resume
python scripts/evaluate_engineering_roi_v2.py --method RiskTemporal --budget 0.20 --resume
```

### TEST evaluation and method comparisons

```bash
python scripts/paper/evaluate_untouched_test.py --method frozen_global --resume
python scripts/paper/evaluate_untouched_test.py --method risktemporal_b10 --resume
python scripts/paper/evaluate_frozen_test_baselines.py --resume
python scripts/paper/evaluate_randomall_test_baseline.py
python scripts/paper/finalize_test_results.py
```

### Mechanism evaluation

```bash
python scripts/evaluate_engineering_roi_v1.py --strategy dynamic_only --budget 0.10 --resume
python scripts/evaluate_engineering_roi_v1.py --strategy support_risk_only --budget 0.10 --resume
python scripts/evaluate_engineering_roi_v2.py --method RiskTemporal --budget 0.10 --resume
```

### H0, statistics, and paper artifacts

```bash
python scripts/paper/evaluate_h0_frozen_methods.py
python scripts/paper/analyze_paired_test_statistics.py
python scripts/paper/build_paper_artifacts.py
```

## Paper results

On the untouched TEST set, relative to FrozenGlobal:

- change-region *h* RelL2 improved in 20/20 cases;
- false-positive wet fraction improved in 20/20 cases;
- final wet IoU improved in 20/20 cases;
- debris-front MAE improved in 19/20 cases; and
- trajectory momentum RelL2 improved in 15/20 cases.

Trajectory-wide *h* RelL2 does not improve, and mean wet IoU is not universally improved.

For H0, the simulated horizon is 24 min and the measured wall-clock time is 27.07 s on an RTX 5060 Laptop GPU, corresponding to an approximately 53.2× simulated-time / wall-time ratio. This is not a speedup relative to the high-fidelity solver.

## Reproducibility

The repository provides formal configurations, model definitions, training and inference scripts, evaluation workflows, paper-result CSV/JSON files, statistical analysis, and publication artifacts. The required external data and checkpoints are described above.

## Citation

If you use this repository, please cite the accompanying manuscript.

## License

This repository is released under the GNU Affero General Public License v3.0. See [LICENSE](LICENSE).
