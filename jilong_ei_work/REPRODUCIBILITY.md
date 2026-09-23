# Reproducibility

Use the approved Conda base environment with CUDA/PyTorch. The release metadata and machine-readable audit files are under `paper_results/release_metadata/` and `paper_results/audit/`.

From a runtime workspace containing the ignored data and model artifacts:

```powershell
D:\Anaconda\python.exe scripts\paper\audit_frozen_configuration.py --runtime-root . --git-root <release-repository>
D:\Anaconda\python.exe scripts\paper\evaluate_untouched_test.py --method frozen_global --resume
D:\Anaconda\python.exe scripts\paper\evaluate_untouched_test.py --method risktemporal_b10 --resume
D:\Anaconda\python.exe scripts\paper\finalize_test_results.py
D:\Anaconda\python.exe scripts\paper\evaluate_h0_frozen_methods.py
D:\Anaconda\python.exe scripts\paper\build_paper_artifacts.py
```

The audit is a blocking gate. TEST and H0 must not be used to change thresholds, budget, checkpoints, selector rules, guards, projection, or model weights.
