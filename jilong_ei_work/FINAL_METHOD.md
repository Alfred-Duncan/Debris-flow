# Final method

The final deployable configuration is **RiskTemporal-v1 B10** at commit `146184d1fbbf366c0c9c40066a56753b398b10f4`.

Frozen chain: Global Operator (`best.pt`, step 5000, STAGE_C, K=4) → Support-Risk ROI → spatial de-redundancy → one-step temporal refresh → Local Corrector (`best.pt`, update 4000) → Wet-Support-Preserving Guard → Momentum State Guard → physical projection.

The fixed B10 configuration is `configs/risk_temporal_final_v1.json`. B05/B10/B20 are reported as budget sensitivity; only B10 is the primary final configuration. DepthGuard is a negative ablation, not part of the deployed method.
