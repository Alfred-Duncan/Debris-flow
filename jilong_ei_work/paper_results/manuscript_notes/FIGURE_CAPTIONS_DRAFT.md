# Figure captions draft

Figure 1. Frozen RiskTemporal-v1 B10 inference architecture.

Figure 2. Completed VAL budget sensitivity. B10 is the selected engineering operating point, not a mathematical optimum.

Figure 3. Case-paired untouched TEST improvements for completed frozen baselines.

Figure 4. Untouched TEST runtime and separate engineering-accuracy trade-offs.

Figure 5. Completed VAL closed-loop failure mechanism using EngineeringROI B10, DynamicOnly B10, SupportRiskOnly B10, and RiskTemporal-v1 B10 from their completed frozen artifacts. Aggressive dynamically targeted local refinement can destabilize long-horizon autoregressive rollout; support-risk restriction improves stability, while one-step temporal refresh yields a more balanced operating point. The figure does not claim universal superiority of RiskTemporal.

Figure 6. H0 field evolution against the documented high-fidelity numerical reference, with a shared depth color scale.

Figure 7. H0 sampled debris-front positions for the documented high-fidelity numerical reference, Frozen Global, and RiskTemporal-v1 B10. Front is the formal debris-front definition: max(route chainage) among cells with h > 0.1 m, c > 0.05, and finite route chainage, reported in km. These sampled positions use the same definition as the formal metric; aggregate full-rollout MAE values are reported separately.

Figure 8. H0 station-arrival limitation.
