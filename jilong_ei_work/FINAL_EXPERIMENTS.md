# Final experiments

Frozen final VAL results are recorded at `19b63d10449cefdaeb0ebe0fdd7a84ed31797c4a`. The paper stage evaluates the untouched TEST split with Frozen Global and RiskTemporal-v1 B10, and applies the same frozen pair to the independent `PARK_V2_H0` historical anchor.

Historical roles are explicit: SupportRisk/Temporal are ablations; DepthGuard is a negative ablation; EngineeringROI-v1 and DynamicOnly are failure-mechanism evidence; Oracle and RandomPerfect are feasibility diagnostics, not deployable methods. H0 is an event-constrained numerical-reference demonstration, not calibration or complete real-event validation.
