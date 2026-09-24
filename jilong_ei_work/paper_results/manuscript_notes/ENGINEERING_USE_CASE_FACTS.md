# Engineering use-case facts

## Scope

This method is a rapid post-detection hazard-evolution forecasting layer, not a hazard-initiation detector. It assumes an organization already has, or can access, terrain products, remote sensing, rainfall/hydrological and geological-hazard monitoring, numerical simulation environments, historical scenario databases, high-fidelity physical models, and operational computing infrastructure.

Once an event has been detected and an initial hazard state, source estimate, or upstream-model estimate is available, the intended task is rapid downstream scenario evolution: propagation, dynamically changing regions, debris-front evolution, possible wet-area expansion, false-positive wet extent, and multiple near-term scenarios.

## Rolling workflow

Existing monitoring / remote sensing / numerical-modeling system -> current hazard state / source estimate -> Frozen Global Operator -> rapid full-domain coarse rollout -> Support-Risk ROI allocation -> Temporal Refresh -> budgeted local correction -> physical and support guards -> rapid downstream hazard-evolution scenario -> updated monitoring state -> repeat rolling forecast.

The method is intended as a rapid scenario-updating layer coupled to existing monitoring and high-fidelity modeling infrastructure. High-fidelity numerical models remain responsible for physics, offline reconstruction, scenario generation, and reference simulation; RiskTemporal provides fast online or near-online scenario rollout, dynamic-region updates, front evolution, and risk-focused local refinement.

## H0 runtime context and limits

For the recorded H0 runtime environment, RiskTemporal-v1 B10 completed the existing 1440-s (24-min) simulated hazard-evolution rollout in approximately 27.1 s: a simulated-horizon/wall-clock ratio of approximately 53.2. This is a real-time progression ratio, not a speedup claim over D-Claw or any other high-fidelity solver.

This study does not establish automatic initiation detection, exact failure-time prediction, operational warning issuance, precise station-arrival prediction, field deployment, field-observation validation, or replacement of governmental high-fidelity systems. H0 station response remains prematurely timed; the appropriate interpretation is propagation trend and spatial hazard evolution, not exact minute-level warning.
