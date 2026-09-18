# Jilong final 64 m source-only reconstruction

Status: **BLOCKED BEFORE SIMULATION**.

This is the event-constrained, 64 m, fixed-S0, time-dependent-inflow D-Claw
case planned for the accepted Gate A / Gate B terrain handoff. No D-Claw
frames, qinit source, smoke run, or source-only sweep has been created.

The local server inspection found no configuration-driven D-Claw mechanism for
injecting a time-dependent interior S0 source. The older case-level boundary
routine only prescribes a state on a rectangular external domain boundary and
contains non-event hard-coded inlet properties. It cannot be reused as the
required two-parameter (volume, duration) source without introducing
unsupported inlet depth and momentum magnitude assumptions.

See the reports directory for the environment check and exact blocker.
