# Jilong final 64 m source-only reconstruction

Status: **STOPPED AFTER FINAL C3 SMOKE MASS-CONSISTENCY FAILURE**.

This case uses the approved unit-Froude computational inflow closure at the
north external boundary nearest S0. It is not an observed or calibrated inlet
state. The six-case design varies only mixture volume and pulse duration.

The numerical inflow preflight passes all prescribed volume and peak-flux
tolerances. The final C3 smoke test reads the accepted 64 m terrain, reaches
120 s, and propagates downstream, but retains only 0.734e6 m3 at t=90 s for a
2.0e6 m3 prescribed source. The north-boundary Riemann flux is therefore not
broadly consistent with the closure's prescribed discharge. No C1--C6 sweep
was run; changing physics or closure parameters is outside this task.

See reports/INFLOW_CLOSURE.md, reports/INFLOW_PREFLIGHT.md, and
reports/SMOKE_TEST.md.
