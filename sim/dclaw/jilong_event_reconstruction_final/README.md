# Jilong final 64 m source-only reconstruction

Status: **EXTERNAL-BOUNDARY METHOD REJECTED AFTER FINAL C3 MASS-BUDGET TEST**.

This case uses the approved unit-Froude computational inflow closure at the
north external boundary nearest S0. It is not an observed or calibrated inlet
state. The six-case design varies only mixture volume and pulse duration.

The final authorized fixed-supercritical-normal C3 smoke test passes the
local D-Claw characteristic check and reaches 120 s, but fails the prescribed
mass budget: retained domain volumes at 30, 60, and 90 s are 1,825, 751,498,
and 856,547 m3 versus targets of 500,000, 1,500,000, and 2,000,000 m3.
No C1--C6 sweep was run. The next method is conservative fixed-S0 source-zone
injection; no further ghost-state external-boundary variants are permitted.

See reports/INFLOW_CLOSURE.md, reports/INFLOW_PREFLIGHT.md, and
reports/SMOKE_TEST.md.
