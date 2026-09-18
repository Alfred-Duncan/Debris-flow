# Jilong final 64 m source-only reconstruction

Status: **STOPPED AFTER FAILED CORRECTED SMOKE TEST**.

This case uses the approved unit-Froude computational inflow closure at the
north external boundary nearest S0. It is not an observed or calibrated inlet
state. The six-case design varies only mixture volume and pulse duration.

The numerical inflow preflight passes all prescribed volume and peak-flux
tolerances. The corrected C3 smoke test read the accepted 64 m terrain and
wrote frame 0, but then failed at the first boundary input read because the
case generator had written literal backslash-n separators. The generator is
corrected, but the task permits only one corrected smoke rerun; no third run
and no C1--C6 sweep was performed.

See reports/INFLOW_CLOSURE.md, reports/INFLOW_PREFLIGHT.md, and
reports/SMOKE_TEST.md.
