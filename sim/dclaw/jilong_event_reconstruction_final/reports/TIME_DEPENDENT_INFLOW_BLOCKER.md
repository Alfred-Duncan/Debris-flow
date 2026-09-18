# Superseded blocker note

The prior blocker was scientifically correct when no transient inlet closure
had been approved. It is superseded by the explicitly authorized
unit-Froude/critical-flow computational external-boundary closure implemented
in this case. The active stop condition is instead recorded in SMOKE_TEST.md:
the permitted corrected C3 smoke rerun failed in boundary input parsing before
the first timestep, so no additional rerun or full sweep was performed.
