# Superseded blocker note

The prior blocker was scientifically correct when no transient inlet closure
had been approved. It is superseded by the explicitly authorized
unit-Froude/critical-flow computational external-boundary closure implemented
in this case. The active stop condition is instead recorded in SMOKE_TEST.md:
the final supercritical-normal C3 smoke run completed but retained only 42.8
percent of the prescribed source volume at t=90 s. The external-boundary
method is rejected; no full sweep was performed.
