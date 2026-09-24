# E4 valley-v3 result

Classification: **LOCAL_VALLEY_GEOMETRY_DOES_NOT_RESOLVE_MOBILITY**.

Final time: 900 s; frames: 31; max depth: 23.9006 m; max speed: 13.6236 m/s.
Final local-axis progress/fraction: 0.000 m / 0.0000.

Recommended next action: `STOP_DCLAW_CALIBRATION_AND_USE_EVENT_CONSTRAINED_SCENARIO_FRAMING`.

The watchdog had two 60-second checks, zero stalls, and no runtime failure. Its final keyword scan matched normal termination wording (`stopping`) after exit code 0 and 31 frames; this is retained as a false-positive audit note.

Physics, terrain, source, and entrainment rate were not modified.

Finite-field check: PASS. Missing front coordinates, where present, mean no source-connected h>0.001 m cell mapped onto the local axis; they are not NaN/Inf solution fields.
