# PARK_V2_H0 run report

Completed one 30 m upper-domain H0 run only: 1440 s, 5 s stored outputs, 6560 adaptive steps, 98.5 s local CUDA wall time.  Config: `configs/PARK_V2_H0.json`.

The run uses 35.42 Mm3, 20% ice, zero initial free water, zero release speed, a 30 s source duration, pre-event restart, v2 density/melt semantics and the released transport settings.  It is a historical parity attempt, not calibration.

The released v2 comparison has 6560 steps and final volume 42451017.188 m3; this reimplementation has 42327752.344 m3 (-0.2904%).  See the scorecard and figures for station and time-series parity.
