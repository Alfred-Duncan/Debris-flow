# Park v2 paper-usage guide

**Observation:** cite Gyirong CCTV `463 +/- 90 s` and the stated Rasu/Syab timing windows only as observational constraints.  Do not call Park’s simulated 390 s an observation.

**Park model inference/output:** the 35.42 Mm3 source, 20% ice, at-rest progressive release, mass-conserving melt formulation, H0 model arrivals and output series are Park v2 model assumptions/results.  The 30 s manuscript reference reports 390 s at Gyirong and 1050 s at Syabrubesi; the released 5 s visualization rerun reports 365 s and 1025 s respectively.

**Our reproduction:** `PARK_V2_H0` is an isolated, documented reimplementation, evaluated against the public Zenodo numerical series.  It is acceptable as the initial historical H0 teacher reference with its B classification, but it is not observational truth or author-code identity.

**Terrain and downstream limit:** upper terrain is a conditioned raw DSM: priority-flood/D8 drainage with 0.0005 minimum slope.  Below Syabrubesi, the revised profile switches to `route_b2_w1`, a selected 1-D compound-section routing representation; it is not a single validated 2-D inundation map.  The pre-event river state depends on conditional monsoon/baseflow spinup assumptions.  State these limits in the Chinese EI paper.

**Must not claim:** no verified temporary-dam mechanism; no pixelwise match to unpublished v2 2-D frames; no inferred observation from video; no calibration to force 390 s.
