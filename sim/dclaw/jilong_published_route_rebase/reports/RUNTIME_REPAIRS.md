# Runtime repairs

1. C3 initial launch: `topo.data` resolved the new terrain filename from the case root. No frame was written. Added a byte-identical case-root copy of the unmodified new-domain TT3 and reran; physical inputs unchanged.
2. Watchdog wrapper initially watched a transient shell rather than `xdclaw`; C3 itself completed successfully with 31 frames. E4 launcher corrected with `exec`, with unchanged simulation inputs.
