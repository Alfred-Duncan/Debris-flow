# C3 smoke test

Status: **FAIL**. No full C1--C6 run was started.

The first attempted smoke invocation did not start D-Claw because Make invoked
setrun.py without the shell-only run variables. This was a clear run-wrapper
error. The allowed corrected smoke rerun used the case-local active_run.json
file and reached D-Claw initialization successfully:

- accepted 64 m terrain was read;
- the S0-cropped 168 by 248 grid was constructed;
- frame 0 was written at t=0;
- initial domain mass was zero, as required for an external inflow case.

It then stopped at the first case-local bc2amr call because inflow.data
contained literal backslash-n separators rather than line terminators. GNU
Fortran reported: Bad real number in item 1 of list input, at bc2amr.f90 line
15. This is a file-format implementation error, not a terrain, material, or
source-closure failure.

setrun.py has been corrected to write real line terminators for future use, but
no third smoke execution was made: the task permits only one corrected smoke
rerun. Therefore smoke status remains FAIL and all C1--C6 production runs are
not run.
