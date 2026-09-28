# Conservative fixed-S0 source implementation

## Frozen architecture

The production case uses **CONSERVATIVE_FIXED_S0_SOURCE_ZONE**. It does not use `bc2amr.f90`, ghost-state discharge, `inflow.data`, a qinit source, a velocity multiplier, or a speed cap. The active `Makefile` excludes installed `dclaw/src/2d/dig/src2.f90` and compiles the case-local `src2.f90`; the installed original copied before modification had SHA-256 `639d8664dcf8f65d9e9649524a39de47214072e4d595a0653aa9e4f3b06cda61`.

`setrun.py` sets standard Clawpack extrapolation on south/east/west and the standard configured `wall` condition on the cropped northern/upstream boundary. Maximum AMR level remains 1; Richardson and flag-based refinement are disabled. Entrainment remains off.

## Verified source call interval

The linked local GeoClaw/AMRClaw call site is `/root/src/clawpack/geoclaw/src/2d/shallow/stepgrid.f`, lines 205–239. It first calls `step2`, then updates `q` by flux differencing, then—when `method(5)==1`—calls:

```fortran
call src2(..., q, maux, aux, time, dt)
```

Thus the case-local injection is applied after the hyperbolic update over **[t, t+dt]**. The new `src2.f90` computes `DeltaV=F(clamp(t+dt))-F(clamp(t))`; it never uses `Q(t)*dt` and has no persistent cumulative-volume counter.

The adaptive-step audit is limited to the active configuration: level 1 only and both AMR error-estimation/refinement flags disabled, so `prepregstep`/`prepbigstep` do not create an additional source update path. The linked `/root/src/clawpack/geoclaw/src/2d/shallow/advanc.f` advances accepted grid time after `stepgrid`; future dt selection is based on reported CFL. Because the source is after the hyperbolic CFL estimate, `dt_max=0.5 s` is an implementation-stability guard: it prevents an initially dry, zero-wave-speed grid from proposing a multi-second step before the first finite source increment can influence the next CFL estimate. It is not a physical speed cap or a source-parameter change. Final mass evidence is the model-state ledger at output times, not a diagnostic call count.

## Incremental state and order

Within the clearly delimited `BEGIN/END JILONG CONSERVATIVE SOURCE` block, only physical `i=1..mx`, `j=1..my` cells whose physical centers match the three JSON cells are updated. For every exact analytic interval increment:

| state | additive increment |
|---|---|
| `q(h)` | `dh=DeltaV/(3 dx dy)` |
| `q(hu)` | `dh U_src tangent_x` |
| `q(hv)` | `dh U_src tangent_y` |
| `q(hm)` | `0.62 dh` |
| `q(pb)` | `rho_f g dh` |
| `q(hchi)` | 0 |
| `q(bdif)` | 0 |

This agrees with the local D-Claw meanings imported by `src2`: `i_h`, `i_hu`, `i_hv`, `i_hm`, `i_pb`, `i_hchi`, and `i_bdif`. The injection precedes the unchanged original physical source-term loop, so newly injected material undergoes the normal same-step D-Claw source physics. `U_src` is evaluated at the actual interval midpoint using only the locked unit-Froude computational closure with `W_ref=148.81741474020365 m`.

The copied physical source code is otherwise unchanged. Source matching has an explicit failure stop for an off-grid or ambiguous cell; it cannot silently inject into ghost cells.
