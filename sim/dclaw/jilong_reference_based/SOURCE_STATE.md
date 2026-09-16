# Moving-entry qinit state verification

The installed D-Claw source was inspected locally at `/root/src/clawpack/dclaw` before this case was changed.

`src/2d/dig/digclaw_module.f90` hard-codes the installed non-Boussinesq state ordering as:

| Fortran q index | State |
|---:|---|
| 1 | `h`, flow depth |
| 2 | `hu`, x momentum |
| 3 | `hv`, y momentum |
| 4 | `hm`, solid-volume depth |
| 5 | `pb`, basal pore-fluid pressure |
| 6 | `hchi` |
| 7 | `Delta b` |

The installed `QinitDClawData` API accepts a TT3 raster for each q index. Its documentation and `qinit.f90` specify an important representation rule: qinit files for q2, q3, q4, and q5 contain **u**, **v**, **m**, and **pb/h**, respectively; after reading, `qinit.f90` multiplies those values by `h` to form conserved `hu`, `hv`, `hm`, and `pb`.

Accordingly the moving source supplies qinit files for `h`, `u`, `v`, and `m=0.62`. Its actual initialized conserved state is `hu=h*(speed*DIRECTION_x)`, `hv=h*(speed*DIRECTION_y)`, and `hm=h*0.62`. No q5 file is needed: the unchanged `init_ptype=0` applies hydrostatic `pb=rho_f*g*h` after qinit. This is native qinit behavior; no custom boundary or Riemann-code change is used.

Every M1--M6 case is first initialized in a zero-time native-qinit gate. The gate reads `fort.q0000`, computes the mean speed over cells with `h > 0.1*hmax`, and requires it to match the requested entry speed within 2%. Only then does the six-run 600-s batch begin.
