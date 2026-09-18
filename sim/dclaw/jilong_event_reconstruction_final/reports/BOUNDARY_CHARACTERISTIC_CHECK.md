# North-boundary characteristic check

Result: **PASS**.

The installed local D-Claw source sets kappa=1 in
dclaw/src/2d/dig/digclaw_module.f90. In
dclaw/src/2d/dig/riemannsolvers_dclaw.f, the gravity scale is
sqrt(geps*h), with geps=gz*eps and eps=kappa+(1-kappa)*gamma. For this case
bed_normal=0, so gz=grav; with kappa=1, the scale is c=sqrt(g*h).

For the NORTH external boundary, the inward normal is negative global y. The
prescribed normal velocity is v_n=-Fr_n*c=-1.20*c. The uniform-state gravity
characteristics are:

| Characteristic | Speed |
|---|---:|
| v_n-c | -2.20*c |
| v_n+c | -0.20*c |

Both are inward, and the slower characteristic retains the requested 0.20*c
inward margin. The three inlet cells therefore receive a fully incoming
supercritical-normal ghost state. Other north-boundary cells are reflective:
scalar state and tangential momentum are copied, while normal momentum is
reversed.
