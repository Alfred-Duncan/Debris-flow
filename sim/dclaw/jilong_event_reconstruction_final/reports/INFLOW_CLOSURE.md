# Unit-Froude computational inflow closure

This is a computational closure, not an observed Jilong inlet state.
The 160 m S0 section is an engineering inspection support, not an observed
channel width. The computational boundary support is its nearest 64 m
discretization.

## Geometry

- Selected external boundary: NORTH; inward normal: [0.0, -1.0]
- S0 downstream tangent from first 160 m: (0.631849985, -0.775090702)
- alpha = dot(t_hat, n_in): 0.775090702
- Domain: x=[331950.0, 342702.0], y=[3127516.0, 3143388.0] m
- S0-to-boundary offset: 4.493 m
- Inlet cells (0-based raster columns): [31, 32, 33]
- B_actual: 192.0 m; W_eff: 148.817415 m

## State closure

This is a fixed supercritical-normal computational inflow closure used
to make the truncated upstream boundary fully incoming; it is not observed.
For Q(t)>0: U_n=1.2 sqrt(g h) and
h=[Q/(B_actual Fr_n sqrt(g))]^(2/3). Total U=U_n/alpha and velocity
follows the fixed corridor tangent. Q values yielding h <= the
D-Claw dry tolerance (0.001 m) are dry/no-inflow.

D-Claw source relation: dclaw/src/2d/dig/qinit.f90 sets hm=m0*h when
no hm raster is supplied; with init_ptype=0 and bed_normal=0 it sets
pb=rho_f*grav*h. hchi is initialized only when segregation=1, so this
case (segregation=0) uses hchi=0. bdif remains its initialized zero.
