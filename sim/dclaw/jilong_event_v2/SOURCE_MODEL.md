# Phase 2C source model and fixed diagnostics

V2 uses one case-level, finite-duration left-boundary inflow, not a moving `qinit` ellipse. The inlet plane is at x=334021.9 m (EPSG:32645), 29.9 m upstream of the fixed model entry (334051.8, 3143392.5 m), on the same 64-m-aligned river transect. No constitutive or Riemann solver was changed.

A 256 m ribbon centred at y=3143392.5 m receives `Q(t)=Q_peak*(1-|2t/T_in-1|)` for `0<=t<=T_in`; imposed volume is `Q_peak*T_in/2`. Only `Q_peak` and `T_in` vary. Fixed, provisional state construction is h=10 m, solid fraction 0.63, species fraction 0.5, hydrostatic basal pore pressure, and velocity consistent with the boundary flux. None are claimed observations.

Entrainment-enabled cases use the documented old-style implemented D-Claw setting: `entrainment=1`, `entrainment_method=0`, rate 0.2, `me=0.65`, and a uniform provisional 2-m entrainable layer. The control has entrainment off. The accumulated entrained-volume state is not used because its output identity was not independently verified; it is reported as unavailable.

## Hydraulic port-proxy

| Feature | EPSG:32645 coordinate (m) |
|---|---:|
| Official Jilong Port landmark | (340837.9, 3129051.7) |
| Hydraulic port-proxy (downstream mapped fourth-order-river terminus) | (340571.826, 3129640.776) |
| Landmark-to-proxy separation | 646.379 m |
| Model entry to proxy mapped-route distance | 17.624 km |

The proxy is a mapped-channel diagnostic section, not a surveyed official port cross section. Front progress is measured only from wet cells (h>0.01 m) within a fixed 320-m mapped-river corridor; isolated off-corridor cells cannot set the front.
