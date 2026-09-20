# D-Claw parameter semantics check

This check was completed before entering reconstruction Families B, C, or D. The
installed case generator maps `active_run.json` values to `dclaw.data` and
`geoclaw.data`; the solver sources below establish their physical role.

| Parameter | Source / subroutine | Minimal code evidence | Effect expected within the declared ladder |
|---|---|---|---|
| `entrainment_rate` | `reference_sources/dclaw-official/src/2d/dig/entrainment.f90`, `ent_dclaw4` lines 29, 71, 78 | `dh = entrainment_rate*dt*(t1bot-t2top)/(rhoe*beta*vnorm)` | Increasing the positive rate nominally increases entrained thickness per eligible moving cell, subject to the remaining erodible layer cap. |
| `phi` | `reference_sources/dclaw-official/src/2d/dig/digclaw_module.f90`, lines 478–482 | `tau = sig_eff*tan(phi + psi)` | Decreasing phi nominally decreases Coulomb resistance for the same effective stress and dilatancy state, so increases mobility. |
| Manning coefficient | local `src2.f90`, lines 347–362 | `gamma = ...*(gz*coeffmanning**2)/(h**(7/3)); hu=hu/dgamma; hv=hv/dgamma` | Decreasing positive Manning coefficient reduces the quadratic friction damping and nominally increases mobility. |

No solver source was modified. The rate, phi and Manning values are confined to
the predeclared discrete ladder; these are sensitivity/reconstruction controls,
not independently measured values.
