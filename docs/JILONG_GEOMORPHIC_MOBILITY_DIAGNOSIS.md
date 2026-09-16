# Jilong moving-entry geomorphic and mobility diagnosis

## Scope and reproducibility

This is a post-processing diagnosis of the existing moving-entry D-Claw cases M1--M6 only.  No D-Claw input, terrain, river vector, constitutive parameter, or source state was changed, and **no new solver run was performed**.  The analysis is implemented by `sim/dclaw/jilong_reference_based/geomorphic_diagnosis.py`; it reads existing `fort.q` outputs and the already verified 12.5 m DEM.

The postprocessor stores selected 0, 60, 120, 240, 420 and 600 s fields (`h`, speed, `u`, `v`, solid fraction, wet mask) for M1/M3/M4/M6 in server-only compressed archives.  The tracked time series provide the reproducible summary for M1--M6.

## Actual simulated path versus mapped river

The M3/M6 footprints do not remain in the mapped fourth-order-river corridor.  Their leading wet material turns onto the DEM-controlled route; at 600 s the conventional mapped-vector front is 2.393 km for M3 and 2.690 km for M6, whereas the same fields project 3.068 km and 3.359 km respectively along the independent DEM path.  M4 and M5 likewise reach 3.359 km along that path while their mapped-vector fronts remain 2.649 km.  Thus the older 1.67--2.61 km metric measures departure from the vector corridor as well as propagation; it is not a path-independent physical stopping distance.

The source was initialized with a southeast tangent from the vector, but the computed wet-footprint directions turn in response to the DEM.  The plan-view contours and velocity arrows in `outputs/reference_based/geomorphic_mobility_summary.png` show this directly; the marked 1.67, 2.32, 2.57 and 2.61 km points are projections on the vector, not observed flow-front points.

## Independent DEM path and offset audit

An 8-neighbour D8/priority-flood drainage tree was derived from the 64 m solver DEM, using only the lowest valid DEM boundary cells as outlets.  Zero-valued TT3 padding was excluded.  The start cell was chosen by DEM flow accumulation within 256 m of the model entry (159.8 m from it); no river-vector geometry was used to trace the path.  The first 5 km was sampled at 32 m spacing in `geomorphic_profile.csv`.

| DEM-path versus mapped-vector statistic, first 5 km | Result |
|---|---:|
| Median horizontal offset | 423.4 m |
| Maximum horizontal offset | 698.9 m |
| Stations with offset >100 m | 151 / 157 |
| Stations with offset >200 m | 132 / 157 |
| Stations with offset >400 m | 82 / 157 (first at 1.60 km) |
| DEM path length inside the model domain | 10.89 km |

This is a real representation conflict in the present downstream set-up: the DEM-derived route and river vector diverge substantially.  It does **not** prove that either one is the observed event thalweg.  In particular, the DEM route eventually seeks a cropped-domain boundary outlet, so its long-distance continuation cannot be promoted to an observed Jilong route without an independently checked corridor and outlet.

The leading M3/M6 material is consistent with the DEM route in the limited sense relevant here: its route-projected front continues 0.68--0.71 km beyond the vector-projected front.  Whole-footprint median distances do not give an unambiguous winner because much of the wet volume is an upstream, laterally spread deposit (for final M6: 241 m to vector versus 304 m to DEM path).  The front, velocity directions, and plan view—not a footprint median alone—are the basis for the mismatch finding.

## Geomorphology near the apparent 2.6 km limit

The 1.5--3.5 km segment of the DEM path is not a clean steep-confined-to-wide-fan transition.  Its DEM cross-section width is 416--608 m (mean 537 m) and cross-valley relief is 152--467 m (mean 329 m).  Around 2.5--2.9 km, the derived path has a local 25 m adverse rise (2726.6 to 2751.8 m between 2.56 and 2.72 km), followed by a fall to 2711.3 m at 2.82 km; 250 m slopes switch sign and reach about +/-0.11.  Width remains approximately 544 m through that interval rather than showing a unique widening at 2.6 km.

This is not a one-cell elevation spike: it occupies multiple 64 m grid cells and has hundreds of metres of cross-valley relief.  But it also occurs after the DEM route has already separated by roughly 0.43 km from the mapped river and is routed toward a domain boundary.  It therefore cannot be interpreted as evidence that the historical downstream flow encountered a real depositional sill or fan at 2.6 km.  Available local hydrology/vector files do not resolve a confluence, drainage-area jump, or observed channel transition there.

## Mass, mobility, spreading and classifications

The full engineering diagnostics are in `mobility_timeseries_M1.csv` through `mobility_timeseries_M6.csv`: wet/moving volume, centre of mass, velocity quantiles, wet/high-depth area, and potential/kinetic energy proxies.  Representative findings are below.  Volumes are wet mixture volumes above the 0.01 m diagnostic threshold; energy is a diagnostic proxy, not a conserved total energy.

| Case | Behaviour after early advance | 600 s state | Required classification |
|---|---|---|---|
| M1 | Vector and DEM-route fronts settle by 240 s; wet area declines from 610,304 to 454,656 m2 | 87,586 m3 above 2 m/s; mean moving speed 3.15 m/s | RUNOUT_STOP |
| M2 | Same pattern, with late wet-area redistribution | 83,211 m3 above 2 m/s; mean moving speed 2.95 m/s | RUNOUT_STOP |
| M3 | DEM-route front continues 2.578 to 3.068 km from 240 to 600 s while vector front only reaches 2.393 km | 120,956 m3 above 2 m/s; mean moving speed 2.89 m/s | CONTINUING_DOWNSTREAM_FLOW |
| M4 | DEM-route front advances 3.113 to 3.359 km after the vector metric is near its plateau; then wet area contracts | 106,274 m3 above 2 m/s | VECTOR_DIAGNOSTIC_MISMATCH |
| M5 | Same mismatch; route front is 3.359 km versus vector 2.649 km | 90,332 m3 above 2 m/s | VECTOR_DIAGNOSTIC_MISMATCH |
| M6 | DEM-route front reaches 3.359 km by 240 s versus vector 2.690 km; wet area subsequently contracts 1.585 to 1.323 km2 | 104,915 m3 above 2 m/s; mean moving speed 3.90 m/s | VECTOR_DIAGNOSTIC_MISMATCH |

M6 illustrates real deposition/spreading after the leading advance: wet area grows from 0.250 km2 at 0 s to 1.585 km2 at 240 s, and the kinetic-energy proxy falls from approximately 2.0e12 J at 0 s to 0.10e12 J at 600 s.  However, appreciable moving mass and multi-metre-per-second mean moving velocity remain at 600 s.  The evidence therefore supports dissipation and lateral accommodation in the numerical field, but not a clean, natural, whole-flow stop at the vector distance.  M3 is still advancing at the end of the saved run.

## Physics audit

| Physical process | Expected relevance downstream | Included in current case? | Evidence available? | Consequence / priority |
|---|---|---|---|---|
| Initial momentum | First-order at an already moving entry | Yes: native qinit velocity, 8/15/25 m/s | Entry velocity is not publicly measured | Present mathematically; source value remains scenario-only |
| Basal pore pressure and dissipation | Potentially important | D-Claw state exists; hydrostatic initial pressure and generic permeability are used | No event entry pressure/permeability | Important uncertainty, not diagnosed as the present path mismatch |
| Granular friction | Important | Yes, but transferred reference parameters (`phi=38 deg`, etc.) | No event calibration | Important uncertainty |
| Valley confinement and curvature | First-order | DEM resolves them at 64 m; river vector is diagnostic only | Local geometry conflicts between representations | **First issue to resolve** |
| Entrainment / erosion/deposition | Potentially important over the historical chain | Entrainment disabled; deposition evolves only through the model state | Literature supports qualitative entrainment, no quantitative law | Cannot be enabled defensibly yet |
| Ambient river water, dilution, water uptake | Potentially important after main-river entry | No explicit ambient river flow/interactions | No entry hydrograph or mixing observation | Important future-process uncertainty |
| Changing solid fraction / debris-flow transformation | Potentially important | D-Claw evolves state, but source starts uniform at 0.62 | Entry composition unavailable | Scenario limitation |

## Scientific classification and consequence

**Primary diagnosis: A. DIAGNOSTIC_PATH_ARTIFACT.**  The apparent 2.6 km limit is primarily a river-vector projection artifact: the model front continues substantially farther on the independently derived DEM drainage route.  This classification does not claim the current DEM route is the historical route, nor that propagation to Jilong Port is established.

**Secondary diagnosis: none assigned.**  A terrain/vector representation conflict and unconstrained source/physics are material uncertainties, but the existing fields do not separate them sufficiently to justify class B, D, or E as a second causal diagnosis.

No new D-Claw run is scientifically justified under classification A.  The corrected, path-dependent diagnostic runout from the existing fields is **3.359 km along the DEM-derived route for M4--M6** (M3: 3.068 km at 600 s), compared with **2.649--2.690 km** on the mapped-vector metric.  It is not an observed event runout and must not be used as a port-arrival or calibration target.

The current simulator is **not scientifically usable for engineering scenario generation** beyond controlled sensitivity demonstrations, because its source-to-corridor geometry is not yet verified.  The one recommended next step is to construct and independently validate one georeferenced downstream corridor/entry-to-port thalweg from the native DEM plus observed channel/impact evidence, then recompute the same diagnostics before changing any physics or source parameter.
