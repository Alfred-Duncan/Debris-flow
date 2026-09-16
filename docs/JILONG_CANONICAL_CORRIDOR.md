# Canonical real-event corridor: Gate A assessment

## Result

**Hard Gate A: NOT PASSED.**  A defensible, approximately 15-km-scale physical main-river-entry-to-Jilong-Port corridor cannot be established from the currently downloaded spatial evidence without introducing an unvalidated connection or selecting an unobserved entry section.  Accordingly, no canonical corridor, control sections, final D-Claw domain, inflow boundary, or event simulation was created.

## Spatial evidence inspected from the port upstream

1. The official port point is available and projects to EPSG:32645 approximately `(340571.8, 3129640.8)` m.
2. The native continuous 12.5 m DEM covers both the port and the previously provisional entry region; its full extent is approximately 167.7 by 120.4 km.  The 8 m DEM is in another projection and covers a limited basin, not a continuous downstream entry-to-port corridor.
3. The fourth-order river layer is a two-part `MultiLineString`, not one continuous surveyed line at the critical downstream connection.  Its two parts terminate near `(337809.8, 3131780.1)` m and start near `(338031.6, 3131303.7)` m, leaving an approximately 0.53 km spatial gap.  The previous diagnostic workflow treated disconnected components using a straight connector; that is explicitly prohibited for the final corridor.
4. The supplied coarse hydrology grids do not resolve the official port cell: the port and the start of the downstream vector component are NoData in the flow-direction, flow-accumulation and downstream-length rasters.  They therefore cannot independently bridge the gap or validate a port-connected thalweg.
5. The core-area polygon is an impact/study boundary, not a channel, entry section, scour corridor, or hydraulic cross-section.  The supplied before/after images are not georeferenced rasters.  No local layer provides a georeferenced continuous event path or a mapped post-event main-river-entry confluence.

## Why a DEM-only route is not accepted as canonical

The full DEM can support drainage analysis, but a DEM-derived low path alone cannot establish that a route is the observed 26 August downstream main-river stage.  At the critical vector gap, choosing a particular DEM valley line would connect two disconnected vector components without independent channel or event-path evidence.  It would be a numerical route-selection assumption, not a validated real-event corridor.  The prior cropped-DEM D8 diagnosis showed exactly why unvalidated raster routing must not be promoted to a historical thalweg.

Likewise, the prior provisional entry was explicitly a geometric model snap, not a publicly observed physical confluence.  Its mapped route/connector length was about 18.27 km, which is materially different from the soft about-15-km estimate and cannot be reconciled by relocating S0 merely to match that estimate.

## Missing evidence needed to pass Gate A

- A georeferenced surveyed or image-interpreted main-river channel/thalweg through the roughly 0.53 km vector discontinuity to the port;
- a georeferenced event-path, scour, or impact corridor linking the inferred post-entry flow to that channel;
- a defined main-river-entry cross-section, or a defensible mapped confluence/control section; and
- channel cross-sections or an orthorectified post-event image sufficient to distinguish the active channel from nearby valley-floor terrain.

Until these are available, `data/processed/jilong_canonical_corridor.geojson`, control sections S0--S4, a corridor profile, and a canonical corridor map would falsely imply a level of spatial certainty that the data do not provide.
