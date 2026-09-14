# Evidence-based downstream model-entry definition

## Decision

The preferred **model entry** is the fourth-order-river centerline snap of the existing provisional baseline point:

| Item | Coordinate / value |
|---|---|
| CRS | EPSG:32645 — WGS 84 / UTM zone 45N |
| Current provisional baseline entry | `(334051.5, 3143392.2)` m |
| Preferred model entry | `(334051.8, 3143392.5)` m |
| Preferred model entry, WGS 84 (reference only) | `85.305943° E, 28.406422° N` |
| Shift from baseline point | approximately 0.4 m, to the mapped fourth-order-river centerline |
| Status of old point | **Acceptable**; the new point is a geometric snap, not a material relocation. |

This is not an official main-river-entry coordinate.  It is an evidence-based **model-entry** location for the downstream domain, selected without reference to a desired travel time and without assigning any source state.

## Spatial evidence and method

1. The existing baseline point is inside the downloaded official core-area boundary and lies only approximately 0.4 m from the downloaded fourth-order-river centerline after all vectors are transformed to EPSG:32645.
2. The mapped fourth-order river is continuous from that point toward Jilong Port.  The selected downstream route follows this vector and then uses a clearly labelled 0.646 km straight connector from the vector terminus to the official port point; the connector is not an invented river reach.
3. The China Geological Survey report describes the event path as the Cuojian River through Guobaxiaqu and Donglin Zangbu to Jilong Port.  It corroborates a connected downstream corridor, but it does not publish a surveyed confluence/entry cross-section.  See [CGS/MNR report](https://www.cgs.gov.cn/ywdt/ddyw/202609/t20260908_868093.html).
4. The downloaded disaster-point layer and core-area boundary identify the port and affected study area.  No separate, georeferenced tributary/debris-corridor vector or surveyed entry line was present in the existing downloaded files.  The supplied before/after images are not georeferenced rasters, so they were not used to claim a precise confluence coordinate.
5. The diagnostic map is [model_entry_map.png](../outputs/phase2b_prep/model_entry_map.png).  It shows the DEM hillshade, official core-area boundary, mapped fourth-order river, existing baseline point, preferred model-entry point, entry uncertainty segment, port, and selected route.

## Distance and uncertainty

The route distance from the preferred model entry to the mapped river terminus is **17.624 km**.  The official port point is **0.646 km** from that terminus.  The displayed mapped-route-plus-connector distance is therefore **18.270 km**.  Straight-line entry-to-port distance is **15.865 km** and is included only as a secondary geometric reference.

The NCDC rapid estimate of “about 15 km” is not used to move this point: the report does not publish the entry cross-section from which its distance was measured.  The model route length must follow the downloaded geometry, not a time/distance target.

Point confidence is **moderate for model geometry, low for the physical-event confluence**.  The working uncertainty is a deliberately short **1.0 km along-river model-entry zone** (500 m on either side of the preferred point), not an official uncertainty measurement:

| Zone position | EPSG:32645 coordinate (m) |
|---|---|
| Upstream end | `(333625.5, 3143653.8)` |
| Preferred centre | `(334051.8, 3143392.5)` |
| Downstream end | `(334433.1, 3143069.2)` |

This zone records the absence of a surveyed public entry section.  It must not be converted into an entry velocity, depth, discharge, volume, or hydrograph.

## DEM and local-direction checks

- **DEM source near the preferred entry:** the original 8 m DEM does not cover this point.  In the existing baseline input construction, the location is filled by the native 12.5 m DEM; the 65 × 65-pixel 12.5 m neighbourhood is fully valid.  No nearest-neighbour residual fill is used at the preferred point.
- **Artifact check:** no no-data hole, isolated extreme value, or obvious terrain artifact was found at the preferred point in the 12.5 m DEM.  Fine-scale values along the mapped centerline fluctuate locally in the narrow valley, so they should not be interpreted as a surveyed channel cross-section.  A visible mismatch can occur where the 8 m and 12.5 m rasters are value-merged elsewhere; the diagnostic map deliberately uses the continuous 12.5 m hillshade and shows 8 m availability only as a pale-cyan overlay.
- **Downhill direction:** sampled 12.5 m elevations fall overall from about 2,809 m at the entry to about 2,724 m after 1 km and about 2,541 m after 5 km along the mapped route.  Despite local raster/centerline variation, the local and reach-scale direction is downhill toward Jilong Port.

## Suitability for the next stage

The location is suitable as a **geometric model-entry location** for a later downstream D-Claw source-definition exercise.  It is not sufficient to define a D-Claw source/boundary condition: the event-time `Q(t)`, depth, velocity, solid fraction, volume, pressure state, material properties, and entrainment parameters remain unknown.  No simulation or baseline input was changed while making this determination.
