# Event timeline and Seqiong-station lock (Phase 2C)

## Locked event anchors

| Anchor | Time | Evidence class | Source / interpretation | D-Claw use |
|---|---:|---|---|---|
| High-elevation initiation | about 10:52, 2026-08-26 | A. DIRECT / OFFICIAL OBSERVATION | Xizang disaster press release (2026-08-30): ice-rock avalanche initiated near 5200 m and followed the Purepu Zangbu / Donglin Zangbu pathway toward Jilong Port. | Whole-chain context only; it predates this artificial downstream inlet. |
| Seqiong (色穷) water-level data interruption | 10:55 | A. DIRECT / OFFICIAL OBSERVATION | Xizang Water Resources Department station-data status. This is an interruption time, **not** a picked debris-front arrival. | Soft event-time anchor only, conditional on an independent station geolocation. |
| Jilong Port impact | about 10:59 | B. PEER-REVIEWED RECONSTRUCTION / ESTIMATE | Derived as approximately 10:52 + the approximately 7 min full-chain result in the Science Bulletin reconstruction / ITP-CAS summary (DOI: 10.1360/CSB-2026-1255). No separate official 10:59 statement was recovered in this fast check. | Whole-chain consistency context only. |

The 22 km / 7 min result is a high-source-to-port reconstruction. It is **not** an entry-to-port travel-time observation. Conversely, the earlier about-15-km / about-30-min / 8.3 m/s rapid assessment is retained as a historical early estimate only; it is not the primary timing-calibration target for V2.

## Local station-data check

The already downloaded `hydrology.zip` was enumerated. It contains only hydrologic terrain rasters (flow direction, accumulated area, river width, upstream/downstream length and their metadata). It contains no point feature, station list, station coordinate, monitoring metadata, or field bearing 色穷 / Seqiong / 东林藏布 water-level-station identity.

**Result: no georeferenced Seqiong station coordinate was recovered from existing downloaded data.** It cannot be transformed to EPSG:32645 without inventing a location. Therefore neither its route position nor an entry → Seqiong → port ordering is established.

## Route-distance status

| Route | Distance | Status |
|---|---:|---|
| Model entry → Jilong Port | about 18.27 km (17.624 km mapped river plus 0.646 km connector) | C. SECONDARY ESTIMATE / model-geometry diagnostic from the existing mapped route. |
| Model entry → Seqiong | Unknown | D. UNKNOWN / NOT PUBLICLY AVAILABLE in the downloaded data. |
| Seqiong → Jilong Port | Unknown | D. UNKNOWN / NOT PUBLICLY AVAILABLE in the downloaded data. |

The currently documented model entry was not moved to manufacture timing agreement. In V2 the finite inlet plane is the same river transect, 29.9 m upstream in the aligned 64 m grid.

## Usable timing differences

The possibly informative ~10:55 → ~10:59 difference is approximately 240 s, but it is **not usable as a spatially resolved D-Claw target yet**: Seqiong has no recovered coordinate and 10:55 is a data interruption rather than an exact arrival pick. The V2 calculation therefore reports station diagnostics as unavailable, not as zero or a fabricated route estimate.

The simulation must still be physically compatible with the approximately 420 s whole source-to-port reconstruction. Since the downstream inlet begins after high-elevation initiation, this does not determine a unique inlet-to-port time; a simulated 1800 s downstream travel time is nevertheless strongly inconsistent with the later whole-chain reconstruction.
