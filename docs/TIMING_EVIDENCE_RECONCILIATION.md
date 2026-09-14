# Timing-evidence reconciliation: 7 min versus 30 min

## Scope

This note resolves terminology only for the planned **downstream, post-main-river-entry** modelling stage.  It does not create a hydrograph, select a velocity, change a D-Claw case, or treat either travel-time number as a measured boundary condition.

## Exact wording and interpretation of the reported figures

| Number | Exact source wording | Publication date and evidence priority | Physical stage and stated endpoints | Measurement / reconstruction status | Appropriate use for the downstream model |
|---|---|---|---|---|---|
| about 22 km | “沿约22公里深切沟谷高速运动” | 2026-09-01. Priority 1: ITP/CAS official summary of the peer-reviewed *Science Bulletin* reconstruction, [S1]. | Complete high-elevation ice-rock-avalanche--debris-flow chain: the immediately preceding clause starts with the mass descending from the approximately 5,200 m source; the narrative ends at Jilong Port. | **B. PEER-REVIEWED RECONSTRUCTION / ESTIMATE.** The source describes a multi-source quantitative reconstruction, not a surveyed downstream reach length. | **Context only.** It is not a main-river-entry-to-port route length. |
| about 7 min | “最终于约7分钟内抵达并冲击吉隆口岸。” | 2026-09-01. Priority 1: ITP/CAS official summary of the peer-reviewed *Science Bulletin* reconstruction, [S1]. | The same complete chain: high-elevation failure, descent/entrainment through the approximately 22 km incised valley, transformation near approximately 4,000 m, then impact at Jilong Port. | **B. PEER-REVIEWED RECONSTRUCTION / ESTIMATE.** It is a reconstructed whole-chain elapsed time, not an arrival pick at a defined main-river entry section. | **Context only** for this downstream-only domain; it may be relevant to a future full-chain reconstruction. |
| about 15 km | “距离约15公里” | 2026-08-27. Priority 1: NCDC/NIEER CAS official emergency rapid assessment, [S2]. | The sentence explicitly defines the interval as from debris flow “进入主河” to Jilong Port.  Neither the entry coordinate nor cross-section is published. | **C. SECONDARY ESTIMATE.** NCDC says the assessment was based on remote-sensing images and eyewitness video; route-picking method and uncertainty are not published. | **Soft consistency target only.** It describes the correct downstream stage but cannot define an exact model-entry location. |
| about 30 min | “本次灾害从泥石流进入主河到吉隆口岸，演进时间约30分钟” | 2026-08-27. Priority 1: NCDC/NIEER CAS official emergency rapid assessment, [S2]. | Downstream propagation after the debris flow entered the main river, ending at Jilong Port. | **C. SECONDARY ESTIMATE.** A rapid assessment, not a direct gauge/video time pair tied to a published entry cross-section. | **Soft consistency target only.** It is the only public time explicitly assigned to the downstream stage. |
| about 8.3 m s^-1 | “平均速度约8.3m/s” | 2026-08-27. Priority 1: NCDC/NIEER CAS official emergency rapid assessment, [S2]. | Same “entered main river” to Jilong Port interval as the 15 km and 30 min figures. | **C. SECONDARY ESTIMATE.** It is reported as a reach average and is arithmetically consistent with the rounded 15 km / 30 min figures; it is not an entry velocity. | **Soft consistency target only** for future reach-average propagation.  It must not initialize `u`, `v`, or momentum. |
| about 3–5 × 10^4 m^3 s^-1 | “吉隆口岸断面泥石流峰值流量约3-5万立方米每秒。” | 2026-08-27. Priority 1: NCDC/NIEER CAS official emergency rapid assessment, [S2]. | Jilong Port cross-section at the event peak; it is not labelled as an entry discharge. | **C. SECONDARY ESTIMATE.** No rating curve, time stamp, section geometry, hydrograph, or uncertainty distribution is public in the cited page. | **Soft consistency target only** for a future port-peak diagnostic.  It cannot yield `Q(t)` or an entry volume. |

### China Geological Survey check

The 2026-09-08 China Geological Survey / Ministry of Natural Resources report provides a spatial-process corroboration but no replacement value for any of the six numbers.  Its relevant wording is: “错坚河发生了高位冰崩碎屑流，铲刮沟道冰碛物，沿着沟道向下游的郭巴峡曲、东林藏布沟高速运动，冲击了吉隆口岸。” [S3]  It supports treating the event as a connected tributary--main-river--port pathway, but does not publish an entry cross-section, timing pick, travel distance, or discharge.

## Why the figures do not conflict numerically

They are not measurements of the same start/end interval.  The later peer-reviewed reconstruction assigns approximately 22 km and approximately 7 min to the **full source-to-port hazard chain**.  The earlier NCDC rapid assessment assigns approximately 15 km, approximately 30 min, and approximately 8.3 m s^-1 to the **post-entry downstream stage**.  The public material does not explain why the earlier rapid estimate is slower, nor does it georeference “entering the main river.”  It is therefore not valid to subtract one duration from the other, average them, infer a phase velocity, or move a model entry point to force either reported distance.

### Recommended downstream timing constraint

**A. Superseded for Phase 2C.**

The qualification is essential: it is a **soft consistency target**, not a hard calibration target.  Phase 2C uses the later whole-chain reconstruction as the higher-priority event context: approximately 7 min is source-to-port, not model-entry-to-port.  The about-30-min rapid assessment remains historical, lower-confidence context only and is not the primary timing-calibration target.  A potentially useful differential anchor (Seqiong data interruption about 10:55 to port impact about 10:59) cannot be used until the station is independently geolocated; the interruption is not an exact debris-front pick.

## Sources

- **[S1]** Institute of Tibetan Plateau Research, Chinese Academy of Sciences, [Joint research team systematically reveals the amplification mechanism of the Gyirong “8·26” cross-border debris-flow disaster](https://itpcas.cas.cn/new_kycg/new_kyjz/202609/t20260907_8279155.html), published 2026-09-01.  It identifies the associated peer-reviewed paper as Guo et al., *Science Bulletin*, DOI: [10.1360/CSB-2026-1255](https://doi.org/10.1360/CSB-2026-1255).
- **[S2]** National Cryosphere Desert Data Center / Northwest Institute of Eco-Environment and Resources, CAS, [Emergency response for the Gyirong County debris-flow disaster](https://nieer.cas.cn/xwdt/kydt/202608/t20260827_8265995.html), published 2026-08-27.
- **[S3]** China Geological Survey, Ministry of Natural Resources, [China Geological Survey supports post-disaster geological-hazard investigation and assessment in Gyirong](https://www.cgs.gov.cn/ywdt/ddyw/202609/t20260908_868093.html), published 2026-09-08.
