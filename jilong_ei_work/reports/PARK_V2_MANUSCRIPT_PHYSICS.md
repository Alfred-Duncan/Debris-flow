# Park v2 manuscript physics extraction

All statements below are restricted to the revised EarthArXiv v2 manuscript (DOI `10.31223/X5250R`) and its Zenodo v2 support; no generic debris-flow parameters were substituted.

| Topic | v2 evidence and value | Evidence class |
| --- | --- | --- |
| H0 source | 35.42 Mm3 distributed on the UNOSAT source area (about 1.96 km2; about 18 m thick), 20% ice / 80% rock and no initial free water | MANUSCRIPT v2 Methods; `upper30h.json` metadata |
| Source treatment | material enters at rest; `upper_visual_5s` adds it for 30 s with `release_speed=0` | ZENODO_V2_README; ZENODO_V2_OUTPUT |
| State/density | state includes `h, hu, hv, hc, hi, H`; `rho_m=1000+1650c-1750i` kg m-3 for water/rock/ice densities 1000/2650/900 | MANUSCRIPT v2 Methods, mixture/melt equations |
| Melt conversion | melt `delta`: `hi'=hi-delta`, `hc'=hc-delta`, `h'=h-0.1delta`; rescale discharge to retain velocity | MANUSCRIPT v2 Methods, melting equations |
| Heat | mechanical-energy transfer to heat, river 12 C, rock 3 C, air/surface flux 300 W m-2, latent heat based on 334 kJ kg-1 | MANUSCRIPT v2 Methods; ZENODO_V2_OUTPUT args |
| Entrainment | K=0.0074; speed threshold 6 m/s; concentration threshold 0.7; 5 m erodible depth | MANUSCRIPT v2 Methods; ZENODO_V2_OUTPUT args |
| Resistance/deposition | mu_s=0; Manning `n_w=.0208`, `n_d=.0145`; settling below 1.833 m/s with 337 s time-scale | MANUSCRIPT v2 Methods; ZENODO_V2_OUTPUT args |
| Terrain | upper 30 m terrain is raw DSM with priority-flood/D8 conditioning, 0.0005 minimum slope; 26 x 28 km upper domain | MANUSCRIPT v2 Methods; `upper30h.json` metadata |
| Spinup/baseflow | H0 restarts a pre-event river spinup; v2 provenance publishes its hash but not the file.  The local H0 regenerates the documented 4 h spinup with the verified static input. | MANUSCRIPT v2 Methods; ZENODO_V2_PROVENANCE; local parity check |
| Timing | observational constraint: Gyirong CCTV 463 +/- 90 s.  v2 model: manuscript 30 s output 390/390/1050 s at Gyirong/Rasu/Syab; v2 5 s support 365/390/1025 s. | MANUSCRIPT v2 Results; ZENODO_V2_OUTPUT `upper5_comparison.json` |

The publication says v2 source code was not released.  Equations are sufficient for this isolated, parity-tested implementation, but not for claiming author-code identity.
