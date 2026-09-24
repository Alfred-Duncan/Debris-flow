# Server environment and handoff check

Checked locally on the AutoDL server after entering the existing login-shell
D-Claw environment. A non-login shell does not export CLAW and was not used for
this verification.

| Item | Result |
|---|---|
| Python | /root/miniconda3/bin/python, Python 3.12.3 |
| CLAW | /root/src/clawpack |
| Clawpack import | PASS: /root/src/clawpack/clawpack/__init__.py |
| GeoClaw import | PASS: /root/src/clawpack/geoclaw/src/python/geoclaw/__init__.py |
| D-Claw source | /root/src/clawpack/dclaw |
| Compiler | /usr/bin/gfortran, GNU Fortran 11.4.0 |
| Build tool | /usr/bin/make, GNU Make 4.3 |
| Previous runnable executable | PASS: sim/dclaw/jilong_reference_based/xdclaw exists |
| Inventory | 12 CPU cores, 90 GB RAM, NVIDIA GeForce RTX 3080 Ti (not used by D-Claw) |

## Handoff integrity

The full local snapshot at /root/autodl-tmp/Jilong_DClaw_Handoff was verified
with the CRLF-normalized SHA256SUMS.txt manifest. All 15 payload files passed.

| File | SHA-256 | Result |
|---|---|---|
| terrain/final/jilong_copernicus_glo30_utm45_v1.tif | b74f07535e790d88fe1a64467a43475ee091df989ca609a9d4c791f0bf1b685e | PASS |
| terrain/final/jilong_copernicus_32m_v1.tif | d4929b6a85b3ed2a8115980c162209795535f578a15c5e5d48f46e4a0c024feb | PASS |
| terrain/final/jilong_copernicus_64m_v1.tif | 53a7f882464eb0757bf24e45069f370c2f8198e145d0b281eacfccf660e55cfd | PASS |
| terrain/source/Copernicus_DSM_COG_10_N28_00_E085_00_DEM.tif | 1590255a0ae7e8c1f49b277e287032a18a2e32c8e13c4c3298ed458f851cd3c7 | PASS |

The 64 m raster is EPSG:32645 with 64 m cells and bounds
(331950, 3127516, 342702, 3145500). The accepted S0 control section lies
within that raster. No terrain was altered, converted, or run at 32 m.

## Locked baseline for an approved source implementation

The local reference basis fixes rho_f=1100 kg m-3, rho_s=2700 kg m-3,
m0=0.62, mcrit=0.64, mref=0.60, kref=1e-11 m2, phi=38 deg, delta=0.01,
mu=0.005 Pa s, alpha_c=0.05, c1=1, and sigma_0=1000 Pa; entrainment is off.

This baseline is documented only; no final case was initialized. The old
speed_limit=30.0 setting is not adopted in this blocked case.
