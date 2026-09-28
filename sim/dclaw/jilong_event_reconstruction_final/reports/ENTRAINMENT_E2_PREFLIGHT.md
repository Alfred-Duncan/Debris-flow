# E2 entrainment preflight

- Terrain: accepted unchanged 64 m computational tt3.
- Gate-B profile: full S0-S4 authoritative profile; robust-low points use its canonical tangent, signed normal offset, and continuous segment interpolation.
- Rule: distance <= 256 m, z <= robust floor + 20 m, and S0-S4 chainage only.
- Erodible cells: **928**; area: **3801088 m2**; maximum available thickness-volume: **7602176 m3**.
- D-Claw available-erodible-thickness aux index: **7** (Fortran one-based i_ent, Cartesian i_dig=2).
- h_e is exactly 2.0 m in mask cells and 0 elsewhere. All values are finite and non-negative.
- Frozen source support remains 11 cells, hash c3cc89cf2431d82e1c8cf522c2c702510d75e4802e8eb2705feb52b8dbc2d8ea.
