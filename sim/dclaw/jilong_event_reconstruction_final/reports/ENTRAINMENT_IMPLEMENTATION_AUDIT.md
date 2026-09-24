# Entrainment implementation audit

The audited installed Cartesian D-Claw source sets i_dig=2 and i_ent=i_dig+5=7 (Fortran one-based). src2.f90 reads available erodible thickness as aux(i_ent,i,j) and increments q(i_bdif,i,j). The generated type-3 auxinit raster is therefore assigned to aux slot 7. E2 uses built-in entrainment=1, entrainment_method=0, entrainment_rate=0.20, and me=0.62; segregation remains disabled.
