import csv, math
from pathlib import Path
C=Path(__file__).resolve().parents[1]
V,T,W,g=2e6,90.,192.,9.81
q=.5*math.pi*V/T; h=((q/W)**2/g)**(1/3); u=math.sqrt(g*h)
rows=[dict(K_u=k,Q_peak_m3s=q,h_ref_peak_m=h,u_peak_ms=k*u,relative_momentum_factor=k,relative_kinetic_energy_per_mass_factor=k*k) for k in (1.,1.5,2.,3.,4.)]
with (C/'results/MOMENTUM_SOURCE_SCENARIOS.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
