"""One front-progression plot for the complete Phase-2C-fix batch."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from route_diagnostics import ROUTE_DISTANCE_M
case=Path(__file__).resolve().parent
fig,ax=plt.subplots(figsize=(8.2,5),constrained_layout=True)
for row in csv.DictReader((case/'CANDIDATE_RUNS.csv').open()):
 p=case/'runs'/row['run_id']/'front_progress.csv'
 if p.exists():
  d=list(csv.DictReader(p.open()));ax.plot([float(x['time_s']) for x in d],[float(x['front_distance_m'])/1000 for x in d],label=row['run_id'])
ax.axhline(ROUTE_DISTANCE_M/1000,color='black',ls='--',lw=1,label='hydraulic port-proxy')
ax.set(xlabel='simulation time (s)',ylabel='mapped-river front distance from entry (km)',xlim=(0,600));ax.grid(alpha=.25);ax.legend(ncol=2,fontsize=8)
fig.savefig(case.parents[2]/'outputs/phase2c_fix/front_progress_batch.png',dpi=180)
