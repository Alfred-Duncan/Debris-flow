"""Single compact Phase 2C diagnostic plot."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
case=Path(__file__).resolve().parent; rows=list(csv.DictReader((case/'CANDIDATE_RUNS.csv').open()))
labels=[r['run_id'].replace('run_','R').replace('_','\n') for r in rows]
speeds=[float(r['domain_max_speed_ms']) for r in rows]; vols=[float(r['inflow_volume'])/1e6 for r in rows]
colors=['#2a9d8f' if r['stable']=='True' else '#e76f51' for r in rows]
fig,ax=plt.subplots(1,2,figsize=(10,4.5),constrained_layout=True)
ax[0].bar(range(len(rows)),speeds,color=colors);ax[0].axhline(60,color='#555',ls='--',lw=1,label='transparent high-speed screen')
ax[0].set_xticks(range(len(rows)),[f'R{i+1}' for i in range(len(rows))]);ax[0].set_ylabel('domain maximum speed (m s$^{-1}$)');ax[0].legend(fontsize=8)
ax[1].bar(range(len(rows)),vols,color=colors);ax[1].set_xticks(range(len(rows)),[f'R{i+1}' for i in range(len(rows))]);ax[1].set_ylabel('imposed triangular-pulse volume ($10^6$ m$^3$)')
for a in ax: a.grid(axis='y',alpha=.25)
fig.suptitle('Phase 2C: all four 600-s cases did not reach Jilong Port; green=screened usable, red=high-speed rejection')
out=case.parents[2]/'outputs/phase2c/event_v2_timing.png';fig.savefig(out,dpi=180);print(out)
