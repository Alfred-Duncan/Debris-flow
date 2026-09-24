"""Layout-only rebuild of manuscript figures from completed static artifacts.

No model code, checkpoints, evaluator, rollout, inference, or training is used.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[2]; PAPER=ROOT/'paper_results'
COLORS=['#4C78A8','#F58518','#54A24B','#E45756']
def out(fig,folder,name,svg=False):
    folder.mkdir(parents=True,exist_ok=True); fig.savefig(folder/f'{name}.png',dpi=360,bbox_inches='tight',facecolor='white');fig.savefig(folder/f'{name}.pdf',bbox_inches='tight',facecolor='white')
    if svg:fig.savefig(folder/f'{name}.svg',bbox_inches='tight',facecolor='white')
    plt.close(fig)
def labels(ax,title,x='Budget (%)',y='Error'):
    if title == 'Runtime per case (s)' and y == 'Error': y='Runtime (s)'
    ax.set_title(title,fontsize=11,pad=8,fontweight='semibold');ax.set_xlabel(x,fontsize=9);ax.set_ylabel(y,fontsize=9);ax.grid(axis='y',alpha=.22);ax.tick_params(labelsize=8)
def short(m):
    return {'trajectory_h_rel_l2':'Trajectory h RelL2','trajectory_momentum_rel_l2':'Trajectory momentum RelL2','change_region_h_rel_l2':'Change-region h RelL2','change_region_momentum_rel_l2':'Change-region momentum RelL2','mixture_volume_relative_error':'Volume relative error','debris_front_mae_km':'Front MAE (km)','mean_wet_iou':'Mean wet IoU','false_positive_wet_fraction':'False-positive wet fraction','case_wall_runtime_seconds':'Runtime per case (s)'}.get(m,m)
def panel_bars(data, methods, metrics, folder, name, title, svg=False):
    fig,axs=plt.subplots(2,2,figsize=(10,6.6),constrained_layout=True); x=np.arange(len(methods))
    for ax,m in zip(axs.flat,metrics):
        vals=[data.loc[data.method.eq(k),m].iloc[0] for k in methods];ax.bar(range(len(methods)),vals,color=COLORS[:len(methods)])
        ax.set_xticks(range(len(methods)),[k.replace('_B10','').replace('RiskTemporal','Risk Temporal') for k in methods],rotation=18,ha='right',fontsize=8);labels(ax,short(m),x='',y=short(m));
        if max(vals)/max(min(vals),1e-12)>30:ax.set_yscale('log')
    fig.suptitle(title,fontsize=14,fontweight='bold');out(fig,folder,name,svg)
def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    figdir=PAPER/'figures'; final=PAPER/'final_figures'; budget=pd.read_csv(PAPER/'budget_sensitivity'/'risktemporal_b05_b10_b20.csv').sort_values('budget_fraction'); xs=budget.budget_fraction*100
    # Legacy budget: readable 2x4 grid; no title/x-label collisions.
    legacy=['trajectory_h_rel_l2','trajectory_momentum_rel_l2','mixture_volume_relative_error','debris_front_mae_km','mean_wet_iou','false_positive_wet_fraction','case_wall_runtime_seconds']
    fig,axs=plt.subplots(2,4,figsize=(13,6.8),constrained_layout=True)
    for ax,m in zip(axs.flat,legacy):ax.plot(xs,budget[m],'-o',color=COLORS[0],lw=2,ms=5);labels(ax,short(m));ax.set_xticks(xs,['B05','B10','B20'])
    axs.flat[-1].axis('off');fig.suptitle('RiskTemporal-v1 budget sensitivity (completed VAL)',fontsize=15,fontweight='bold');out(fig,figdir,'figure_A_budget_sensitivity')
    # Final budget focuses on engineering trade-offs.
    finalm=['trajectory_h_rel_l2','trajectory_momentum_rel_l2','change_region_h_rel_l2','mixture_volume_relative_error','debris_front_mae_km','case_wall_runtime_seconds']
    fig,axs=plt.subplots(2,3,figsize=(11.5,6.8),constrained_layout=True)
    for ax,m in zip(axs.flat,finalm):ax.plot(xs,budget[m],'-o',color=COLORS[0],lw=2,ms=5);labels(ax,short(m));ax.set_xticks(xs,['B05','B10','B20'])
    fig.suptitle('Budget sensitivity: B10 is the selected engineering operating point',fontsize=14,fontweight='bold');out(fig,final,'figure_02_budget_sensitivity',True)
    test=pd.read_csv(PAPER/'test'/'test_summary.csv');methods=['FrozenGlobal','RiskTemporal_B10']
    panel_bars(test,methods,['trajectory_h_rel_l2','trajectory_momentum_rel_l2','mixture_volume_relative_error','debris_front_mae_km'],figdir,'figure_B_primary_test_comparison','Untouched TEST: Frozen Global and RiskTemporal-v1 B10')
    fail=pd.read_csv(PAPER/'failure_cases'/'failure_mechanism_summary.csv'); fm=['EngineeringROI_B10','DynamicOnly_B10','SupportRiskOnly_B10']
    panel_bars(fail,fm,['trajectory_h_rel_l2','trajectory_momentum_rel_l2','mixture_volume_relative_error','debris_front_mae_km'],figdir,'figure_C_failure_mechanism','Development failure mechanisms (completed VAL)')
    abl=pd.read_csv(PAPER/'ablations'/'supportrisk_temporal_depthguard.csv'); am=['SupportRisk_Base_B10','SupportRisk_Temporal_B10','SupportRisk_DepthGuard_B10']
    fig,axs=plt.subplots(1,3,figsize=(12,4.3),constrained_layout=True)
    for ax,m in zip(axs,['change_region_h_rel_l2','mixture_volume_relative_error','debris_front_mae_km']):
        vals=[abl.loc[abl.method.eq(k),m].iloc[0] for k in am];ax.bar(range(3),vals,color=COLORS[:3]);ax.set_xticks(range(3),['Support\nBase','Temporal\nRefresh','Depth\nGuard'],fontsize=8);labels(ax,short(m),x='',y=short(m))
    fig.suptitle('Support-risk refinement variants (completed VAL)',fontsize=14,fontweight='bold');out(fig,figdir,'figure_D_temporal_ablation')
    # Runtime accuracy legacy and per-case robustness.
    fig,axs=plt.subplots(1,2,figsize=(10,4.2),constrained_layout=True)
    for ax,m in zip(axs,['mixture_volume_relative_error','debris_front_mae_km']):
        ax.scatter(budget.case_wall_runtime_seconds,budget[m],s=60,color=COLORS[0]);
        for _,r in budget.iterrows():ax.annotate(f"B{int(r.budget_fraction*100):02d}",(r.case_wall_runtime_seconds,r[m]),xytext=(5,5),textcoords='offset points',fontsize=8)
        labels(ax,short(m),x='Runtime per case (s)',y=short(m))
    fig.suptitle('Budget runtime/accuracy trade-off',fontsize=14,fontweight='bold');out(fig,figdir,'figure_F_runtime_accuracy')
    cases=pd.read_csv(PAPER/'test'/'test_case_metrics.csv'); piv=cases.pivot(index='scenario_id',columns='method',values='trajectory_h_rel_l2');fig,ax=plt.subplots(figsize=(5.4,5),constrained_layout=True);ax.scatter(piv.FrozenGlobal,piv.RiskTemporal_B10,color=COLORS[0]);lim=max(piv.max())*1.05;ax.plot([0,lim],[0,lim],'--',color='#666',lw=1);ax.set(xlim=(0,lim),ylim=(0,lim),xlabel='Frozen Global trajectory h RelL2',ylabel='RiskTemporal B10 trajectory h RelL2');ax.grid(alpha=.22);fig.suptitle('Case-level trajectory h comparison',fontsize=13,fontweight='bold');out(fig,figdir,'figure_G_case_robustness')
    # Final Figure 01: arrows occupy their own gaps, never cross text.
    fig,ax=plt.subplots(figsize=(13,3.2));ax.set_axis_off();boxes=[('Frozen\nglobal operator','#D7E8F5'),('Support-risk\nROI ranking','#FDE7BD'),('Temporal refresh\n+ guards','#D8EFD5'),('Budgeted local\ncorrection','#F6D2D2')]; positions=[.05,.30,.55,.80]
    for (text,color),x in zip(boxes,positions):ax.add_patch(FancyBboxPatch((x,.32),.16,.34,boxstyle='round,pad=.018,rounding_size=.03',fc=color,ec='#444',lw=1.4,transform=ax.transAxes));ax.text(x+.08,.49,text,ha='center',va='center',fontsize=12,transform=ax.transAxes)
    for x in [.21,.46,.71]:ax.annotate('',xy=(x+.075,.49),xytext=(x,.49),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',lw=1.7,color='#444'))
    ax.set_title('Frozen RiskTemporal-v1 B10 inference path',fontsize=17,fontweight='bold',pad=18);out(fig,final,'figure_01_method_architecture',True)
    # Final paired improvements: clear legend outside plotting area.
    imp=pd.read_csv(PAPER/'statistics'/'test_case_improvements_final.csv'); metrics=['trajectory_h_rel_l2','false_positive_wet_fraction','final_wet_iou'];fig,axs=plt.subplots(1,3,figsize=(12.5,3.8),constrained_layout=True)
    for ax,m in zip(axs,metrics):
        for base,col in zip(['FrozenGlobal','RandomAll_B10','SupportRisk_Base_B10'],COLORS[:3]):
            v=imp[(imp.baseline==base)&(imp.metric==m)].improvement.to_numpy();ax.scatter(np.arange(1,len(v)+1),v,s=20,label=base.replace('_B10',''),color=col)
        ax.axhline(0,color='#555',lw=.8);labels(ax,short(m),x='TEST case',y='Paired improvement')
    handles,labs=axs[-1].get_legend_handles_labels();fig.legend(handles,labs,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.10),fontsize=8);fig.suptitle('Case-paired untouched TEST improvements',fontsize=14,fontweight='bold');out(fig,final,'figure_03_paired_improvements',True)
    # Final runtime trade-off: labels offset and no in-plot legend.
    summ=pd.read_csv(PAPER/'test'/'test_summary_all_methods.csv');order=['FrozenGlobal','RandomAll_B10','SupportRisk_Base_B10','RiskTemporal_B10'];summ=summ.set_index('method').loc[order].reset_index();fig,axs=plt.subplots(1,3,figsize=(13,3.9),constrained_layout=True)
    for ax,m in zip(axs,['debris_front_mae_km','change_region_h_rel_l2','trajectory_momentum_rel_l2']):
        for i,r in summ.iterrows():ax.scatter(r.case_wall_runtime_seconds,r[m],s=70,color=COLORS[i]);ax.annotate(r.method.replace('_B10','').replace('SupportRisk_Base','Support base'), (r.case_wall_runtime_seconds,r[m]),xytext=(5,5),textcoords='offset points',fontsize=7)
        labels(ax,short(m),x='Runtime per case (s)',y=short(m))
    fig.suptitle('Untouched TEST runtime and engineering trade-offs',fontsize=14,fontweight='bold');out(fig,final,'figure_04_runtime_tradeoff',True)
    # Figure 06: GridSpec gives the shared colorbar a dedicated column.
    fields=PAPER/'h0'/'fields';times=[240,480,720,960,1200,1440];keys=[('reference','Reference'),('FrozenGlobal','Frozen Global'),('RiskTemporal_B10','RiskTemporal B10')];arr=[np.load(fields/f'{k}_t{t:04d}s.npz')['h'] for t in times for k,_ in keys];vmax=np.nanpercentile(np.concatenate([a.ravel() for a in arr]),99.5)
    fig=plt.figure(figsize=(10.4,15.8),layout='constrained');gs=fig.add_gridspec(6,4,width_ratios=[1,1,1,.065]);im=None
    for i,t in enumerate(times):
        for j,(k,label) in enumerate(keys):
            ax=fig.add_subplot(gs[i,j]);a=np.load(fields/f'{k}_t{t:04d}s.npz')['h'];im=ax.imshow(np.ma.masked_less_equal(a,1e-4),vmin=0,vmax=vmax,cmap='viridis');ax.set_axis_off();
            if i==0:ax.set_title(label,fontsize=10,pad=6)
            if j==0:ax.text(-.10,.5,f'{t} s',rotation=90,transform=ax.transAxes,ha='center',va='center',fontsize=9)
    cax=fig.add_subplot(gs[:,3]);fig.colorbar(im,cax=cax,label='Flow depth h (m), shared scale');fig.suptitle('H0 field evolution against documented numerical reference',fontsize=14,fontweight='bold');out(fig,final,'figure_06_h0_field_evolution',True)
    # Final station-arrival limitation: concise station labels and an external legend.
    station=pd.read_csv(PAPER/'h0'/'stations'/'h0_station_metrics.csv'); groups=list(station.method.unique()); stations=list(station.station.unique()); x=np.arange(len(stations)); width=.72/max(len(groups),1)
    fig,ax=plt.subplots(figsize=(9.4,4.2),constrained_layout=True)
    for i,m in enumerate(groups):
        g=station[station.method.eq(m)].set_index('station').loc[stations];ax.bar(x+(i-(len(groups)-1)/2)*width,g.arrival_error_s,width,label=m.replace('_B10',''),color=COLORS[i])
    ax.set_xticks(x,[s.replace('_','\n') for s in stations],fontsize=8);labels(ax,'H0 station arrival limitation',x='',y='Arrival error (s)');ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),frameon=False,fontsize=8);out(fig,final,'figure_08_h0_station_arrival',True)
    # Existing H0 products rebuilt with exterior legends.
    h0=pd.read_csv(PAPER/'h0'/'h0_summary.csv');fig,axs=plt.subplots(1,2,figsize=(9,3.8),constrained_layout=True)
    for ax,m in zip(axs,['debris_front_mae_km','mixture_volume_relative_error']):ax.bar(h0.method,h0[m],color=COLORS[:len(h0)]);labels(ax,short(m),x='',y=short(m));ax.tick_params(axis='x',rotation=20)
    fig.suptitle('H0 engineering summary',fontsize=14,fontweight='bold');out(fig,figdir,'figure_H_h0_field_comparison')
    front=pd.read_csv(PAPER/'h0'/'h0_debris_front_evolution.csv');fig,ax=plt.subplots(figsize=(7.2,4),constrained_layout=True)
    for i,(m,g) in enumerate(front.groupby('method',sort=False)):ax.plot(g.time_s,g.debris_front_km,'-o',label=m,color=COLORS[i])
    labels(ax,'H0 debris-front propagation',x='Time (s)',y='Debris-front chainage (km)');ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),frameon=False,fontsize=8);out(fig,figdir,'figure_H_h0_front_evolution')
    hydro=pd.read_csv(PAPER/'h0'/'stations'/'station_hydrographs.csv');col='gyirong_cctv_arrival_Q';fig,ax=plt.subplots(figsize=(7.2,4),constrained_layout=True)
    for i,(m,g) in enumerate(hydro.groupby('method')):ax.plot(g.time_s,g[col],label=m,color=COLORS[i])
    labels(ax,'H0 station hydrograph',x='Time (s)',y='Section discharge proxy');ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),frameon=False,fontsize=8);out(fig,figdir,'figure_H_h0_hydrograph')
    volume=pd.read_csv(PAPER/'h0'/'metrics'/'volume_evolution_proxy.csv');fig,ax=plt.subplots(figsize=(7.2,4),constrained_layout=True)
    for i,(m,g) in enumerate(volume.groupby('method')):ax.plot(g.time_s,g.wet_volume_proxy_m3/1e6,label=m,color=COLORS[i])
    labels(ax,'H0 wet-volume evolution',x='Time (s)',y='Wet-volume proxy (Mm³)');ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),frameon=False,fontsize=8);out(fig,figdir,'figure_H_h0_volume_evolution')
if __name__=='__main__':main()
