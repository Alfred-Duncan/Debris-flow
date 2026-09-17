"""Create visual-only human-review materials for the upstream Gate A anomaly."""
from __future__ import annotations

import csv
import sys

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from shapely.geometry import Point
from shapely.ops import substring

sys.path.insert(0, r"E:\Alfred\Jilong_GateA\scripts")
from finalize_gate_a import (RAW, FIG, REVIEW, DOWNLOADS, CRS, PORT, OLD_ENTRY,
                             show_image, read_dem, hillshade, decorate)

S0 = Point(334051.79369854234, 3143392.492568097)
S4 = Point(340571.8257777202, 3129640.775894154)
ANOMALY = Point(337303.8418827355, 3139798.3985810857)


def river_parts():
    options=[]
    for path in (RAW / "basic_geography").rglob("*.shp"):
        frame=gpd.read_file(path).to_crs(CRS)
        if set(frame.geom_type).issubset({"LineString","MultiLineString"}):
            options.append((frame.geometry.union_all().distance(PORT),frame))
    river=min(options,key=lambda item:item[0])[1]
    parts=[p for geom in river.geometry for p in (geom.geoms if geom.geom_type=="MultiLineString" else [geom])]
    return river, sorted(parts,key=lambda line:line.distance(OLD_ENTRY))[0]


def find_rise(upstream):
    start_distance=upstream.project(S0)
    segment=substring(upstream,start_distance,upstream.length)
    distances=np.arange(0.,segment.length+12.5,12.5); distances[-1]=segment.length
    points=[segment.interpolate(float(d)) for d in distances]
    dem=next((RAW/'dem_12p5m').rglob('*.tif'))
    with rasterio.open(dem) as ds:
        z=np.array([v[0] for v in ds.sample([(p.x,p.y) for p in points])],float)
        valid=np.isfinite(z)&(z!=ds.nodata)
    z[~valid]=np.nan
    rises=z[20:]-z[:-20]
    # Restrict to the previously documented direct-contradiction vicinity.
    candidates=[i for i in range(len(rises)) if points[i].distance(ANOMALY)<400]
    idx=max(candidates,key=lambda i:rises[i])
    return points[idx], points[idx+20], float(z[idx]), float(z[idx+20])


def draw_official(ax, river, rise_start, rise_end, use_labels=True):
    first=True
    for geom in river.geometry:
        for line in (geom.geoms if geom.geom_type=="MultiLineString" else [geom]):
            ax.plot(*line.xy,color="#00d9ff",lw=1.5,label="OFFICIAL VECTOR UNDER REVIEW" if first else "_nolegend_")
            first=False
    ax.plot([rise_start.x,rise_end.x],[rise_start.y,rise_end.y],color="#ff3b30",lw=3,label="250 m adverse-rise interval")
    ax.scatter([ANOMALY.x],[ANOMALY.y],color="red",edgecolors="white",s=70,zorder=6,label="ANOMALY")
    ax.scatter([rise_start.x,rise_end.x],[rise_start.y,rise_end.y],color="#ffb000",edgecolors="black",s=48,zorder=6)
    if use_labels:
        ax.annotate("RISE_START",xy=(rise_start.x,rise_start.y),xytext=(8,8),textcoords="offset points",color="white",weight="bold",fontsize=8)
        ax.annotate("RISE_END",xy=(rise_end.x,rise_end.y),xytext=(8,-12),textcoords="offset points",color="white",weight="bold",fontsize=8)


def main():
    FIG.mkdir(parents=True,exist_ok=True); REVIEW.mkdir(parents=True,exist_ok=True)
    river, upstream=river_parts()
    rise_start,rise_end,z_start,z_end=find_rise(upstream)
    # Adjacent official vertices bracket the suspicious vector piece.
    coords=list(upstream.coords)
    nearest=min(range(len(coords)),key=lambda i:Point(coords[i]).distance(ANOMALY))
    upstream_end=Point(coords[max(0,nearest-1)])
    downstream_start=Point(coords[min(len(coords)-1,nearest+1)])
    pre=next(DOWNLOADS.rglob('05m.tif')); post=next(DOWNLOADS.rglob('*composite.tif'))
    dem=next((RAW/'dem_12p5m').rglob('*.tif'))
    close=(ANOMALY.x-750,ANOMALY.y-750,ANOMALY.x+750,ANOMALY.y+750)
    medium=(ANOMALY.x-1500,ANOMALY.y-1500,ANOMALY.x+1500,ANOMALY.y+1500)
    context=(ANOMALY.x-2500,ANOMALY.y-2500,ANOMALY.x+2500,ANOMALY.y+2500)

    # 21: medium image review.
    fig,ax=plt.subplots(figsize=(11,11));show_image(ax,pre,medium,"Upstream geometry QA failure v2 — pre-event 0.5 m DOM")
    draw_official(ax,river,rise_start,rise_end);ax.annotate("UPSTREAM / toward S0",xy=(ANOMALY.x-750,ANOMALY.y+750),xytext=(ANOMALY.x-1180,ANOMALY.y+1180),color="white",arrowprops={"arrowstyle":"->","color":"white"},weight="bold")
    ax.annotate("DOWNSTREAM / toward S4",xy=(ANOMALY.x+500,ANOMALY.y-500),xytext=(ANOMALY.x+800,ANOMALY.y-1100),color="white",arrowprops={"arrowstyle":"->","color":"white"},weight="bold")
    ax.legend(loc="upper left",fontsize=8);decorate(ax,medium);fig.savefig(FIG/'21_upstream_geometry_QA_failure_v2.png',dpi=230);plt.close(fig)

    # 22: uncluttered close inspection.
    fig,ax=plt.subplots(figsize=(11,11));show_image(ax,pre,close,"Upstream anomaly — close pre-event 0.5 m DOM")
    draw_official(ax,river,rise_start,rise_end,use_labels=False)
    ax.scatter([upstream_end.x,downstream_start.x],[upstream_end.y,downstream_start.y],color=["yellow","lime"],edgecolors="black",s=58,zorder=6)
    ax.annotate("UPSTREAM_SEGMENT_END",xy=(upstream_end.x,upstream_end.y),xytext=(8,8),textcoords="offset points",color="yellow",weight="bold",fontsize=8)
    ax.annotate("DOWNSTREAM_SEGMENT_START",xy=(downstream_start.x,downstream_start.y),xytext=(8,-12),textcoords="offset points",color="lime",weight="bold",fontsize=8)
    decorate(ax,close);fig.savefig(FIG/'22_upstream_anomaly_pre_event_close.png',dpi=250);plt.close(fig)

    # 23: system context.
    fig,ax=plt.subplots(figsize=(12,12));show_image(ax,pre,context,"Upstream anomaly — pre-event 0.5 m DOM context")
    draw_official(ax,river,rise_start,rise_end);ax.scatter([S0.x],[S0.y],color="red",s=55,label="S0")
    ax.annotate("toward S4",xy=(ANOMALY.x+900,ANOMALY.y-900),xytext=(ANOMALY.x+1500,ANOMALY.y-1600),color="white",arrowprops={"arrowstyle":"->","color":"white"},weight="bold")
    ax.legend(loc="upper left",fontsize=8);decorate(ax,context);fig.savefig(FIG/'23_upstream_anomaly_pre_event_context.png',dpi=220);plt.close(fig)

    # 24: terrain-only review.
    z,db=read_dem(dem,medium);fig,ax=plt.subplots(figsize=(11,11));ax.imshow(hillshade(z),extent=(db[0],db[2],db[1],db[3]),cmap='gray')
    draw_official(ax,river,rise_start,rise_end);ax.text(.02,.02,f"Native 12.5 m DEM\nRISE_START: {z_start:.0f} m\nRISE_END: {z_end:.0f} m",transform=ax.transAxes,color='white',bbox={'facecolor':'black','alpha':.7})
    ax.legend(loc='upper left',fontsize=8);ax.set_title('Upstream anomaly — native 12.5 m DEM hillshade');decorate(ax,medium);fig.savefig(FIG/'24_upstream_anomaly_hillshade.png',dpi=230);plt.close(fig)

    # 25: clean paired multi-evidence review; no proposed trace.
    fig,axes=plt.subplots(1,2,figsize=(20,10));show_image(axes[0],pre,medium,'Pre-event 0.5 m DOM');draw_official(axes[0],river,rise_start,rise_end);axes[0].legend(loc='upper left',fontsize=7);decorate(axes[0],medium)
    axes[1].imshow(hillshade(z),extent=(db[0],db[2],db[1],db[3]),cmap='gray');draw_official(axes[1],river,rise_start,rise_end);axes[1].legend(loc='upper left',fontsize=7);axes[1].set_title('Native 12.5 m DEM valley context');decorate(axes[1],medium)
    fig.tight_layout();fig.savefig(FIG/'25_upstream_anomaly_multievidence.png',dpi=220);plt.close(fig)

    # 26: produce only when the source overlaps the AOI; state cloud limitation.
    fig,ax=plt.subplots(figsize=(11,11));covered=show_image(ax,post,medium,'Upstream anomaly — post-event PlanetScope 3 m')
    if covered:
        draw_official(ax,river,rise_start,rise_end);ax.text(.02,.02,'POST-EVENT VIEW PARTLY/HEAVILY CLOUD OBSCURED',transform=ax.transAxes,color='white',bbox={'facecolor':'black','alpha':.75})
        ax.legend(loc='upper left',fontsize=8);decorate(ax,medium);fig.savefig(FIG/'26_upstream_anomaly_post_event_3m.png',dpi=220)
    plt.close(fig)

    points=[
      ('ANOMALY',ANOMALY,'documented upstream vector/terrain contradiction'),
      ('RISE_START',rise_start,'start of sampled 250 m adverse-rise interval'),
      ('RISE_END',rise_end,'end of sampled 250 m adverse-rise interval'),
      ('UPSTREAM_SEGMENT_END',upstream_end,'official vertex bracketing suspicious segment upstream'),
      ('DOWNSTREAM_SEGMENT_START',downstream_start,'official vertex bracketing suspicious segment downstream')]
    with (REVIEW/'UPSTREAM_ANOMALY_REVIEW_POINTS.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['point_id','x','y','role','description']);w.writeheader()
        for point_id,point,description in points:w.writerow({'point_id':point_id,'x':point.x,'y':point.y,'role':point_id,'description':description})
    (REVIEW/'UPSTREAM_ANOMALY_REVIEW_README.txt').write_text(
      'Human reviewer should inspect Figures 21–26 and answer:\n1. Does the official vector visibly leave the physical channel?\n2. What visible channel should be followed instead?\n3. Which visible upstream point should anchor the correction?\n4. Which visible downstream point should anchor the correction?\n5. Is the route unambiguous enough for manual digitization?\n',encoding='utf-8')
    # ASCII range wording avoids inherited encoding damage in the pasted task text.
    (REVIEW/'UPSTREAM_ANOMALY_REVIEW_README.txt').write_text(
      'Human reviewer should inspect Figures 21 through 26 and answer:\n'
      '1. Does the official vector visibly leave the physical channel?\n'
      '2. What visible channel should be followed instead?\n'
      '3. Which visible upstream point should anchor the correction?\n'
      '4. Which visible downstream point should anchor the correction?\n'
      '5. Is the route unambiguous enough for manual digitization?\n', encoding='utf-8')
    print('POST_EVENT_COVERAGE = ' + ('PARTIAL' if covered else 'NO'))

if __name__=='__main__':main()
