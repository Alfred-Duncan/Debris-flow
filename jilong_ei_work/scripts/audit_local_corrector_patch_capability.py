"""VAL-only, one-step patch capability audit; intentionally contains no training."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.dataset import FrameStore,INPUT,build_features,scenario_rows
from src.global_operator_v2.frame_adapter import static_and_exogenous
from src.global_operator_v2.losses import project_physical
from src.global_operator_v2.oracle_refinement import PatchLayout,oracle_patch_scores,select_random
from src.local_corrector.model import JilongLocalCorrector
from src.local_corrector.patches import core,extract
from src.local_corrector.trainer import apply_learned_correction
from src.local_corrector.v1_1 import true_front_patch
from scripts.run_global_v2_oracle_refinement import load_model,predict
SEED=20260920; NAMES=('h','hu','hv','c','ice','dz')
def ten(x,d):return torch.from_numpy(np.asarray(x,np.float32)).unsqueeze(0).to(d)
def pick_front2(layout,truth,active,route):
    one=true_front_patch(layout,truth,active,route)
    if one is None:return []
    debris=((truth[0,0]>.1)&(truth[0,3]>.05)&active[0,0].bool()).cpu().numpy();front=route[debris].max();zone=np.isfinite(route)&(np.abs(route-front)<=1000)
    return sorted([p for p in layout.eligible if zone[p.r0:p.r1,p.c0:p.c1].any()],key=lambda p:(-int(zone[p.r0:p.r1,p.c0:p.c1].sum()),p.patch_id))[:2]
def classify(x):return 'strong_positive' if x>=.2 else 'positive' if x>0 else ('severely_harmful' if x<=-.2 else ('harmful' if x<=-.05 else 'neutral'))
def main(a):
    if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED')
    dev=torch.device('cuda');out=ROOT/'results/local_corrector_patch_audit';out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'results/local_corrector_v1_1/validation_manifest.json').read_text(encoding='utf-8-sig'));ids=manifest['patch_val_case_ids'];times=manifest['transition_times']
    rows=scenario_rows('VAL');rows=rows[rows.scenario_id.isin(ids)].sort_values('scenario_id').reset_index(drop=True)
    if len(rows)!=8 or set(rows['split'])!={'VAL'} or rows.scenario_id.str.contains('TEST|H0|HOLDOUT',case=False).any():raise RuntimeError('AUDIT_VAL_SCOPE_REQUIRED')
    global_model,tr,norm,delta,_=load_model(dev);static_np,_=static_and_exogenous(INPUT);static=ten(static_np,dev);active=static[:,1:2];layout=PatchLayout(static_np[1]);route=np.load(INPUT)['route_chainage_m']
    cn=json.loads((ROOT/'models/local_corrector_v1_1/correction_normalization.json').read_text());local_scale=torch.tensor(cn['scales'],device=dev)[None,:,None,None]
    best=torch.load(ROOT/'models/local_corrector_v1_1/best.pt',map_location='cpu',weights_only=False);c4=torch.load(ROOT/'models/local_corrector_v1_1/candidate_4000.pt',map_location='cpu',weights_only=False)
    if any(not torch.equal(best['model'][k],c4['model'][k]) for k in best['model']):raise RuntimeError('BEST_4000_MISMATCH')
    updates=[int(a.checkpoint)] if a.checkpoint else [2000,4000,6000];store=FrameStore(rows);records=[]
    for update in updates:
      model=JilongLocalCorrector(47).to(dev);model.load_state_dict(torch.load(ROOT/f'models/local_corrector_v1_1/candidate_{update}.pt',map_location=dev,weights_only=False)['model']);model.eval()
      for ci,(_,row) in enumerate(rows.iterrows()):
       params=ten(np.asarray([row[p] for p in ('volume_scale','ice_fraction','erosion_K','n_debris','dep_tau_s')],np.float32),dev)
       for t in times:
        prev=ten(store.frame(row,max(0,t-10)),dev);cur=ten(store.frame(row,t),dev);truth=ten(store.frame(row,t+10),dev);prov=predict(global_model,prev,cur,static,params,t/1440.,tr,norm,active);pt,tt,ct=tr.encode(prov),tr.encode(truth),tr.encode(cur);feat=build_features(prev,cur,static,params,torch.tensor([t/1440.],device=dev),tr,norm);scores,_,_=oracle_patch_scores(pt,tt,prov,truth,active,layout,delta.scales);eligible=list(layout.eligible)
        fp=(prov[0,0]>=.05)&(truth[0,0]<.05)&active[0,0].bool();fp_rank=sorted(eligible,key=lambda p:(-float(((prov[0,0,p.r0:p.r1,p.c0:p.c1]-truth[0,0,p.r0:p.r1,p.c0:p.c1]).abs()*fp[p.r0:p.r1,p.c0:p.c1]).sum()),-int(fp[p.r0:p.r1,p.c0:p.c1].sum()),p.patch_id))
        cats={'RANDOM':list(select_random(layout,2,SEED+ci*10000+t)),'HIGH_ERROR':[eligible[i] for i in np.argsort(-scores)[:2]],'TRUE_FRONT':pick_front2(layout,truth,active,route),'FALSE_POSITIVE_WET':[p for p in fp_rank[:2] if fp[p.r0:p.r1,p.c0:p.c1].any()]}
        median=float(np.median(scores))
        for typ,patches in cats.items():
         for p in patches:
          corrected_t,predn,sat=apply_learned_correction(model,feat,pt,ct,(p,),layout,delta.scales,cn['scales'],cn['bounds'],active,global_model);corrected=project_physical(tr.decode(corrected_t),active);m=core(extract(active,p,layout),layout)[0,0].bool();a0=core(extract(pt,p,layout),layout)[0];a1=core(extract(corrected_t,p,layout),layout)[0];target=core(extract(tt,p,layout),layout)[0];den=m.sum().item()*6
          eb=(((a0-target)/local_scale[0]).square()*m).sum().item()/max(den,1);ea=(((a1-target)/local_scale[0]).square()*m).sum().item()/max(den,1);rel=(eb-ea)/max(eb,1e-12)
          pp=core(extract(prov,p,layout),layout)[0];cp=core(extract(corrected,p,layout),layout)[0];tp=core(extract(truth,p,layout),layout)[0];wet=lambda x:x[0]>=.05;pw,cw,tw=wet(pp)&m,wet(cp)&m,wet(tp)&m;inter=lambda x,y:int((x&y).sum());fpb=int((pw&~tw).sum());fpa=int((cw&~tw).sum());perb=(((a0-target)/local_scale[0]).square()*m).sum((1,2)).sqrt()/m.float().sum().sqrt();pera=(((a1-target)/local_scale[0]).square()*m).sum((1,2)).sqrt()/m.float().sum().sqrt()
          rec={'checkpoint_update':update,'scenario_id':row.scenario_id,'time_s':t,'patch_type':typ,'patch_id':p.patch_id,'oracle_score':float(scores[eligible.index(p)]),'oracle_percentile':float((scores<=scores[eligible.index(p)]).mean()),'E_before':eb,'E_after':ea,'relative_improvement':rel,'classification':classify(rel),'wet_iou_before':inter(pw,tw)/max(inter(pw|tw,pw|tw),1),'wet_iou_after':inter(cw,tw)/max(inter(cw|tw,cw|tw),1),'wet_precision_before':inter(pw,tw)/max(int(pw.sum()),1),'wet_precision_after':inter(cw,tw)/max(int(cw.sum()),1),'wet_recall_before':inter(pw,tw)/max(int(tw.sum()),1),'wet_recall_after':inter(cw,tw)/max(int(tw.sum()),1),'fp_wet_before':fpb,'fp_wet_after':fpa,'fp_reduction':(fpb-fpa)/max(fpb,1),'active_core_cells':int(m.sum()),'truth_wet_cells':int(tw.sum()),'provisional_wet_cells':int(pw.sum()),'corrected_wet_cells':int(cw.sum())}
          for i,n in enumerate(NAMES):rec[f'{n}_before']=float(perb[i]);rec[f'{n}_after']=float(pera[i]);rec[f'mean_abs_corr_{n}']=float(predn[0,i].abs()[m].mean());rec[f'saturation_{n}']=float(sat[i])
          records.append(rec)
    df=pd.DataFrame(records);df.to_csv(out/'patch_audit_rows.csv',index=False);g=df.groupby(['checkpoint_update','patch_type']);summary=g.agg(sample_count=('patch_id','size'),positive_improvement_rate=('relative_improvement',lambda x:(x>0).mean()),strong_positive_rate=('relative_improvement',lambda x:(x>=.2).mean()),harmful_rate=('relative_improvement',lambda x:(x<=-.05).mean()),severely_harmful_rate=('relative_improvement',lambda x:(x<=-.2).mean()),mean_relative_improvement=('relative_improvement','mean'),median_relative_improvement=('relative_improvement','median'),fp_reduction=('fp_reduction','mean')).reset_index();summary.to_csv(out/'patch_audit_summary.csv',index=False);summary[summary.checkpoint_update.eq(4000)].to_csv(out/'best4000_patch_capability.csv',index=False);summary.to_csv(out/'checkpoint_patch_evolution.csv',index=False)
    best4=summary[summary.checkpoint_update.eq(4000)].set_index('patch_type');targeted=[x for x in ('HIGH_ERROR','TRUE_FRONT','FALSE_POSITIVE_WET') if x in best4.index];good=[x for x in targeted if best4.loc[x,'positive_improvement_rate']>=.65 and best4.loc[x,'median_relative_improvement']>0];cap='STRONG' if targeted and all(best4.loc[x,'positive_improvement_rate']>=.7 and best4.loc[x,'median_relative_improvement']>0 for x in targeted) and (best4.positive_improvement_rate.mean()>=.7) else ('TARGETED' if len(good)>=2 else 'WEAK');report={'best_update':4000,'patch_corrector_capability':cap,'engineering_selector_ready':'YES' if cap in ('STRONG','TARGETED') else 'NO'};(ROOT/'reports/LOCAL_CORRECTOR_PATCH_CAPABILITY.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',choices=('2000','4000','6000'));p.add_argument('--all',action='store_true');a=p.parse_args();main(a)
