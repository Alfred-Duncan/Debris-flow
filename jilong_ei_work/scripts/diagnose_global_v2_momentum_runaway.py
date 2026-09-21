"""Read-only fixed-VAL momentum-envelope diagnosis for V2 checkpoints."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2 import dataset as dataset_module
from src.global_operator_v2.dataset import FrameStore,build_features,feature_names
from src.global_operator_v2.frame_adapter import static_and_exogenous
from src.global_operator_v2.losses import project_physical
from src.global_operator_v2.model import JilongGlobalOperatorV2
from src.global_operator_v2.momentum import fit_momentum_envelope
from src.global_operator_v2.transforms import PhysicalTransform,FeatureNormalizer

SAMPLE_TIMES=tuple(sorted(set((0,10,20,30,60,120,180,240,*range(300,1441,60)))))

def tensor(value,device):return torch.from_numpy(np.asarray(value)).unsqueeze(0).to(device)
def rows_for(root,split):
 index=pd.read_csv(root/'data/scenario_index.csv');design=pd.read_csv(root/'configs/scenario_design/JILONG_EI_SCENARIOS.csv')
 return index.merge(design[['scenario_id',*dataset_module.PARAMS,'design_index']],on='scenario_id',validate='one_to_one').query('split == @split').reset_index(drop=True)
def configure(root):
 dataset_module.ROOT=root;dataset_module.INPUT=root/'data/downloads/park_v2/inputs/upper30h.npz'
def load_model(path,device,guard=None):
 checkpoint=torch.load(path,map_location='cpu',weights_only=False);semantics=checkpoint.get('model_semantics',{});delta=semantics.get('delta_bounds') if semantics.get('bounded_residual') else None
 model=JilongGlobalOperatorV2(len(feature_names()),**checkpoint['architecture'],delta_bounds=delta,momentum_state_bounds=guard).to(device);missing,unexpected=model.load_state_dict(checkpoint['model'],strict=False)
 if unexpected or set(missing)-{'momentum_state_bounds'}:raise RuntimeError(f'checkpoint load mismatch {path}: {missing} {unexpected}')
 model.eval();return model,PhysicalTransform.from_dict(checkpoint['transform']),FeatureNormalizer.from_dict(checkpoint['feature_normalizer']),checkpoint
def fit_envelope(root,train,store,static,transform):
 active=np.asarray(static[1],bool);rng=np.random.default_rng(20260920);hu=[];hv=[];speed=[]
 for _,row in train.sort_values('scenario_id').iterrows():
  for time_s in SAMPLE_TIMES:
   state=store.frame(row,time_s);wet=np.flatnonzero((active&(state[0]>=.03)).ravel())
   if wet.size>4096:wet=rng.choice(wet,4096,replace=False)
   if not wet.size:continue
   h=state[0].ravel()[wet];u=state[1].ravel()[wet];v=state[2].ravel()[wet]
   hu.append(np.abs(np.arcsinh(u/transform.hu_scale)));hv.append(np.abs(np.arcsinh(v/transform.hv_scale)));speed.append(np.sqrt((u/h)**2+(v/h)**2))
 envelope=fit_momentum_envelope(np.concatenate(hu),np.concatenate(hv),np.concatenate(speed));envelope.update({'sampling':{'times_s':list(SAMPLE_TIMES),'wet_threshold_m':.03,'max_wet_cells_per_frame':4096,'seed':20260920},'sample_count':int(sum(len(x) for x in hu))})
 return envelope
@torch.no_grad()
def diagnose_checkpoint(name,path,root,val,static,guard,envelope,device):
 model,transform,norm,checkpoint=load_model(path,device);store=FrameStore(val);active=static[:,1:2];records=[];runaways=[]
 for _,row in val.iterrows():
  index=int(np.where(store.rows.scenario_id.eq(row.scenario_id))[0][0]);previous,current,_,params,time,_=store.sample(index,0);previous,current,params=(tensor(x,device) for x in (previous,current,params));consecutive=0;first=None
  for step in range(1,145):
   features=build_features(previous,current,static,params,torch.tensor([time+(step-1)/144.],device=device),transform,norm);encoded=model(features,transform.encode(current));pred=project_physical(transform.decode(encoded),active);target=tensor(store.frame(row,step*10),device)
   mask=active.bool();wet=mask[:,0]&(pred[:,0]>=.03);hu=pred[:,1][mask[:,0]];hv=pred[:,2][mask[:,0]];eh=encoded[:,1][mask[:,0]];ev=encoded[:,2][mask[:,0]]
   if wet.any():speed=torch.sqrt((pred[:,1][wet]/pred[:,0][wet]).square()+(pred[:,2][wet]/pred[:,0][wet]).square());max_speed=float(speed.max().cpu());p99=float(torch.quantile(speed,.99).cpu())
   else:max_speed=p99=0.
   target_l2=float(torch.sqrt((target[:,1:3].square()*active).sum()).cpu());pred_l2=float(torch.sqrt((pred[:,1:3].square()*active).sum()).cpu());error_l2=float(torch.sqrt(((pred[:,1:3]-target[:,1:3]).square()*active).sum()).cpu())
   max_ehu=float(eh.abs().max().cpu());max_ehv=float(ev.abs().max().cpu());hit=max(max_ehu/guard[0],max_ehv/guard[1],max_speed/max(envelope['speed_mps']['q9999'],1e-12))>1.5;consecutive=consecutive+1 if hit else 0
   if consecutive>=3 and first is None:first=step-2
   records.append({'scenario_id':row.scenario_id,'checkpoint':name,'checkpoint_step':checkpoint.get('global_step'),'checkpoint_stage':checkpoint.get('stage_name'),'step':step,'time_s':step*10,'target_momentum_l2':target_l2,'prediction_momentum_l2':pred_l2,'momentum_error_l2':error_l2,'max_abs_hu':float(hu.abs().max().cpu()),'max_abs_hv':float(hv.abs().max().cpu()),'max_speed_mps':max_speed,'p99_speed_mps':p99,'max_abs_encoded_hu':max_ehu,'max_abs_encoded_hv':max_ehv,'encoded_hu_guard_ratio':max_ehu/guard[0],'encoded_hv_guard_ratio':max_ehv/guard[1],'speed_q9999_ratio':max_speed/max(envelope['speed_mps']['q9999'],1e-12)})
   previous,current=current,pred
  if first is not None:runaways.append({'checkpoint':name,'scenario_id':row.scenario_id,'first_runaway_step':first})
 return records,runaways
def main(args):
 root=Path(args.runtime_root).resolve();configure(root);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');formal=root/'models/global_operator_v2';train=rows_for(root,'TRAIN');val=pd.read_csv(root/'configs/global_operator_v2_val_subset.csv');train_store=FrameStore(train);static_np,_=static_and_exogenous(root/'data/downloads/park_v2/inputs/upper30h.npz');static=torch.from_numpy(static_np).unsqueeze(0).to(device)
 reference=torch.load(formal/'last.pt',map_location='cpu',weights_only=False);transform=PhysicalTransform.from_dict(reference['transform']);envelope=fit_envelope(root,train,train_store,static_np,transform);guard=[envelope['hu']['guard'],envelope['hv']['guard']];(formal/'momentum_state_envelope.json').write_text(json.dumps(envelope,indent=2))
 records=[];runaways=[]
 for name in ('stage_a_best.pt','stage_b_best.pt','stage_c_best.pt','last.pt'):
  path=formal/name
  if path.exists():
   a,b=diagnose_checkpoint(path.stem,path,root,val,static,guard,envelope,device);records.extend(a);runaways.extend(b)
 out=pd.DataFrame(records);out.to_csv(root/'results/global_operator_v2/momentum_runaway_by_case.csv',index=False)
 report={'real_runaway':bool(runaways),'first_runaway_step':min((x['first_runaway_step'] for x in runaways),default=None),'runaway_scenarios':runaways,'momentum_envelope':envelope,'checkpoints':sorted(out.checkpoint.unique().tolist()),'records':int(len(out))};(root/'reports/MOMENTUM_RUNAWAY_DIAGNOSTIC.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--runtime-root',required=True);main(parser.parse_args())
