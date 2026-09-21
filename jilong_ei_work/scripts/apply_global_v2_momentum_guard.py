"""Migrate one paused V2 continuation to the persisted TRAIN-only momentum guard."""
from __future__ import annotations
import argparse,importlib.util,json,sys
from pathlib import Path
import pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.dataset import FrameStore,feature_names,scenario_rows
from src.global_operator_v2.frame_adapter import static_and_exogenous
from src.global_operator_v2.model import JilongGlobalOperatorV2
from src.global_operator_v2.pipeline import build_scheduler,checkpoint_agreement,load_state
from src.global_operator_v2.trainer import restore_checkpoint,save_checkpoint
from src.global_operator_v2.transforms import DeltaNormalization,FeatureNormalizer,PhysicalTransform
from src.global_operator_v2.validation import validate_all_val

def load_runner():
 spec=importlib.util.spec_from_file_location('global_v2_runner',ROOT/'scripts/run_global_operator_v2_full.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner);return runner
def guarded_model(checkpoint,bounds,version,device):
 sem=checkpoint.get('model_semantics',{});delta=sem.get('delta_bounds') if sem.get('bounded_residual') else None
 model=JilongGlobalOperatorV2(len(feature_names()),**checkpoint['architecture'],delta_bounds=delta,momentum_state_bounds=bounds).to(device);model.momentum_envelope_version=version;model.delta_normalization=checkpoint.get('delta_normalization');return model
def load_with_state(path,bounds,version,state,device):
 checkpoint=torch.load(path,map_location='cpu',weights_only=False);model=guarded_model(checkpoint,bounds,version,device);opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4);sched=build_scheduler(opt,500,state.total_planned_updates);restored=restore_checkpoint(path,model,opt,sched);return model,opt,sched,restored
def main(args):
 root=Path(args.runtime_root).resolve();formal=root/'models/global_operator_v2';envelope=json.loads((formal/'momentum_state_envelope.json').read_text());bounds=[float(envelope['hu']['guard']),float(envelope['hv']['guard'])];version=envelope['envelope_version'];device=torch.device('cuda')
 runner=load_runner();state=load_state(formal/'run_state.json',runner.initial_state());rows=scenario_rows('TRAIN');subset=pd.read_csv(root/'configs/global_operator_v2_val_subset.csv');store=FrameStore(subset);transform=PhysicalTransform.from_dict(json.loads((formal/'transform.json').read_text()));normalizer=FeatureNormalizer.from_dict(json.loads((formal/'feature_normalization.json').read_text()))
 candidate_rows=[];selected=None
 for name in ('stage_a_best.pt','stage_b_best.pt','stage_c_best.pt','last.pt'):
  path=formal/name
  if not path.exists():continue
  model,opt,sched,checkpoint=load_with_state(path,bounds,version,state,device);records,summary=validate_all_val(model,store,transform,normalizer,device,rows=subset,steps=144)
  row={'checkpoint':name,'original_global_step':checkpoint['global_step'],'original_stage':checkpoint['stage_name'],**summary};candidate_rows.append(row)
  if summary.get('finite_rollout') and (selected is None or summary['J_val']<selected[0]['J_val']):selected=(row,model,opt,sched,checkpoint)
 if selected is None:raise RuntimeError('NO_FINITE_GUARDED_CANDIDATE')
 frame=pd.DataFrame(candidate_rows);report=root/'reports';report.mkdir(parents=True,exist_ok=True);frame.to_csv(report/'MOMENTUM_GUARD_CANDIDATE_REEVAL.csv',index=False)
 row,model,opt,sched,checkpoint=selected;best_path=formal/'best_candidate.pt';save_checkpoint(best_path,model,opt,sched,row['original_global_step'],transform,runner.CFG,normalizer,checkpoint['stage_name'],row['J_val'],checkpoint['stage_step'],checkpoint['stage_updates_total'],checkpoint['architecture'],str(best_path),checkpoint.get('stage_best_score'))
 last_model,last_opt,last_sched,last_checkpoint=load_with_state(formal/'last.pt',bounds,version,state,device);state.best_val_score=row['J_val'];state.best_checkpoint_path=str(best_path);runner.save_resume_pair(last_model,last_opt,last_sched,transform,normalizer,state);checkpoint_agreement(state,torch.load(formal/'last.pt',map_location='cpu',weights_only=False))
 migration={'migration':'ADD_TRAIN_ONLY_MOMENTUM_STATE_GUARD','pause_step':state.global_step,'momentum_state_bounds':bounds,'momentum_envelope_version':version,'selected_candidate':row,'checkpoint_agreement':'PASS'};(report/'GLOBAL_V2_CHECKPOINT_MIGRATION.json').write_text(json.dumps(migration,indent=2));print(json.dumps(migration,indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--runtime-root',required=True);main(parser.parse_args())
