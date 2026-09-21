"""Low-cost regressions for momentum guard migration and best metadata durability."""
from __future__ import annotations
import importlib.util,tempfile,sys
from pathlib import Path
import numpy as np,torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import src.global_operator_v2.model as model_module
from src.global_operator_v2.momentum import fit_momentum_envelope
from src.global_operator_v2.pipeline import PipelineState
from src.global_operator_v2.trainer import restore_checkpoint,save_checkpoint
from src.global_operator_v2.transforms import FeatureNormalizer,PhysicalTransform

class Fixed(nn.Module):
 def __init__(self,values):super().__init__();self.values=torch.tensor(values,dtype=torch.float32);self.weight=nn.Parameter(torch.zeros(()))
 def forward(self,x):return self.values[None,:,None,None].expand(x.shape[0],-1,x.shape[-2],x.shape[-1])+self.weight*0
class FakeFNO(nn.Module):
 def __init__(self,*args):super().__init__();self.project=nn.ModuleList((nn.Conv2d(1,1,1),nn.Conv2d(1,6,1)))
 def forward(self,x):return self.project[-1](x)
model_module._rvpi_import=lambda path:FakeFNO
JilongGlobalOperatorV2=model_module.JilongGlobalOperatorV2
def runner():
 spec=importlib.util.spec_from_file_location('runner_guard',ROOT/'scripts/run_global_operator_v2_full.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def model(bounds=None,raw=(0,0,0,0,0,0)):
 out=JilongGlobalOperatorV2(1,width=4,modes=2,depth=1,delta_bounds=[10]*6,momentum_state_bounds=bounds);out.fno=Fixed(raw);return out
def test_guard_projection_and_activation():
 current=torch.zeros(1,6,2,2);x=torch.zeros(1,1,2,2)
 plain=model((2.,3.),(0,1.,-2.,0,0,0));out=plain(x,current);assert torch.allclose(out[:,1],torch.tanh(torch.ones_like(out[:,1])/10)*10) and torch.allclose(out[:,2],torch.tanh(-2*torch.ones_like(out[:,2])/10)*10) and plain.last_momentum_guard_activation_fraction==0.
 clipped=model((2.,3.),(0,4.,-5.,0,0,0));out=clipped(x,current);assert torch.equal(out[:,1],2*torch.ones_like(out[:,1])) and torch.equal(out[:,2],-3*torch.ones_like(out[:,2])) and clipped.last_momentum_guard_activation_fraction==1.
def test_envelope_coverage():
 values=np.arange(1,10001,dtype=float);envelope=fit_momentum_envelope(values,values*2,values/10);assert envelope['hu']['coverage']>=.9999 and envelope['hv']['coverage']>=.9999
def test_checkpoint_migration_and_semantics():
 tr=PhysicalTransform(1,1,1,.05);norm=FeatureNormalizer(0,1,1,1,[0]*5,[1]*5);config={'x':1}
 with tempfile.TemporaryDirectory() as directory:
  path=Path(directory)/'old.pt';old=model();opt=torch.optim.AdamW(old.parameters(),lr=.01);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.);save_checkpoint(path,old,opt,sched,7,tr,config,norm,'STAGE_C',4.2,3,4,{'width':4,'modes':2,'depth':1})
  payload=torch.load(path,weights_only=False);payload['model'].pop('momentum_state_bounds');payload['model_semantics'].pop('momentum_state_guard');payload['model_semantics'].pop('momentum_state_bounds');torch.save(payload,path)
  unguarded=model();restore_checkpoint(path,unguarded);guarded=model((2.,3.));migration=restore_checkpoint(path,guarded);assert migration['checkpoint_migration']=='ADD_TRAIN_ONLY_MOMENTUM_STATE_GUARD'
  new=Path(directory)/'guarded.pt';save_checkpoint(new,guarded,opt,sched,7,tr,config,norm,'STAGE_C',4.2,3,4,{'width':4,'modes':2,'depth':1});same=model((2.,3.));restored=restore_checkpoint(new,same);assert restored['model_semantics']['momentum_state_guard'] is True and restored['model_semantics']['momentum_state_bounds']==[2.,3.] and 'checkpoint_migration' not in restored
def test_best_metadata_survives_rollback():
 r=runner()
 with tempfile.TemporaryDirectory() as directory:
  old=r.FORMAL;r.FORMAL=Path(directory);state=PipelineState('v','hash',architecture={'width':1},current_stage='STAGE_C',stage_step=3,global_step=7,stage_updates_total=4,best_val_score=9.)
  torch.save({'stage_name':'STAGE_C','stage_step':3,'global_step':7,'stage_updates_total':4,'architecture':{'width':1},'best_val_score':4.34,'best_checkpoint_path':'old','stage_best_score':4.34,'config_hash':'hash'},r.FORMAL/'last.pt');torch.save({'best_val_score':4.19,'best_metric':4.19},r.FORMAL/'best_candidate.pt')
  r.rollback_state_to_last_checkpoint(state);assert state.best_val_score==4.19 and state.best_checkpoint_path==str(r.FORMAL/'best_candidate.pt');r.FORMAL=old
def main():
 test_guard_projection_and_activation();test_envelope_coverage();test_checkpoint_migration_and_semantics();test_best_metadata_survives_rollback();print('PASS momentum guard')
if __name__=='__main__':main()
