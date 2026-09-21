"""Fast regression tests for long-horizon stabilization primitives."""
from __future__ import annotations
import importlib.util
import tempfile
from pathlib import Path
from unittest.mock import patch
import sys
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.losses import project_physical,normalized_delta_huber
from src.global_operator_v2.pipeline import training_sample_for_step
from src.global_operator_v2.transforms import DeltaNormalization
from src.global_operator_v2.transforms import PhysicalTransform,FeatureNormalizer,fit_delta_normalization
import pandas as pd

def runner_module():
 spec=importlib.util.spec_from_file_location('v2_runner',ROOT/'scripts/run_global_operator_v2_full.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def test_projection():
 x=torch.full((1,6,2,2),99.);x[:,0,0,0]=-.5;x[:,3]=.2;x[:,4]=.9;active=torch.tensor([[[[1.,0.],[0.,1.]]]])
 result=project_physical(x,active)
 assert torch.equal(result*(1-active),torch.zeros_like(result))
 assert (result[:,4]<=result[:,3]).all() and (result[:,3]<=1).all() and result[0,0,0,0]==0

def test_delta_loss():
 current=torch.zeros((1,6,2,2));target=torch.ones_like(current);active=torch.ones((1,1,2,2));scales=[1]*6
 assert normalized_delta_huber(target,current,target,current,active,scales).item()==0
 assert normalized_delta_huber(current,current,target,current,active,scales).item()>0

def test_bounds_and_sampler():
 norm=DeltaNormalization([1]*6,[2]*6,.1,.9995,.995,.999,1.2,12)
 values=torch.tensor([1.99,-2.,.5]);assert float((values.abs()<=norm.bounds[0]).float().mean())==1
 sampler={'source_active_fraction':.20,'early_fraction':.20,'source_starts_s':[0,10,20],'early_start_range_s':[30,120]}
 one=training_sample_for_step(20260920,123,160,4,burn_in_choices=(0,4,8),sampler=sampler);two=training_sample_for_step(20260920,123,160,4,burn_in_choices=(0,4,8),sampler=sampler);assert one==two
 samples=[training_sample_for_step(20260920,step,160,1,burn_in_choices=(0,),sampler=sampler) for step in range(20000)]
 source=sum(item['sampler_stratum']=='source_active' for item in samples)/len(samples);assert .17<=source<=.23
 source_samples=[training_sample_for_step(20260920,step,160,6,burn_in_choices=(0,6,12,24),sampler=sampler) for step in range(20000)];assert all(item['burn_in_steps']==0 and item['gradient_start_time_s'] in (0,10,20) for item in source_samples if item['sampler_stratum']=='source_active')

def test_sparse_delta_fit():
 class Store:
  def frame(self,row,time_s):
   result=torch.zeros((6,10,10)).numpy()
   if time_s>0:result[3,0,0]=.25;result[4,0,1]=.125;result[5,0,2]=.05
   return result
 rows=pd.DataFrame([{'split':'TRAIN','scenario_id':'one'}]);static=torch.zeros((6,10,10)).numpy();static[1]=1
 fitted=fit_delta_normalization(rows,Store(),static,PhysicalTransform(),samples_per_case=4,max_cells=100,quantile=.995,bound_quantile=.999)
 assert fitted.scales[3]>1e-6 and fitted.scales[4]>1e-6 and fitted.scales[5]>1e-6
 assert fitted.change_threshold>1e-6 and all(value['coverage_all_samples']>=.999 for value in fitted.per_channel.values())

def test_burn_graph_and_capacity_telemetry():
 # Stateless burn selection is enough to guarantee resume reproduces future exposure.
 future=[training_sample_for_step(20260920,step,160,6,burn_in_choices=(0,6,12,24)) for step in range(900,930)]
 assert future==[training_sample_for_step(20260920,step,160,6,burn_in_choices=(0,6,12,24)) for step in range(900,930)]
 runner=runner_module();model=torch.nn.Linear(2,2);before={k:v.detach().clone() for k,v in model.state_dict().items()}
 probe=runner.capacity_probe(model,probe_fn=lambda *_:{'K':1,'safe':True,'status':'SAFE'})
 assert probe and all(torch.equal(before[k],v) for k,v in model.state_dict().items())
 with tempfile.TemporaryDirectory() as directory:
  target=Path(directory)/'training_log.csv';original=Path.open
  def locked(path,*args,**kwargs):
   if path==target:raise PermissionError('locked')
   return original(path,*args,**kwargs)
  with patch.object(Path,'open',locked):assert not runner.safe_append_csv(target,{'x':1})
  assert list(Path(directory).glob('training_log_fallback_*.jsonl'))

def test_runtime_model_zero_init_and_checkpoint():
 try:import rvpi
 except ModuleNotFoundError:return
 from src.global_operator_v2.model import JilongGlobalOperatorV2
 from src.global_operator_v2.trainer import save_checkpoint,restore_checkpoint
 model=JilongGlobalOperatorV2(35,width=8,modes=4,depth=1,delta_bounds=[.1,.2,.3,.4,.5,.6]);features=torch.randn(1,35,16,16);current=torch.randn(1,6,16,16);assert torch.allclose(model(features,current),current,atol=1e-7)
 model.delta_normalization={'scales':[1]*6,'bounds':[.1,.2,.3,.4,.5,.6]};optimizer=torch.optim.AdamW(model.parameters());scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
 with tempfile.TemporaryDirectory() as directory:
  path=Path(directory)/'checkpoint.pt';normalizer=FeatureNormalizer(0.,1.,1.,1.,[0]*5,[1]*5);save_checkpoint(path,model,optimizer,scheduler,0,PhysicalTransform(),{},normalizer)
  restored=JilongGlobalOperatorV2(35,width=8,modes=4,depth=1,delta_bounds=[.1,.2,.3,.4,.5,.6]);restore_checkpoint(path,restored);assert restored.bounded_residual and torch.allclose(restored(features,current),model(features,current))

if __name__=='__main__':
 test_projection();test_delta_loss();test_bounds_and_sampler();test_sparse_delta_fit();test_burn_graph_and_capacity_telemetry();test_runtime_model_zero_init_and_checkpoint();print('PASS stabilization')
