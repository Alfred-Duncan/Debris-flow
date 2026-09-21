"""Mocked PRECHECK-to-FREEZE transition and capacity fallback tests."""
from __future__ import annotations
import importlib.util,sys,tempfile
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.pipeline import PipelineState,apply_capacity_result,checkpoint_agreement,load_state
from src.global_operator_v2.trainer import is_improvement
from src.global_operator_v2.transforms import PhysicalTransform,FeatureNormalizer
from src.global_operator_v2.validation import validate_one_step
spec=importlib.util.spec_from_file_location('runner',ROOT/'scripts/run_global_operator_v2_full.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
def main():
    called=[]
    def mocked_probe(model,k,loss):
        called.append(k);return {'K':k,'status':'OOM' if k==4 else 'SAFE','safe':k<4}
    probes=runner.capacity_probe(None,lambda k:None,mocked_probe);state=apply_capacity_result(PipelineState('v','h',current_stage='ARCHITECTURE'),probes)
    assert called==[1,2,4] and state.total_planned_updates==4000
    assert runner.core_stage_path(state)==['ARCHITECTURE','STAGE_A','STAGE_B','VAL_CONFIRM','FREEZE']
    best=None
    for score in (.8,.6,.75):
        if is_improvement(score,best):best=score
    assert best==.6
    # Checkpoint/run-state pair is identical both mid-stage and at the boundary.
    with tempfile.TemporaryDirectory() as directory:
        old=runner.FORMAL;runner.FORMAL=Path(directory);model=torch.nn.Linear(1,1);opt=torch.optim.AdamW(model.parameters(),lr=.01);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.);tr=PhysicalTransform(1,1,1,.05);norm=FeatureNormalizer(0,1,1,1,[0]*5,[1]*5);state=PipelineState('v',runner.cfg_hash(),current_stage='STAGE_A',stage_updates_total=2000)
        runner.save_resume_pair(model,opt,sched,tr,norm,state);state.stage_step=250;state.global_step=250;runner.save_resume_pair(model,opt,sched,tr,norm,state);restored=load_state(runner.FORMAL/'run_state.json',state);checkpoint_agreement(restored,torch.load(runner.FORMAL/'last.pt',weights_only=False));runner.enter_stage(model,opt,sched,tr,norm,state,'STAGE_B');restored=load_state(runner.FORMAL/'run_state.json',state);checkpoint_agreement(restored,torch.load(runner.FORMAL/'last.pt',weights_only=False));assert restored.stage_best_score is None and restored.current_stage=='STAGE_B';state.stage_step=50;state.global_step=300;runner.rollback_state_to_last_checkpoint(state);checkpoint_agreement(state,torch.load(runner.FORMAL/'last.pt',weights_only=False));assert state.stage_step==0 and state.global_step==250;runner.FORMAL=old
    # Fixed cheap-validation times never fall back to t=0.
    class Store:
        def __init__(self):self.rows=pd.DataFrame([{'scenario_id':'v','split':'VAL'}]);self.static=np.ones((6,1,1),np.float32);self.seen=[]
        def sample(self,index,t):self.seen.append(t);x=np.zeros((6,1,1),np.float32);return x,x,x,np.ones(5,np.float32),t/1440.,self.rows.iloc[0]
    class Model(torch.nn.Module):
        def forward(self,features,encoded):return encoded
    store=Store();validate_one_step(Model(),store,PhysicalTransform(1,1,1,.05),FeatureNormalizer(0,1,1,1,[0]*5,[1]*5),torch.device('cpu'),times_s=(120,300,600,900,1200));assert store.seen==[120,300,600,900,1200]
if __name__=='__main__':main()
