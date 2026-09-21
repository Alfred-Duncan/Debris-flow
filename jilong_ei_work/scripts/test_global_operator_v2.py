"""Deterministic CPU unit tests for V2 semantics; no data generation or training."""
from __future__ import annotations
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.transforms import PhysicalTransform,FeatureNormalizer
from src.global_operator_v2.dataset import build_features,feature_names
from src.global_operator_v2.losses import state_loss,wet_loss,integral_loss,amplitude_guard,project_physical
def main():
 torch.manual_seed(20260920);x=torch.tensor([[[[0.,1.]],[[0.,-2.]],[[0.,3.]],[[.2,.4]],[[.1,.2]],[[0.,-.1]]]],dtype=torch.float32);tr=PhysicalTransform(2,3,4,.05);roundtrip=tr.decode(tr.encode(x));assert torch.allclose(x,roundtrip,atol=1e-6) and torch.equal(tr.encode(torch.zeros_like(x)),torch.zeros_like(x))
 active=torch.ones((1,1,2));perfect,_=state_loss(tr.encode(x),tr.encode(x),x,x,active);assert perfect<1e-8 and wet_loss(x,x,active)<1e-8 and integral_loss(x,x,active,900)[0]<1e-8 and amplitude_guard(tr.encode(x),tr.encode(x),active)<1e-8
 assert amplitude_guard(1.5*tr.encode(x),tr.encode(x),active)<1e-8 and amplitude_guard(3*tr.encode(x),tr.encode(x),active)>0
 a=x.clone();b=x.flip(-1);b[:,0]=a[:,0];b[:,3:5]=a[:,3:5];assert integral_loss(a,b,active,900)[0]<1e-8
 norm=FeatureNormalizer(0,1,1,1,[0]*5,[1]*5);static=torch.zeros((1,6,1,2));static[:,1]=1;static[:,2]=1;params=torch.ones((1,5));params[:,2]=1;features=build_features(x,x,static,params,torch.tensor([0.]),tr,norm);assert features.shape[1]==len(feature_names())==35 and features[:,32].nonzero().shape[0]>0
 late=build_features(x,x,static,params,torch.tensor([1.]),tr,norm);assert late[:,32].abs().sum()==0
 p=project_physical(torch.tensor([[[[-1.]],[[1.]],[[1.]],[[2.]],[[2.]],[[.2]]]]));assert p[:,0].item()==0 and p[:,1:5].abs().sum()==0 and torch.allclose(p[:,5],torch.tensor([[[.2]]]))
 print(json.dumps({'status':'PASS','transform_roundtrip':True,'loss_tests':True,'feature_channels':35,'source_forcing_tests':True,'projection_tests':True},indent=2))
if __name__=='__main__':main()
