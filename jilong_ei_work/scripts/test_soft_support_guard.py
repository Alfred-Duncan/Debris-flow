import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.local_corrector.soft_support_guard import apply_soft_support_guard
class T:
 def encode(self,x):return x
 def decode(self,x):return x
def main():
 p=torch.zeros(1,6,5,5); l=p.clone(); l[:,:,2,2]=1; l[:,:,2,3]=1; cur=p.clone(); cur[:,0,2,1]=.1; active=torch.ones(1,1,5,5); out,t=apply_soft_support_guard(p,l,cur,p,active,T(),radius_cells=1,alpha_front=.5); assert out[0,0,2,2]==.5 and out[0,0,2,3]==0 and t['fraction_zone_B']>0 and t['fraction_zone_C']>0; print('PASS soft support guard')
if __name__=='__main__':main()
