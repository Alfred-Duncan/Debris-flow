from __future__ import annotations
import torch
import torch.nn.functional as F

def project_physical(x: torch.Tensor, dry: float=.03) -> torch.Tensor:
    y=x.clone(); y[:,0].clamp_(min=0); y[:,3:5].clamp_(0,1)
    total=y[:,3]+y[:,4]; over=total>1
    y[:,3][over] /= total[over]; y[:,4][over] /= total[over]
    y[:,1:3] *= (y[:,0:1] >= dry).to(y.dtype)
    return y

def v2_loss(pred_t, target_t, pred_p, target_p, active, dry=.03, wet_tau=.02):
    a=active[:,None].float(); union=((pred_p[:,0]>=dry)|(target_p[:,0]>=dry))[:,None].float()
    weight=a*(.1+.9*union)
    point=F.smooth_l1_loss(pred_t,target_t,reduction="none")
    field=(point*weight).sum((0,2,3))/weight.sum((0,2,3)).clamp_min(1); state=field.mean()
    wetp=torch.sigmoid((pred_p[:,0]-dry)/wet_tau); wett=torch.sigmoid((target_p[:,0]-dry)/wet_tau)
    wet=F.mse_loss(wetp*a,wett*a)
    # Active-domain integrals regulate mass/momentum amplitudes without global standardization.
    integrals=((pred_p-target_p).abs()*a).mean((0,2,3)).mean()
    anchor=F.smooth_l1_loss(pred_p[:,0]*a,target_p[:,0]*a)
    amplitude=(pred_p.abs()*a).mean()
    total=state+.10*wet+.05*integrals+.50*anchor+.01*amplitude
    return total,{"total":total.detach(),"state":state.detach(),"wet":wet.detach(),"integrals":integrals.detach(),"anchor":anchor.detach(),"amplitude":amplitude.detach(),"channels":field.detach()}
