"""Deterministic TRAIN-only envelope helpers for momentum-state safeguards."""
from __future__ import annotations
import numpy as np

QUANTILES=(('q50',.50),('q90',.90),('q99',.99),('q999',.999),('q9999',.9999))

def _stats(values):
 values=np.asarray(values,dtype=np.float64).reshape(-1)
 if not values.size:raise ValueError('momentum envelope requires samples')
 return {name:float(np.quantile(values,q)) for name,q in QUANTILES}|{'max_sampled':float(values.max()),'sample_count':int(values.size)}

def fit_momentum_envelope(absolute_encoded_hu,absolute_encoded_hv,speed):
 """Fit conservative guards and verify their empirical TRAIN-only coverage."""
 hu=np.asarray(absolute_encoded_hu,dtype=np.float64).reshape(-1);hv=np.asarray(absolute_encoded_hv,dtype=np.float64).reshape(-1);speed=np.asarray(speed,dtype=np.float64).reshape(-1)
 def guarded(values):
  stats=_stats(values);guard=max(1.5*stats['q9999'],1.1*stats['q999'])
  if float(np.mean(values<=guard))<.9999:guard=max(guard,float(values.max()))
  return stats|{'guard':float(guard),'coverage':float(np.mean(values<=guard))}
 return {'envelope_version':'TRAIN_ONLY_MOMENTUM_STATE_ENVELOPE_V1','fit_scope':'TRAIN_ONLY','hu':guarded(hu),'hv':guarded(hv),'speed_mps':_stats(speed)}
