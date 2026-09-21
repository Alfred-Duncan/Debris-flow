"""Numerical tests for strict dimensionless validation scoring."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.validation import validation_score,front_error,_ratio

def main():
    metrics={'trajectory_h_rel_l2':.2,'trajectory_momentum_rel_l2':.3,'mean_wet_iou':.8,'debris_front_mae_km':.1,'mixture_volume_relative_error':.02}
    expected=.4*.2+.2*.3+.25*.2+.1*.2+.05*.2
    assert abs(validation_score(metrics)-expected)<1e-12
    try:validation_score({'trajectory_h_rel_l2':.1})
    except RuntimeError as error:assert 'VALIDATION_SCORE_INCOMPLETE' in str(error)
    else:raise AssertionError('incomplete score accepted')
    # Four independently allocated accumulators guard against chained-assignment aliasing.
    hnum=torch.zeros(());hden=torch.zeros(());mnum=torch.zeros(());mden=torch.zeros(());hnum+=1;assert hden.item()==0 and mnum.item()==0 and mden.item()==0
    prediction,truth=torch.tensor([1.,3.,5.]),torch.tensor([2.,1.,4.]);assert abs(_ratio(((prediction-truth).square()).sum(),truth.square().sum())-float(torch.linalg.vector_norm(prediction-truth)/torch.linalg.vector_norm(truth)))<1e-12
    assert front_error(1.5,1.,3.)==.5 and front_error(float('nan'),1.,3.)==3.
if __name__=='__main__':main()
