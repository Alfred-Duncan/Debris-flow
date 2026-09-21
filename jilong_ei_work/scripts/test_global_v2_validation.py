"""Numerical tests for strict dimensionless validation scoring."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.validation import validation_score

def main():
    metrics={'trajectory_h_rel_l2':.2,'trajectory_momentum_rel_l2':.3,'mean_wet_iou':.8,'debris_front_mae_km':.1,'mixture_volume_relative_error':.02}
    expected=.4*.2+.2*.3+.25*.2+.1*.2+.05*.2
    assert abs(validation_score(metrics)-expected)<1e-12
    try:validation_score({'trajectory_h_rel_l2':.1})
    except RuntimeError as error:assert 'VALIDATION_SCORE_INCOMPLETE' in str(error)
    else:raise AssertionError('incomplete score accepted')
if __name__=='__main__':main()
