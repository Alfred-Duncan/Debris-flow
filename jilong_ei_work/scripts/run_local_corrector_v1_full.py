"""Single resume-safe entry point for JILONG LOCAL CORRECTOR V1."""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.fit_local_corrector_normalization import main as fit_normalization
from scripts.train_local_corrector_v1 import train


def _paths():
    directory=ROOT/'models/local_corrector_v1'
    return directory,directory/'correction_normalization.json',directory/'last.pt',directory/'best.pt'


def _check_frozen():
    checkpoint=ROOT/'models/global_operator_v2/best.pt'
    if not checkpoint.exists() or checkpoint.stat().st_size<1_000_000:
        raise RuntimeError('FROZEN_GLOBAL_CHECKPOINT_MISSING')


def main(args):
    _check_frozen(); directory,normalization,last,best=_paths();directory.mkdir(parents=True,exist_ok=True)
    if args.evaluate_only:
        if not best.exists(): raise RuntimeError('BEST_LOCAL_CHECKPOINT_MISSING')
        print('EVALUATION_READY',best);return
    if not normalization.exists(): fit_normalization()
    if args.smoke:
        train(smoke=True,resume=args.resume,total_updates=20)
        print('SMOKE_TRAINING_PASS',last);return
    train(smoke=False,resume=args.resume,total_updates=6000)
    shutil.copy2(last,best)
    print('FORMAL_TRAINING_COMPLETE',best)


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smoke',action='store_true');group.add_argument('--formal',action='store_true');group.add_argument('--evaluate-only',action='store_true')
    parser.add_argument('--resume',action='store_true');main(parser.parse_args())
