"""V1.1 launcher: fresh model, audited V1 TRAIN-only normalization."""
from __future__ import annotations
import argparse,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.dataset import scenario_rows
from src.local_corrector.v1_1 import audit_normalization
from scripts.train_local_corrector_v1 import train

def prepare():
    source=ROOT/'models/local_corrector_v1/correction_normalization.json';out=ROOT/'models/local_corrector_v1_1';out.mkdir(parents=True,exist_ok=True)
    data=json.loads(source.read_text());audit_normalization(data,scenario_rows('TRAIN').scenario_id.tolist());data['reused_from_v1']=True
    (out/'correction_normalization.json').write_text(json.dumps(data,indent=2))

def main(args):
    if not (ROOT/'models/global_operator_v2/best.pt').exists():raise RuntimeError('FROZEN_GLOBAL_CHECKPOINT_MISSING')
    prepare();train(smoke=args.smoke,resume=args.resume,total_updates=20 if args.smoke else 6000,version='v1_1')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--formal',action='store_true');p.add_argument('--resume',action='store_true');main(p.parse_args())
