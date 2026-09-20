"""Formal-only sealed final-holdout generator. It refuses to run in code-only mode."""
from __future__ import annotations
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main(formal: bool):
    if not formal: raise RuntimeError('Refusing final-holdout generation: pass --formal only on the approved formal-training day.')
    design={'seed':20260920,'design_indices':list(range(201,221)),'ids':[f'FINAL_HOLDOUT_{i:03d}' for i in range(1,21)],'status':'SEALED_PENDING_PHYSICS'}
    out=ROOT/'outputs/final_holdout_v2';out.mkdir(parents=True,exist_ok=True);(out/'MANIFEST.json').write_text(json.dumps(design,indent=2));print(json.dumps(design))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--formal',action='store_true');main(p.parse_args().formal)
