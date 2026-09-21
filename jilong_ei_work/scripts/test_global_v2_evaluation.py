"""Synthetic tests for immutable freeze and exactly-once holdout locks."""
from __future__ import annotations
import hashlib,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.global_operator_v2.evaluation import final_holdout_lock,freeze_best_candidate,sha256

def main():
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);source=root/'candidate.pt';source.write_bytes(b'candidate');target=root/'best.pt';manifest=root/'manifest.json'
        assert freeze_best_candidate(source,target,manifest,{'config_hash':'x'})==sha256(target)
        lock=root/'lock.json';final_holdout_lock(lock,target,'x','m')
        try:final_holdout_lock(lock,target,'x','m')
        except RuntimeError as error:assert str(error)=='FINAL_HOLDOUT_ALREADY_EVALUATED'
        else:raise AssertionError('second primary holdout evaluation accepted')
if __name__=='__main__':main()
