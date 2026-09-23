"""TEST-only wrapper around the pre-existing frozen EngineeringROI-v1 RandomAll B10."""
from argparse import Namespace
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts import evaluate_engineering_roi_v1 as frozen
def main():
 original_rows,original_manifest=frozen.scenario_rows,frozen.manifest_for
 frozen.scenario_rows=lambda _:original_rows('TEST').assign(split='VAL')
 def manifest(*a,**k):
  value=original_manifest(*a,**k);value['scope']='FROZEN_BASELINE_TEST';return value
 frozen.manifest_for=manifest
 try:frozen.main(Namespace(strategy='random_all',budget=.1,shard_index=0,shard_count=0,merge_shards=False,output_dir='paper_results/test/baselines/randomall_b10',resume=True,source_code_sha='146184d1fbbf366c0c9c40066a56753b398b10f4'))
 finally:frozen.scenario_rows, frozen.manifest_for=original_rows,original_manifest
if __name__=='__main__':main()
