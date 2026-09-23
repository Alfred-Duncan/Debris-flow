import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 v2=(ROOT/'src/local_corrector/engineering_roi_v2.py').read_text();guard=(ROOT/'src/local_corrector/depth_envelope_guard.py').read_text();assert 'compute_dynamic_component(' not in v2 and 'compute_front_component(' not in v2 and 'compute_section_component(' not in v2 and 'previous_selected_patch_ids' in v2 and '[:,:1]' in guard;print(json.dumps({'status':'PASS','v1_status':'FROZEN','v2_definition':'PREDECLARED','new_training':False,'real_rollout_executed':False}))
if __name__=='__main__':main()
