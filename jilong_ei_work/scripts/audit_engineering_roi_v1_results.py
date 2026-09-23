"""Read-only formal VAL result audit and report generator."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];FORMAL_SHA='fbb6bdef1140f9faf755cc8e2318780686c4a177'
METHODS=('FrozenGlobal','RandomAll_B05','RandomAll_B10','RandomAll_B20','RandomSupport_B05','RandomSupport_B10','RandomSupport_B20','EngineeringROI_B05','EngineeringROI_B10','EngineeringROI_B20','DynamicOnly_B10','FrontOnly_B10','SupportRiskOnly_B10','SectionOnly_B10','EngineeringROI_NoDiversity_B10')
def main():
    root=ROOT/'results/engineering_roi_v1';runs=root/'runs';manifests=[]
    for method in METHODS:
        run=runs/method
        if not (run/'RUN_COMPLETE.json').exists():raise RuntimeError('FORMAL_VAL_RESULT_AUDIT_FAIL')
        manifests.append(json.loads((run/'run_manifest.json').read_text()))
    keys=('code_sha','config_sha256','global_checkpoint_sha256','local_checkpoint_sha256','local_normalization_sha256','support_guard_version','full_scenario_ids','patch_layout_rows','patch_layout_cols')
    if any(manifest.get('code_sha')!=FORMAL_SHA for manifest in manifests) or any(any(manifest.get(key)!=manifests[0].get(key) for key in keys) for manifest in manifests[1:]):raise RuntimeError('FORMAL_VAL_RESULT_AUDIT_FAIL')
    case=pd.read_csv(root/'final_case_metrics.csv');timeline=pd.read_csv(root/'selection_timeline.csv');summary=pd.read_csv(root/'final_method_summary.csv');runtime=pd.read_csv(root/'runtime_summary.csv')
    if len(case)!=300 or set(case.method)!=set(METHODS) or not (case.groupby('method').size()==20).all() or len(timeline)!=40320 or not (timeline.groupby(['method','scenario_id']).size()==144).all():raise RuntimeError('FORMAL_VAL_RESULT_AUDIT_FAIL')
    report={'formal_code_sha':FORMAL_SHA,'config_sha256':manifests[0]['config_sha256'],'global_checkpoint_sha256':manifests[0]['global_checkpoint_sha256'],'local_checkpoint_sha256':manifests[0]['local_checkpoint_sha256'],'local_normalization_sha256':manifests[0]['local_normalization_sha256'],'support_guard_version':manifests[0]['support_guard_version'],'scope':'VAL_ONLY','case_ids':manifests[0]['full_scenario_ids'],'methods':list(METHODS),'main_method_summary':summary.to_dict('records'),'runtime_summary':runtime.to_dict('records'),'formal_val_result_audit':'PASS'}
    reports=ROOT/'reports';reports.mkdir(exist_ok=True);(reports/'ENGINEERING_ROI_V1_FORMAL_VAL.json').write_text(json.dumps(report,indent=2,allow_nan=True));lines=['# EngineeringROI-v1 Formal VAL','','## Provenance',f'- Formal code SHA: `{FORMAL_SHA}`',f"- Config SHA256: `{report['config_sha256']}`",'- Scope: 20 VAL cases; 15 predefined methods.','','## Main Results','',summary.to_markdown(index=False),'','## Runtime','',runtime.to_markdown(index=False),'','## Limitations','- This report is VAL-only; no TEST, H0, or FINAL_HOLDOUT results are included.'];(reports/'ENGINEERING_ROI_V1_FORMAL_VAL.md').write_text('\n'.join(lines));print('FORMAL_VAL_RESULT_AUDIT = PASS')
if __name__=='__main__':main()
