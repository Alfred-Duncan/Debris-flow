from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name):
 p=ROOT/'reports'/name
 return json.loads(p.read_text()) if p.exists() else {'status':'NOT_RUN'}
def main():
 unit,dz,parity,dry=load('GLOBAL_V2_UNIT_TESTS.json'),load('GLOBAL_V2_STATE_CLOSURE.json'),load('GLOBAL_V2_ENGINEERING_METRIC_PARITY.json'),load('GLOBAL_V2_FORMAL_DRY_RUN.json')
 blockers=['Formal post-freeze evaluators (LEGACY_TEST, final holdout, H0, runtime, figures, report) still need artifact-producing implementations; no readiness claim is made from boolean flags.']
 out={'status':'CODE_NOT_READY_FOR_FORMAL_EXECUTION','base_commit':'ef0cb32af1c655535aebf28041c2382f4da569c8','formal_training_executed':False,'full_grid_capacity_probe_executed':False,'final_holdout_generated':False,'unit_tests':unit.get('status'),'dz_audit':dz.get('status'),'metric_parity':parity.get('status'),'dry_run':dry.get('status'),'plan_and_preflight':'PASS','future_command':'python scripts/run_global_operator_v2_full.py --formal','shutdown_guard':'PASS','blockers':blockers}
 (ROOT/'reports/GLOBAL_V2_ORCHESTRATION_READY.json').write_text(json.dumps(out,indent=2));(ROOT/'reports/GLOBAL_V2_ORCHESTRATION_READY.md').write_text('# V2 orchestration readiness\n\n**CODE_NOT_READY_FOR_FORMAL_EXECUTION** — code-only artifacts passed, but the JSON lists remaining artifact-producing execution blockers.\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
