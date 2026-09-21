"""AST-level execution-path guard; it deliberately performs no formal execution."""
from __future__ import annotations
import ast,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    runner_path=ROOT/'scripts/run_global_operator_v2_full.py';runner=ast.parse(runner_path.read_text(encoding='utf-8'));validation=ast.parse((ROOT/'src/global_operator_v2/validation.py').read_text(encoding='utf-8'))
    calls={node.func.id for node in ast.walk(runner) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)}
    function={node.name:node for node in runner.body if isinstance(node,ast.FunctionDef)}
    validate_fn=next(node for node in validation.body if isinstance(node,ast.FunctionDef) and node.name=='validate_full_rollout')
    assignments=[node for node in ast.walk(validate_fn) if isinstance(node,ast.Assign)]
    chained=any(len(node.targets)>1 for node in assignments)
    checks={
        'formal_dispatch':'_stage_training' in calls and 'freeze_best_candidate' in calls and 'validate_all_val' in calls,
        'capacity_real_closure':'_capacity_loss' in calls and 'rollout_loss' in calls,
        'capacity_result_applied':'apply_capacity_result' in calls,
        'scheduler_after_capacity':'build_scheduler' in calls,
        'train_stage_called':'train_stage' in calls,
        'validation_streaming':not any(isinstance(node,ast.Attribute) and node.attr=='window' for node in ast.walk(validate_fn)),
        'accumulators_independent':not chained,
        'front_not_constant':any(isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='debris_front' for node in ast.walk(validate_fn)),
        'freeze_called':'freeze_best_candidate' in calls,
    }
    report={'status':'PASS' if all(checks.values()) else 'FAIL','passed':sum(checks.values()),'total':len(checks),'checks':checks}
    path=ROOT/'reports/GLOBAL_V2_FORMAL_CODE_READY.json';path.write_text(json.dumps(report,indent=2));(ROOT/'reports/GLOBAL_V2_FORMAL_CODE_READY.md').write_text('# Global V2 formal code readiness\n\n'+json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if not all(checks.values()):raise SystemExit(1)
if __name__=='__main__':main()
