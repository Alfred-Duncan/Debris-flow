"""Static code-readiness guard; it deliberately performs no formal execution."""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    files=list((ROOT/'src/global_operator_v2').glob('*.py'))+[ROOT/'scripts/run_global_operator_v2_full.py']
    text='\n'.join(path.read_text(encoding='utf-8') for path in files)
    checks={
        'no_not_implemented':'NotImplementedError' not in text,
        'no_formal_handler_placeholder':'FORMAL_HANDLER_REQUIRED' not in text,
        'physical_ice_component':'torch.minimum(y[:,4],y[:,3])' in text,
        'capacity_oom_handling':'torch.cuda.OutOfMemoryError' in text,
        'capacity_result_applied':'apply_capacity_result(state,probes)' in text,
        'stateless_sampler':'SeedSequence([seed,global_step])' in text,
        'scheduler_effective_total':'state.total_planned_updates' in text,
        'streaming_validation':'store.window' not in (ROOT/'src/global_operator_v2/validation.py').read_text(encoding='utf-8'),
        'strict_validation_score':'VALIDATION_SCORE_INCOMPLETE' in text,
        'atomic_checkpoint':'os.replace(tmp,path)' in (ROOT/'src/global_operator_v2/trainer.py').read_text(encoding='utf-8'),
        'freeze_copies_candidate':'freeze_best_candidate' in text,
        'final_holdout_lock':'FINAL_HOLDOUT_ALREADY_EVALUATED' in text,
        'shutdown_guard':'shutdown_allowed' in text,
        'final_holdout_isolated':'final_holdout' not in (ROOT/'src/global_operator_v2/validation.py').read_text(encoding='utf-8').lower(),
    }
    report={'status':'PASS' if all(checks.values()) else 'FAIL','passed':sum(checks.values()),'total':len(checks),'checks':checks}
    path=ROOT/'reports/GLOBAL_V2_FORMAL_CODE_READY.json';path.write_text(json.dumps(report,indent=2));(ROOT/'reports/GLOBAL_V2_FORMAL_CODE_READY.md').write_text('# Global V2 formal code readiness\n\n'+json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if not all(checks.values()):raise SystemExit(1)
if __name__=='__main__':main()
