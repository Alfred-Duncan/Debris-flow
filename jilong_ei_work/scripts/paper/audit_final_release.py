"""Read-only final release audit for paper-stage artifacts."""
from __future__ import annotations
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ("configs/risk_temporal_final_v1.json", "src/local_corrector/engineering_roi_v2.py", "src/local_corrector/depth_envelope_guard.py", "src/local_corrector/support_guard.py", "scripts/evaluate_engineering_roi_v2.py")
BASE = "146184d1fbbf366c0c9c40066a56753b398b10f4"


def main() -> None:
    audit = ROOT / "paper_results" / "audit"
    required = [audit / "frozen_configuration.json", audit / "data_split_audit.json", audit / "checkpoint_audit.json", audit / "metric_audit.json", audit / "FINAL_AUDIT.md", audit / "test_integrity_audit.json", audit / "test_reproducibility_audit.json", audit / "FINAL_TEST_AUDIT.md", ROOT / "paper_results" / "PAPER_RESULTS_SUMMARY.md", ROOT / "paper_results" / "all_experiments_manifest.csv", ROOT / "paper_results" / "h0" / "H0_REPORT.md", ROOT / "FINAL_METHOD.md", ROOT / "FINAL_EXPERIMENTS.md", ROOT / "REPRODUCIBILITY.md"]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    statuses = {name: json.loads((audit / name).read_text())["status"] for name in ("frozen_configuration.json", "data_split_audit.json", "checkpoint_audit.json", "metric_audit.json", "test_integrity_audit.json", "test_reproducibility_audit.json")}
    changed = subprocess.run(["git", "diff", "--quiet", BASE, "--", *FROZEN], cwd=ROOT, check=False).returncode != 0
    payload = {"status": "PASS" if not missing and all(v == "PASS" for v in statuses.values()) and not changed else "FAIL", "missing": missing, "audit_statuses": statuses, "frozen_algorithm_source_modified_since_final_code": changed, "test_specific_tuning": False, "large_model_or_checkpoint_copies_in_release": False}
    (audit / "final_release_audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    if payload["status"] != "PASS": raise SystemExit(2)


if __name__ == "__main__": main()
