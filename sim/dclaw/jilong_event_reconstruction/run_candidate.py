"""Run exactly one Phase-2B source candidate and append reproducible diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

CASE = Path(__file__).resolve().parent


def run(cmd: list[str], log) -> None:
    process = subprocess.run(cmd, cwd=CASE, stdout=log, stderr=subprocess.STDOUT, text=True)
    if process.returncode:
        raise RuntimeError(f"command failed ({process.returncode}): {' '.join(cmd)}")


def main(run_id: str, depth_m: float, speed_ms: float) -> None:
    run_dir = CASE / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    log_path = run_dir / "run.log"
    started = time.monotonic()
    try:
        with log_path.open("w") as log:
            run([sys.executable, "build_inputs.py", "--depth-m", str(depth_m), "--speed-ms", str(speed_ms)], log)
            run([sys.executable, "setrun.py"], log)
            if not (CASE / "xdclaw").exists():
                run(["make"], log)
            if (CASE / "_output").exists():
                shutil.rmtree(CASE / "_output")
            # The standard Clawpack make target is marked by .output; remove
            # that marker whenever archived output is absent so each candidate
            # actually executes rather than being treated as up to date.
            (CASE / ".output").unlink(missing_ok=True)
            run(["make", ".output"], log)
            run([sys.executable, "postprocess.py", "--output", "_output", "--summary", str(run_dir / "diagnostics.json")], log)
    except Exception as exc:
        (run_dir / "FAILED.txt").write_text(str(exc) + "\n")
        raise
    runtime_s = time.monotonic() - started
    result = json.loads((run_dir / "diagnostics.json").read_text())
    stable = bool(result["stable_fields"] and result["domain_max_speed_ms"] <= 36.0)
    row = {
        "run_id": run_id, "source_depth_m": depth_m, "initial_speed_ms": speed_ms,
        "reached_port": result["reached_port"], "arrival_time_s": result["arrival_time_s"],
        "port_peak_discharge_m3s": result["port_peak_discharge_m3s"],
        "port_peak_depth_m": result["port_peak_depth_m"], "port_peak_speed_ms": result["port_peak_speed_ms"],
        "source_volume_m3": result["source_volume_m3"], "runtime_s": runtime_s, "stable": stable,
    }
    csv_path = CASE / "CANDIDATE_RUNS.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row), lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    shutil.move(str(CASE / "_output"), str(run_dir / "_output"))
    (run_dir / "candidate_row.json").write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--depth-m", type=float, required=True)
    parser.add_argument("--speed-ms", type=float, required=True)
    args = parser.parse_args()
    main(args.run_id, args.depth_m, args.speed_ms)
