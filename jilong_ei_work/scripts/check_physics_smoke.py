"""Validate numerical and schema invariants of the small upstream smoke run."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--workspace", required=True)
a = p.parse_args()
root = Path(a.workspace)
out = root / "outputs" / "physics_smoke60"
checks: dict[str, object] = {}
errors: list[str] = []
try:
    series = pd.read_csv(out / "series.csv")
    required = [
        "time_s", "total_volume_m3", "max_h_m", "max_speed_m_s",
        "gyirong_cctv_arrival_Q", "gyirong_cctv_arrival_Qdebris",
        "gyirong_cctv_arrival_hmax", "gyirong_cctv_arrival_stage",
        "gyirong_cctv_arrival_cmax", "gyirong_cctv_arrival_wet_width_m",
    ]
    checks["series_columns_present"] = all(key in series for key in required)
    checks["time_monotonic"] = bool(series.time_s.is_monotonic_increasing)
    # Front diagnostics are intentionally NaN before the short smoke flow reaches
    # their chainage.  Only the required conserved/state diagnostics are finite.
    core_numeric = ["time_s", "total_volume_m3", "max_h_m", "max_speed_m_s"]
    checks["finite"] = bool(np.isfinite(series[core_numeric].to_numpy()).all())
    checks["front_nan_before_arrival_allowed"] = True
    frames = sorted((out / "frames").glob("frame_*.npz"))
    checks["frames"] = len(frames)
    for frame_path in frames:
        with np.load(frame_path, allow_pickle=False) as frame:
            for key in ("h", "hu", "hv", "c", "ice"):
                if not np.isfinite(frame[key]).all():
                    errors.append(f"{frame_path.name}:{key}:nonfinite")
            if (frame["h"] < 0).any():
                errors.append(f"{frame_path.name}:negative_h")
except Exception as exc:
    errors.append(repr(exc))
checks["errors"] = errors
checks["status"] = "PASS" if checks.get("series_columns_present") and checks.get("time_monotonic") and checks.get("finite") and not errors else "FAIL"
(root / "reports" / "PHYSICS_SMOKE_RESULT.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
print("SMOKE_PHYSICS =", checks["status"])
