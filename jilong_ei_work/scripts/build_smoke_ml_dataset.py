"""Build adjacent-frame ML transition samples; intentionally too small for training."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--workspace", required=True)
a = p.parse_args()
root = Path(a.workspace)
sys.path.insert(0, str(root))
from src.physics_frame_adapter import static_tensor, dynamic_tensor

inp = root / "data" / "downloads" / "corridor60s.npz"
frames = sorted((root / "outputs" / "physics_smoke60" / "frames").glob("frame_*.npz"))
out = root / "data" / "ml_smoke"
out.mkdir(parents=True, exist_ok=True)
static, meta = static_tensor(inp)
paths = []
for i, (left, right) in enumerate(zip(frames[:-1], frames[1:])):
    state_t, m0 = dynamic_tensor(left, tuple(meta["shape"]))
    state_tp1, m1 = dynamic_tensor(right, tuple(meta["shape"]))
    path = out / f"sample_{i:04d}.npz"
    np.savez_compressed(
        path, static=static, state_t=state_t, state_tp1=state_tp1,
        dt_s=np.float32(m1["time_s"] - m0["time_s"]),
        parameters_json=json.dumps({"run_id": "physics_smoke60"}),
        metadata_json=json.dumps({**meta, "time_s": m0["time_s"], "run_id": "physics_smoke60", "scenario_parameters": {"source": "original-code software smoke"}}),
    )
    paths.append(path)
values = []
for path in paths:
    with np.load(path, allow_pickle=False) as sample:
        values.append(np.concatenate([sample["static"].ravel(), sample["state_t"].ravel(), sample["state_tp1"].ravel()]))
flat = np.concatenate(values) if values else np.array([], dtype=np.float32)
report = {
    "status": "PASS" if paths and np.isfinite(flat).all() else "FAIL",
    "number_samples": len(paths),
    "static_shape": list(static.shape),
    "state_shape": list(state_t.shape) if paths else None,
    "static_channels": meta["static_channels"],
    "dynamic_channels": ["h", "hu", "hv", "c", "ice"],
    "dtype": "float32",
    "min": float(flat.min()) if flat.size else None,
    "max": float(flat.max()) if flat.size else None,
    "nan": int(np.isnan(flat).sum()),
    "inf": int(np.isinf(flat).sum()),
    "disk_bytes": sum(path.stat().st_size for path in paths),
}
(root / "reports" / "ML_DATASET_SMOKE.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report))
