"""Instantiate upstream sparse solver only; never enter its time loop."""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--workspace", required=True)
a = p.parse_args()
root = Path(a.workspace)
up = root / "external" / "langtang-2026-cascade"
inp = root / "data" / "downloads" / "corridor60s.npz"
sys.path.insert(0, str(up / "swe"))
spec = importlib.util.spec_from_file_location("solver_sparse", up / "swe" / "solver_sparse.py")
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
data = np.load(inp, allow_pickle=False)
active = data["active"]
try:
    model = mod.SparseSWE(data["z"], active, 60.0, device="cuda", mu_s=0.0, n_w=0.0208, n_d=0.0145)
    result = {
        "status": "PASS",
        "LOCAL_GPU_LIMIT": False,
        "active_cells": int(model.n),
        "grid_shape": list(data["z"].shape),
        "input_bytes": int(inp.stat().st_size),
        "static_gpu_allocated_bytes": int(torch.cuda.memory_allocated()),
        "static_gpu_reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "device": torch.cuda.get_device_name(0),
        "vram_total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
    }
except torch.OutOfMemoryError as exc:
    result = {
        "status": "OOM",
        "LOCAL_GPU_LIMIT": True,
        "error": str(exc),
        "active_cells": int(active.sum()),
        "grid_shape": list(data["z"].shape),
    }
(root / "reports" / "CORRIDOR60_MEMORY_PREFLIGHT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result))

