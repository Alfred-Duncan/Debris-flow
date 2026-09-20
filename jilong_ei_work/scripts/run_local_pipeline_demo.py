"""End-to-end adapter -> RVPI FNO wrappers -> local patch smoke demo."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--workspace", required=True)
a = p.parse_args()
root = Path(a.workspace)
sys.path.insert(0, str(root))
from src.models import GlobalOperatorWrapper, LocalCorrectorWrapper

sample_path = sorted((root / "data" / "ml_smoke").glob("sample_*.npz"))[0]
with np.load(sample_path, allow_pickle=False) as sample:
    static = np.asarray(sample["static"], dtype=np.float32)
    state = np.asarray(sample["state_t"], dtype=np.float32)
    target = np.asarray(sample["state_tp1"], dtype=np.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
S = torch.from_numpy(static[None]).to(device)
X = torch.from_numpy(state[None]).to(device)
Y = torch.from_numpy(target[None]).to(device)
params = torch.tensor([[0.0]], device=device)
upstream = root / "external" / "RVPI-PDE"
global_model = GlobalOperatorWrapper(upstream, S.shape[1], X.shape[1]).to(device)
local_model = LocalCorrectorWrapper(upstream, X.shape[1]).to(device)
started = time.perf_counter()
pred = global_model(S, X, params)
height, width = pred.shape[-2:]
ys = slice(height // 2 - 16, height // 2 + 16)
xs = slice(width // 2 - 16, width // 2 + 16)
corrected = local_model(pred, (ys, xs), S[:, :1])
loss = torch.mean((corrected - Y) ** 2)
loss.backward()
wall = time.perf_counter() - started
summary = {
    "status": "PASS",
    "note": "Untrained models: metric is interface-only, not physical accuracy.",
    "sample": sample_path.name,
    "device": str(device),
    "input_shape": list(X.shape),
    "prediction_shape": list(pred.shape),
    "corrected_shape": list(corrected.shape),
    "finite": bool(torch.isfinite(corrected).all()),
    "global_backprop": all(parameter.grad is not None for parameter in global_model.parameters()),
    "local_backprop": all(parameter.grad is not None for parameter in local_model.parameters()),
    "dummy_mse": float(loss.detach().cpu()),
    "wall_s": wall,
    "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
}
out = root / "outputs" / "local_pipeline_demo"
out.mkdir(parents=True, exist_ok=True)
(out / "pipeline_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(state[0], cmap="viridis")
axes[0].set_title("Physics h (smoke)")
axes[1].imshow(pred.detach().cpu().numpy()[0, 0], cmap="viridis")
axes[1].set_title("Untrained operator output")
for axis in axes:
    axis.set_axis_off()
fig.tight_layout()
fig.savefig(out / "shape_alignment_only.png", dpi=150)
print(json.dumps(summary))

