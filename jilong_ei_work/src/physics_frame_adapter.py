"""Dense, orientation-audited adapter for Langtang sparse output frames."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

DYNAMIC_NAMES = ("h", "hu", "hv", "c", "ice")


def _grid_meta(input_npz: Path) -> dict[str, Any]:
    meta = json.loads(input_npz.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(input_npz, allow_pickle=False) as inp:
        meta["shape"] = list(inp["z"].shape)
    return meta


def static_tensor(input_npz: str | Path, include_z_raw: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    """Return static channels as float32 [C, H, W]."""
    input_npz = Path(input_npz)
    with np.load(input_npz, allow_pickle=False) as inp:
        names = ["z"] + (["z_raw"] if include_z_raw and "z_raw" in inp.files else []) + ["source_mask", "active"]
        channels = [np.asarray(inp[name], dtype=np.float32) for name in names]
    meta = _grid_meta(input_npz)
    meta["dx_m"] = float(meta["cell_m"])
    meta["static_channels"] = names
    meta["flattening"] = "numpy C-order / upstream model.flat = row * cols + col"
    meta["row_axis"] = "grid row; northing decreases as row increases"
    return np.stack(channels, axis=0), meta


def dynamic_tensor(frame_npz: str | Path, shape: tuple[int, int]) -> tuple[np.ndarray, dict[str, Any]]:
    """Reconstruct upstream wet-cell sparse fields to float32 [5, H, W]."""
    frame_npz = Path(frame_npz)
    h, w = shape
    with np.load(frame_npz, allow_pickle=False) as frame:
        idx = np.asarray(frame["idx"], dtype=np.int64)
        if idx.size and (idx.min() < 0 or idx.max() >= h * w):
            raise ValueError("upstream frame idx lies outside declared grid")
        arrays: list[np.ndarray] = []
        for key in DYNAMIC_NAMES:
            field = np.zeros(h * w, dtype=np.float32)
            field[idx] = np.asarray(frame[key], dtype=np.float32)
            arrays.append(field.reshape((h, w), order="C"))
        metadata = {"time_s": float(frame["time_s"]), "frame_path": str(frame_npz), "dynamic_channels": list(DYNAMIC_NAMES)}
    return np.stack(arrays, axis=0), metadata


def frame_sample(input_npz: str | Path, frame_npz: str | Path, run_id: str, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return one uniform ML-side record without writing to an upstream path."""
    static, meta = static_tensor(input_npz)
    state, dynamic_meta = dynamic_tensor(frame_npz, tuple(meta["shape"]))
    return {"static": static, "state": state, "metadata": {**meta, **dynamic_meta, "run_id": run_id, "parameters": parameters or {}}}


def orientation_audit(input_npz: str | Path, frame_npz: str | Path) -> dict[str, Any]:
    """Check C-order reconstruction, finite values, and non-negative depth."""
    static, meta = static_tensor(input_npz)
    state, dyn = dynamic_tensor(frame_npz, tuple(meta["shape"]))
    with np.load(frame_npz, allow_pickle=False) as raw:
        idx = np.asarray(raw["idx"], dtype=np.int64)
    restored = np.flatnonzero(state[0].reshape(-1, order="C") > 0)
    return {
        "status": "PASS" if np.array_equal(restored, idx) and np.isfinite(state).all() and (state[0] >= 0).all() else "FAIL",
        "shape": meta["shape"],
        "idx_count": int(idx.size),
        "idx_min": int(idx.min()) if idx.size else None,
        "idx_max": int(idx.max()) if idx.size else None,
        "c_order_roundtrip": bool(np.array_equal(restored, idx)),
        "row_col_formula": "row=idx//cols, col=idx%cols",
        "northing_orientation": "metadata extent [xmin,xmax,ymin,ymax]; row 0 is northern edge",
        "frame_time_s": dyn["time_s"],
        "static_channels": meta["static_channels"],
        "dynamic_channels": dyn["dynamic_channels"],
        "finite": bool(np.isfinite(state).all()),
        "nonnegative_h": bool((state[0] >= 0).all()),
        "active_cells": int((static[-1] > 0).sum()),
    }
