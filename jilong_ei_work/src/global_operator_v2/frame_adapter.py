"""Lossless V2 bridge from sparse solver frames to dense physical fields.

The solver stores mixture concentration as ``c`` and ice fraction as ``ice``
only at wet cells.  V2 intentionally retains the dense, cumulative ``dz``
field so that the predictor state exposes bed evolution rather than treating
topography as immutable.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np

STATE_NAMES = ("h", "hu", "hv", "c", "ice", "dz")

def _dense(idx: np.ndarray, values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape[0] * shape[1], dtype=np.float32)
    if idx.size:
        if int(idx.min()) < 0 or int(idx.max()) >= out.size:
            raise ValueError("sparse index outside input grid")
        out[idx.astype(np.int64)] = values.astype(np.float32)
    return out.reshape(shape, order="C")

def input_metadata(input_npz: str | Path) -> dict[str, Any]:
    p = Path(input_npz)
    metadata = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(p, allow_pickle=False) as data:
        metadata["shape"] = list(data["z"].shape)
        metadata["available_static_fields"] = list(data.files)
    return metadata

def static_and_exogenous(input_npz: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Return physical static maps [z0, active, source, channel, lateral_q, inflow]."""
    p = Path(input_npz); meta = input_metadata(p)
    with np.load(p, allow_pickle=False) as data:
        shape = data["z"].shape
        inflow = np.zeros(shape, dtype=np.float32)
        for item in meta.get("external_inflows", []):
            r, c = int(item["row"]), int(item["col"])
            if not (0 <= r < shape[0] and 0 <= c < shape[1]):
                raise ValueError(f"external inflow outside grid: {item}")
            inflow[r, c] += float(item["q_m3_s"])
        names = ["z_initial", "active", "source_mask", "channel", "lateral_q_active_m3_s", "external_inflow_q_m3_s"]
        arr = np.stack((data["z"], data["active"], data["source_mask"], data["channel"],
                        data["lateral_q_active_m3_s"], inflow), axis=0).astype(np.float32)
    meta.update({"static_channels": names, "cell_m": float(meta["cell_m"]),
                 "flattening": "numpy C-order; row=idx//cols, col=idx%cols"})
    return arr, meta

def dynamic_state(frame_npz: str | Path, shape: tuple[int, int]) -> tuple[np.ndarray, dict[str, Any]]:
    """Dense state [h, hu, hv, c, ice, dz], with dry/outside values exactly zero."""
    p = Path(frame_npz)
    with np.load(p, allow_pickle=False) as frame:
        idx = np.asarray(frame["idx"], dtype=np.int64)
        fields = [_dense(idx, np.asarray(frame[k]), shape) for k in ("h", "hu", "hv", "c", "ice")]
        dz = _dense(np.asarray(frame["dz_idx"], dtype=np.int64), np.asarray(frame["dz"]), shape)
        meta = {"time_s": float(frame["time_s"]), "frame_path": str(p), "state_channels": list(STATE_NAMES),
                "wet_sparse_count": int(idx.size), "dz_sparse_count": int(np.asarray(frame["dz_idx"]).size)}
    return np.stack((*fields, dz), axis=0), meta

def physical_state_from_final(final_npz: str | Path) -> dict[str, np.ndarray]:
    """Expose all stored final solver quantities without pretending hq is known in frames."""
    with np.load(final_npz, allow_pickle=False) as f:
        return {name: np.asarray(f[name], dtype=np.float32) for name in f.files if name != "time_s"}
