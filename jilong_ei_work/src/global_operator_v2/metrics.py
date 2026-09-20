"""Physics-unit engineering diagnostics; route data is evaluation-only."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch

def station_discharge(state: np.ndarray, transect: dict, cell_m: float=30.) -> float:
    """Q = sum (hu*north + hv*east) * cell width over supplied station line."""
    rows=np.asarray(transect["rows"],int); cols=np.asarray(transect["cols"],int); tangent=np.asarray(transect["tangent"],float)
    normal=np.array([-tangent[1],tangent[0]])
    return float(np.sum((state[1,rows,cols]*normal[0]+state[2,rows,cols]*normal[1])*cell_m)*float(transect.get("flux_scale",1.)))

def route_front(state: np.ndarray, route_chainage_m: np.ndarray, active: np.ndarray, dry: float=.03) -> float | None:
    valid=(state[0]>=dry)&active&np.isfinite(route_chainage_m)
    return float(np.max(route_chainage_m[valid])) if valid.any() else None

def arrival_time(times_s: np.ndarray, states: list[np.ndarray], rows: np.ndarray, cols: np.ndarray, dry: float=.03) -> float | None:
    for t,s in zip(times_s,states):
        if np.any(s[0,rows,cols]>=dry): return float(t)
    return None

def load_transects(path: str | Path) -> dict: return json.loads(Path(path).read_text(encoding="utf-8"))
