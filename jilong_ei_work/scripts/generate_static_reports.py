from __future__ import annotations
import csv, ctypes, hashlib, json, platform, subprocess, sys
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[1]
reports, ext, downloads = root / "reports", root / "external", root / "data" / "downloads"

def cmd(*parts: str) -> str:
    return subprocess.check_output(parts, text=True, encoding="utf-8", errors="replace").strip()

def repo_record(name: str) -> dict:
    path = ext / name
    return {"repo": name, "remote_url": cmd("git", "-C", str(path), "remote", "get-url", "origin"), "commit_sha": cmd("git", "-C", str(path), "rev-parse", "HEAD"), "branch": cmd("git", "-C", str(path), "branch", "--show-current"), "dirty": bool(cmd("git", "-C", str(path), "status", "--porcelain"))}

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

(reports / "EXTERNAL_REPOSITORIES.json").write_text(json.dumps({"repositories":[repo_record("langtang-2026-cascade"),repo_record("RVPI-PDE")],"policy":"Both directories are read-only upstreams; no tracked upstream file is modified."},indent=2)+"\n",encoding="utf-8")
gpu = subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv,noheader"],text=True).strip()
class MemoryStatus(ctypes.Structure):
    _fields_=[("dwLength",ctypes.c_ulong),("dwMemoryLoad",ctypes.c_ulong),("ullTotalPhys",ctypes.c_ulonglong),("ullAvailPhys",ctypes.c_ulonglong),("ullTotalPageFile",ctypes.c_ulonglong),("ullAvailPageFile",ctypes.c_ulonglong),("ullTotalVirtual",ctypes.c_ulonglong),("ullAvailVirtual",ctypes.c_ulonglong),("ullAvailExtendedVirtual",ctypes.c_ulonglong)]
memory=MemoryStatus(); memory.dwLength=ctypes.sizeof(MemoryStatus); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
(reports / "LOCAL_HARDWARE.json").write_text(json.dumps({"os":platform.platform(),"cpu":platform.processor(),"ram_total_bytes":int(memory.ullTotalPhys),"python":sys.version,"conda_environment":"base","gpu_nvidia_smi":gpu,"torch":torch.__version__,"cuda_runtime":torch.version.cuda,"torch_cuda_available":torch.cuda.is_available(),"torch_device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU","vram_total_bytes":int(torch.cuda.get_device_properties(0).total_memory) if torch.cuda.is_available() else 0},indent=2)+"\n",encoding="utf-8")

files=[("corridor60s.npz",24181719,"22549201","required preprocessed 60 m sparse-solver input"),("corridor60s.json",5589,"22549201","grid metadata; reused from matching GitHub upstream"),("spinup60s_c/final_state.npz",None,"22549201","historical checkpoint absent from inspected manifest"),("prodp60_series.csv",423887,"22549201","small original-production comparison series"),("prodp60_result.json",7630,"22549201","small original-production parameter/result record")]
with (reports / "ZENODO_FILE_MANIFEST.csv").open("w",newline="",encoding="utf-8") as handle:
    writer=csv.DictWriter(handle,fieldnames=["filename","size_bytes","record","purpose","downloaded","sha256"]); writer.writeheader()
    for name,size,record,purpose in files:
        path=downloads / Path(name).name
        writer.writerow({"filename":name,"size_bytes":size or "","record":record,"purpose":purpose,"downloaded":path.exists(),"sha256":sha(path) if path.exists() else ""})

(reports / "EXTERNAL_MODEL_VERSION_NOTE.md").write_text("""# External model version note

| Quantity | ORIGINAL CODE RELEASE | REVISED MODEL / REVISED ANIMATIONS | Reproducible locally now | Benchmark use |
|---|---|---|---|---|
| Release state | Historical production command releases rock-ice over 30 s. | README states revised animations add source material at rest. | Original command only. | Software/data-interface smoke only. |
| Release speed | Historical command uses release-speed 150; source code initializes downslope momentum from that speed. | Revised note describes material added at rest. | 150 m/s path is reproducible; at-rest implementation is not supplied in current code. | Do not mix semantics. |
| Melting | Historical code supports heat/melt terms. | Revised archive says melting is mass-conserving. | Original code release. | No event-validation claim in this workspace. |

The 120-s run validates software, CUDA, output frames, and tensor interfaces. It is not a final event reconstruction or a validated reference simulation.
""",encoding="utf-8")

(reports / "PHYSICS_OUTPUT_SCHEMA.md").write_text("""# Physics output schema

Definitions are read from upstream swe/solver_sparse.py; this workspace does not alter them.

| Name | Class | Upstream semantics |
|---|---|---|
| h, hu, hv, c, ice, bed_change | cell fields in sparse frames | Wet-cell fields indexed by row-major idx; c=hc/h, ice=hi/h; bed_change=z-z0. |
| speed | cell field in sparse frames | sqrt(u squared + v squared), where u=hu/h and v=hv/h. |
| Q, Qdebris, hmax, stage, cmax, wet_width_m | transect metrics in series.csv | Flux and geometry across author-defined transect cells, prefixed by station ID. |
| total_volume_m3, debris_volume_m3, entrained_m3, ice_volume_m3, melted_m3 | scalar time diagnostics | Domain-integrated model quantities. |
| debris_front_route_km, debris_front_upper_km, flood_front_route_km | scalar time diagnostics | Front locations using stored chainage fields. |

Sparse frame orientation: idx = row * cols + col; dense fields are [C, rows, cols] using NumPy C-order.
""",encoding="utf-8")

(reports / "ENGINEERING_METRIC_SCHEMA.json").write_text(json.dumps({"status":"PASS","available_prefix":"gyirong_cctv_arrival","metrics":{"arrival_time":"Derived later from threshold rule; not asserted by 120-s smoke.","peak_Q":"transect Q maximum","peak_Qdebris":"transect Qdebris maximum","peak_stage_or_rise":"transect stage field","max_depth":"transect hmax/global max_h_m","wet_width":"transect wet_width_m","front_position":"three front chainage diagnostics","runtime":"series/result wall time"},"smoke_scope":"schema/path verification; non-arrival at Gyirong during short run is normal"},indent=2)+"\n",encoding="utf-8")

(reports / "RVPI_REUSE_MAP.md").write_text("""# RVPI-PDE reuse map

| Component | Source file | Source class/function | Decision | Required adaptation / excluded reason |
|---|---|---|---|---|
| Global neural operator | rvpi/models/fno.py | FNO2d | REUSE_WITH_ADAPTER | Reused through project wrapper; field channels differ from upstream data. |
| Conditional operator | rvpi/models/conditional_fno.py | ConditionalFNO2d | REUSE_WITH_ADAPTER | Candidate for later conditioned histories; not needed for first smoke. |
| Local correction proposal | rvpi/models/fno.py | FNO2d | REUSE_WITH_ADAPTER | Wrapper applies it only to a manually selected patch. |
| Autoregressive rollout pattern | rvpi/pipelines/shallow_water.py | coarse rollout helpers | REUSE_WITH_ADAPTER | Future transition dataset can adopt it after scenarios exist. |
| Selector/RV-PI/policy/budget logic | rvpi/models/set_aware_selector.py, macro_policy.py, rvpi/env | RL components | DO_NOT_REUSE_FOR_CHINESE_EI | Explicitly excluded in this phase. |

Only the non-RL FNO building block is imported in the smoke demo. No selector, policy, reward, or RL training loop is run.
""",encoding="utf-8")
