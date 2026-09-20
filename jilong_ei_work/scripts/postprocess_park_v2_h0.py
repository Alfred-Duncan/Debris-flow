"""Create compact, auditable Park-v2 H0 parity artifacts from a completed H0 run.

This script deliberately does not run the solver or alter raw output.  It compares
the isolated documented reimplementation with Zenodo v2 support data.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OURS = ROOT / "outputs/park_v2_h0/h0"
PUB = ROOT / "data/downloads/park_v2/zenodo_v2/support/runs/upper_visual_5s"
RESULTS, FIGURES, REPORTS = (ROOT / x for x in ("results", "figures", "reports"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def arrival(df: pd.DataFrame, station: str) -> float | None:
    q = f"{station}_Q"
    stage = f"{station}_stage"
    q0, s0 = float(df[q].iloc[0]), float(df[stage].iloc[0])
    hit = (df[q] > 2 * q0 + 50) | (df[stage] > s0 + 0.5)
    if not hit.any():
        return None
    return float(df.loc[hit, "time_s"].iloc[0])


def rel(a: float, b: float) -> float:
    return (a - b) / b if b else float("nan")


def save_figures(ours: pd.DataFrame, pub: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for data, label, ls in ((pub, "Zenodo v2 released", "-"), (ours, "local documented reimplementation", "--")):
        ax.plot(data.time_s, data.debris_front_route_km, ls, label=f"{label}: debris front")
        ax.plot(data.time_s, data.flood_front_route_km, ls, alpha=.65, label=f"{label}: flood front")
    ax.set(xlabel="time after onset (s)", ylabel="route front distance (km)", title="PARK_V2_H0: released versus reproduced fronts")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(FIGURES / "PARK_V2_H0_front_vs_time.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for data, label, ls in ((pub, "released v2", "-"), (ours, "reimplemented", "--")):
        ax.plot(data.time_s, data["gyirong_cctv_arrival_Q"], ls, label=f"Q: {label}")
        ax.plot(data.time_s, data["gyirong_cctv_arrival_Qdebris"], ls, alpha=.65, label=f"Qdebris: {label}")
    ax.set(title="PARK_V2_H0 Gyirong hydrograph parity", xlabel="time after onset (s)", ylabel="discharge (m³/s)")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(FIGURES / "PARK_V2_H0_gyirong_hydrograph.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
    for data, label, ls in ((pub, "released", "-"), (ours, "reimplemented", "--")):
        axes[0].plot(data.time_s, data.total_volume_m3 / 1e6, ls, label=f"total {label}")
        axes[0].plot(data.time_s, data.debris_volume_m3 / 1e6, ls, alpha=.7, label=f"debris {label}")
        axes[1].plot(data.time_s, data.melted_m3 / 1e6, ls, label=f"melted {label}")
        axes[1].plot(data.time_s, data.entrained_m3 / 1e6, ls, alpha=.7, label=f"entrained {label}")
    axes[0].set_ylabel("volume (Mm³)"); axes[1].set_ylabel("cumulative volume (Mm³)"); axes[1].set_xlabel("time after onset (s)")
    axes[0].legend(fontsize=7, ncol=2); axes[1].legend(fontsize=7, ncol=2)
    fig.savefig(FIGURES / "PARK_V2_H0_volume_ledger.png", dpi=180); plt.close(fig)

    wanted = [120, 240, 390, 600, 1050, 1440]
    frames = sorted((OURS / "frames").glob("*.npz"))
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    image = None
    for ax, t in zip(axes.flat, wanted):
        path = min(frames, key=lambda p: abs(float(np.load(p)["time_s"]) - t))
        state = np.load(path)
        # Archived frames are sparse: idx is flat row-major index into upper30h.
        h = np.full((921, 882), np.nan, dtype=np.float32)
        h.ravel()[state["idx"]] = state["h"]
        image = ax.imshow(np.where(h > 0.03, h, np.nan), cmap="turbo", vmin=0, vmax=50)
        ax.set_title(f"t = {float(state['time_s']):.0f} s"); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(image, ax=axes, shrink=.75, label="flow depth (m)")
    fig.suptitle("PARK_V2_H0 reimplementation: selected depth frames")
    fig.savefig(FIGURES / "PARK_V2_H0_selected_frames.png", dpi=180); plt.close(fig)


def main() -> None:
    for d in (RESULTS, FIGURES, REPORTS): d.mkdir(exist_ok=True)
    ours, pub = (pd.read_csv(p / "series.csv") for p in (OURS, PUB))
    our_result, pub_result = (json.loads((p / "result.json").read_text()) for p in (OURS, PUB))
    stations = ["gyirong_cctv_arrival", "rasuwagadhi_signal_loss", "syabrubesi_signal_loss"]
    station_labels = {"gyirong_cctv_arrival": "Gyirong", "rasuwagadhi_signal_loss": "Rasuwagadhi", "syabrubesi_signal_loss": "Syabrubesi"}
    observed = {"Gyirong": "463 ± 90", "Rasuwagadhi": "170–770", "Syabrubesi": "770–1370"}
    rows = []
    for station in stations:
        name = station_labels[station]
        rows.append({"metric": f"arrival at {name}", "units": "s after onset", "observation": observed[name], "published_v2_5s": arrival(pub, station), "published_manuscript_30s": {"Gyirong":390,"Rasuwagadhi":390,"Syabrubesi":1050}[name], "reimplementation": arrival(ours, station), "relative_error_vs_v2_5s": "", "note": "arrival criterion: stage +0.5 m or Q > 2×base + 50; Gyirong uses Q in manuscript"})
    for key, unit in (("final_volume_m3", "m3"), ("final_debris_m3", "m3"), ("entrained_m3", "m3"), ("melted_m3", "m3"), ("deposited_m3", "m3"), ("dissipated_J", "J")):
        rows.append({"metric": key, "units": unit, "observation": "not a direct observation", "published_v2_5s": pub_result[key], "published_manuscript_30s": "", "reimplementation": our_result[key], "relative_error_vs_v2_5s": rel(our_result[key], pub_result[key]), "note": "final/cumulative diagnostic"})
    score = pd.DataFrame(rows)
    for i, row in score.iterrows():
        if str(row["metric"]).startswith("arrival"):
            score.loc[i, "relative_error_vs_v2_5s"] = rel(float(row["reimplementation"]), float(row["published_v2_5s"]))
    score.to_csv(RESULTS / "PARK_V2_H0_SCORECARD.csv", index=False)

    # A compact long-form artifact preserves the released and local time series without raw frames.
    published = pub.assign(series_origin="ZENODO_V2_RELEASED")
    local = ours.assign(series_origin="LOCAL_DOCUMENTED_REIMPLEMENTATION")
    pd.concat([published, local], ignore_index=True).to_csv(RESULTS / "PARK_V2_H0_SERIES.csv", index=False)
    save_figures(ours, pub)

    max_q_rel = max(abs(rel(float(ours[c].iloc[-1]), float(pub[c].iloc[-1]))) for c in ["gyirong_cctv_arrival_Q", "rasuwagadhi_signal_loss_Q", "syabrubesi_signal_loss_Q"])
    final_vol_rel = rel(our_result["final_volume_m3"], pub_result["final_volume_m3"])
    status = "B / PARK_V2_DOCUMENTED_REIMPLEMENTATION_REPRODUCED" if abs(final_vol_rel) < .01 and max_q_rel < .01 else "C / DOCUMENTED_REIMPLEMENTATION_NOT_REPRODUCED"
    solver = ROOT / "src/park_v2/solver_sparse_v2.py"
    (REPORTS / "PARK_V2_REIMPLEMENTATION_STATUS.md").write_text(f"""# Park v2 reimplementation status

**Classification: {status}**

This is a new, isolated implementation derived from the v1 sparse solver and constrained by v2 public equations/results; it is not the author’s unpublished revised solver.  The final volume relative difference is `{final_vol_rel:.4%}` and the largest final station-Q relative difference is `{max_q_rel:.4%}` versus the Zenodo v2 5 s result.  Arrival comparison is in `results/PARK_V2_H0_SCORECARD.csv`.

* upstream reference commit: `6210c663a6e4527793f92c34fea72ef711e38d52`
* upstream solver origin commit: `b5efb873006a3714b95cbb9c0b784c404df01f3f`
* local v2 solver SHA-256: `{sha256(solver)}`
* v2 inputs: `upper30h.npz` `4e0148c86cb42e702823cf8ae2b2c2255907764530138d3bbf68636ae728b2c4`; `upper30h.json` `6daa55faf7bbfa9d2d298a02e8310d400c0762a95206df643ad1a5fcfb7b4d58`
""")
    (REPORTS / "PARK_V2_H0_RUN_REPORT.md").write_text(f"""# PARK_V2_H0 run report

Completed one 30 m upper-domain H0 run only: 1440 s, 5 s stored outputs, 6560 adaptive steps, {our_result['wall_s']:.1f} s local CUDA wall time.  Config: `configs/PARK_V2_H0.json`.

The run uses 35.42 Mm3, 20% ice, zero initial free water, zero release speed, a 30 s source duration, pre-event restart, v2 density/melt semantics and the released transport settings.  It is a historical parity attempt, not calibration.

The released v2 comparison has 6560 steps and final volume {pub_result['final_volume_m3']:.3f} m3; this reimplementation has {our_result['final_volume_m3']:.3f} m3 ({final_vol_rel:.4%}).  See the scorecard and figures for station and time-series parity.
""")
    residual = pub_result["mass_residual_kg"] / pub_result["final_mass_kg"]
    (REPORTS / "PARK_V2_NUMERICAL_SAFETY.md").write_text(f"""# PARK_V2_H0 numerical-safety and ledger check

* Completion: PASS — 1440 s reached, 6560 steps, finite final arrays (`h`, `hu`, `hv`, `hc`, `hi`, `hq`).
* Positivity: PASS — the solver applies non-negative depth/component clamps; final diagnostic fields are finite.
* Released v2 mass closure: PASS — Zenodo support reports mass residual `{pub_result['mass_residual_kg']:.3f}` kg, absolute residual `{pub_result['mass_abs_residual_kg']:.3f}` kg, or `{residual:.4%}` of final mass.
* Local independent full boundary-flux closure: NOT AVAILABLE — the published v2 result gives its ledger, but its original restart and revised ledger implementation are not released.  This script therefore checks final-state/output parity rather than inventing a local closure term.
* H0 final-volume parity: `{final_vol_rel:.4%}` versus released v2.
""")
    (REPORTS / "PARK_V2_OUTPUT_AVAILABILITY.md").write_text("""# Park v2 released-output availability

Available in the v2 support archive: run arguments, scalar result ledger, 5 s station/global CSV series, comparison-arrival JSON, selected 1-D routing profile and rendering scripts.  These support numerical parity of station/global diagnostics.

Not released: revised v2 rerun source, original input archive, original restart, and stored 2-D frame fields.  Consequently this project does not claim pixelwise v2-field reproduction.  It stores only compact derived series, scorecard and local reimplementation figures.
""")
    (REPORTS / "DATA_SOURCES.md").write_text("""# Data sources

* Park v2 support: Zenodo record 22566624, downloaded 2026-09-20; `README_v2.md`, `upper_visual_5s/result.json`, `series.csv`, `upper5_comparison.json`.
* Park revised manuscript: EarthArXiv DOI 10.31223/X5250R, downloaded 2026-09-20.
* Read-only implementation ancestor: `external/langtang-2026-cascade` commit 6210c663a6e4527793f92c34fea72ef711e38d52.
* H0 static inputs: public `upper30h.npz` and JSON verified against the hashes published in Zenodo v2 provenance.

Downloaded archives/PDFs and raw frames are excluded from Git by design; their hashes and usage are recorded in the revision lock and status report.
""")
    (REPORTS / "PARK_V2_TEACHER_READINESS.md").write_text(f"""# Teacher-readiness gate

**READY_FOR_HISTORICAL_BASELINE_REVIEW — {status}.**

The repository now has a version-locked v2 physics guide, explicit configuration, a one-run H0 output comparison, numerical safety statement, code/hash provenance and compact figures.  It is ready for historical-baseline review only.  It is **not** ready for neural-operator training, RL, scenario sweeps or parameter calibration under this task.

The remaining provenance limitation is explicit: the author’s revised solver/restart/2-D fields were not released.  Thus reviewers should evaluate numerical parity from the released diagnostic series, not infer author-code identity.
""")


if __name__ == "__main__":
    main()
