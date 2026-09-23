"""Run frozen Global and RiskTemporal-v1 B10 on the independent H0 anchor."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_engineering_roi_v2 as frozen
from scripts.paper.evaluate_untouched_test import setup
from scripts.run_global_v2_oracle_refinement import append_station, station_row, station_summary, summarize
from src.global_operator_v2.frame_adapter import dynamic_state
from src.global_operator_v2.oracle_refinement import StreamingMetrics

OUT = ROOT / "paper_results" / "h0"
SAMPLES = {0, 240, 480, 720, 960, 1200, 1440}
H0_ID = "PARK_V2_H0"


class H0Frames:
    def __init__(self):
        self.paths = []
        for path in (ROOT / "outputs" / "park_v2_h0" / "h0_global_eval_reference" / "frames").glob("frame_*.npz"):
            with np.load(path, allow_pickle=False) as frame: self.paths.append((float(frame["time_s"]), path))
        if not self.paths: raise RuntimeError("H0_REFERENCE_FRAMES_MISSING")
        self.paths.sort()
    def frame(self, time_s: int, shape: tuple[int, int]) -> np.ndarray:
        _, path = min(self.paths, key=lambda item: abs(item[0] - time_s))
        value, meta = dynamic_state(path, shape)
        # The archived first adaptive solver frame is timestamped at 0.973 s;
        # all requested 10-s rollout frames otherwise match exactly.  This is
        # reference-frame alignment only, never a method parameter adjustment.
        if abs(meta["time_s"] - time_s) > 1.0: raise RuntimeError("H0_REFERENCE_CADENCE_MISMATCH")
        return value


def write_field(method: str, time_s: int, state: torch.Tensor | np.ndarray) -> None:
    value = state[0].detach().cpu().numpy() if isinstance(state, torch.Tensor) else state
    np.savez_compressed(OUT / "fields" / f"{method}_t{time_s:04d}s.npz", h=value[0], hu=value[1], hv=value[2], c=value[3], ice=value[4], dz=value[5])


@torch.no_grad()
def run(label: str) -> dict:
    device, config, _, global_model, transform, normalizer, delta, _, static_np, static, _, normalization, _, local, route, transects, metadata = setup()
    frames = H0Frames(); shape = tuple(static_np.shape[-2:]); params = frozen.tensor(np.asarray([1.0, 0.2, 0.0074, 0.0145, 337.0], np.float32), device)
    previous = frozen.tensor(frames.frame(0, shape), device); current = previous.clone(); state = frozen.initial_temporal_state()
    metric = StreamingMetrics(metadata.active, route, delta.change_threshold); predicted, teacher, timeline, volume = [], [], [], []
    truth0 = frozen.tensor(frames.frame(0, shape), device); append_station(predicted, 0, station_row(truth0, static_np[0], transects)); append_station(teacher, 0, station_row(truth0, static_np[0], transects)); write_field("reference", 0, truth0); write_field(label, 0, truth0)
    timing = {key: [] for key in frozen.TIMING_STEP_KEYS}; started = time.perf_counter()
    for step in range(144):
        time_s = (step + 1) * 10
        provisional, global_seconds = frozen.timed_cuda(lambda: frozen.predict(global_model, previous, current, static, params, step / 144.0, transform, normalizer, metadata.active), device)
        if label == "RiskTemporal_B10":
            encoded_current, encoded_provisional = transform.encode(current), transform.encode(provisional)
            corrected, selected, state, telemetry, transition = frozen.v2_transition(previous, current, provisional, encoded_current, encoded_provisional, metadata, config, 0.10, state, "RiskTemporal", local, frozen.PatchLayout(static_np[1]), delta, normalization, static, params, step / 144.0, transform, normalizer, global_model, device)
        else:
            corrected, selected, telemetry = provisional, (), {"actual_active_fraction": 0.0, "selected_patch_consecutive_overlap_count": 0.0, "depth_guard_activation_fraction": 0.0}
            transition = {"roi_scoring_runtime_ms": 0.0, "local_correction_runtime_ms": 0.0, "support_guard_runtime_ms": 0.0, "depth_guard_runtime_ms": 0.0}
        truth_current = frozen.tensor(frames.frame(step * 10, shape), device); truth_next = frozen.tensor(frames.frame(time_s, shape), device)
        metric.add(corrected, truth_current, truth_next, transform)
        append_station(predicted, time_s, station_row(corrected, static_np[0], transects)); append_station(teacher, time_s, station_row(truth_next, static_np[0], transects))
        timing["global_steps_ms"].append(1000 * global_seconds); timing["roi_steps_ms"].append(transition["roi_scoring_runtime_ms"]); timing["local_steps_ms"].append(transition["local_correction_runtime_ms"]); timing["support_steps_ms"].append(transition["support_guard_runtime_ms"]); timing["depth_steps_ms"].append(transition["depth_guard_runtime_ms"])
        if label == "RiskTemporal_B10": timeline.append(telemetry | transition | {"method": label, "time_s": time_s, "selected_patch_ids": ";".join(str(p.patch_id) for p in selected), "selected_count": len(selected), "budget_max_count": 19})
        for name, value in (("reference", truth_next), (label, corrected)):
            arr = value[0, 0].detach().cpu().numpy(); volume.append({"method": name, "time_s": time_s, "wet_volume_proxy_m3": float((arr * static_np[1]).sum() * 900.0)})
        if time_s in SAMPLES: write_field("reference", time_s, truth_next); write_field(label, time_s, corrected)
        previous, current = current, corrected
    wall = time.perf_counter() - started
    case = metric.result() | {"method": label, "scenario_id": H0_ID, "budget_fraction": 0.10 if label == "RiskTemporal_B10" else 0.0, "selected_patch_count": 19.0 if label == "RiskTemporal_B10" else 0.0, "case_wall_runtime_seconds": wall}
    stations = [item | {"method": label, "scenario_id": H0_ID} for item in station_summary(predicted, teacher, transects)]
    return {"case": case, "stations": stations, "timeline": timeline, "volume": volume, "predicted": predicted, "teacher": teacher, "wall": wall, "timing": timing}


def main() -> None:
    for name in ("fields", "stations", "sections", "metrics", "roi", "figures"): (OUT / name).mkdir(parents=True, exist_ok=True)
    if (OUT / "h0_summary.csv").exists(): raise RuntimeError("H0_OUTPUT_EXISTS_REFUSE_TO_RERUN")
    all_runs = [run("FrozenGlobal"), run("RiskTemporal_B10")]
    cases = pd.DataFrame([item["case"] for item in all_runs]); stations = pd.DataFrame([row for item in all_runs for row in item["stations"]]); timeline = pd.DataFrame([row for item in all_runs for row in item["timeline"]]); volume = pd.DataFrame([row for item in all_runs for row in item["volume"]])
    hydro = pd.concat([pd.DataFrame(item["predicted"]).assign(method=item["case"]["method"], series_role="frozen_method") for item in all_runs] + [pd.DataFrame(all_runs[0]["teacher"]).assign(method="reference", series_role="high_fidelity_numerical_reference")], ignore_index=True)
    cases.to_csv(OUT / "metrics" / "h0_case_metrics.csv", index=False); stations.to_csv(OUT / "stations" / "h0_station_metrics.csv", index=False); hydro.to_csv(OUT / "stations" / "station_hydrographs.csv", index=False); volume.to_csv(OUT / "metrics" / "volume_evolution_proxy.csv", index=False); timeline.to_csv(OUT / "roi" / "selection_timeline.csv", index=False)
    qcols = ["time_s", "method", "series_role"] + [col for col in hydro.columns if col.endswith("_Q")]
    hydro[qcols].to_csv(OUT / "sections" / "section_discharge.csv", index=False)
    summary = cases.copy(); summary["historical_anchor"] = True; summary["reference_type"] = "PARK_V2_DOCUMENTED_REIMPLEMENTATION_REPRODUCED"; summary.to_csv(OUT / "h0_summary.csv", index=False)
    report = {"status": "PASS", "case_id": H0_ID, "purpose": "historical anchor / event-constrained reconstruction against a high-fidelity numerical reference", "not_claimed": ["ground-truth disaster reconstruction", "fully validated real event"], "methods": ["FrozenGlobal", "RiskTemporal-v1 B10"], "reference": "PARK_V2_DOCUMENTED_REIMPLEMENTATION_REPRODUCED", "runtime_seconds": {item["case"]["method"]: item["wall"] for item in all_runs}, "risktemporal_temporal_overlap": float(timeline.selected_patch_consecutive_overlap_count.mean()), "risktemporal_depth_guard_activation": float(timeline.depth_guard_activation_fraction.mean())}
    (OUT / "H0_REPORT.md").write_text("# H0 historical anchor\n\n**PASS** — frozen methods were evaluated without calibration against the documented Park-v2 numerical reference. This is an event-constrained engineering demonstration, not a ground-truth disaster reconstruction or a fully validated real event.\n", encoding="utf-8")
    (OUT / "metrics" / "h0_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__": main()
