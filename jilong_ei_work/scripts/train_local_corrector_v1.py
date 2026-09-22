"""Resume-safe TRAIN-only Local Corrector V1 optimizer."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.global_operator_v2.dataset import FrameStore, INPUT, build_features, scenario_rows
from src.global_operator_v2.frame_adapter import static_and_exogenous
from src.global_operator_v2.losses import project_physical
from src.global_operator_v2.oracle_refinement import PatchLayout, oracle_patch_scores, select_random
from src.local_corrector.model import JilongLocalCorrector
from src.local_corrector.trainer import apply_learned_correction, correction_loss
from src.local_corrector.v1_1 import restore_local_rng, snapshot_local_rng, true_front_patch
from scripts.run_global_v2_oracle_refinement import load_model, predict

SEED = 20260920
STATE_NAMES = ("h", "hu", "hv", "c", "ice", "dz")


def _tensor(value, device):
    return torch.from_numpy(np.asarray(value, np.float32)).unsqueeze(0).to(device)


def _phase(step: int) -> str:
    return "A" if step < 2000 else "B"


def _probabilities(step: int):
    return (("teacher", .70), ("coarse", .30), ("refined", 0.)) if _phase(step) == "A" else (("teacher", .50), ("coarse", .25), ("refined", .25))


def _choose_mode(rng, step):
    names, probs = zip(*_probabilities(step))
    return str(rng.choice(names, p=probs))


def _pick_training_patches(layout, provisional_t, truth_t, provisional_p, truth_p, active, delta, rng, route_chainage_m=None):
    """TRAIN_ONLY_PATCH_SAMPLER: 4 random, 2 error, 1 front, 1 FP wet."""
    eligible = list(layout.eligible)
    selected = []
    def add(p):
        if p not in selected:
            selected.append(p)
    for p in rng.choice(eligible, size=min(4, len(eligible)), replace=False):
        add(p)
    scores, _, _ = oracle_patch_scores(provisional_t, truth_t, provisional_p, truth_p, active, layout, delta.scales)
    for index in np.argsort(-scores, kind="stable"):
        add(eligible[int(index)])
        if len(selected) >= 6:
            break
    front=true_front_patch(layout,truth_p,active,route_chainage_m) if route_chainage_m is not None else None
    if front is not None:add(front)
    fp = (provisional_p[0, 0].ge(.05) & truth_p[0, 0].lt(.05)).detach().cpu().numpy()
    fp_rank = sorted(eligible, key=lambda p: (-int(fp[p.r0:p.r1, p.c0:p.c1].sum()), p.patch_id))
    for p in fp_rank:
        if fp[p.r0:p.r1, p.c0:p.c1].any():
            add(p); break
    for p in rng.permutation(eligible):
        add(p)
        if len(selected) == 8:
            break
    return tuple(selected[:8])


@torch.no_grad()
def _burn(mode, row, store, params, static, active, global_model, local_model, transform, normalizer,
          normalization, delta, layout, device, rng, target_t):
    """Construct a no-future-leakage state at target_t from an older truth pair."""
    if mode == "teacher" or target_t == 0:
        previous = _tensor(store.frame(row, max(0, target_t - 10)), device)
        return previous, _tensor(store.frame(row, target_t), device)
    k_choices = (1, 2, 4) if mode == "coarse" else (1, 2, 4, 8)
    k = min(int(rng.choice(k_choices)), target_t // 10)
    start = target_t - k * 10
    previous = _tensor(store.frame(row, max(0, start - 10)), device)
    current = _tensor(store.frame(row, start), device)
    for burn_step in range(k):
        fraction = (start + burn_step * 10) / 1440.
        provisional = predict(global_model, previous, current, static, params, fraction, transform, normalizer, active)
        if mode == "refined":
            features = build_features(previous, current, static, params, torch.tensor([fraction], device=device), transform, normalizer)
            patches = select_random(layout, layout.count_for_budget(.10), SEED + int(row.design_index) * 100000 + target_t * 10 + burn_step)
            encoded, _, _ = apply_learned_correction(local_model, features, transform.encode(provisional), transform.encode(current), patches, layout, delta.scales, normalization["scales"], normalization["bounds"], active, global_model)
            provisional = project_physical(transform.decode(encoded), active)
        previous, current = current, provisional
    return previous, current


def _save(path, model, optimizer, scheduler, update, best_score=None, rng=None):
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "update": int(update), "phase": _phase(max(update - 1, 0)), "best_score": best_score,
             "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    if rng is not None:state['local_rng_state']=snapshot_local_rng(rng)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train(smoke=False, resume=False, total_updates=None, version='v1'):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    rows = scenario_rows("TRAIN")
    if len(rows) != 160 or not set(rows["split"]).issubset({"TRAIN"}) or rows.scenario_id.str.contains("TEST|H0", case=False).any():
        raise RuntimeError("TRAIN_SCOPE_REQUIRED")
    out = ROOT / f"models/local_corrector_{version}"
    normalization = json.loads((out / "correction_normalization.json").read_text())
    if normalization.get("fit_scope") != "TRAIN_ONLY":
        raise RuntimeError("LOCAL_NORMALIZATION_NOT_TRAIN_ONLY")
    store = FrameStore(rows); global_model, transform, normalizer, delta, _ = load_model(device)
    global_model.eval()
    static_np, _ = static_and_exogenous(INPUT)
    static = _tensor(static_np, device); active = static[:, 1:2]
    layout = PatchLayout(static_np[1], 16, 16)
    model = JilongLocalCorrector(47).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    total = int(total_updates or (20 if smoke else 6000))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1., (s + 1) / 250.) * (.5 + .5 * np.cos(np.pi * max(s - 250, 0) / max(total - 250, 1))))
    start = 0
    last = out / "last.pt"
    if resume:
        if not last.exists():
            raise RuntimeError("RESUME_REQUESTED_BUT_LAST_CHECKPOINT_MISSING")
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"]); scheduler.load_state_dict(ck["scheduler"])
        random.setstate(ck["python_rng"]); np.random.set_state(ck["numpy_rng"]); torch.set_rng_state(ck["torch_rng"])
        if ck.get("cuda_rng") is not None: torch.cuda.set_rng_state_all(ck["cuda_rng"])
        start = int(ck["update"])
    rng = restore_local_rng(ck['local_rng_state']) if resume and ck.get('local_rng_state') is not None else np.random.default_rng(SEED + start)
    log_path = ROOT / f"results/local_corrector_{version}/training_log.jsonl"; log_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(INPUT) as source:route_chainage_m=np.asarray(source['route_chainage_m'],np.float32)
    for update in range(start, total):
        model.train(); mode = _choose_mode(rng, update); index = int(rng.integers(len(rows))); row = rows.iloc[index]
        target_t = int(rng.integers(0, 144)) * 10
        params = _tensor(np.asarray([row[p] for p in ("volume_scale", "ice_fraction", "erosion_K", "n_debris", "dep_tau_s")], np.float32), device)
        previous, current = _burn(mode, row, store, params, static, active, global_model, model, transform, normalizer, normalization, delta, layout, device, rng, target_t)
        truth = _tensor(store.frame(row, target_t + 10), device)
        fraction = target_t / 1440.
        with torch.no_grad():
            provisional = predict(global_model, previous, current, static, params, fraction, transform, normalizer, active)
            features = build_features(previous, current, static, params, torch.tensor([fraction], device=device), transform, normalizer)
            patches = _pick_training_patches(layout, transform.encode(provisional), transform.encode(truth), provisional, truth, active, delta, rng, route_chainage_m)
        optimizer.zero_grad(set_to_none=True)
        loss, details = correction_loss(model, features, transform.encode(provisional), transform.encode(truth), transform.encode(current), provisional, truth, patches, layout, delta.scales, normalization["scales"], normalization["bounds"], active, delta.change_threshold, global_model)
        if not torch.isfinite(loss):
            _save(last, model, optimizer, scheduler, update, rng=rng)
            raise RuntimeError("NONFINITE_LOCAL_LOSS")
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.).detach())
        if not np.isfinite(grad):
            _save(last, model, optimizer, scheduler, update, rng=rng)
            raise RuntimeError("NONFINITE_LOCAL_GRADIENT")
        optimizer.step(); scheduler.step()
        record = {"update": update + 1, "phase": _phase(update), "state_mode": mode, "loss": float(loss.detach()), "grad_norm": grad,
                  "saturation": details["saturation"], "lr": optimizer.param_groups[0]["lr"]}
        if (update + 1) % 20 == 0 or update == start:
            print(json.dumps(record), flush=True)
            with log_path.open("a", encoding="utf8") as handle: handle.write(json.dumps(record) + "\n")
        if (update + 1) % 500 == 0 or update + 1 == total:
            _save(last, model, optimizer, scheduler, update + 1, rng=rng)
            if version != 'v1':shutil.copy2(last,out/f'candidate_{update+1:04d}.pt')
    return last


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--updates", type=int)
    args = parser.parse_args(); train(args.smoke, args.resume, args.updates)
