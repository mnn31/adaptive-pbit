# Author: Manan Gupta <mnn@yogins.com>
"""Main experiment suite. All results under results/v2/.

Phases (run with --phase N, resumable; existing outputs are skipped ONLY
if their recorded git SHA matches HEAD, preventing version mixing):
  0  train baselines (val split) + correlation-aware models
  1  E1 ideal anchors
  2  E2 sampling-ratio sweep, settle + streaming modes (+ const-tau variant)
  3  E3 heterogeneity (both centerings, independent seeds)
  4  E4 correlation-aware evals (test + val for selection)
  5  steady-state verification (lane-count invariance)
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from smtjnn.data import splits
from smtjnn.infer import evaluate_lanes, evaluate_ideal
from smtjnn.smtj import IdealSource, TelegraphSource
from smtjnn.train import train, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v2"

R0_GRID = [10, 4, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02]
SETTLE = 4.0

_SPLITS = {}


def data(ds):
    if ds not in _SPLITS:
        _SPLITS[ds] = splits(ds)
    return _SPLITS[ds]


def sha():
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                   cwd=ROOT).decode().strip()


def done(path):
    if not path.exists():
        return False
    return json.loads(path.read_text()).get("git") == sha()


def save(path, rec):
    rec = dict(rec, git=sha())
    path.write_text(json.dumps(rec))
    print(path.name, rec.get("acc"), flush=True)


def lane_params(T, r0):
    lanes = 100 if T <= 4 else 200
    if np.isfinite(r0) and r0 > 0:
        warm = int(min(100, max(10, np.ceil(6.0 / (T * r0)))))
    else:
        warm = 5
    return lanes, warm


def eval_stream(model, ds, src, T, split="test"):
    trx, _, vx, vy, tx, ty = data(ds)
    xs, ys = (vx, vy) if split == "val" else (tx, ty)
    lanes, warm = lane_params(T, src.r0)
    return evaluate_lanes(model, xs, ys, src, T=T, lanes=lanes,
                          warmup_x=trx, warmup_per_lane=warm)


def eval_settle(model, ds, src, T, split="test"):
    trx, _, vx, vy, tx, ty = data(ds)
    xs, ys = (vx, vy) if split == "val" else (tx, ty)
    lanes, _ = lane_params(T, src.r0)
    return evaluate_lanes(model, xs, ys, src, T=T, lanes=lanes,
                          warmup_x=trx, warmup_per_lane=0)


# ---------------------------------------------------------------- phase 0
def phase0():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    for ds, seeds in [("mnist", range(5)), ("fashion", range(3))]:
        for s in seeds:
            name = f"v2_{ds}_seed{s}.pt"
            if not (ROOT / "results" / name).exists():
                train(dataset=ds, epochs=30, seed=s, device=dev,
                      ckpt_name=name)
    for tr0, seeds in [(0.5, range(3)), (0.2, range(3)), (0.1, range(1))]:
        for s in seeds:
            name = f"v2_mnist_na{tr0}_seed{s}.pt"
            if not (ROOT / "results" / name).exists():
                src = TelegraphSource(r0=tr0, streaming=True,
                                      noise_seed=123 + s)
                train(dataset="mnist", epochs=30, seed=s, device="cpu",
                      ckpt_name=name, source=src)


# ---------------------------------------------------------------- phase 1
def phase1():
    for ds, seeds in [("mnist", range(5)), ("fashion", range(3))]:
        _, _, _, _, tx, ty = data(ds)
        for s in seeds:
            model = load_checkpoint(ROOT / "results" / f"v2_{ds}_seed{s}.pt")
            for T in [1, 2, 4, 8, 16, 32, 64]:
                p = OUT / f"e1_{ds}_s{s}_T{T}.json"
                if done(p):
                    continue
                acc = evaluate_ideal(model, tx, ty, IdealSource(seed=s), T=T)
                save(p, dict(ds=ds, model_seed=s, T=T, acc=acc))


# ---------------------------------------------------------------- phase 2
def phase2(ds="mnist"):
    model = load_checkpoint(ROOT / "results" / f"v2_{ds}_seed0.pt")
    for mode in ["settle", "stream"]:
        for r0 in R0_GRID:
            for T in [1, 2, 4, 8, 16, 32, 64]:
                nseeds = 5 if T in (1, 32) else 3
                for ns in range(nseeds):
                    p = OUT / f"e2_{ds}_{mode}_r{r0}_T{T}_ns{ns}.json"
                    if done(p):
                        continue
                    src = TelegraphSource(
                        r0=r0, streaming=(mode == "stream"),
                        settle_ratio=SETTLE, noise_seed=ns)
                    acc = (eval_stream if mode == "stream"
                           else eval_settle)(model, ds, src, T)
                    save(p, dict(ds=ds, mode=mode, r0=r0, T=T, ns=ns,
                                 settle=SETTLE, acc=acc))
    # settle-ratio sweep at key points
    for rho in [1.0, 2.0, 8.0]:
        for r0 in [0.02, 0.2]:
            for T in [1, 32]:
                for ns in range(3):
                    p = OUT / f"e2rho_{ds}_rho{rho}_r{r0}_T{T}_ns{ns}.json"
                    if done(p):
                        continue
                    src = TelegraphSource(r0=r0, streaming=False,
                                          settle_ratio=rho, noise_seed=ns)
                    acc = eval_settle(model, ds, src, T)
                    save(p, dict(ds=ds, mode="settle", rho=rho, r0=r0, T=T,
                                 ns=ns, acc=acc))
    # const-tau variant, streaming
    for r0 in R0_GRID:
        for T in [1, 32]:
            for ns in range(3):
                p = OUT / f"e2ct_{ds}_r{r0}_T{T}_ns{ns}.json"
                if done(p):
                    continue
                src = TelegraphSource(r0=r0, streaming=True, noise_seed=ns,
                                      bias_independent_tau=True)
                acc = eval_stream(model, ds, src, T)
                save(p, dict(ds=ds, mode="stream_ct", r0=r0, T=T, ns=ns,
                             acc=acc))


# ---------------------------------------------------------------- phase 3
def phase3(ds="mnist"):
    model = load_checkpoint(ROOT / "results" / f"v2_{ds}_seed0.pt")
    T = 32
    for centering in ["mean-tau", "median"]:
        for r0 in [0.05, 0.1, 0.2, 0.5]:
            for sd in [0.0, 0.25, 0.5, 1.0, 1.5]:
                for chip in range(5):
                    p = OUT / (f"e3_{ds}_{centering}_r{r0}_sd{sd}"
                               f"_c{chip}.json")
                    if done(p):
                        continue
                    src = TelegraphSource(
                        r0=r0, streaming=True, sigma_delta=sd,
                        chip_seed=chip, noise_seed=1000 + 17 * chip,
                        centering=centering)
                    acc = eval_stream(model, ds, src, T)
                    save(p, dict(ds=ds, centering=centering, r0=r0,
                                 sigma_delta=sd, chip=chip, T=T, acc=acc))


# ---------------------------------------------------------------- phase 4
def phase4(ds="mnist"):
    models = [(tr0, s) for tr0, ss in [(0.5, range(3)), (0.2, range(3)),
                                       (0.1, range(1))] for s in ss]
    for tr0, s in models:
        model = load_checkpoint(ROOT / "results" /
                                f"v2_{ds}_na{tr0}_seed{s}.pt")
        for split in ["val", "test"]:
            for er0 in [1, 0.5, 0.2, 0.1, 0.05, 0.02]:
                for T in [1, 8, 32]:
                    for ns in range(3):
                        p = OUT / (f"e4_{ds}_tr{tr0}_s{s}_{split}"
                                   f"_er{er0}_T{T}_ns{ns}.json")
                        if done(p):
                            continue
                        src = TelegraphSource(r0=er0, streaming=True,
                                              noise_seed=ns)
                        acc = eval_stream(model, ds, src, T, split=split)
                        save(p, dict(ds=ds, train_r0=tr0, model_seed=s,
                                     split=split, eval_r0=er0, T=T, ns=ns,
                                     acc=acc))


# ---------------------------------------------------------------- phase 5
def phase5(ds="mnist"):
    model = load_checkpoint(ROOT / "results" / f"v2_{ds}_seed0.pt")
    trx, _, _, _, tx, ty = data(ds)
    for lanes in [50, 100, 200, 500]:
        p = OUT / f"e5conv_{ds}_lanes{lanes}.json"
        if done(p):
            continue
        src = TelegraphSource(r0=0.05, streaming=True, noise_seed=0)
        acc = evaluate_lanes(model, tx, ty, src, T=32, lanes=lanes,
                             warmup_x=trx, warmup_per_lane=50)
        save(p, dict(ds=ds, lanes=lanes, r0=0.05, T=32, warm=50, acc=acc))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True)
    ap.add_argument("--dataset", default="mnist")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fn = [phase0, phase1, phase2, phase3, phase4, phase5][args.phase]
    if args.phase in (0, 1):
        fn()
    else:
        fn(args.dataset)
    print(f"phase {args.phase} complete in {time.time() - t0:.0f}s",
          flush=True)
