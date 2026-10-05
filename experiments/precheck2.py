# Author: Manan Gupta <mnn@yogins.com>
"""Week-1 pre-check v2: escalation arms after precheck v1 came in under the
go/no-go threshold (stopping error 0.07x-0.20x nominal, trending up as r0
falls, mean samples falling — right signature, diluted magnitude).

Diagnosis encoded here:
(a) Arrhenius devices self-decorrelate under bias (tau ~ 1/cosh I), so
    saturated (easy) neurons are fast AND deterministic; the effect lives in
    the easy-plane class (bias_independent_tau=True) where bias does not
    speed the device up.
(b) The guarantee is per-image; aggregate error is diluted by easy images.
    Stratify stopping error by the ideal-source reference margin.
(c) Streaming mode removes the per-input settle decorrelation.

Arms: (1) arrhenius+settle (v1 continuity), (2) easyplane+settle,
(3) easyplane+streaming. r0 in {inf, 2, 1, 0.5, 0.25}. Val split only.
"""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from apbit.data import splits
from apbit.diagnostics import vote_lag1_acf
from apbit.infer_adaptive import evaluate_lanes_adaptive, reference_decisions
from apbit.smtj import IdealSource, TelegraphSource
from apbit.stopping import DirichletStop
from apbit.train import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CKPT = RESULTS / "sbnn_mnist_seed0.pt"

ALPHA = 0.05
T_MAX = 64
T_MIN = 4
T_REF = 1024
N_IMAGES = 2000
LANES = 100
R0S = [np.inf, 2.0, 1.0, 0.5, 0.25]
MARGIN_CUT = 0.9  # ideal top1-top2 vote-share margin below this = "marginal"
NOISE_SEED = 1000
CHIP_SEED = 0


def git_sha():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def stratified(out, ref, keep, marginal):
    d = out["decisions"]
    res = {}
    for name, m in (("all", keep), ("marginal", keep & marginal),
                    ("easy", keep & ~marginal)):
        n = int(m.sum())
        res[name] = {
            "n": n,
            "stopping_error": float((d[m] != ref[m]).mean()) if n else None,
            "mean_samples": float(out["samples_used"][m].mean()) if n else None,
        }
    return res


def make_source(kind, r0, seed):
    if np.isinf(r0):
        return IdealSource(seed=seed)
    if kind == "arrhenius_settle":
        return TelegraphSource(r0=r0, chip_seed=CHIP_SEED, noise_seed=seed)
    if kind == "easyplane_settle":
        return TelegraphSource(r0=r0, chip_seed=CHIP_SEED, noise_seed=seed,
                               bias_independent_tau=True)
    if kind == "easyplane_streaming":
        return TelegraphSource(r0=r0, chip_seed=CHIP_SEED, noise_seed=seed,
                               bias_independent_tau=True, streaming=True)
    raise ValueError(kind)


def main():
    t0 = time.time()
    model = load_checkpoint(CKPT)
    model.eval()
    trx, _, vx, vy, _, _ = splits("mnist")
    xs, ys = vx[:N_IMAGES], vy[:N_IMAGES]
    warm = trx[:2 * LANES]

    ref, tie, margin = reference_decisions(
        model, xs, IdealSource(seed=7), t_ref=T_REF, batch=1000,
        return_margin=True)
    keep = ~tie
    marginal = margin < MARGIN_CUT
    print(f"marginal images (ideal margin < {MARGIN_CUT}): "
          f"{int(marginal.sum())}/{N_IMAGES}", flush=True)

    arms = ["arrhenius_settle", "easyplane_settle", "easyplane_streaming"]
    rows = []
    for kind in arms:
        for r0 in R0S:
            tag = "inf" if np.isinf(r0) else str(r0)
            src = make_source(kind, r0, NOISE_SEED)
            kw = {}
            if kind == "easyplane_streaming" and not np.isinf(r0):
                kw = {"warmup_x": warm, "warmup_per_lane": 2}
            out = evaluate_lanes_adaptive(
                model, xs, ys, src,
                lambda: DirichletStop(alpha=ALPHA, k_classes=10,
                                      t_min=T_MIN, t_max=T_MAX),
                t_max=T_MAX, lanes=LANES, collect_votes=True,
                reference=ref, reference_ties=tie, **kw)
            acf_m, acf_n = vote_lag1_acf(out["votes"], marginal)
            row = {"arm": kind, "r0": tag,
                   "strata": stratified(out, ref, keep, marginal),
                   "task_accuracy": out["accuracy"],
                   "early_stop_frac": out["early_stop_frac"],
                   "marginal_vote_lag1_acf": acf_m,
                   "acf_n_images": acf_n}
            rows.append(row)
            s = row["strata"]
            print(f"{kind:22s} r0={tag:5s} err(all)="
                  f"{s['all']['stopping_error']:.4f} err(marginal)="
                  f"{s['marginal']['stopping_error']} n_marg="
                  f"{s['marginal']['n']} acf={acf_m if acf_n else 'na'}",
                  flush=True)

    payload = {"git_sha": git_sha(),
               "config": {"alpha": ALPHA, "t_max": T_MAX, "t_min": T_MIN,
                          "t_ref": T_REF, "n_images": N_IMAGES,
                          "lanes": LANES, "margin_cut": MARGIN_CUT,
                          "split": "val[:2000]", "noise_seed": NOISE_SEED,
                          "chip_seed": CHIP_SEED},
               "marginal_count": int(marginal.sum()),
               "rows": rows,
               "wall_seconds": time.time() - t0}
    outp = RESULTS / "precheck2_seed0.json"
    outp.write_text(json.dumps(payload, indent=1))
    print(f"wrote {outp} ({payload['wall_seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
