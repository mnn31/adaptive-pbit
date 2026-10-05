# Author: Manan Gupta <mnn@yogins.com>
"""Week-1 pre-check (design.md): effect-size go/no-go.

Measures the STOPPING ERROR (vs ideal-source reference decisions, the
quantity nominal alpha bounds) of the naive i.i.d. DirichletStop rule on a
trained MNIST SBNN as device correlation grows (r0 sweep), plus vote lag-1
autocorrelation and fixed-N task accuracy for the "accuracy still fine"
overlay. VALIDATION split only — test remains untouched.

Go/no-go (design.md): if stopping-error divergence at r0 <= 2 is under
~1.5x nominal alpha, escalate (smaller alpha / larger t_max / streaming)
before committing to the full grid.
"""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from apbit.data import splits
from apbit.infer import evaluate_ideal
from apbit.infer_adaptive import evaluate_lanes_adaptive, reference_decisions
from apbit.smtj import IdealSource, TelegraphSource
from apbit.stopping import DirichletStop
from apbit.train import load_checkpoint, train

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CKPT = RESULTS / "sbnn_mnist_seed0.pt"

ALPHA = 0.05
T_MAX = 64
T_MIN = 4
T_REF = 1024
N_IMAGES = 2000
LANES = 100
R0S = [np.inf, 8.0, 4.0, 2.0, 1.0, 0.5]
NOISE_SEED = 1000
CHIP_SEED = 0


def git_sha():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def vote_lag1_acf(votes):
    """Mean over images of lag-1 ACF of the top-class indicator sequence."""
    acfs = []
    for row in votes:
        top = np.bincount(row).argmax()
        x = (row == top).astype(float)
        if x.std() == 0:
            continue
        acfs.append(np.corrcoef(x[:-1], x[1:])[0, 1])
    return float(np.mean(acfs)) if acfs else float("nan")


def main():
    t0 = time.time()
    if not CKPT.exists():
        print("training checkpoint (30 epochs, cpu)...", flush=True)
        train(dataset="mnist", seed=0, ckpt_name=CKPT.name)
    model = load_checkpoint(CKPT)
    model.eval()

    _, _, vx, vy, _, _ = splits("mnist")
    xs, ys = vx[:N_IMAGES], vy[:N_IMAGES]

    print("reference decisions (ideal, T_ref=%d)..." % T_REF, flush=True)
    ref, tie = reference_decisions(model, xs, IdealSource(seed=7),
                                   t_ref=T_REF, batch=1000)
    ref_acc = float((ref == ys.numpy()).mean())
    print(f"reference label-accuracy {ref_acc:.4f}, ties {int(tie.sum())}",
          flush=True)

    rows = []
    for r0 in R0S:
        tag = "inf" if np.isinf(r0) else str(r0)
        if np.isinf(r0):
            src_a = IdealSource(seed=NOISE_SEED)
            src_f = IdealSource(seed=NOISE_SEED + 1)
        else:
            src_a = TelegraphSource(r0=r0, chip_seed=CHIP_SEED,
                                    noise_seed=NOISE_SEED)
            src_f = TelegraphSource(r0=r0, chip_seed=CHIP_SEED,
                                    noise_seed=NOISE_SEED + 1)
        out = evaluate_lanes_adaptive(
            model, xs, ys, src_a,
            lambda: DirichletStop(alpha=ALPHA, k_classes=10,
                                  t_min=T_MIN, t_max=T_MAX),
            t_max=T_MAX, lanes=LANES, collect_votes=True,
            reference=ref, reference_ties=tie)
        acf = vote_lag1_acf(out["votes"])
        # fixed-N task accuracy at the same r0 (the "still fine" overlay)
        if np.isinf(r0):
            fixed_acc = evaluate_ideal(model, xs, ys, src_f, T=T_MAX)
        else:
            from apbit.infer import evaluate_lanes
            fixed_acc = evaluate_lanes(model, xs, ys, src_f, T=T_MAX,
                                       lanes=LANES)
        row = {
            "r0": tag,
            "stopping_error": out["stopping_error"],
            "stopping_error_over_alpha": out["stopping_error"] / ALPHA,
            "reference_tie_count": out["reference_tie_count"],
            "adaptive_task_accuracy": out["accuracy"],
            "fixed64_task_accuracy": float(fixed_acc),
            "mean_samples": out["mean_samples"],
            "early_stop_frac": out["early_stop_frac"],
            "vote_lag1_acf": acf,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    payload = {
        "git_sha": git_sha(),
        "config": {"alpha": ALPHA, "t_max": T_MAX, "t_min": T_MIN,
                   "t_ref": T_REF, "n_images": N_IMAGES, "lanes": LANES,
                   "rule": "DirichletStop(beta top-2)", "split": "val[:2000]",
                   "mode": "settle (default settle_ratio=4)",
                   "noise_seed": NOISE_SEED, "chip_seed": CHIP_SEED},
        "reference_label_accuracy": ref_acc,
        "rows": rows,
        "wall_seconds": time.time() - t0,
    }
    outp = RESULTS / "precheck_seed0.json"
    outp.write_text(json.dumps(payload, indent=1))
    print(f"wrote {outp} ({payload['wall_seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
