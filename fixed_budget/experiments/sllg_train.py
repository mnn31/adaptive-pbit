# Author: Manan Gupta <mnn@yogins.com>
"""Correlation-aware training with sLLG macrospin activations.

The same recipe as the paper's telegraph-trained networks (run_v2 phase 0):
30 epochs, Adam 1e-3 with cosine decay, straight-through estimator, batch
256, training seed s, device noise seed 123 + s, hidden activations drawn
in streaming mode at read ratio r0_train with the device state carried
across minibatches. Only the source differs: macrospin neurons of
--dev-dir/device.json, r0 in units of that device's tau_c(0). Nothing is
selected: the network after the last epoch is kept, and the validation
split is only logged.

Writes --dev-dir/nets/sllg_na{r0}_seed{s}.pt and a training record
(per-epoch loss and validation accuracy) next to it.

    python experiments/sllg_train.py --dev-dir results/sllg_matched \\
        --r0 0.2 --seed 0 --gpu cuda:1
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from smtjnn import sllg
from smtjnn.sllg import Macrospin, MacrospinSource
from smtjnn.train import train

ROOT = Path(__file__).resolve().parents[1]


# Train one network on the macrospin of dev_dir at read ratio r0 and training seed
# `seed`, then write the checkpoint and its training record. Skips if the checkpoint
# exists.
def main(dev_dir, r0, seed, gpu, dt=1e-12):
    dev_dir = Path(dev_dir).resolve()
    ckpt = dev_dir / "nets" / f"sllg_na{r0:g}_seed{seed}.pt"
    if ckpt.exists():
        print(ckpt, "exists")
        return
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    drec = json.loads((dev_dir / "device.json").read_text())
    d = drec["device"]
    dev = Macrospin(d["delta"], d["moment_per_T"], d["alpha"])
    src = MacrospinSource(r0, dev, drec["tau_c0_s"],
                          drec["tesla_per_unit_i"], dt=dt, streaming=True,
                          noise_seed=123 + seed, device=gpu)
    lines = []

    # print a training line and keep it for the record
    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    t0 = time.time()
    train(dataset="mnist", epochs=30, seed=seed, device=gpu,
          ckpt_name=str(ckpt), source=src,
          log=log)
    files = [Path(sllg.__file__), Path(__file__)]
    inside = ckpt.is_relative_to(ROOT)
    rec = {"checkpoint": str(ckpt.relative_to(ROOT) if inside else ckpt),
           "train_r0": r0,
           "seed": seed, "noise_seed": 123 + seed, "dt_s": dt,
           "device": d, "tau_c0_s": drec["tau_c0_s"],
           "tesla_per_unit_i": drec["tesla_per_unit_i"],
           "recipe": "30 epochs, Adam 1e-3, cosine decay, batch 256, "
                     "straight-through estimator, streaming source",
           "log": lines, "seconds": round(time.time() - t0, 1),
           "code_sha256": {f.name: hashlib.sha256(f.read_bytes())
                           .hexdigest()[:16] for f in files},
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    ckpt.with_suffix(".json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-dir", default=str(ROOT / "results"
                                             / "sllg_matched"))
    ap.add_argument("--r0", type=float, nargs="+", default=[0.2])
    ap.add_argument("--seed", type=int, nargs="+", default=[0])
    ap.add_argument("--gpu", default="cuda:1")
    args = ap.parse_args()
    for r0 in args.r0:
        for s in args.seed:
            main(args.dev_dir, r0, s, args.gpu)
