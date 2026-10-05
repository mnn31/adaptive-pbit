# Author: Manan Gupta <mnn@yogins.com>
"""Supplementary verification runs:
A. Cross-network replication of key E2 cells over model seeds 1-4.
F. Lane-invariance at the production warmup (warm=10).
H. Correlation-aware models evaluated at rule-compliant r0=4 (true nominal).
"""

from pathlib import Path

from run_v2 import (OUT, data, done, save, eval_stream, eval_settle,
                    lane_params)
from smtjnn.infer import evaluate_lanes
from smtjnn.smtj import TelegraphSource
from smtjnn.train import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def phaseA(ds="mnist"):
    for seed in [1, 2, 3, 4]:
        model = load_checkpoint(ROOT / "results" / f"v2_{ds}_seed{seed}.pt")
        for mode in ["settle", "stream"]:
            for r0 in [2, 0.5, 0.1, 0.02]:
                for T in [1, 32]:
                    for ns in range(3):
                        p = OUT / (f"e2x_{ds}_{mode}_r{r0}_T{T}"
                                   f"_ms{seed}_ns{ns}.json")
                        if done(p):
                            continue
                        src = TelegraphSource(r0=r0,
                                              streaming=(mode == "stream"),
                                              settle_ratio=4.0, noise_seed=ns)
                        acc = (eval_stream if mode == "stream"
                               else eval_settle)(model, ds, src, T)
                        save(p, dict(ds=ds, mode=mode, r0=r0, T=T,
                                     model_seed=seed, ns=ns, acc=acc))
    # E3 replication over model seeds at the headline cell
    for seed in [1, 2]:
        model = load_checkpoint(ROOT / "results" / f"v2_{ds}_seed{seed}.pt")
        for centering in ["median", "mean-tau"]:
            for sd in [0.0, 1.5]:
                for chip in range(3):
                    p = OUT / (f"e3x_{ds}_{centering}_r0.05_sd{sd}"
                               f"_ms{seed}_c{chip}.json")
                    if done(p):
                        continue
                    src = TelegraphSource(r0=0.05, streaming=True,
                                          sigma_delta=sd, chip_seed=chip,
                                          noise_seed=1000 + 17 * chip,
                                          centering=centering)
                    acc = eval_stream(model, ds, src, 32)
                    save(p, dict(ds=ds, centering=centering, r0=0.05,
                                 sigma_delta=sd, model_seed=seed, chip=chip,
                                 T=32, acc=acc))


def phaseF(ds="mnist"):
    model = load_checkpoint(ROOT / "results" / f"v2_{ds}_seed0.pt")
    trx, _, _, _, tx, ty = data(ds)
    for lanes in [50, 100, 200, 500]:
        p = OUT / f"e5convw10_{ds}_lanes{lanes}.json"
        if done(p):
            continue
        src = TelegraphSource(r0=0.05, streaming=True, noise_seed=0)
        acc = evaluate_lanes(model, tx, ty, src, T=32, lanes=lanes,
                             warmup_x=trx, warmup_per_lane=10)
        save(p, dict(ds=ds, lanes=lanes, r0=0.05, T=32, warm=10, acc=acc))


def phaseH(ds="mnist"):
    for tr0, seeds in [(0.5, range(3)), (0.2, range(3)), (0.1, range(1))]:
        for s in seeds:
            model = load_checkpoint(ROOT / "results" /
                                    f"v2_{ds}_na{tr0}_seed{s}.pt")
            for T in [1, 32]:
                for ns in range(3):
                    p = OUT / f"e4r_{ds}_tr{tr0}_s{s}_er4_T{T}_ns{ns}.json"
                    if done(p):
                        continue
                    src = TelegraphSource(r0=4, streaming=True, noise_seed=ns)
                    acc = eval_stream(model, ds, src, T)
                    save(p, dict(ds=ds, train_r0=tr0, model_seed=s,
                                 split="test", eval_r0=4, T=T, ns=ns,
                                 acc=acc))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    phaseF()
    phaseH()
    phaseA()
    print("extras complete", flush=True)
