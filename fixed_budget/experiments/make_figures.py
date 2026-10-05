"""Paper figures beyond fig_e2 (which analyze_sweep.py makes):
fig_model    — telegraph traces at three sampling ratios (mechanism illustration)
fig_e3       — heterogeneity: accuracy vs sigma_delta per r0, stream vs reset
fig_e4       — correlation-aware training: streaming accuracy vs eval r0
fig_pareto   — accuracy vs normalized throughput 1/(T*r0), frontier + NA points
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIGS = ROOT / "paper" / "figs"

C = {"blue": "#0072B2", "verm": "#D55E00", "green": "#009E73", "grey": "0.45"}
plt.rcParams.update({
    "font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
    "lines.linewidth": 1.4, "figure.dpi": 200,
})


def fig_model():
    from smtjnn.smtj import TelegraphSource
    import torch
    fig, axes = plt.subplots(3, 1, figsize=(3.45, 2.6), sharex=True)
    n = 120
    # slowly-varying drive so p_eq is visible
    t = np.arange(n)
    p_drive = 0.5 + 0.45 * np.sin(2 * np.pi * t / n)
    for ax, r0, label in zip(axes, [10, 0.5, 0.05],
                             ["$r_0=10$ (i.i.d. regime)",
                              "$r_0=0.5$ ($2\\tau$ rule)",
                              "$r_0=0.05$ (overclocked)"]):
        src = TelegraphSource(r0=r0, noise_seed=5)
        s = [src.sample(torch.tensor([[p]]), layer=0).item() for p in p_drive]
        ax.step(t, s, where="mid", color=C["blue"], lw=0.9)
        ax.plot(t, 2 * p_drive - 1, color=C["verm"], lw=1.0, ls="--")
        ax.set_ylim(-1.5, 1.5)
        ax.set_yticks([-1, 1])
        ax.text(0.99, 0.82, label, transform=ax.transAxes, ha="right", fontsize=7)
        ax.grid(True, lw=0.3, alpha=0.4)
    axes[1].set_ylabel("state $s$  /  $2p_{eq}\\!-\\!1$ (dashed)")
    axes[-1].set_xlabel("query index")
    fig.tight_layout(h_pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_model.{ext}", bbox_inches="tight")


def fig_e3():
    agg = defaultdict(list)
    for p in (RES / "hetero").glob("*.json"):
        r = json.loads(p.read_text())
        mode = "stream" if r.get("streaming", "_stream" in p.name) else "reset"
        if "streaming" not in r:
            mode = "stream" if "_stream" in p.name else "reset"
        agg[(mode, r["r0"], r["sigma_delta"])].append(r["acc"])
    sds = [0.0, 0.25, 0.5, 1.0, 1.5]
    r0s = [0.05, 0.1, 0.2, 0.5]
    colors = ["#0072B2", "#D55E00", "#009E73", "#555555"]
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    for r0, col in zip(r0s, colors):
        for mode, ls, mk in [("stream", "-", "o"), ("reset", "--", "^")]:
            mu = [np.mean(agg[(mode, r0, sd)]) for sd in sds
                  if agg[(mode, r0, sd)]]
            sd_ = [np.std(agg[(mode, r0, sd)]) for sd in sds
                   if agg[(mode, r0, sd)]]
            if len(mu) == len(sds):
                ax.errorbar(sds, mu, yerr=sd_, color=col, ls=ls, marker=mk,
                            markersize=3, capsize=2,
                            label=f"$r_0$={r0} {mode}" if mode == "stream" else None)
    ax.set_xlabel(r"device barrier spread $\sigma_\Delta$ ($k_BT$)")
    ax.set_ylabel("test accuracy")
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    ax.grid(True, lw=0.3, alpha=0.4)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_e3_hetero.{ext}", bbox_inches="tight")


def fig_e4():
    agg = defaultdict(list)
    for p in (RES / "noise_aware").glob("*_stream.json"):
        r = json.loads(p.read_text())
        agg[(r["train_r0"], r["eval_r0"])].append(r["acc"])
    base = defaultdict(list)
    for p in (RES / "sweeps").glob("mnist_r*_T32_stream_*ns[0-9].json"):
        r = json.loads(p.read_text())
        base[str(r["r0"])].append(r["acc"])
    evals = ["inf", "2", "1", "0.5", "0.2", "0.1", "0.05", "0.02"]
    x = np.arange(len(evals))
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    ax.plot(x, [np.mean(base[e]) for e in evals], color="k", marker="D",
            markersize=3, label="ideal-trained")
    for tr, col, mk in [(0.5, C["blue"], "o"), (0.2, C["verm"], "s"),
                        (0.1, C["green"], "^")]:
        ax.plot(x, [np.mean(agg[(tr, e)]) for e in evals], color=col,
                marker=mk, markersize=3, label=f"trained @ $r_0$={tr}")
    ax.set_xticks(x, ["$\\infty$"] + evals[1:])
    ax.set_xlabel(r"evaluation sampling ratio $r_0$ (streaming, $T=32$)")
    ax.set_ylabel("test accuracy")
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    ax.grid(True, lw=0.3, alpha=0.4)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_e4_training.{ext}", bbox_inches="tight")


def fig_pareto():
    """Physically-accounted throughput (images per tau_corr):
    streaming: 1/(T*r0).  reset: 1/(T*r0 + 2) — the re-equilibration wait
    between images costs ~2*tau, capping reset at 0.5 images/tau.
    """
    ideal = defaultdict(list)   # (r0, T, mode) -> accs
    for p in (RES / "sweeps").glob("mnist_r*_ns[0-9].json"):
        r = json.loads(p.read_text())
        if r["r0"] != "inf":
            mode = "stream" if r["streaming"] else "reset"
            ideal[(float(r["r0"]), r["T"], mode)].append(r["acc"])
    pts = []
    for (r0, T, mode), a in ideal.items():
        thr = 1.0 / (T * r0) if mode == "stream" else 1.0 / (T * r0 + 2)
        pts.append((thr, np.mean(a), r0, T, mode))

    na = defaultdict(list)      # (er0, T, tr) -> accs  (streaming only)
    for p in (RES / "noise_aware_T").glob("*_stream.json"):
        r = json.loads(p.read_text())
        na[(float(r["eval_r0"]), r["T"], r["train_r0"])].append(r["acc"])
    for p in (RES / "noise_aware").glob("*_stream.json"):
        r = json.loads(p.read_text())
        if r["eval_r0"] != "inf":
            na[(float(r["eval_r0"]), 32, r["train_r0"])].append(r["acc"])
    na_pts = {}                 # (er0, T) -> best-model point
    for (er0, T, tr), accs in na.items():
        m = np.mean(accs)
        key = (er0, T)
        if key not in na_pts or m > na_pts[key][1]:
            na_pts[key] = (1.0 / (T * er0), m, er0, T, "na")

    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    for mode, col, mk, lab in [("reset", C["green"], "^", "ideal-trained, reset"),
                               ("stream", C["blue"], "o", "ideal-trained, streaming")]:
        sub = [p for p in pts if p[4] == mode]
        ax.scatter([p[0] for p in sub], [p[1] for p in sub], s=9, marker=mk,
                   color=col, alpha=0.65, label=lab)
    ax.scatter([v[0] for v in na_pts.values()], [v[1] for v in na_pts.values()],
               s=12, marker="s", color=C["verm"], alpha=0.85,
               label="corr.-aware, streaming")
    allp = pts + list(na_pts.values())
    allp.sort(key=lambda p: -p[0])
    front, best = [], -1
    for p in allp:
        if p[1] > best:
            front.append(p)
            best = p[1]
    front.sort(key=lambda p: p[0])
    ax.plot([p[0] for p in front], [p[1] for p in front], color="k", lw=0.8,
            ls=":", zorder=1, label="Pareto frontier")
    ax.axvline(0.5, color=C["grey"], lw=0.7, ls="--")
    ax.text(0.5, 0.30, r" $2\tau$-rule ceiling", rotation=90, fontsize=6.5,
            color="0.3", transform=ax.get_xaxis_transform(), va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel(r"throughput (images per $\tau_{corr}$)")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.2, 1.0)
    ax.legend(frameon=False, fontsize=6, loc="lower left")
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_pareto.{ext}", bbox_inches="tight")

    rule_best = {}
    for tgt in [0.97, 0.95, 0.90]:
        # rule-compliant: r0 >= 2 in either mode, correct accounting
        okr = [p[0] for p in pts if p[2] >= 2 and p[1] >= tgt]
        ok_any = [p[0] for p in allp if p[1] >= tgt]
        if okr and ok_any:
            rule_best[tgt] = (max(okr), max(ok_any))
            print(f"target {tgt:.0%}: rule {max(okr):.3f}/tau -> "
                  f"achievable {max(ok_any):.3f}/tau "
                  f"({max(ok_any)/max(okr):.1f}x)")


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_model()
    fig_e3()
    fig_e4()
    fig_pareto()
    print("figures written to", FIGS)
