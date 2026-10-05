# Author: Manan Gupta <mnn@yogins.com>
"""Analysis: tables + all paper figures from results/v2.

Figures:
  fig_e2_correlation  settle vs streaming accuracy vs r0 (MNIST only)
  fig_e3_hetero       both centerings + effective-refresh collapse
  fig_e4_training     correlation-aware robustness (streaming, T=32)
  fig_pareto          accuracy vs throughput, settle charged rho per input
Tables printed: E2 summary, E4 selection (val-chosen), energy, rule-relative
speedups (rule boundary r0 = 4: Daniels 2-tau in MEAN DWELL time).
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results" / "v2"
FIGS = ROOT / "paper" / "figs"

C = {1: "#0072B2", 8: "#D55E00", 32: "#009E73"}
MK = {1: "o", 8: "s", 32: "^"}
SETTLE_RHO = 4.0
RULE_R0 = 4.0

plt.rcParams.update({
    "font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
    "lines.linewidth": 1.4, "figure.dpi": 300,
})


def rows(prefix, **filters):
    out = []
    for p in V2.glob(prefix + "_*.json"):
        r = json.loads(p.read_text())
        if all(r.get(k) == v for k, v in filters.items()):
            out.append(r)
    return out


def agg(rs, keys, val="acc"):
    d = defaultdict(list)
    for r in rs:
        d[tuple(r[k] for k in keys)].append(r[val])
    return {k: (np.mean(v), np.std(v), len(v)) for k, v in d.items()}


# ---------------------------------------------------------------- E2
def fig_e2(ds="mnist"):
    e2 = agg(rows("e2", ds=ds), ["mode", "r0", "T"])
    e1 = agg(rows("e1", ds=ds, model_seed=0), ["T"])
    r0s = sorted({k[1] for k in e2})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, mode, title in [(axes[0], "settle",
                             f"Settle mode ($\\rho={SETTLE_RHO:g}$)"),
                            (axes[1], "stream", "Streaming (steady state)")]:
        for T in [1, 8, 32]:
            pts = [(r, *e2[(mode, r, T)][:2]) for r in r0s
                   if (mode, r, T) in e2]
            if not pts:
                continue
            xs, mu, sd = zip(*pts)
            ax.errorbar(xs, mu, yerr=sd, color=C[T], marker=MK[T],
                        markersize=3.5, capsize=2, label=f"$T={T}$")
            if (T,) in e1:
                ax.axhline(e1[(T,)][0], color=C[T], ls=":", lw=0.7, alpha=0.6)
        ax.axvline(RULE_R0, color="0.4", ls="--", lw=0.8)
        ax.text(RULE_R0, 0.97, r" $2\tau$ rule ", rotation=0, fontsize=7,
                color="0.3", ha="right", va="top",
                transform=ax.get_xaxis_transform())
        ax.set_xscale("log")
        ax.set_xlabel(r"sampling ratio  $r_0 = \Delta t / \tau_{corr}(0)$")
        ax.set_title(title, fontsize=8)
        ax.grid(True, which="both", lw=0.3, alpha=0.4)
    axes[0].set_ylabel("test accuracy")
    axes[0].legend(frameon=False, loc="lower right", fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_e2_correlation.{ext}", bbox_inches="tight")

    print(f"\n=== E2 {ds} (mean±std [n]) ===")
    print(f"{'mode':>7} {'r0':>6} " + " ".join(f"T={T:<3}        "
                                               for T in [1, 8, 32]))
    for mode in ["settle", "stream"]:
        for r in r0s:
            cells = []
            for T in [1, 8, 32]:
                v = e2.get((mode, r, T))
                cells.append(f"{v[0]:.4f}±{v[1]:.4f}[{v[2]}]" if v else "  --  ")
            print(f"{mode:>7} {r:>6} " + " ".join(cells))
    ct = agg(rows("e2ct", ds=ds), ["r0", "T"])
    if ct:
        print("const-tau streaming variant:")
        for T in [1, 32]:
            line = {r: round(ct[(r, T)][0], 4) for r in r0s if (r, T) in ct}
            print(f"  T={T}: {line}")
    rho = agg(rows("e2rho", ds=ds), ["rho", "r0", "T"])
    if rho:
        print("settle-ratio sweep:", {k: round(v[0], 4)
                                      for k, v in sorted(rho.items())})


# ---------------------------------------------------------------- E3
def baseline_interp(ds="mnist", T=32):
    """Interpolator over the sigma=0 streaming curve: log(r0) -> accuracy."""
    e2 = agg(rows("e2", ds=ds, mode="stream"), ["r0", "T"])
    pts = sorted((r0, v[0]) for (r0, t), v in e2.items() if t == T)
    xs = np.log([p[0] for p in pts])
    ys = [p[1] for p in pts]
    return lambda r0: float(np.interp(np.log(r0), xs, ys))


def fig_e3(ds="mnist"):
    e3 = agg(rows("e3", ds=ds), ["centering", "r0", "sigma_delta"])
    sds = [0.0, 0.25, 0.5, 1.0, 1.5]
    cols = {0.05: "#0072B2", 0.1: "#D55E00", 0.2: "#009E73", 0.5: "#555555"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for cen, ls, mk in [("median", "-", "o"), ("mean-tau", "--", "s")]:
        for r0 in [0.05, 0.1, 0.2, 0.5]:
            pts = [(sd, *e3[(cen, r0, sd)][:2]) for sd in sds
                   if (cen, r0, sd) in e3]
            if not pts:
                continue
            xs, mu, sderr = zip(*pts)
            axes[0].errorbar(xs, mu, yerr=sderr, color=cols[r0], ls=ls,
                             marker=mk, markersize=3, capsize=2,
                             label=f"$r_0$={r0}" if cen == "median" else None)
    axes[0].set_xlabel(r"barrier spread $\sigma_\Delta$ ($k_BT$)")
    axes[0].set_ylabel("test accuracy")
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower left",
                   title="solid: median-fixed\ndashed: mean-$\\tau$-fixed",
                   title_fontsize=6)
    axes[0].grid(True, lw=0.3, alpha=0.4)
    # offset-prediction test: measured vs sigma=0 baseline at the
    # median-device effective ratio r0_eff = r0 * e^{sd^2/2} (mean-tau)
    # or r0 (median centering)
    interp = baseline_interp(ds)
    for cen, mk in [("median", "o"), ("mean-tau", "s")]:
        for r0 in [0.05, 0.1, 0.2, 0.5]:
            for sd in sds:
                if (cen, r0, sd) in e3:
                    r_eff = r0 * (np.exp(sd ** 2 / 2)
                                  if cen == "mean-tau" else 1.0)
                    axes[1].scatter(interp(r_eff), e3[(cen, r0, sd)][0],
                                    s=12, marker=mk, color=cols[r0],
                                    alpha=0.8)
    lim = [0.6, 1.0]
    axes[1].plot(lim, lim, color="0.5", lw=0.7, ls="--")
    axes[1].set_xlim(lim), axes[1].set_ylim(lim)
    axes[1].set_xlabel("predicted from $\\sigma_\\Delta{=}0$ curve at "
                       "$r_0\\, e^{\\sigma_\\Delta^2/2}$")
    axes[1].set_ylabel("measured accuracy")
    axes[1].grid(True, lw=0.3, alpha=0.4)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_e3_hetero.{ext}", bbox_inches="tight")
    print("\n=== E3 (see figure) ===")


# ---------------------------------------------------------------- E4
def e4_selection(ds="mnist"):
    """Val-based selection: for each (eval_r0, T), pick (train_r0) by val
    accuracy averaged over model seeds; report test accuracy of the pick."""
    val = agg(rows("e4", ds=ds, split="val"), ["train_r0", "eval_r0", "T"])
    test = agg(rows("e4", ds=ds, split="test"), ["train_r0", "eval_r0", "T"])
    sel = {}
    for (tr, er, T), v in val.items():
        key = (er, T)
        if key not in sel or v[0] > val[(sel[key], er, T)][0]:
            sel[key] = tr
    return sel, test


def fig_e4(ds="mnist"):
    test = agg(rows("e4", ds=ds, split="test"), ["train_r0", "eval_r0", "T"])
    base = agg(rows("e2", ds=ds, mode="stream"), ["r0", "T"])
    ers = [1, 0.5, 0.2, 0.1, 0.05, 0.02]
    x = np.arange(len(ers))
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    bl = [base[(e, 32)][0] for e in ers if (e, 32) in base]
    ax.plot(x[:len(bl)], bl, color="k", marker="D", markersize=3,
            label="ideal-trained")
    for tr, col, mk in [(0.5, "#0072B2", "o"), (0.2, "#D55E00", "s"),
                        (0.1, "#009E73", "^")]:
        pts = [(i, *test[(tr, e, 32)][:2]) for i, e in enumerate(ers)
               if (tr, e, 32) in test]
        if pts:
            xs, mu, sd = zip(*pts)
            ax.errorbar(xs, mu, yerr=sd, color=col, marker=mk, markersize=3,
                        capsize=2, label=f"trained @ $r_0$={tr}")
    ax.set_xticks(x, [str(e) for e in ers])
    ax.set_xlabel(r"evaluation $r_0$ (streaming, $T=32$)")
    ax.set_ylabel("test accuracy")
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    ax.grid(True, lw=0.3, alpha=0.4)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_e4_training.{ext}", bbox_inches="tight")
    print("\n=== E4 test (mean±std over model seeds x noise seeds) ===")
    for tr in [0.5, 0.2, 0.1]:
        line = {e: f"{test[(tr, e, 32)][0]:.4f}" for e in ers
                if (tr, e, 32) in test}
        print(f"  train {tr}: {line}")


# ---------------------------------------------------------------- Pareto
def fig_pareto(ds="mnist"):
    e2 = agg(rows("e2", ds=ds), ["mode", "r0", "T"])
    sel, test = e4_selection(ds)
    pts = []
    for (mode, r0, T), v in e2.items():
        thr = (1.0 / ((T - 1) * r0 + SETTLE_RHO) if mode == "settle"
               else 1.0 / (T * r0))
        pts.append((thr, v[0], r0, T, mode))
    na_pts = []
    for (er, T), tr in sel.items():
        v = test.get((tr, er, T))
        if v:
            na_pts.append((1.0 / (T * er), v[0], er, T, f"na{tr}"))
    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    for mode, col, mk, lab in [("settle", "#009E73", "^", "ideal-trained, settle"),
                               ("stream", "#0072B2", "o", "ideal-trained, streaming")]:
        sub = [p for p in pts if p[4] == mode]
        ax.scatter([p[0] for p in sub], [p[1] for p in sub], s=9, marker=mk,
                   color=col, alpha=0.6, label=lab)
    ax.scatter([p[0] for p in na_pts], [p[1] for p in na_pts], s=12,
               marker="s", color="#D55E00", alpha=0.9,
               label="corr.-aware (val-selected)")
    allp = pts + na_pts
    allp.sort(key=lambda p: -p[0])
    front, best = [], -1
    for p in allp:
        if p[1] > best:
            front.append(p)
            best = p[1]
    front.sort(key=lambda p: p[0])
    ax.plot([p[0] for p in front], [p[1] for p in front], "k:", lw=0.8,
            label="Pareto frontier")
    rule_ceiling = 1.0 / (1 * RULE_R0)
    ax.axvline(rule_ceiling, color="0.45", lw=0.7, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"throughput (images per $\tau_{corr}$)")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.15, 1.0)
    ax.legend(frameon=False, fontsize=6, loc="lower left")
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_pareto.{ext}", bbox_inches="tight")

    print("\n=== rule-relative speedups (rule: r0>=4, either mode) ===")
    for tgt in [0.97, 0.95, 0.90]:
        okr = [p for p in pts if p[2] >= RULE_R0 and p[1] >= tgt]
        oka = [p for p in allp if p[1] >= tgt]
        if okr and oka:
            br, ba = max(p[0] for p in okr), max(p[0] for p in oka)
            wa = max(oka, key=lambda p: p[0])
            print(f"  {tgt:.0%}: rule {br:.3f} -> {ba:.3f} ({ba/br:.1f}x) "
                  f"via {wa[4]} r0={wa[2]} T={wa[3]} acc={wa[1]:.4f}")


# ---------------------------------------------------------------- energy
def energy_table():
    hidden = 384  # 256 + 128 stochastic units per pass
    print("\n=== randomness-generation energy per inference ===")
    print("T    bits      sMTJ(0.03pJ/b)  GPU-PRNG(3nJ/b)  CPU-PRNG(30nJ/b)")
    for T in [1, 8, 32]:
        bits = hidden * T
        print(f"{T:<4} {bits:<9} {bits*0.03e-12*1e9:.4f} nJ      "
              f"{bits*3e-9*1e6:.1f} uJ         {bits*30e-9*1e6:.1f} uJ")


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_e2()
    fig_e3()
    fig_e4()
    fig_pareto()
    energy_table()
    conv = rows("e5conv")
    if conv:
        print("\nlane-invariance:", {r["lanes"]: round(r["acc"], 4)
                                     for r in conv})
