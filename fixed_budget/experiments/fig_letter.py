# Author: Manan Gupta <mnn@yogins.com>
"""Figures 2 and 3 of the letter, single column (3.4 in), from stored records.

Reads results/v2 (telegraph model, one JSON per configuration and seed; mean
and standard deviation over seeds as in experiments/analyze_v2.py) and the
macrospin columns of results/sllg_matched/classifier_table.json. Figure 1 is
drawn by experiments/fig_validation.py.

    python experiments/fig_letter.py OUT_DIR [fig2 fig3 dispersion]

fig2 and fig3 are drawn by default. dispersion draws the device-to-device
dispersion sweep (dispersion.pdf), whose result the letter states in the
text of Sec. III-C without a figure.

Macrospin points are open black-edged symbols: the same networks with every
hidden neuron a stochastic-LLG macrospin (experiments/sllg_classifier.py).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results" / "v2"
MACRO = ROOT / "results" / "sllg_matched" / "classifier_table.json"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("figs")

W = 3.4  # single column, inches
SETTLE = 4.0
RULE_R0 = 4.0
C = {1: "#0072B2", 8: "#D55E00", 32: "#009E73"}
MK = {1: "o", 8: "s", 32: "^"}
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "pdf.fonttype": 42, "figure.dpi": 300,
})
R0LAB = r"Sampling ratio, $r_0=\Delta t/\tau_{\mathrm{corr}}(0)$"


def rows(prefix, **f):
    out = []
    for p in V2.glob(prefix + "_*.json"):
        r = json.loads(p.read_text())
        if all(r.get(k) == v for k, v in f.items()):
            out.append(r)
    return out


def agg(rs, keys):
    d = defaultdict(list)
    for r in rs:
        d[tuple(r[k] for k in keys)].append(r["acc"])
    return {k: (np.mean(v), np.std(v), len(v)) for k, v in d.items()}


def panel(ax, s):
    ax.text(-0.18, 1.04, s, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight",
                metadata={"Creator": None, "Producer": None,
                          "CreationDate": None})
    plt.close(fig)


def macro_rows(net, mode, n, settle=SETTLE):
    """Macrospin accuracy (r0, mean, sd in %) of one network, mode and N."""
    table = json.loads(MACRO.read_text())
    pts = {}
    for r in table["rows"]:
        if ((r["net"], r["mode"], r["T"], r["settle"]) == (net, mode, n, settle)
                and r.get("sllg") and not r["group"].startswith("dt")):
            # one noise seed on record gives no spread
            pts[r["r0"]] = (r["sllg"]["mean"], r["sllg"]["sd"] or 0.0)
    return [(r0, *pts[r0]) for r0 in sorted(pts)]


def macro_transfer(trained_on, train_r0, tested_on, n=32):
    """Correlation-aware networks in table transfer_2x2: r0 -> (mean, sd) in %."""
    table = json.loads(MACRO.read_text())
    return {t["r0"]: (t[f"tested_on_{tested_on}"]["mean"],
                      t[f"tested_on_{tested_on}"]["sd"])
            for t in table["transfer_2x2"]
            if (t["trained_on"], t["train_r0"], t["T"]) == (trained_on,
                                                            train_r0, n)}


# ---------------------------------------------------------------- Fig. 2
def fig2():
    e2 = agg(rows("e2", ds="mnist"), ["mode", "r0", "T"])
    e1 = agg(rows("e1", ds="mnist", model_seed=0), ["T"])
    ct = agg(rows("e2ct", ds="mnist"), ["r0", "T"])
    r0s = sorted({k[1] for k in e2})
    fig, axes = plt.subplots(2, 1, figsize=(W, 3.7), sharex=True)
    for ax, mode, title, lab in [
            (axes[0], "settle", "Settle mode (settle $4\\,\\tau_{\\mathrm{corr}}(0)$)", "(a)"),
            (axes[1], "stream", "Streaming mode (no settle)", "(b)")]:
        for T in [1, 8, 32]:
            pts = [(r, *e2[(mode, r, T)][:2]) for r in r0s if (mode, r, T) in e2]
            xs, mu, sd = zip(*pts)
            ax.errorbar(xs, 100 * np.array(mu), yerr=100 * np.array(sd),
                        color=C[T], marker=MK[T], ms=3, capsize=1.5,
                        label=f"$N={T}$" + (", Arrhenius" if mode == "stream" else ""))
            if (T,) in e1:
                ax.axhline(100 * e1[(T,)][0], color=C[T], ls=":", lw=0.7)
            mac = macro_rows("ideal", mode, T)
            if mac and not (mode == "settle" and T == 1):
                xs, mu, sd = zip(*mac)
                ax.errorbar(xs, mu, yerr=sd, ls="none", marker=MK[T], ms=4.5,
                            mfc="none", mec="k", mew=0.8, ecolor="k",
                            capsize=1.5, zorder=5)
        if mode == "stream":
            pts = sorted((r, v[0]) for (r, T), v in ct.items() if T == 32)
            ax.plot([q[0] for q in pts], [100 * q[1] for q in pts], color="0.35",
                    ls="--", marker="v", ms=3, lw=1.0,
                    label="$N=32$, bias-indep.")
        if mode == "stream":
            ax.plot([], [], ls="none", marker="o", ms=4.5, mfc="none",
                    mec="k", mew=0.8, label="macrospin neurons")
        ax.axvline(RULE_R0, color="0.4", ls="--", lw=0.7)
        ax.text(RULE_R0 * 0.93, 0.55 if mode == "stream" else 0.62,
                r"$2\tau$ rule", ha="right", fontsize=8, color="0.3",
                transform=ax.get_xaxis_transform())
        ax.set_xscale("log")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(title, fontsize=8)
        ax.grid(True, which="both", lw=0.3, alpha=0.4)
        panel(ax, lab)
    axes[0].set_ylim(95.6, 98.2)
    axes[1].set_ylim(0, 100)
    # legends kept clear of the curves and of the 2-tau-rule line (r0 = 4)
    axes[0].legend(frameon=False, loc="lower center", fontsize=8, ncol=3,
                   handletextpad=0.3, columnspacing=1.0)
    axes[1].legend(frameon=False, fontsize=8, loc="upper center",
                   bbox_to_anchor=(0.45, -0.36), ncol=2, handletextpad=0.3,
                   columnspacing=0.8)
    axes[1].set_xlabel(R0LAB)
    fig.tight_layout(h_pad=0.6)
    save(fig, "gupta2")


# ------------------------------------------- dispersion (not in the letter)
def dispersion():
    e3 = agg(rows("e3", ds="mnist"), ["centering", "r0", "sigma_delta"])
    sds = [0.0, 0.25, 0.5, 1.0, 1.5]
    cols = {0.05: "#0072B2", 0.1: "#D55E00", 0.2: "#009E73", 0.5: "#555555"}
    mks = {0.05: "o", 0.1: "s", 0.2: "^", 0.5: "D"}  # shape identifies r0 in grayscale
    fig, ax = plt.subplots(figsize=(W, 2.1))
    for cen, ls in [("median", "-"), ("mean-tau", "--")]:
        for r0 in [0.05, 0.1, 0.2, 0.5]:
            pts = [(sd, *e3[(cen, r0, sd)][:2]) for sd in sds if (cen, r0, sd) in e3]
            xs, mu, se = zip(*pts)
            ax.errorbar(xs, 100 * np.array(mu), yerr=100 * np.array(se),
                        color=cols[r0], ls=ls, marker=mks[r0], ms=3.5,
                        capsize=1.5, mec=cols[r0],
                        mfc="white" if cen == "median" else cols[r0],
                        label=f"$r_0={r0}$" if cen == "median" else None)
    ax.set_xlabel(r"Barrier spread, $\sigma_\Delta$ ($k_BT$)")
    ax.set_ylabel("Test accuracy (%)")
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=2,
              title="solid, open: median device fixed\n"
                    "dashed, filled: mean $\\tau_{\\mathrm{corr}}$ fixed",
              title_fontsize=8, alignment="left")
    ax.set_ylim(15, 100)
    ax.grid(True, lw=0.3, alpha=0.4)
    fig.tight_layout()
    save(fig, "dispersion")


# ---------------------------------------------------------------- Fig. 3
def e4_selection():
    val = agg(rows("e4", ds="mnist", split="val"), ["train_r0", "eval_r0", "T"])
    test = agg(rows("e4", ds="mnist", split="test"), ["train_r0", "eval_r0", "T"])
    sel = {}
    for (tr, er, T), v in val.items():
        if (er, T) not in sel or v[0] > val[(sel[(er, T)], er, T)][0]:
            sel[(er, T)] = tr
    return sel, test


def fig3():
    test = agg(rows("e4", ds="mnist", split="test"), ["train_r0", "eval_r0", "T"])
    base = agg(rows("e2", ds="mnist", mode="stream"), ["r0", "T"])
    ers = [1, 0.5, 0.2, 0.1, 0.05, 0.02]
    x = np.arange(len(ers))
    fig, axes = plt.subplots(2, 1, figsize=(W, 3.4))
    ax = axes[0]
    # colour: training activations; solid, filled: evaluated on telegraph
    # neurons; dotted, open: evaluated on macrospin neurons
    ax.plot(x, [100 * base[(e, 32)][0] for e in ers], color="k", marker="D",
            ms=3, label="ideal (independent)")
    for tr, col, mk in [(0.5, "#0072B2", "o"), (0.2, "#D55E00", "s"),
                        (0.1, "#009E73", "^")]:
        pts = [(i, *test[(tr, e, 32)][:2]) for i, e in enumerate(ers)
               if (tr, e, 32) in test]
        xs, mu, sd = zip(*pts)
        ax.errorbar(xs, 100 * np.array(mu), yerr=100 * np.array(sd), color=col,
                    marker=mk, ms=3, capsize=1.5, label=f"telegraph, {tr}")
    pink = "#CC79A7"
    mtel = macro_transfer("macrospin", 0.2, "telegraph")
    xs = [i for i, e in enumerate(ers) if e in mtel]
    ax.errorbar(xs, [mtel[ers[i]][0] for i in xs],
                yerr=[mtel[ers[i]][1] for i in xs], color=pink, marker="v",
                ms=3, capsize=1.5, label="macrospin, 0.2")
    mac = {r0: (mu, sd) for r0, mu, sd in macro_rows("ideal", "stream", 32)}
    for pts, col, mk in [
            (mac, "k", "D"),
            (macro_transfer("telegraph", 0.2, "macrospin"), "#D55E00", "s"),
            (macro_transfer("macrospin", 0.2, "macrospin"), pink, "v")]:
        xs = [i for i, e in enumerate(ers) if e in pts]
        ax.errorbar(xs, [pts[ers[i]][0] for i in xs],
                    yerr=[pts[ers[i]][1] for i in xs], color=col, ls=":",
                    lw=0.9, marker=mk, ms=4.5, mfc="white", mec=col, mew=0.8,
                    capsize=1.5)
    ax.plot([], [], color="0.3", ls=":", lw=0.9, marker="o", ms=4.5,
            mfc="white", mec="0.3", mew=0.8, label="on macrospin neurons")
    ax.set_xticks(x, [str(e) for e in ers])
    ax.set_xlabel(r"Evaluation sampling ratio, $r_0$ (streaming, $N=32$)")
    ax.set_ylabel("Test accuracy (%)")
    order = ["ideal (independent)", "telegraph, 0.5", "telegraph, 0.2",
             "telegraph, 0.1", "macrospin, 0.2", "on macrospin neurons"]
    handles = dict(zip(*ax.get_legend_handles_labels()[::-1]))
    # the lower left is free of data: every curve is above 90% there
    ax.legend([handles[k] for k in order], order, frameon=False, fontsize=8,
              loc="lower left", handletextpad=0.4, labelspacing=0.25,
              borderaxespad=0.2)
    ax.set_ylim(0, 102)
    ax.grid(True, lw=0.3, alpha=0.4)
    panel(ax, "(a)")

    e2 = agg(rows("e2", ds="mnist"), ["mode", "r0", "T"])
    sel, _ = e4_selection()
    pts = [((1.0 / ((T - 1) * r0 + SETTLE) if m == "settle" else 1.0 / (T * r0)),
            v[0], m) for (m, r0, T), v in e2.items()]
    na = [(1.0 / (T * er), test[(tr, er, T)][0], "na")
          for (er, T), tr in sel.items() if (tr, er, T) in test]
    ax = axes[1]
    for m, col, mk, lab in [("settle", "#009E73", "^", "ideal-trained, settle"),
                            ("stream", "#0072B2", "o", "ideal-trained, stream")]:
        sub = [q for q in pts if q[2] == m]
        ax.scatter([q[0] for q in sub], [100 * q[1] for q in sub], s=8,
                   marker=mk, color=col, alpha=0.6, label=lab)
    ax.scatter([q[0] for q in na], [100 * q[1] for q in na], s=10, marker="s",
               color="#D55E00", alpha=0.9, label="correlation-aware")
    allp = sorted(pts + na, key=lambda q: -q[0])
    front, best = [], -1
    for q in allp:
        if q[1] > best:
            front.append(q)
            best = q[1]
    front.sort(key=lambda q: q[0])
    ax.plot([q[0] for q in front], [100 * q[1] for q in front], "k:", lw=0.8,
            label="frontier")
    ax.axvline(1.0 / RULE_R0, color="0.45", lw=0.7, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"Throughput (inputs per $\tau_{\mathrm{corr}}(0)$)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(5, 102)
    # the lower left (slow and inaccurate) holds no points
    ax.legend(frameon=False, fontsize=8, loc="lower left", handletextpad=0.3,
              labelspacing=0.25, borderaxespad=0.2)
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    panel(ax, "(b)")
    fig.tight_layout(h_pad=0.8)
    save(fig, "gupta3")


if __name__ == "__main__":
    names = sys.argv[2:] or ["fig2", "fig3"]
    for name in names:
        globals()[name]()
