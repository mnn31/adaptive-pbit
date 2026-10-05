"""Aggregate E1/E2 sweep JSONs into the headline figure + summary table."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SWEEPS = ROOT / "results" / "sweeps"
FIGS = ROOT / "paper" / "figs"

# Okabe-Ito subset, CVD-validated; markers give grayscale/print encoding
COLOR = {1: "#0072B2", 8: "#D55E00", 32: "#009E73"}
MARKER = {1: "o", 8: "s", 32: "^"}

plt.rcParams.update({
    "font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
    "lines.linewidth": 1.4, "figure.dpi": 200,
})


def load():
    rows = [json.loads(p.read_text()) for p in SWEEPS.glob("*.json")]
    tele = [r for r in rows if "r0" in r]
    ideal = {r["T"]: r["acc"] for r in rows if r.get("source") == "ideal"}
    agg = defaultdict(list)  # (r0, T, streaming) -> [acc]
    for r in tele:
        r0 = np.inf if r["r0"] == "inf" else float(r["r0"])
        agg[(r0, r["T"], r["streaming"])].append(r["acc"])
    return agg, ideal


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    agg, ideal = load()
    r0s = sorted({k[0] for k in agg if np.isfinite(k[0])})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, streaming, title in [(axes[0], False, "Reset per image"),
                                 (axes[1], True, "Streaming")]:
        for T in [1, 8, 32]:
            mu = [np.mean(agg[(r, T, streaming)]) for r in r0s]
            sd = [np.std(agg[(r, T, streaming)]) for r in r0s]
            ax.errorbar(r0s, mu, yerr=sd, color=COLOR[T], marker=MARKER[T],
                        markersize=3.5, capsize=2, label=f"T = {T}")
            iid = np.mean(agg[(np.inf, T, streaming)])
            ax.axhline(iid, color=COLOR[T], ls=":", lw=0.7, alpha=0.6)
        ax.axvline(0.5, color="0.4", ls="--", lw=0.8)
        ax.text(0.5, 0.12, r" $2\tau$ rule", transform=ax.get_xaxis_transform(),
                fontsize=7, color="0.3")
        ax.set_xscale("log")
        ax.set_xlabel(r"sampling ratio  $r_0 = \Delta t \cdot 2f_0$")
        ax.set_title(title, fontsize=8)
        ax.grid(True, which="both", lw=0.3, alpha=0.4)
    axes[0].set_ylabel("test accuracy")
    axes[0].legend(frameon=False, loc="lower right", fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_e2_correlation.{ext}", bbox_inches="tight")

    # summary table
    print(f"{'r0':>8} {'mode':>8} " + " ".join(f"T={T:<2}       " for T in [1, 8, 32]))
    for streaming in [False, True]:
        for r in [np.inf] + r0s:
            cells = []
            for T in [1, 8, 32]:
                v = agg.get((r, T, streaming), [])
                cells.append(f"{np.mean(v):.4f}±{np.std(v):.4f}" if v else "  --  ")
            print(f"{'inf' if np.isinf(r) else r:>8} "
                  f"{'stream' if streaming else 'reset':>8} " + " ".join(cells))
    print("\nideal-source anchors:", {k: round(v, 4) for k, v in sorted(ideal.items())})


if __name__ == "__main__":
    main()
