# Author: Manan Gupta <mnn@yogins.com>
"""Shared measurement helpers for the experiment drivers.

Factored out of experiments/precheck2.py so the pre-check and the full grid
report the SAME numbers from the SAME code — a diagnostic that drifts between
the go/no-go run and the reported grid is a reviewer gift.

`vote_lag1_acf` is the mechanism measurement behind figure F5 and the
deployment story for MarkovSPRT: a chip can estimate the lag-1 autocorrelation
of its own vote indicators from draws it is already taking. It is also what
phase 2a of the grid measures on the validation split to supply the fixed-rho
variant of the corrected rule (rho is a DEVICE measurement, not label
information — see run_grid.py).

Audit r1 item 9 added the rest of the mechanism panel: the lag-k ACF out to
k = 48, the integrated inflation factor it implies (and its AR(1) counterpart,
whose ratio is the predictor of where the first-order correction stops being
enough), the per-image rho histogram behind the pooled mean, and the entropy
RATE of the fitted two-state chain — the quantity that actually earns the word
"entropy" for a correlated stream, as opposed to the marginal entropy of a
single draw.

LABELLING. Every one of these statistics is computed only over images whose
vote indicator FLUCTUATES; a saturated image (same class every pass) has no
defined autocorrelation and is skipped. Panels built from them must say
"images whose votes fluctuate", never "all images".
"""

from math import log, sqrt

import numpy as np
import torch

__all__ = ["vote_lag1_acf", "wilson_interval", "vote_acf_curve",
           "integrated_inflation_factor", "vote_rho_histogram",
           "vote_entropy_rate", "vote_diagnostics",
           "abs_preactivation_summary"]


def vote_lag1_acf(votes, mask=None):
    """NaN-safe mean lag-1 ACF of the per-image top-class vote indicator.

    votes: [N, T] int array of per-pass class votes (as returned by
      `evaluate_lanes_adaptive(..., collect_votes=True)`; the full T-pass
      stream is collected for every image whether or not its lane stopped
      reading, so this measures the DEVICE, not the rule).
    mask: optional [N] bool selecting a stratum (e.g. the marginal images).

    For each selected image the indicator is x_t = 1[vote_t == plurality] and
    the statistic is corr(x[:-1], x[1:]). Images whose indicator is constant
    have no defined autocorrelation and are skipped (this is common in the easy
    stratum, where a saturated network votes one class every pass), as are any
    non-finite results. Returns (mean_acf, n_images_contributing); the mean is
    NaN when nothing contributed, so callers must check the count.
    """
    votes = np.asarray(votes)
    idx = np.arange(votes.shape[0]) if mask is None else np.where(mask)[0]
    acfs = []
    for i in idx:
        row = votes[i]
        x = (row == np.bincount(row).argmax()).astype(float)
        a, b = x[:-1], x[1:]
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        r = np.corrcoef(a, b)[0, 1]
        if np.isfinite(r):
            acfs.append(r)
    return (float(np.mean(acfs)), len(acfs)) if acfs else (float("nan"), 0)


def wilson_interval(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion (design.md: realized
    error rates are reported with Wilson intervals, never Wald's, because the
    realized errors we care about sit near zero where Wald is nonsense)."""
    if n <= 0:
        return (None, None)
    p = k / n
    d = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    half = z * sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return ((centre - half) / d, (centre + half) / d)


# ---------------------------------------------------------------------------
# audit r1 item 9 — the full temporal-correlation panel
# ---------------------------------------------------------------------------

def _fluctuating_indicators(votes, mask=None):
    """[M, T] float indicator matrix for the images whose votes fluctuate.

    Row i is x_t = 1[vote_t == that image's plurality class] for one selected
    image, kept only if the indicator is not constant (a constant row has no
    defined autocorrelation, no transitions to fit, and no entropy rate).
    Returns (X, idx) with idx the original row indices that survived.
    """
    votes = np.asarray(votes)
    idx = (np.arange(votes.shape[0]) if mask is None
           else np.where(np.asarray(mask, dtype=bool))[0])
    rows, keep = [], []
    for i in idx:
        row = votes[i]
        x = (row == np.bincount(row).argmax()).astype(np.float64)
        if 0.0 < x.mean() < 1.0:
            rows.append(x)
            keep.append(int(i))
    if not rows:
        return np.zeros((0, votes.shape[1])), []
    return np.vstack(rows), keep


def _acf_per_row(x: np.ndarray, k: int):
    """Per-row lag-k Pearson correlation of x[:-k] vs x[k:].

    Same estimator as np.corrcoef on the two windows (each window gets its own
    mean and sd), so the k = 1 column reproduces `vote_lag1_acf` exactly.
    Rows with a constant window at this lag are dropped.
    """
    if k >= x.shape[1]:
        return np.zeros(0)
    a, b = x[:, :-k], x[:, k:]
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    sa = np.sqrt((a * a).sum(axis=1))
    sb = np.sqrt((b * b).sum(axis=1))
    ok = (sa > 1e-12) & (sb > 1e-12)
    if not ok.any():
        return np.zeros(0)
    num = (a[ok] * b[ok]).sum(axis=1)
    return num / (sa[ok] * sb[ok])


def vote_acf_curve(votes, mask=None, max_lag: int = 48) -> dict:
    """Mean lag-k vote-indicator ACF for k = 1..max_lag.

    Averaged over images whose votes fluctuate (see module docstring). The
    per-lag image count is reported because it SHRINKS with k: a stream of T
    passes gives T-k pairs, and a row can lose its variance in one of the two
    windows at large k.
    """
    x, _ = _fluctuating_indicators(votes, mask)
    lags = list(range(1, int(max_lag) + 1))
    mean_acf, counts = [], []
    for k in lags:
        vals = _acf_per_row(x, k) if x.shape[0] else np.zeros(0)
        vals = vals[np.isfinite(vals)]
        mean_acf.append(float(vals.mean()) if vals.size else float("nan"))
        counts.append(int(vals.size))
    return {"lags": lags, "mean_acf": mean_acf, "n_images": counts,
            "n_images_fluctuating": int(x.shape[0]),
            "stratum_note": "images whose votes fluctuate"}


def integrated_inflation_factor(mean_acf) -> dict:
    """Variance inflation factor of a correlated mean, from the ACF curve.

    For a stationary stream the variance of the sample mean is inflated over
    the i.i.d. value by IF = 1 + 2 sum_{k>=1} rho_k. Two versions are
    reported and their RATIO is the diagnostic:

      if_true   initial-positive-sequence truncation (Geyer 1992): sum the
                lags up to the first non-positive rho_k. Truncating is the
                standard defence against the estimator's own noise at large
                k, where each rho_k is estimated from few pairs.
      if_full   the untruncated sum over every lag supplied, for contrast.
      if_ar1    (1 + rho_1) / (1 - rho_1) — the inflation a FIRST-ORDER model
                predicts from the same lag-1 number the Markov rule is fed.

    ratio = if_true / if_ar1 > 1 means the stream carries memory beyond lag 1
    and the first-order correction under-corrects; that ratio crossing 1 is
    the predictor of the validity floor reported in the paper.
    """
    rho = [float(v) for v in mean_acf]
    trunc, total = 0, 0.0
    for k, r in enumerate(rho, start=1):
        if not np.isfinite(r) or r <= 0.0:
            break
        total += r
        trunc = k
    full = float(np.nansum([r for r in rho if np.isfinite(r)]))
    r1 = rho[0] if rho and np.isfinite(rho[0]) else float("nan")
    if_ar1 = ((1.0 + r1) / (1.0 - r1)
              if np.isfinite(r1) and r1 < 1.0 else float("nan"))
    if_true = 1.0 + 2.0 * total
    return {"if_true": float(if_true), "if_full": float(1.0 + 2.0 * full),
            "if_ar1": float(if_ar1), "truncation_lag": int(trunc),
            "ratio_true_over_ar1": (float(if_true / if_ar1)
                                    if np.isfinite(if_ar1) and if_ar1 > 0
                                    else None),
            "lag1": float(r1)}


def vote_rho_histogram(votes, mask=None, bins: int = 20) -> dict:
    """Distribution of the PER-IMAGE lag-1 vote ACF behind the pooled mean.

    The rule is fed one pooled rho; this is the spread it is wrong by on any
    individual image. Range is fixed to [-1, 1] so histograms from different
    cells are directly comparable.
    """
    x, _ = _fluctuating_indicators(votes, mask)
    vals = _acf_per_row(x, 1) if x.shape[0] else np.zeros(0)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"n_images": 0, "mean": None, "sd": None, "quantiles": {},
                "histogram_counts": [], "histogram_edges": [],
                "stratum_note": "images whose votes fluctuate"}
    hist, edges = np.histogram(vals, bins=int(bins), range=(-1.0, 1.0))
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    return {
        "n_images": int(vals.size),
        "mean": float(vals.mean()),
        "sd": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        "quantiles": {f"q{q}": float(np.percentile(vals, q)) for q in qs},
        "histogram_counts": [int(c) for c in hist],
        "histogram_edges": [float(e) for e in edges],
        "stratum_note": "images whose votes fluctuate",
    }


def vote_entropy_rate(votes, mask=None) -> dict:
    """Entropy RATE of the fitted two-state vote chain, bits per pass.

    Per image, fit the first-order chain of the vote indicator by transition
    counts, then

        H = - sum_i pi_i sum_j P_ij log2 P_ij

    with pi the empirical marginal (the stationary law of the fitted chain).
    Reported next to the i.i.d. entropy H_iid = H_2(mean indicator), which is
    what a marginal-only accounting would claim. H < H_iid is exactly the
    predictability a correlated stream hands the stopping rule for free, and
    the gap is what "degraded entropy" means quantitatively here.
    """
    x, _ = _fluctuating_indicators(votes, mask)
    if x.shape[0] == 0:
        return {"n_images": 0, "entropy_rate_bits": None,
                "iid_entropy_bits": None, "gap_bits": None,
                "stratum_note": "images whose votes fluctuate"}
    a, b = x[:, :-1].astype(np.int64), x[:, 1:].astype(np.int64)
    rates, iids = [], []
    ln2 = log(2.0)
    for i in range(x.shape[0]):
        n = np.zeros((2, 2))
        np.add.at(n, (a[i], b[i]), 1.0)
        pi = x[i].mean()
        pi_vec = np.array([1.0 - pi, pi])
        h = 0.0
        for s in (0, 1):
            tot = n[s].sum()
            if tot <= 0:
                continue
            p = n[s] / tot
            hs = -sum(q * log(q) / ln2 for q in p if q > 0)
            h += pi_vec[s] * hs
        rates.append(h)
        iids.append(-sum(q * log(q) / ln2 for q in (pi, 1.0 - pi) if q > 0))
    return {
        "n_images": int(x.shape[0]),
        "entropy_rate_bits": float(np.mean(rates)),
        "iid_entropy_bits": float(np.mean(iids)),
        "gap_bits": float(np.mean(iids) - np.mean(rates)),
        "stratum_note": "images whose votes fluctuate",
    }


def vote_diagnostics(votes, mask=None, max_lag: int = 48,
                     hist_bins: int = 20) -> dict:
    """The whole temporal-correlation panel for one vote stream, one call."""
    curve = vote_acf_curve(votes, mask, max_lag=max_lag)
    return {
        "acf_curve": curve,
        "inflation": integrated_inflation_factor(curve["mean_acf"]),
        "rho_histogram": vote_rho_histogram(votes, mask, bins=hist_bins),
        "entropy_rate": vote_entropy_rate(votes, mask),
    }


# ---------------------------------------------------------------------------
# audit r1 item 4 — the |I| distribution the rate ceiling acts on
# ---------------------------------------------------------------------------

@torch.no_grad()
def abs_preactivation_summary(model, xs, source=None, passes: int = 4,
                              batch: int = 500, ceilings=(2.0,)) -> dict:
    """Distribution of |I| at the stochastic layers, per layer and pooled.

    The rate ceiling k_tot = min(2 f cosh I, 1/tau0) only binds where
    cosh|I| >= C, so the honest way to report it is the fraction of
    (image, unit, pass) triples above arccosh(C) — that fraction is how much
    of the network the ceiling actually touches. Re-implements the forward
    sweep of SBNN.forward rather than hooking it, so the recorded I is the
    exact pre-activation the source is handed.
    """
    model.eval()
    n = int(xs.shape[0])
    per_layer, ceil_hits = {}, {float(c): 0 for c in ceilings}
    total = 0
    pooled_sum = pooled_sq = 0.0
    pooled_max = 0.0
    samples = []
    for a in range(0, n, batch):
        x0 = xs[a:a + batch]
        for _ in range(int(passes)):
            h = x0
            for li, layer in enumerate(model.layers[:-1]):
                i = layer(h)
                ai = i.detach().abs().cpu().numpy().astype(np.float64)
                st = per_layer.setdefault(
                    li, {"n": 0, "sum": 0.0, "sq": 0.0, "max": 0.0})
                st["n"] += ai.size
                st["sum"] += float(ai.sum())
                st["sq"] += float((ai * ai).sum())
                st["max"] = max(st["max"], float(ai.max()))
                total += ai.size
                pooled_sum += float(ai.sum())
                pooled_sq += float((ai * ai).sum())
                pooled_max = max(pooled_max, float(ai.max()))
                ch = np.cosh(ai)
                for c in ceil_hits:
                    ceil_hits[c] += int((ch >= c).sum())
                if len(samples) < 40:
                    samples.append(ai.ravel()[::97].copy())
                if source is None:
                    u = torch.rand_like(i)
                    h = torch.where(u < 0.5 * (1.0 + torch.tanh(i)), 1.0, -1.0)
                else:
                    h = source.sample(i, layer=li)
    pooled = np.concatenate(samples) if samples else np.zeros(0)
    qs = [50, 75, 90, 95, 99]
    mean = pooled_sum / total if total else float("nan")
    var = (pooled_sq / total - mean * mean) if total else float("nan")
    return {
        "n_values": int(total),
        "passes": int(passes),
        "n_images": int(n),
        "abs_I_mean": float(mean),
        "abs_I_sd": float(np.sqrt(max(var, 0.0))),
        "abs_I_max": float(pooled_max),
        "abs_I_quantiles_subsampled": {f"q{q}": float(np.percentile(pooled, q))
                                       for q in qs} if pooled.size else {},
        "cosh_ceiling_binding_frac": {f"C={c:g}": (h / total if total else None)
                                      for c, h in ceil_hits.items()},
        "per_layer_abs_I_mean": {str(k): (v["sum"] / v["n"] if v["n"] else None)
                                 for k, v in per_layer.items()},
        "per_layer_abs_I_max": {str(k): v["max"] for k, v in per_layer.items()},
        "note": ("|I| over (image, unit, pass); quantiles from a strided "
                 "subsample, means/max exact. cosh_ceiling_binding_frac is "
                 "the fraction of values with cosh|I| >= C, i.e. the share of "
                 "the network on which the attempt-frequency ceiling binds."),
    }
