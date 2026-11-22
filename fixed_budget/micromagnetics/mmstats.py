# Author: Manan Gupta <mnn@yogins.com>
"""Statistics of simulated nanomagnet trajectories, compared with the telegraph model.

All functions take arrays shaped (time, disk) sampled at a fixed interval dt.
Reads are thresholded: s = sign(m), as a comparator on the TMR signal gives.
The telegraph reference is the letter's two-state model: stationary
P(+) = p, correlation time tau_c, propagator p + (1[s=+] - p) exp(-t/tau_c).
"""

import numpy as np
from scipy.optimize import curve_fit


def read_table(path):
    """Load a mumax3 table.txt into (header, array) quickly."""
    with open(path) as f:
        header = f.readline().lstrip("# ").rstrip("\n").split("\t")
        data = np.array(f.read().split(), dtype=np.float64)
    return header, data.reshape(-1, len(header))


def sign_reads(m):
    """Thresholded reads +-1 (exact zeros, which do not occur in practice, read as +1)."""
    return np.where(m >= 0, 1, -1).astype(np.int8)


def hysteretic_states(m, thr=0.4):
    """State that switches only when m crosses the far threshold (+-thr); starts at sign(m[0])."""
    m = np.asarray(m)
    s = np.empty(m.shape, dtype=np.int8)
    cur = sign_reads(m[0])
    for t in range(m.shape[0]):
        cur = np.where(m[t] > thr, 1, np.where(m[t] < -thr, -1, cur)).astype(np.int8)
        s[t] = cur
    return s


def dwell_times(s, dt):
    """Run lengths of a +-1 series per disk.

    Returns dict state -> (durations, censored, first). censored marks the last
    run of a disk, cut by the window end before it flipped. first marks the run
    already under way at the window start: its flip is seen, so it counts in the
    escape rate, but its length is not a whole dwell.
    """
    out = {1: ([], [], []), -1: ([], [], [])}
    s = np.asarray(s)
    for k in range(s.shape[1]):
        x = s[:, k]
        edges = np.flatnonzero(np.diff(x)) + 1
        starts = np.concatenate([[0], edges])
        ends = np.concatenate([edges, [len(x)]])
        for i, (a, b) in enumerate(zip(starts, ends)):
            d, c, f = out[int(x[a])]
            d.append((b - a) * dt)
            c.append(i == len(starts) - 1)
            f.append(i == 0)
    return {st: (np.array(d), np.array(c, dtype=bool), np.array(f, dtype=bool))
            for st, (d, c, f) in out.items()}


def exp_rate_mle(durations, censored):
    """Escape rate: flips out of a state / time spent in it (exponential MLE, right censoring).

    Pass every run, the first of each disk included. Leaving out the flip that
    ends a first run while keeping its time loses one flip per disk and biases
    the rate low by about (disks) / (flips).
    """
    total = durations.sum()
    n = (~censored).sum()
    return n / total if total > 0 else np.nan


def escape_rates(s, dt):
    """Escape rates (k_out_of_up, k_out_of_down) of a +-1 series (time, disk), and the flip counts.

    The same estimator as exp_rate_mle over every run of dwell_times, written
    without the runs: flips out of a state over the total time spent in it,
    with the time of the first and last run of every disk included.
    The correlation time of the matching telegraph process is 1 / (k_up + k_down).
    """
    s = np.asarray(s)
    a, b = s[:-1], s[1:]
    n_up, n_dn = int(((a == 1) & (b == -1)).sum()), int(((a == -1) & (b == 1)).sum())
    t_up, t_dn = (s == 1).sum() * dt, (s == -1).sum() * dt
    k_up = n_up / t_up if t_up > 0 else np.nan
    k_dn = n_dn / t_dn if t_dn > 0 else np.nan
    return k_up, k_dn, n_up, n_dn


def km_survival(durations, censored):
    """Kaplan-Meier survival curve; returns (times, S(t)) at each completed-run time."""
    order = np.argsort(durations, kind="stable")
    d = durations[order]
    c = censored[order]
    at_risk = len(d) - np.arange(len(d))
    factors = np.where(c, 1.0, 1.0 - 1.0 / at_risk)
    surv = np.cumprod(factors)
    keep = ~c
    return d[keep], surv[keep]


def exp_deviation(durations, censored, rate):
    """Largest gap between the Kaplan-Meier survival and exp(-rate t) (KS-like D).

    Takes whole dwells only (runs that began inside the window).
    """
    t, s = km_survival(durations, censored)
    if len(t) == 0:
        return np.nan
    # Compare on both sides of each step of the empirical curve.
    before = np.concatenate([[1.0], s[:-1]])
    model = np.exp(-rate * t)
    return float(max(np.abs(s - model).max(), np.abs(before - model).max()))


def short_dwell_excess(durations, censored, rate, frac=0.05):
    """Kaplan-Meier fraction of whole dwells shorter than frac / rate, over the exponential value."""
    t, s = km_survival(durations, censored)
    if len(t) == 0:
        return np.nan
    cut = frac / rate
    below = t < cut
    observed = 1 - s[below][-1] if below.any() else 0.0
    return float(observed / (1 - np.exp(-rate * cut)))


def telegraph_series(k_up, k_down, n_steps, n_disks, dt, rng, p0=None):
    """Exact sampled telegraph process (+-1), used as the null model for dwell tests."""
    p = k_up / (k_up + k_down)
    stay_up = p + (1 - p) * np.exp(-(k_up + k_down) * dt)
    stay_dn = 1 - p * (1 - np.exp(-(k_up + k_down) * dt))
    s = np.empty((n_steps, n_disks), dtype=np.int8)
    s[0] = np.where(rng.random(n_disks) < (p if p0 is None else p0), 1, -1)
    for t in range(1, n_steps):
        u = rng.random(n_disks)
        stay = np.where(s[t - 1] == 1, u < stay_up, u < stay_dn)
        s[t] = np.where(stay, s[t - 1], -s[t - 1])
    return s


def autocorr(s, max_lag):
    """Normalized autocorrelation of a (time, disk) series, lags 0..max_lag.

    Uses the pooled mean and variance, so C(0) = 1 and C(t) -> 0 for a stationary
    process; for a +-1 telegraph process C(t) = exp(-t / tau_c).
    """
    x = np.asarray(s, dtype=np.float64)
    x = x - x.mean()
    n = x.shape[0]
    if max_lag >= n:
        raise ValueError("max_lag must be shorter than the series")
    size = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, n=size, axis=0)
    acov = np.fft.irfft(f * np.conj(f), n=size, axis=0)[: max_lag + 1].sum(axis=1)
    acov /= (n - np.arange(max_lag + 1)) * x.shape[1]
    if acov[0] <= 0:
        return np.full(max_lag + 1, np.nan)    # a disk set that never left one state
    return acov / acov[0]


def efold_time(c, dt):
    """First time the autocorrelation falls below 1/e (linear interpolation)."""
    below = np.flatnonzero(c < np.exp(-1))
    if len(below) == 0:
        return np.nan
    i = below[0]
    c0, c1 = c[i - 1], c[i]
    return float((i - 1 + (c0 - np.exp(-1)) / (c0 - c1)) * dt)


def fit_exp_tail(c, dt, lo=0.6, hi=0.2):
    """Fit C(t) = A exp(-t/tau) on the stretch where lo >= C >= hi.

    Returns (tau, A). A < 1 measures correlation lost faster than the
    telegraph decay (in-well motion and barrier-top recrossings).
    """
    t = np.arange(len(c)) * dt
    first_low = np.flatnonzero(c < hi)
    end = first_low[0] if len(first_low) else len(c)
    sel = (c[:end] <= lo) & (c[:end] > 0)
    if sel.sum() < 3:
        return np.nan, np.nan
    slope, icpt = np.polyfit(t[:end][sel], np.log(c[:end][sel]), 1)
    return float(-1 / slope), float(np.exp(icpt))


def stale_read(s, lags):
    """P(s(t+lag) = s(t)) overall and given s(t) = +1 or -1, pooled over disks."""
    s = np.asarray(s)
    rows = []
    for lag in lags:
        a, b = s[:-lag], s[lag:]
        same = a == b
        up = a == 1
        rows.append((same.mean(), same[up].mean() if up.any() else np.nan,
                     same[~up].mean() if (~up).any() else np.nan))
    return np.array(rows)


def telegraph_stale(p, x):
    """Telegraph prediction of stale-read probability at x = lag / tau_c: (overall, from +, from -)."""
    e = np.exp(-np.asarray(x, dtype=np.float64))
    from_up = p + (1 - p) * e
    from_dn = 1 - p * (1 - e)
    return p * from_up + (1 - p) * from_dn, from_up, from_dn


def mz_histogram(m, bins=100):
    """Density of m on [-1, 1] and the effective barrier ln[P(peak) / P(0)] in kT."""
    dens, edges = np.histogram(np.asarray(m).ravel(), bins=bins, range=(-1, 1), density=True)
    centres = (edges[:-1] + edges[1:]) / 2
    mid = dens[np.abs(centres) < 0.1].mean()
    neg, pos = centres < 0, centres > 0
    peaks = (centres[neg][np.argmax(dens[neg])], centres[pos][np.argmax(dens[pos])])
    barrier = np.log(dens.max() / mid) if mid > 0 else np.inf
    return dens, centres, float(barrier), peaks


def well_depths(dens, centres, window=0.6):
    """Depths (kT) of the m < 0 and m > 0 wells: ln of each peak density over the saddle density.

    The saddle is the density minimum within abs(m) < window. nan where the
    saddle bin is empty (a well too rarely visited to resolve).
    """
    inner = np.abs(centres) < window
    j = np.flatnonzero(inner)[np.argmin(dens[inner])]
    if dens[j] <= 0:
        return np.nan, np.nan, float(centres[j])
    with np.errstate(divide="ignore"):
        lo = np.log(dens[:j].max() / dens[j]) if dens[:j].max() > 0 else np.nan
        hi = np.log(dens[j + 1:].max() / dens[j])
    return float(lo), float(hi), float(centres[j])


def brown_relax_ratio(i, delta):
    """tau_c(0) / tau_c(I) from the high-barrier formula for an axially symmetric macrospin.

    Field h = I / (2 delta) tilts the wells to delta (1 -+ h)^2; each escape rate
    carries Brown's (1 - h^2)(1 +- h) prefactor. The relaxation rate is the sum.
    The formula is asymptotic in the barrier, so at a few kT it is only a reference.
    """
    h = np.asarray(i, dtype=np.float64) / (2 * delta)
    h = np.clip(h, -1, 1)
    rate = (1 - h**2) * ((1 + h) * np.exp(-delta * (1 + h) ** 2)
                         + (1 - h) * np.exp(-delta * (1 - h) ** 2))
    return rate / (2 * np.exp(-delta))


def step_response(mean_s, dt, p, s0):
    """Fit mean_s(t) = p + (s0 - p) exp(-t/tau) after a bias step; returns tau.

    mean_s is the ensemble mean of the +-1 reads (so p here is the stationary
    mean of s, i.e. tanh I, and s0 its value at the step). The asymptote p is
    an input, and tau is sensitive to it; step_fit_free fits it instead.
    """
    t = np.arange(len(mean_s)) * dt
    frac = (np.asarray(mean_s) - p) / (s0 - p)
    sel = (frac > 0.1) & (frac < 0.9)
    if sel.sum() < 3:
        return np.nan
    slope = np.polyfit(t[sel], np.log(frac[sel]), 1)[0]
    return float(-1 / slope)


def step_fit_free(mean_s, dt):
    """Least-squares fit of mean_s(t) = s_inf + a exp(-t/tau), all three free; returns (tau, s_inf)."""
    y = np.asarray(mean_s, dtype=np.float64)
    t = np.arange(len(y)) * dt
    tail = y[-(len(y) // 5):].mean()
    a0 = y[0] - tail
    if a0 == 0:
        return np.nan, np.nan
    below = np.flatnonzero((y - tail) / a0 < np.exp(-1))
    tau0 = max(t[below[0]], dt) if len(below) else t[-1] / 3
    try:
        (s_inf, _a, tau), _ = curve_fit(lambda x, c, a, k: c + a * np.exp(-x / k), t, y,
                                        p0=(tail, a0, tau0), maxfev=2000)
    except RuntimeError:
        return np.nan, np.nan
    return float(tau), float(s_inf)
