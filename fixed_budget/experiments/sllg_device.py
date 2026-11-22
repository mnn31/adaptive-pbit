# Author: Manan Gupta <mnn@yogins.com>
"""Device-level characterization of the sLLG macrospin neuron.

The macrospin has three parameters: barrier delta (kT), logit slope (I per
mT of perpendicular field) and alpha. They are options:

  --delta, --slope   set them directly (defaults 3.5 kT and 0.056 per mT,
                     the first device set; records in results/sllg/)
  --part match       choose them from a mumax3 record (default
                     results/micromagnetics/summary.json): the slope is the
                     measured field-to-I slope, and the barrier is the one
                     value at which the macrospin's zero-bias 1/e read
                     autocorrelation time equals the mumax3 tau_c(0), found
                     from Brown's Fokker-Planck equation (sllg.match_barrier).
                     Writes match.json in --out; every later part run with
                     the same --out uses it. Takes about two minutes.

Each part writes compact JSON records to --out (default results/sllg/):

  match      the fitted barrier and slope (see above)
  dt         time-step convergence of tau_c(0), P(+) and <mz^2>
  bias       equilibrium runs at the bias points: mz histogram, dwell times,
             P(+) and the field-to-I map, tau_c(I), read ACF, stale reads
             (each with the telegraph model's value at the same lag)
  step       ensemble step response <sign mz>(t) after a field step, with
             the memory left after one read interval against the telegraph
             model at the realized and at the nominal input
  alpha      zero-bias correlation time at alpha = 0.03
  easyplane  easy-plane control (no perpendicular anisotropy), read along x
  summary    device.json: tau_c(0) and the field-to-I map for the classifier,
             the three time scales at each bias against the model laws,
             replicate spread of tau_c(0)

Definitions: the read is sign(mz) at the read instant; tau_c(0) is the 1/e
time of the zero-bias read autocorrelation, the definition used for the
mumax3 disk, and r0 = read interval / tau_c(0); the field-to-I map is the
least-squares slope of I = atanh<sign mz> against Bz over the bias points
with |I| <= 2. Dwell times and escape rates use the mumax3 analysis's own
estimator (micromagnetics/analyze.py: flips out of a state over the time
spent in it, every run counted, the last one censored).

The read ACF is not a single exponential under bias, so every record
carries three time scales side by side: the 1/e time, the exponential-tail
fit with its amplitude (the weight of the slow component) and the
integrated time (area under the ACF). All three equal tau_c for a
telegraph signal.

    python experiments/sllg_device.py --part match --out results/sllg_matched
    for p in bias step summary; do python experiments/sllg_device.py \
        --part $p --out results/sllg_matched --gpu cuda:1; done
"""

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from smtjnn import sllg
from smtjnn.sllg import (
    Macrospin,
    MacrospinArray,
    fp_tau_1e,
    match_barrier,
    matched_device,
)

ROOT = Path(__file__).resolve().parents[1]
# the mumax3 analysis's dwell estimator, shared so both devices use one definition
sys.path.insert(0, str(ROOT / "micromagnetics"))
mm_analyze = importlib.import_module("analyze")
mmstats = importlib.import_module("mmstats")

OUT = ROOT / "results" / "sllg"
MUMAX_SUMMARY = ROOT / "results" / "micromagnetics" / "summary.json"

ALPHA = 0.1
# barrier (kT) and logit slope (per T); replaced by --delta/--slope or match.json
DEVICE = {"delta": 3.5, "slope": 56.0}
DT = 1e-12
SAMPLE = 10e-12            # sampling interval of the recorded reads
R0_GRID = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 4]
I_TARGETS = [0, 0.5, -0.5, 1, -1, 1.5, -1.5, 2, -2, 3, 4, 6, 8]
STEPS = [(0, 1), (0, 2), (0, 4), (0, 6), (0, 8),
         (-1, 1), (-2, 2), (-4, 4), (-6, 6), (-8, 8)]
# easy-plane control: same disk with Ku = 0. Shape anisotropy 39.05 kT
# (mumax3 demag energy difference of the 80-cell disk), moment Ms V / kT.
EASY_PLANE = Macrospin(delta=-39.05, moment=77.26, alpha=ALPHA)
BLOCKS = 16                # spin blocks for standard errors


# the macrospin for the current DEVICE barrier and slope, at damping alpha
def device_p(alpha=ALPHA):
    return matched_device(DEVICE["delta"], DEVICE["slope"], alpha)


# code hashes, torch version and time, stamped into every record
def provenance():
    files = [Path(sllg.__file__), Path(__file__),
             ROOT / "micromagnetics" / "analyze.py"]
    return {"code_sha256": {f.name: hashlib.sha256(f.read_bytes())
                            .hexdigest()[:16] for f in files},
            "torch": torch.__version__,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}


# write one JSON record to OUT with the provenance attached
def save(name, rec):
    OUT.mkdir(parents=True, exist_ok=True)
    rec = dict(rec, **provenance())
    (OUT / name).write_text(json.dumps(rec, indent=1))
    print("wrote", name, flush=True)


# read back a record that an earlier part wrote to OUT
def load(name):
    return json.loads((OUT / name).read_text())


# device parameters as stored in the records
def dev_dict(dev, dt):
    return {"delta": dev.delta, "moment_per_T": dev.moment,
            "alpha": dev.alpha, "b_anis_T": dev.b_anis, "dt_s": dt}


# ------------------------------------------------------------- simulation
# Equilibrium run of n spins at a held field b_z (T) for `duration` (s). Returns the
# sampled reads and the histograms the bias and easy-plane parts need.
def simulate(dev, b_z, n, duration, dt, seed, gpu, axis=2, burn=2e-9,
             every=None):
    """Record sign(m[axis]) and the hysteretic +-0.4 state of n spins.

    Spins start in exact Boltzmann equilibrium at b_z (so only a short
    burn-in is needed). Returns int8 series of shape [n, samples] on the
    GPU, the histogram (200 bins on [-1, 1]) of m[axis], the histogram of
    the in-plane angle (36 bins) and the time-averaged <mz^2>.
    """
    every = every or round(SAMPLE / dt)
    n_samples = round(duration / (every * dt))
    arr = MacrospinArray(dev, n, dt, seed=seed, device=gpu)
    arr.set_equilibrium(b_z if axis == 2 else 0.0)
    b = torch.full((n,), float(b_z), device=gpu)
    arr.run(b, round(burn / dt))
    sign = torch.empty(n_samples, n, dtype=torch.int8, device=gpu)
    hyst = torch.empty_like(sign)
    hist = torch.zeros(200, dtype=torch.float64, device=gpu)
    phi_hist = torch.zeros(36, dtype=torch.float64, device=gpu)
    mz2 = 0.0
    state = torch.where(arr.m[axis] >= 0, 1, -1).to(torch.int8)
    for t in range(n_samples):
        arr.run(b, every)
        c = arr.m[axis]
        sign[t] = torch.where(c >= 0, 1, -1).to(torch.int8)
        # hysteretic state: changes only when m crosses the far threshold +-0.4
        state = torch.where(c > 0.4, 1, torch.where(c < -0.4, -1, state))
        state = state.to(torch.int8)
        hyst[t] = state
        hist += torch.histc(c, bins=200, min=-1.0, max=1.0).double()
        phi = torch.atan2(arr.m[1], arr.m[0])
        phi_hist += torch.histc(phi, bins=36, min=-np.pi, max=np.pi).double()
        mz2 += (arr.m[2] ** 2).mean().item() / n_samples
    return {"sign": sign.T.contiguous(), "hyst": hyst.T.contiguous(),
            "hist": hist.cpu().numpy(), "phi_hist": phi_hist.cpu().numpy(),
            "mz2": mz2, "sample_s": every * dt}


# Normalized autocorrelation of a +-1 series [n, T] up to max_lag samples, computed
# by FFT separately for each spin block. Returns [BLOCKS, max_lag + 1].
def acf_blocks(series, max_lag):
    n, T = series.shape
    mean = series.float().mean()
    out = []
    for blk in torch.chunk(torch.arange(n, device=series.device), BLOCKS):
        acc = torch.zeros(max_lag + 1, dtype=torch.float64,
                          device=series.device)
        for sub in torch.split(blk, 512):
            x = series[sub].float() - mean
            # zero-padded to 2T so the FFT gives the linear autocovariance
            f = torch.fft.rfft(x, n=2 * T)
            c = torch.fft.irfft(f.abs() ** 2, n=2 * T)[:, :max_lag + 1]
            acc += c.double().sum(0)
        lag = torch.arange(max_lag + 1, device=series.device)
        acc = acc / ((T - lag) * len(blk))
        out.append((acc / acc[0]).cpu().numpy())
    return np.array(out)


# Exponential fit to the ACF over its first stretch from hi down to lo. Returns the
# time constant (s) and the amplitude, or NaNs if the stretch is too short.
def fit_tail(t, c, hi=0.8, lo=0.05):
    below_hi = np.nonzero(c <= hi)[0]
    if below_hi.size == 0:
        return np.nan, np.nan
    i0 = max(int(below_hi[0]), 1)
    below_lo = np.nonzero(c[i0:] < lo)[0]
    i1 = i0 + (int(below_lo[0]) if below_lo.size else c.size - i0)
    if i1 - i0 < 3:
        return np.nan, np.nan
    slope, icpt = np.polyfit(t[i0:i1], np.log(c[i0:i1]), 1)
    return -1.0 / slope, float(np.exp(icpt))


# First time the ACF falls below 1/e, linearly interpolated between samples.
def e_time(t, c):
    k = np.nonzero(c < np.exp(-1))[0]
    if k.size == 0 or k[0] == 0:
        return np.nan
    k = k[0]
    y0, y1 = c[k - 1], c[k]
    return float(t[k - 1] + (y0 - np.exp(-1)) / (y0 - y1) * (t[k] - t[k - 1]))


# Area under the ACF up to index cut (trapezoid), in s.
def integrated_time(t, c, cut):
    return float(np.sum(0.5 * (c[1:cut + 1] + c[:cut]) * np.diff(t[:cut + 1])))


# mean and standard error of the finite entries; None where there are too few
def mean_se(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return [float(x.mean()) if x.size else None, None]
    return [float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size))]


# The three time scales of a read series (1/e, tail fit, integrated area), each
# with a block standard error, and the ACF itself on a log-spaced lag grid.
def acf_summary(series, sample_s, max_time):
    max_lag = min(int(max_time / sample_s), series.shape[1] // 2)
    blocks = acf_blocks(series, max_lag)
    c = blocks.mean(0)
    t = np.arange(max_lag + 1) * sample_s
    tails = [fit_tail(t, b) for b in blocks]
    tau, amp = fit_tail(t, c)
    # integrate to where the mean ACF first drops below 1%; the same cut
    # is used for every block so the block spread is not biased by noise
    low = np.nonzero(c < 0.01)[0]
    cut = int(low[0]) if low.size else max_lag
    k = np.unique(np.round(np.geomspace(1, max_lag, 120)).astype(int))
    return {"tau_tail_s": tau, "tau_tail_se_s": mean_se([a for a, _ in
                                                         tails])[1],
            "tail_amplitude": amp,
            "tail_amplitude_se": mean_se([a for _, a in tails])[1],
            "tau_1e_s": e_time(t, c),
            "tau_1e_se_s": mean_se([e_time(t, b) for b in blocks])[1],
            "tau_int_s": integrated_time(t, c, cut),
            "tau_int_se_s": mean_se([integrated_time(t, b, cut)
                                     for b in blocks])[1],
            "tau_int_cut_s": float(t[cut]),
            "acf_at_100ps": float(c[round(1e-10 / sample_s)]),
            "acf_lag_s": t[k].tolist(), "acf": c[k].tolist(),
            "acf_se": (blocks[:, k].std(0, ddof=1) / np.sqrt(BLOCKS))
            .tolist()}


# Probability that a read is unchanged after `lag` samples, given + or - first, and
# the read ACF at that lag, each as mean and SE over spin blocks.
def stale_reads(series, lag):
    rows = []
    for blk in torch.chunk(torch.arange(series.shape[0],
                                        device=series.device), BLOCKS):
        a, b = series[blk, :-lag], series[blk, lag:]
        up = a > 0
        n_up = up.sum().item()
        n_dn = a.numel() - n_up
        same_up = (up & (b > 0)).sum().item()
        same_dn = (~up & (b < 0)).sum().item()
        af, bf = a.float(), b.float()
        mu = 0.5 * (af.mean() + bf.mean())
        cov = ((af - mu) * (bf - mu)).mean()
        var = 0.5 * (((af - mu) ** 2).mean() + ((bf - mu) ** 2).mean())
        # a block with no reads in a state says nothing about that state
        rows.append([same_up / n_up if n_up else np.nan,
                     same_dn / n_dn if n_dn else np.nan,
                     (cov / var).item()])
    rows = np.array(rows)
    return {k: mean_se(rows[:, j]) for j, k in
            enumerate(["p_same_plus", "p_same_minus", "acf"])}


# The telegraph model's stale-read values at input i after a lag of
# lag_over_tau0 zero-bias correlation times (cosh-law rate).
def telegraph_stale(i, lag_over_tau0):
    p = 0.5 * (1 + np.tanh(i))
    e = float(np.exp(-lag_over_tau0 * np.cosh(i)))
    return {"i": float(i), "p_same_plus": float(p + (1 - p) * e),
            "p_same_minus": float(1 - p + p * e), "acf": e}


# Correlation time at input i (field b_z in T) under the cosh, barrier-shift and
# Brown laws, each normalized to tau0 at zero bias.
def tau_laws(i, b_z, dev, tau0):
    """tau_c at input i under the three model laws, each tau0 at zero bias.

    cosh: the paper's telegraph law. shift: the same with the barrier-shift
    factor exp(i^2 / 4 delta), an empirical form (i is the logit, not
    moment * B). brown: Brown's high-barrier rates for the two wells,
    (1 - h^2)(1 -+ h) exp(-delta (1 -+ h)^2) with h = Bz / B_anis, valid
    below h = 1.
    """
    h = abs(b_z) / dev.b_anis
    out = {"tau_cosh_s": float(tau0 / np.cosh(i)),
           "tau_shift_s": float(tau0 * np.exp(i ** 2 / (4 * dev.delta))
                                / np.cosh(i)),
           "tau_brown_s": None}
    # above h = 1 the minority well is gone and the formula has no meaning
    if h < 1:
        rate = (1 - h ** 2) * sum(
            (1 + s * h) * np.exp(-dev.delta * ((1 + s * h) ** 2 - 1))
            for s in (-1, 1))
        out["tau_brown_s"] = float(2 * tau0 / rate)
    return out


# Escape rate and dwell-time shape for the + and - states of a read series, with
# the same estimator as the mumax3 analysis.
def dwell_stats(series_cpu, sample_s):
    """Escape rate and dwell shape per state, by the mumax3 analysis's estimator.

    rate = flips out of the state / time spent in it, over every run (the
    run under way at the start included, the last one censored); the shape
    statistics (cv, Kaplan-Meier distance D from the exponential, excess of
    very short dwells) use whole dwells only. mean_s = 1 / rate.
    """
    runs = mmstats.dwell_times(series_cpu.T, sample_s)
    out = {}
    for name, st in [("plus", 1), ("minus", -1)]:
        r = mm_analyze.dwell_stats(*runs[st])
        r = {k: (float(v) if isinstance(v, (float, np.floating)) else v)
             for k, v in r.items()}
        r["mean_s"] = 1.0 / r["rate"] if r["rate"] > 0 else None
        out[name] = r
    return out


# Density of m along the read axis from the 200-bin histogram, with the barrier
# from peak over centre density and the weight near the equator.
def histogram_summary(hist):
    centers = np.linspace(-1, 1, 201)
    centers = 0.5 * (centers[1:] + centers[:-1])
    p = hist / hist.sum()
    dens = p / (2.0 / 200)
    # centre density from the four bins around mz = 0
    mid = dens[98:102].mean()
    peak = dens.max()
    return {"centers": centers.tolist(), "density": dens.tolist(),
            "barrier_kT": float(np.log(peak / mid)) if mid > 0 else None,
            "peaks": [float(centers[:100][dens[:100].argmax()]),
                      float(centers[100:][dens[100:].argmax()])],
            "frac_abs_below_0.3": float(p[np.abs(centers) < 0.3].sum()),
            "frac_abs_below_0.5": float(p[np.abs(centers) < 0.5].sum())}


# ------------------------------------------------------------------ parts
# Time-step check: tau_c(0), P(+) and <mz^2> at I = 0 and 1 for four values of dt.
def part_dt(gpu):
    dev = device_p()
    rows = []
    for dt in [0.25e-12, 0.5e-12, 1e-12, 2e-12]:
        for i_t in [0.0, 1.0]:
            b = i_t / DEVICE["slope"]
            r = simulate(dev, b, 8192, 300e-9, dt, seed=11, gpu=gpu)
            p = (r["sign"] > 0).float().mean().item()
            mz2 = r["mz2"]
            acf = acf_summary(r["sign"], r["sample_s"], 60e-9)
            rows.append({"dt_s": dt, "i_target": i_t, "b_T": b,
                         "p_plus": p, "p_plus_exact":
                         sllg.equilibrium_p_plus(dev, b),
                         "mz2": mz2, "mz2_exact":
                         sllg.equilibrium_moments(dev, b)["mz2"],
                         "tau_tail_s": acf["tau_tail_s"],
                         "tau_tail_se_s": acf["tau_tail_se_s"],
                         "tail_amplitude": acf["tail_amplitude"],
                         "tau_1e_s": acf["tau_1e_s"],
                         "tau_int_s": acf["tau_int_s"],
                         "acf_at_100ps": acf["acf_at_100ps"]})
            print(rows[-1], flush=True)
    save("device_dt.json", {"device": dev_dict(dev, None), "n_spins": 8192,
                            "duration_s": 300e-9, "rows": rows})


# One equilibrium run per bias point in I_TARGETS; skips points already on disk so
# the part can be restarted.
def part_bias(gpu, n=16384, duration=400e-9):
    dev = device_p()
    tau0 = None
    for k, i_t in enumerate(I_TARGETS):
        name = f"device_bias_I{i_t:+g}.json"
        if (OUT / name).exists():
            tau0 = tau0 or load("device_bias_I+0.json")["acf"]["tau_1e_s"]
            continue
        b = i_t / DEVICE["slope"]
        r = simulate(dev, b, n, duration, DT, seed=1000 + k, gpu=gpu)
        sign, s_s = r["sign"], r["sample_s"]
        p_blocks = [(sign[blk] > 0).float().mean().item() for blk in
                    torch.chunk(torch.arange(n, device=gpu), BLOCKS)]
        p = float(np.mean(p_blocks))
        acf = acf_summary(sign, s_s, 100e-9)
        if i_t == 0:
            tau0 = acf["tau_1e_s"]
        # telegraph reference at the same lag: once at the input giving the
        # device's own exact P(+) (isolates the dynamics), once at the
        # nominal network input that the field stands for
        i_exact = float(np.arctanh(2 * sllg.equilibrium_p_plus(dev, b) - 1))
        stale = []
        for r0 in R0_GRID:
            lag = max(1, round(r0 * tau0 / s_s))
            x = lag * s_s / tau0
            stale.append(dict(r0=r0, lag_s=lag * s_s,
                              **stale_reads(sign, lag),
                              telegraph=telegraph_stale(i_exact, x),
                              telegraph_nominal=telegraph_stale(i_t, x)))
        rec = {"device": dev_dict(dev, DT), "i_target": i_t, "b_T": b,
               "n_spins": n, "duration_s": duration, "sample_s": s_s,
               "p_plus": [p, float(np.std(p_blocks, ddof=1) /
                                   np.sqrt(BLOCKS))],
               "p_plus_exact": sllg.equilibrium_p_plus(dev, b),
               "i_meas": float(np.arctanh(2 * p - 1)),
               "i_exact": i_exact,
               "hist": histogram_summary(r["hist"]),
               "acf": acf, "tau0_used_s": tau0, "stale": stale,
               "dwell_sign": dwell_stats(sign.cpu().numpy(), s_s),
               "dwell_hyst": dwell_stats(r["hyst"].cpu().numpy(), s_s)}
        save(name, rec)
        print(f"I={i_t:+g}: p={p:.4f} I_meas={rec['i_meas']:.3f} "
              f"tau_tail={acf['tau_tail_s'] * 1e9:.3f} ns "
              f"(amplitude {acf['tail_amplitude']:.3f}) "
              f"tau_1e={acf['tau_1e_s'] * 1e9:.3f} ns "
              f"tau_int={acf['tau_int_s'] * 1e9:.3f} ns", flush=True)
        del r, sign
        torch.cuda.empty_cache()


# Ensemble step response after each field step in STEPS, starting from exact
# equilibrium at the old field. Needs the zero-bias record from part_bias for tau0.
def part_step(gpu, n=262144):
    dev = device_p()
    tau0 = load("device_bias_I+0.json")["acf"]["tau_1e_s"]
    rows = []
    for k, (i0, i1) in enumerate(STEPS):
        b0, b1 = i0 / DEVICE["slope"], i1 / DEVICE["slope"]
        arr = MacrospinArray(dev, n, DT, seed=5000 + k, device=gpu)
        arr.set_equilibrium(b0)
        b = torch.full((n,), b1, device=gpu)
        every = round(SAMPLE / DT)
        n_samples = round(8 * tau0 / SAMPLE)
        blocks = torch.zeros(n_samples + 1, BLOCKS, device=gpu,
                             dtype=torch.float64)
        blocks[0] = torch.stack([c.mean() for c in
                                 torch.chunk(arr.read(), BLOCKS)])
        for t in range(1, n_samples + 1):
            arr.run(b, every)
            blocks[t] = torch.stack([c.mean() for c in
                                     torch.chunk(arr.read(), BLOCKS)])
        blocks = blocks.cpu().numpy()
        s = blocks.mean(1)
        se = blocks.std(1, ddof=1) / np.sqrt(BLOCKS)
        t = np.arange(n_samples + 1) * SAMPLE
        s_inf = 2 * sllg.equilibrium_p_plus(dev, b1) - 1
        s0 = 2 * sllg.equilibrium_p_plus(dev, b0) - 1
        g = (s - s_inf) / (s0 - s_inf)   # remaining memory, 1 at t = 0
        g_se = se / abs(s0 - s_inf)
        k4 = round(4 * tau0 / SAMPLE)
        # the field stands for the nominal input i1; the device's own logit
        # at that field (i1_exact) is smaller once the map bends
        i1_exact = float(np.arctanh(s_inf))
        memory = []
        for r0 in R0_GRID:
            lag = max(1, round(r0 * tau0 / SAMPLE))
            x = lag * SAMPLE / tau0
            memory.append({"r0": r0, "lag_s": lag * SAMPLE,
                           "g": float(g[lag]), "g_se": float(g_se[lag]),
                           "g_telegraph": float(np.exp(-x * np.cosh(i1_exact))),
                           "g_telegraph_nominal":
                           float(np.exp(-x * np.cosh(i1)))})
        keep = np.unique(np.round(np.geomspace(1, n_samples, 150))
                         .astype(int))
        tau_g, amp_g = fit_tail(t, g)
        rows.append({"i_from": i0, "i_to": i1, "b_from_T": b0,
                     "b_to_T": b1, "s0_exact": s0, "s_inf_exact": s_inf,
                     "s_end": float(s[-1]), "s_end_se": float(se[-1]),
                     "g_at_4tau0": float(g[k4]),
                     "i_to_exact": i1_exact,
                     "g_at_4tau0_se": float(g_se[k4]),
                     "g_telegraph_4tau0": float(np.exp(-4 * np.cosh(i1))),
                     "tau_1e_s": e_time(t, g), "tau_tail_s": tau_g,
                     "tail_amplitude": amp_g,
                     "tau_int_s": integrated_time(t, g, n_samples),
                     "tau_cosh_s": tau0 / np.cosh(i1_exact),
                     "tau_cosh_nominal_s": tau0 / np.cosh(i1),
                     "memory": memory,
                     "t_s": t[keep].tolist(), "g": g[keep].tolist(),
                     "s": s[keep].tolist(), "s_se": se[keep].tolist()})
        print(f"step {i0}->{i1}: g(4tau0)={g[k4]:.2e} "
              f"(telegraph {np.exp(-4 * np.cosh(i1)):.2e}) "
              f"tau_1e={rows[-1]['tau_1e_s'] * 1e9:.3f} ns "
              f"cosh law {rows[-1]['tau_cosh_s'] * 1e9:.3f} ns at the "
              f"realized I={i1_exact:.2f}, "
              f"{rows[-1]['tau_cosh_nominal_s'] * 1e9:.3f} ns nominal",
              flush=True)
    save("device_step.json", {"device": dev_dict(dev, DT), "n_spins": n,
                              "tau0_s": tau0, "sample_s": SAMPLE,
                              "rows": rows})


# Zero-bias correlation time at alpha = 0.03 and at the default 0.1.
def part_alpha(gpu):
    rows = []
    for alpha in [0.03, 0.1]:
        dev = device_p(alpha)
        r = simulate(dev, 0.0, 8192, 1200e-9, DT, seed=77, gpu=gpu)
        acf = acf_summary(r["sign"], r["sample_s"], 300e-9)
        rows.append({"device": dev_dict(dev, DT), **{
            k: acf[k] for k in ["tau_tail_s", "tau_tail_se_s",
                                "tail_amplitude", "tau_1e_s", "tau_int_s",
                                "acf_at_100ps"]}})
        print(rows[-1], flush=True)
    save("device_alpha.json", {"n_spins": 8192, "duration_s": 1200e-9,
                               "rows": rows})


# Easy-plane control (no perpendicular anisotropy), read along x, at two time steps.
def part_easyplane(gpu, n=8192, duration=20e-9):
    dev = EASY_PLANE
    rows = []
    for dt in [0.1e-12, 0.05e-12]:
        r = simulate(dev, 0.0, n, duration, dt, seed=31, gpu=gpu, axis=0,
                     burn=1e-9, every=round(5e-12 / dt))
        sx, s_s = r["sign"], r["sample_s"]
        ph = r["phi_hist"] / r["phi_hist"].mean()
        acf = acf_summary(sx, s_s, 2e-9)
        rows.append({
            "dt_s": dt, "sample_s": s_s,
            "mx_hist": histogram_summary(r["hist"]),
            "phi_hist_rel": ph.tolist(),
            "phi_hist_max_dev": float(np.abs(ph - 1).max()),
            "phi_free_energy_range_kT": float(np.log(ph.max() / ph.min())),
            "mz_rms": float(np.sqrt(r["mz2"])),
            "mz_rms_exact": float(np.sqrt(sllg.equilibrium_moments(
                dev, 0.0)["mz2"])),
            "acf_sign_mx": acf,
            "dwell_sign_mx": dwell_stats(sx.cpu().numpy(), s_s),
            "flips_per_ns": float((sx[:, 1:] != sx[:, :-1]).float().sum()
                                  .item() / n / duration * 1e-9)})
        print(f"easy plane dt={dt}: phi dev {rows[-1]['phi_hist_max_dev']:.3f}"
              f" mz rms {rows[-1]['mz_rms']:.3f} "
              f"tau_1e {acf['tau_1e_s'] * 1e12:.1f} ps", flush=True)
    save("device_easyplane.json", {"device": dev_dict(dev, None),
                                   "n_spins": n, "duration_s": duration,
                                   "rows": rows})


# Collect the bias records into device.json: the field-to-I slope and its fit
# quality, tau_c(0) with its replicate spread, and the model laws at each point.
def part_summary():
    dev = device_p()
    pts = [load(f"device_bias_I{i:+g}.json") for i in I_TARGETS]
    b = np.array([p["b_T"] for p in pts])
    i = np.array([p["i_meas"] for p in pts])
    # SE of I from the SE of p: dI/dp = 1 / (2 p (1 - p))
    se = np.array([p["p_plus"][1] / (2 * p["p_plus"][0] *
                                     (1 - p["p_plus"][0])) for p in pts])
    lin = np.abs(i) <= 2
    w = 1.0 / se[lin] ** 2
    slope = float((w * b[lin] * i[lin]).sum() / (w * b[lin] ** 2).sum())
    slope_se = float(1.0 / np.sqrt((w * b[lin] ** 2).sum()))
    chi2 = float((w * (i[lin] - slope * b[lin]) ** 2).sum())
    cubic = np.linalg.lstsq(np.stack([b[lin], b[lin] ** 3], 1), i[lin],
                            rcond=None)[0]
    i_ex = np.array([p["i_exact"] for p in pts])
    zs = (i - i_ex) / se
    z = load("device_bias_I+0.json")
    tau0 = z["acf"]["tau_1e_s"]
    # independent zero-bias runs at the same device and dt (when present):
    # the block SE of one run understates the run-to-run spread
    keys = ["tau_tail_s", "tail_amplitude", "tau_1e_s", "tau_int_s"]
    reps = [{"run": "bias", **{k: z["acf"][k] for k in keys}}]
    if (OUT / "device_dt.json").exists():
        reps += [{"run": "dt", **{k: r[k] for k in keys}}
                 for r in load("device_dt.json")["rows"]
                 if r["i_target"] == 0 and r["dt_s"] == DT]
    if (OUT / "device_alpha.json").exists():
        reps += [{"run": "alpha", **{k: r[k] for k in keys}}
                 for r in load("device_alpha.json")["rows"]
                 if r["device"]["alpha"] == ALPHA]

    # mean and SD of one time scale over the zero-bias replicate runs
    def spread(key, name):
        v = [r[key] for r in reps]
        return {f"{name}_mean_s": float(np.mean(v)),
                f"{name}_sd_s": float(np.std(v, ddof=1)) if len(v) > 1
                else None}

    match = load("match.json") if (OUT / "match.json").exists() else None
    rec = {"device": dev_dict(dev, DT),
           "tau_c0_definition": "1/e time of the zero-bias read "
                                "autocorrelation; r0 = read interval / tau_c0_s",
           "tau_c0_s": tau0,
           "tau_c0_se_s": z["acf"]["tau_1e_se_s"],
           "tau_c0_fokker_planck_s": fp_tau_1e(dev),
           "tau_c0_tail_s": z["acf"]["tau_tail_s"],
           "tau_c0_int_s": z["acf"]["tau_int_s"],
           "tail_amplitude0": z["acf"]["tail_amplitude"],
           "tau_c0_replicates": {"runs": reps, **spread("tau_1e_s", "tau_1e"),
                                 **spread("tau_tail_s", "tail"),
                                 **spread("tau_int_s", "tau_int")},
           "match": match,
           "boltzmann_check": {
               "chi2_abs_i_le_2": float((zs[lin] ** 2).sum()),
               "n_abs_i_le_2": int(lin.sum()),
               "chi2_all": float((zs ** 2).sum()), "n_all": int(zs.size),
               "max_abs_z": float(np.abs(zs).max()),
               "mean_z": float(zs.mean())},
           "slope_per_T": slope, "slope_se_per_T": slope_se,
           "slope_fit_chi2": chi2, "slope_fit_dof": int(lin.sum() - 1),
           "slope_exact_per_T": sllg.logit_slope(dev),
           "cubic_fit": {"a_per_T": float(cubic[0]),
                         "c_per_T3": float(cubic[1])},
           "tesla_per_unit_i": 1.0 / slope,
           "map_note": "Bz = I * tesla_per_unit_i at every I, no clipping; "
                       "tesla_per_unit_i is 1 / slope_per_T, the slope "
                       "measured on these sLLG runs, not the Boltzmann "
                       "value slope_exact_per_T that the moment was set to",
           "tau_note": "the read ACF is not one exponential under bias: "
                       "tau_tail_s is the slow component only, of weight "
                       "tail_amplitude; tau_1e_s and tau_int_s cover the "
                       "whole decay; the model laws are evaluated at i_exact",
           "points": [{"i_target": p["i_target"], "b_T": p["b_T"],
                       "i_meas": p["i_meas"], "i_se": float(s),
                       "i_exact": p["i_exact"],
                       "z_vs_exact": float(zz),
                       **{k: p["acf"][k] for k in
                          ["tau_tail_s", "tau_tail_se_s", "tail_amplitude",
                           "tail_amplitude_se", "tau_1e_s", "tau_1e_se_s",
                           "tau_int_s", "tau_int_se_s"]},
                       **tau_laws(p["i_exact"], p["b_T"], dev, tau0)}
                      for p, s, zz in zip(pts, se, zs)]}
    save("device.json", rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "points"},
                     indent=1))


# Fit the macrospin to a mumax3 record: take its field-to-I slope, then find the
# barrier whose zero-bias 1/e read time equals the mumax3 tau_c(0).
def part_match(mumax_summary):
    mm = json.loads(Path(mumax_summary).read_text())
    tau, tau_se = (x * 1e-9 for x in mm["tau_c0_ns"])
    slope, slope_se = (x * 1e3 for x in mm["field_slope_I_per_mT"])
    dev = match_barrier(tau, slope, ALPHA)
    # barrier range spanned by the mumax3 standard errors
    lo = match_barrier(tau - tau_se, slope, ALPHA).delta
    hi = match_barrier(tau + tau_se, slope, ALPHA).delta
    rec = {"rule": "slope = mumax3 field-to-I slope; delta = the barrier at "
                   "which the macrospin's zero-bias 1/e read "
                   "autocorrelation time (Brown's Fokker-Planck equation) "
                   "equals the mumax3 tau_c(0). Only delta is fitted.",
           "mumax3_record": os.path.relpath(Path(mumax_summary).resolve(),
                                            ROOT),
           "mumax3_tau_c0_ns": mm["tau_c0_ns"],
           "mumax3_slope_I_per_mT": mm["field_slope_I_per_mT"],
           "mumax3_histogram_barrier_kT": mm.get("barrier_kT"),
           "delta": dev.delta, "delta_range_for_tau_se": [lo, hi],
           "slope_per_T": slope, "slope_se_per_T": slope_se,
           "device": dev_dict(dev, DT),
           "fp_tau_1e_s": fp_tau_1e(dev)}
    save("match.json", rec)
    print(json.dumps(rec, indent=1))


# Pick the output directory and the device; match.json in --out wins over
# --delta/--slope for every part but match itself.
def configure(args):
    global OUT
    OUT = Path(args.out).resolve()
    DEVICE.update(delta=args.delta, slope=args.slope * 1e3)
    if args.part != "match" and (OUT / "match.json").exists():
        m = load("match.json")
        DEVICE.update(delta=m["delta"], slope=m["slope_per_T"])
    print(f"device: delta {DEVICE['delta']:.4f} kT, slope "
          f"{DEVICE['slope'] / 1e3:.5f} per mT -> {OUT}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=["match", "dt", "bias", "step", "alpha",
                             "easyplane", "summary"])
    ap.add_argument("--gpu", default="cuda:1")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--delta", type=float, default=3.5,
                    help="barrier in kT (ignored if OUT/match.json exists)")
    ap.add_argument("--slope", type=float, default=0.056,
                    help="logit slope, I per mT (ignored if OUT/match.json "
                         "exists)")
    ap.add_argument("--mumax", default=str(MUMAX_SUMMARY),
                    help="mumax3 summary.json used by --part match")
    args = ap.parse_args()
    configure(args)
    t0 = time.time()
    if args.part == "summary":
        part_summary()
    elif args.part == "match":
        part_match(args.mumax)
    else:
        globals()[f"part_{args.part}"](args.gpu)
    print(f"{args.part} done in {time.time() - t0:.0f}s", flush=True)
