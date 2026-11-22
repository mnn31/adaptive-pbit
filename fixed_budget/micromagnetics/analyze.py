# Author: Manan Gupta <mnn@yogins.com>
"""Reduce packed mumax3 runs to the compact records in results/micromagnetics/.

Usage: python analyze.py RUNDIR [RUNDIR ...] OUTDIR

The equilibrium runs of the perpendicular disk (sets eq, dt25 and dt12) differ
only in the time step, so they are grouped by it. The PRODUCTION_FS group
(25 fs) gives every record under bias; the 50 fs and 12.5 fs groups are the
time-step check in checks.json. Steps, the easy-plane control and the
spin-torque runs are treated the same way: the 25 fs runs give the record and
the 50 fs runs are kept beside it under a _dt50 name. Spin-torque points are
compared with the zero-bias point of their own time step. The array-pitch
control is a zero-bias 50 fs run and is compared with the 50 fs zero-bias point.

Every number is computed per thermal seed (one mumax3 process, 64 disks) and
reported as the seed mean with its standard error. The correlation time of a
point is the 1/e time of the read autocorrelation; tau_c(0) from the
zero-field runs sets the read-ratio scale r0 = lag / tau_c(0). Two cross-checks
are kept: the dwell-rate time 1/(k_up + k_down) and an exponential fit to the
autocorrelation between C = 0.6 and 0.2, whose amplitude measures the fast loss.

The 1/e time is read off the same autocorrelation that the stale-read test
measures, so a telegraph prediction built on it agrees at r0 = 1 by
construction. The stale-read records therefore also carry the prediction built
on the dwell-rate time, which comes from counted flips (flips out of each
state over the time spent in it) and not from the autocorrelation. Rows where
the lag equals the correlation time a prediction is built on are marked
by_construction. Step-response numbers are given for three choices of the
asymptote (fitted, equilibrium runs, segment tail) with bootstrap errors.
Where the minority state is never visited the input I is not measured; the
step record then carries I and the cosh-law ratio for several extrapolations
of the field-to-I map, and their spread is the uncertainty of that ratio.
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mmstats as ms
import numpy as np
from scipy.special import i0, i1

KT = 1.380649e-23 * 300
CELL_VOL = 2e-9 * 2e-9 * 1e-9
R0_GRID = (0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 4)
LOG_LAGS_NS = np.unique(np.round(np.geomspace(0.01, 80, 60), 2))
N_NULL = 200              # telegraph replicas per seed for the dwell-test null
N_BOOT = 300              # bootstrap resamples of the disk-steps in a step response
MEMORY_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
PRODUCTION_FS = 25.0      # time step of the production set; 50 and 12.5 fs are the check
EQ_SETS = ("eq", "dt25", "dt12")
WORKERS = min(32, os.cpu_count() or 1)


# Seed mean and standard error (nan-safe; SE is nan for a single seed).
def mean_se(values):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if len(v) == 0:
        return [np.nan, np.nan]
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
    return [float(v.mean()), float(se)]


# Parameters of a job, given its path without extension.
def meta_of(stem):
    return json.loads(Path(f"{stem}.json").read_text())


# Packed run -> (metadata, time step of the table in ns, times, magnetization after
# burn-in).
def load(stem):
    meta = meta_of(stem)
    z = np.load(f"{stem}.npz")
    t = z["t"] * 1e9
    dt = float(np.median(np.diff(t)))
    m = z["m"].astype(np.float32)
    keep = t >= meta["burn_ns"] - 1e-6
    return meta, dt, t[keep] - meta["burn_ns"], m[keep]


# Mean equal-time correlation of reads between nearest-neighbour disks.
def neighbour_corr(s, nx, ny):
    x = s.astype(np.float64)
    x = (x - x.mean(0)) / (x.std(0) + 1e-12)
    grid = x.reshape(len(x), nx, ny)
    pairs = [(grid[:, :-1, :], grid[:, 1:, :]), (grid[:, :, :-1], grid[:, :, 1:])]
    return float(np.mean([(a * b).mean() for a, b in pairs]))


# 95th percentile of a null sample and the fraction of it at or above d (nan-safe).
def null_stats(null, d):
    null = np.asarray(null, dtype=np.float64)
    null = null[np.isfinite(null)]
    if len(null) < 20 or not np.isfinite(d):
        return np.nan, np.nan
    return float(np.quantile(null, 0.95)), float(np.mean(null >= d))


# Escape rate from every run, and the shape of the whole dwells (those begun inside the
# window).
def dwell_stats(durations, censored, first):
    rate = float(ms.exp_rate_mle(durations, censored))
    d, c = durations[~first], censored[~first]
    done = d[~c]
    # The window cuts long dwells, so cv of the completed ones sits a little below 1 even
    # for an exponential; compare it with cv_null_mean.
    return {"n_flips": int((~censored).sum()), "n_dwells": len(done), "rate": rate,
            "cv": float(done.std() / done.mean()) if len(done) > 1 else np.nan,
            "D": ms.exp_deviation(d, c, rate),
            "short_excess": ms.short_dwell_excess(d, c, rate)}


# Dwell statistics of a +-1 series per state, with the same statistics of a telegraph
# null.
def dwell_block(s, dt, rng, null=True):
    dw = ms.dwell_times(s, dt)
    out = {key: dwell_stats(*dw[st]) for st, key in ((1, "up"), (-1, "down"))}
    if null and out["up"]["n_dwells"] > 20 and out["down"]["n_dwells"] > 20:
        # Same rates, disks, window and sampling, but an exact telegraph process.
        n = s.shape[1]
        sim = ms.telegraph_series(out["down"]["rate"], out["up"]["rate"], s.shape[0],
                                  n * N_NULL, dt, rng)
        reps = [ms.dwell_times(sim[:, j * n:(j + 1) * n], dt) for j in range(N_NULL)]
        for st, key in ((1, "up"), (-1, "down")):
            stats = [dwell_stats(*r[st]) for r in reps]
            dnull = np.array([x["D"] for x in stats])
            out[key]["D_null"] = dnull
            out[key]["D_null_mean"] = float(np.nanmean(dnull))
            out[key]["D_null_p95"], out[key]["D_p_value"] = null_stats(dnull, out[key]["D"])
            out[key]["cv_null_mean"] = float(np.nanmean([x["cv"] for x in stats]))
            out[key]["short_excess_null_mean"] = float(np.nanmean([x["short_excess"] for x in stats]))
            out[key]["short_excess_null_sd"] = float(np.nanstd([x["short_excess"] for x in stats]))
    return out


# All single-seed statistics of a two-state (m_z) run.
def seed_record(meta, dt, m, rng, max_lag_ns=90.0):
    s = ms.sign_reads(m)
    max_lag = min(round(max_lag_ns / dt), len(s) // 2)
    c = ms.autocorr(s, max_lag)
    tau_tail, amp = ms.fit_exp_tail(c, dt)
    dens, centres, barrier, peaks = ms.mz_histogram(m)
    depth_minus, depth_plus, saddle = ms.well_depths(dens, centres)
    p = float((s == 1).mean())
    with np.errstate(divide="ignore"):
        i_meas = float(np.arctanh(2 * p - 1))    # infinite if the minority state never appears
    rec = {"name": meta["name"], "p_up": p, "I": i_meas, "minority_frac": min(p, 1 - p),
           "mean_abs_m": float(np.abs(m).mean()), "barrier_kT": barrier, "peaks": list(peaks),
           "well_minus_kT": depth_minus, "well_plus_kT": depth_plus, "saddle_m": saddle,
           "frac_abs_below_0p3": float((np.abs(m) < 0.3).mean()),
           "tau_efold": ms.efold_time(c, dt), "tau_tail": tau_tail, "tail_amp": amp,
           "tau_int": float(np.trapezoid(np.clip(c, 0, None), dx=dt)),
           "acf_0p1ns": float(c[round(0.1 / dt)]),
           "disk_ns": float(len(s) * dt * s.shape[1]),
           "neighbour_corr": neighbour_corr(s, meta["nx"], meta["ny"]),
           "dwell_hyst": dwell_block(ms.hysteretic_states(m), dt, rng),
           "dwell_sign": dwell_block(s, dt, rng, null=False)}
    h = rec["dwell_hyst"]
    rec["tau_dwell"] = 1 / (h["up"]["rate"] + h["down"]["rate"])
    rec["flips"] = h["up"]["n_flips"] + h["down"]["n_flips"]
    # Stationarity after the discarded start: P(+) in the first 20 ns, the next 40 ns and the rest.
    a, b = round(20 / dt), round(60 / dt)
    rec["p_up_blocks"] = [float((x == 1).mean()) if len(x) else np.nan for x in (s[:a], s[a:b], s[b:])]
    lags = np.clip(np.round(LOG_LAGS_NS / dt).astype(int), 1, max_lag)
    curves = {"acf": c[np.unique(lags)], "acf_lag_ns": np.unique(lags) * dt, "hist": dens,
              "hist_centres": centres}
    return rec, s, curves


# Seed means and errors of the scalar fields of a list of seed records.
def aggregate(recs):
    out = {}
    for key in ("p_up", "I", "minority_frac", "mean_abs_m", "barrier_kT", "well_minus_kT",
                "well_plus_kT", "saddle_m", "frac_abs_below_0p3", "tau_efold",
                "tau_tail", "tail_amp", "tau_int", "acf_0p1ns", "neighbour_corr", "tau_dwell"):
        out[key] = mean_se([r[key] for r in recs])
    for kind in ("dwell_hyst", "dwell_sign"):
        for st in ("up", "down"):
            for key in ("rate", "cv", "D", "short_excess", "D_null_mean", "D_null_p95",
                        "D_p_value", "cv_null_mean", "short_excess_null_mean",
                        "short_excess_null_sd"):
                vals = [r[kind][st].get(key, np.nan) for r in recs]
                out[f"{kind}_{st}_{key}"] = mean_se(vals)
            for key in ("n_flips", "n_dwells"):
                out[f"{kind}_{st}_{key}"] = int(sum(r[kind][st][key] for r in recs))
            # Null of the seed-mean D: average the telegraph replicas across seeds.
            nulls = [r[kind][st].pop("D_null") for r in recs if "D_null" in r[kind][st]]
            if len(nulls) == len(recs):
                p95, pval = null_stats(np.mean(nulls, axis=0), out[f"{kind}_{st}_D"][0])
                out[f"{kind}_{st}_seedmean_D_null_p95"] = p95
                out[f"{kind}_{st}_seedmean_D_p_value"] = pval
    out["p_up_blocks"] = [mean_se([r["p_up_blocks"][j] for r in recs]) for j in range(3)]
    out["I_all_seeds_finite"] = bool(all(np.isfinite(r["I"]) for r in recs))
    p = np.mean([r["p_up"] for r in recs])
    with np.errstate(divide="ignore"):
        out["I_pooled"] = float(np.arctanh(2 * p - 1))
    out["flips"] = int(sum(r["flips"] for r in recs))
    out["disk_us"] = float(sum(r["disk_ns"] for r in recs) / 1e3)
    out["seeds"] = len(recs)
    return out


# Stale-read probabilities on the r0 grid, measured per seed and predicted by the
# telegraph propagator for each correlation time in `taus`.
def stale_block(series, dt, p, tau0, taus):
    """Stale-read probabilities at the r0 grid, measured and from the telegraph propagator.

    The lags are r0 x tau0 with tau0 the zero-bias 1/e time. taus maps a label
    to the [value, error] of a correlation time; each gives one telegraph
    prediction, with the error carried through.
    """
    rows = []
    own = taus["telegraph_efold"][0]      # this point's own 1/e time
    for r0 in R0_GRID:
        lag = max(1, round(r0 * tau0 / dt))
        if lag >= min(len(s) for s in series) // 2:
            continue
        per_seed = np.array([ms.stale_read(s, [lag])[0] for s in series])
        meas = [mean_se(per_seed[:, j]) for j in range(3)]
        acf = mean_se([ms.autocorr(s, lag)[-1] for s in series])
        row = {"r0": r0, "lag_ns": lag * dt,
               "measured": {"all": meas[0], "up": meas[1], "down": meas[2]}, "acf": acf}
        for label, (tau, tau_se) in taus.items():
            pred = ms.telegraph_stale(p, lag * dt / tau)
            # At a lag of one 1/e time the measured C is 1/e by definition, so a prediction
            # built on that time agrees there whatever the dynamics.
            row[label] = {"tau_ns": tau, "all": float(pred[0]), "up": float(pred[1]),
                          "down": float(pred[2]), "acf": float(np.exp(-lag * dt / tau)),
                          "by_construction": bool(abs(tau / own - 1) < 0.01
                                                  and abs(lag * dt / tau - 1) < 0.05)}
            if np.isfinite(tau_se):
                hi = ms.telegraph_stale(p, lag * dt / (tau + tau_se))[0]
                lo = ms.telegraph_stale(p, lag * dt / (tau - tau_se))[0]
                row[label]["all_se"] = float(abs(hi - lo) / 2)
        rows.append(row)
    return rows


# Input I to use at field b (mT): measured, mirrored from -b, or from the field fit.
def model_I(points, fit, b):
    """Input I of the telegraph model at field b (mT), with a note of where it comes from.

    The measured logit where every seed visits the minority state (mirroring the
    opposite field if only that one was run); otherwise the field-fit line.
    """
    for key, sign in ((b, 1), (-b, -1)):
        if key in points and points[key]["I_all_seeds_finite"]:
            return sign * points[key]["I"][0], "measured" if sign == 1 else "mirrored"
    return fit["slope_per_mT"][0] * b + fit["intercept"][0], "field_fit"


# Reduce one run; the dwell-test null is drawn from the run's own seed, so results are
# reproducible.
def seed_job(stem):
    meta, dt, _, m = load(stem)
    rec, s, c = seed_record(meta, dt, m, np.random.default_rng(meta["seed"]))
    return meta, dt, rec, s, c


# Group runs by bias and reduce each group; returns (points, curves, seed series).
def two_state_set(stems):
    groups = {}
    for stem in stems:
        meta = meta_of(stem)
        key = meta.get("stt_j") or meta["bz_mT"]
        groups.setdefault(key, []).append(stem)
    with ProcessPoolExecutor(WORKERS) as pool:
        done = dict(zip(stems, pool.map(seed_job, stems)))
    points, curves, series = {}, {}, {}
    for key, members in sorted(groups.items()):
        recs, ss, cs = [], [], []
        for stem in sorted(members):
            meta, dt, rec, s, c = done[stem]
            recs.append(rec)
            ss.append(s)
            cs.append(c)
        points[key] = dict(aggregate(recs), dt_table_ns=dt, sim_dt_fs=meta["dt"] * 1e15,
                           seed_records=recs)
        # Runs of different length reach different maximum lags; keep the common part.
        curves[key] = {k: np.mean([c[k][:min(len(x[k]) for x in cs)] for c in cs], axis=0)
                       for k in cs[0]}
        series[key] = (dt, ss)
    return points, curves, series


# Finished jobs whose name starts with prefix, as paths without extension.
def done_stems(rundirs, prefix):
    return sorted(p.parent / p.name[:-5] for d in rundirs for p in d.glob(f"{prefix}*.done"))


# Finished jobs of the given sets, grouped by time step in fs.
def by_time_step(rundirs, sets):
    out = {}
    for stem in done_stems(rundirs, ""):
        meta = meta_of(stem)
        if meta["set"] in sets:
            out.setdefault(round(meta["dt"] * 1e15, 2), []).append(stem)
    return out


# Points without their per-seed records.
def strip(points):
    return {k: {f: v for f, v in pt.items() if f != "seed_records"} for k, pt in points.items()}


# At biases holding both, the production rerun (eq_ runs) against the earlier 25 fs
# check (dt25_ runs).
def rerun_vs_check(points):
    out = {}
    for k, pt in points.items():
        recs = {pre: [r for r in pt["seed_records"] if r["name"].startswith(pre)]
                for pre in ("eq_", "dt25_")}
        if all(len(v) > 1 for v in recs.values()):
            row = {pre.rstrip("_"): {f: mean_se([r[f] for r in v]) for f in ("I", "tau_efold", "tau_dwell")}
                   | {"seeds": len(v)} for pre, v in recs.items()}
            row["eq_minus_dt25"] = {f: difference(row["eq"][f], row["dt25"][f])
                                    for f in ("I", "tau_efold", "tau_dwell")}
            out[k] = row
    return out


# Least-squares fit of i to the given powers of b: coefficients and their covariance.
def power_fit(b, i, powers):
    a = np.stack([b ** p for p in powers], axis=1)
    coef = np.linalg.lstsq(a, i, rcond=None)[0]
    resid = i - a @ coef
    cov = np.linalg.inv(a.T @ a) * (resid @ resid) / (len(b) - len(powers))
    return {"powers": list(powers), "coef": coef.tolist(), "cov": cov.tolist()}


# [I, error] at field b (mT) under each map of field_fit; largest_measured bounds abs(I)
# from below.
def extrapolated_I(maps, b):
    out = {}
    for label, m in maps.items():
        if label == "largest_measured":
            out[label] = [float(np.sign(b) * m[0]), m[1]]
        else:
            x = np.array([float(b) ** q for q in m["powers"]])
            out[label] = [float(x @ m["coef"]), float(np.sqrt(x @ np.array(m["cov"]) @ x))]
    return out


# Field-to-I map: the linear fit used for every model comparison, its curvature
# test, and I at the fields where it was not measured.
def field_fit(points):
    """Fit I against B_z over abs(I) <= 2 (ordinary least squares, errors from the residuals).

    With two to four seeds per point the seed errors are themselves too noisy to
    use as weights; they are kept in I_all. The cubic coefficient tests curvature,
    and the slope over all finite points shows any bend beyond I = 2.
    """
    b = np.array(list(points), dtype=np.float64)
    i = np.array([points[k]["I"][0] for k in points])
    e = np.array([points[k]["I"][1] for k in points])
    # A point counts as measured only if every seed visited the minority state.
    finite = np.isfinite(i) & np.array([points[k].get("I_all_seeds_finite", True) for k in points])
    sel = finite & (np.abs(i) <= 2.1)
    coef, cov = np.polyfit(b[sel], i[sel], 1, cov=True)
    resid = i[sel] - np.polyval(coef, b[sel])
    c3, cov3 = np.polyfit(b[sel], i[sel], 3, cov=True)
    slope_all = np.polyfit(b[finite], i[finite], 1)[0]
    # Fields with no measured logit: I under three maps, and the largest measured abs(I)
    # as a lower bound (I rises with field). The cubics are odd, a B + c B^3, because
    # I(-B) = -I(B). The spread of the three is the uncertainty of I there.
    top = int(np.argmax(np.where(finite, np.abs(i), -1)))
    maps = {"linear": power_fit(b[sel], i[sel], (1, 0)),
            "odd_cubic_absI_le_2p1": power_fit(b[sel], i[sel], (1, 3)),
            "odd_cubic_all_measured": power_fit(b[finite], i[finite], (1, 3)),
            "largest_measured": [float(abs(i[top])), float(e[top])]}
    extrap = {float(x): extrapolated_I(maps, x) for x in b[~finite]}
    # I(+B) + I(-B) should vanish; the gap is given in units of its seed error.
    sym = {int(k): difference(points[k]["I"], [-points[-k]["I"][0], points[-k]["I"][1]])
           for k in points if k > 0 and -k in points}
    return {"points_used": int(sel.sum()), "symmetry": sym,
            "slope_per_mT": [float(coef[0]), float(np.sqrt(cov[0, 0]))],
            "intercept": [float(coef[1]), float(np.sqrt(cov[1, 1]))],
            "residual_rms": float(np.sqrt(np.mean(resid ** 2))),
            "residual_max": float(np.abs(resid).max()),
            "cubic_coef_per_mT3": [float(c3[0]), float(np.sqrt(cov3[0, 0]))],
            "slope_all_finite_points": float(slope_all),
            "extrapolation_maps": maps, "extrapolated_I": extrap,
            "I_over_B_per_point": {float(x): float(y / x) for x, y in zip(b[finite], i[finite]) if x},
            "I_all": [[float(x), float(y), float(z)] for x, y, z in zip(b, i, e)]}


# tau(I) cosh(I) / tau(0) with its error; all three arguments are [value, error] pairs.
def cosh_ratio(tau, tau0, i):
    r = ratio(tau, tau0)
    # Error of cosh I from the error of I: d(cosh I) / cosh I = tanh(I) dI.
    i_se = i[1] if np.isfinite(i[1]) else 0.0
    return [r[0] * np.cosh(i[0]), np.cosh(i[0]) * np.hypot(r[1], r[0] * np.tanh(i[0]) * i_se)]


# One row per bias point comparing the measured correlation times with the model laws.
def tau_vs_I(points, ref, delta, fit=None):
    """Measured tau_c(I) against the Arrhenius (cosh), bias-independent and macrospin predictions.

    ref is the zero-bias point. The pred_* columns scale its 1/e time; each
    *_over_cosh ratio is tau(I) cosh(I) / tau(0) with both times from the same
    estimator, so 1 means the cosh law holds for that estimator.
    few_flips marks points where the read is nearly single-state and its
    1/e time is set by brief excursions across m_z = 0, not by switching.
    With a field fit, I comes from model_I; without one (current bias) it is
    the measured logit. The ratio errors carry the error of I where I is
    measured or mirrored; a field-fit I is an extrapolation with no usable
    error, and none is added.
    """
    tau0 = ref["tau_efold"][0]
    rows = []
    for k, pt in points.items():
        i, source = model_I(points, fit, k) if fit else (pt["I"][0], "measured")
        row = {"bias": k, "I": pt["I"], "I_model": i, "I_model_source": source,
               "flips": pt["flips"], "few_flips": pt["flips"] < 30, "acf_0p1ns": pt["acf_0p1ns"],
               "tau_tail": pt["tau_tail"], "tau_efold": pt["tau_efold"],
               "tau_dwell": pt["tau_dwell"], "tail_amp": pt["tail_amp"],
               "pred_cosh": tau0 / np.cosh(i), "pred_constant": tau0,
               "pred_brown": tau0 / float(ms.brown_relax_ratio(i, delta)),
               "well_minus_kT": pt["well_minus_kT"], "well_plus_kT": pt["well_plus_kT"],
               # Uniaxial macrospin: wells delta (1 -+ h)^2 with h = I / (2 delta).
               "pred_well_minus_kT": delta * (1 - i / (2 * delta)) ** 2 if abs(i) < 2 * delta else 0.0,
               "pred_well_plus_kT": delta * (1 + i / (2 * delta)) ** 2 if abs(i) < 2 * delta else 0.0}
        for src in ("tau_tail", "tau_dwell", "tau_efold"):
            if np.isfinite(pt[src][0]):
                rel = pt[src][0] / ref[src][0]
                row[f"{src}_over_cosh"] = rel * np.cosh(i)
                i_err = {"measured": pt["I"][1], "mirrored": points.get(-k, pt)["I"][1]}.get(source, np.nan)
                row[f"{src}_over_cosh_se"] = cosh_ratio(pt[src], ref[src], [i, i_err])[1]
                row[f"{src}_over_brown"] = rel * float(ms.brown_relax_ratio(i, delta))
        rows.append(row)
    return rows


# Bootstrap standard error (nan-safe spread of the resampled values).
def boot_se(values):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(v.std(ddof=1)) if len(v) > 1 else np.nan


# Step-response record of one set of disk-steps, with bootstrap errors.
def step_record(units, dt, s_eq, tau0, tau_pred, rng):
    """Relaxation of the ensemble read mean after a field step, for three asymptotes.

    units is (disk-steps, time) of +-1 reads, time 0 at the step. s_eq is the
    [value, error] of the stationary read mean from the equilibrium runs. The
    asymptote is in turn fitted (three-parameter exponential), taken from the
    equilibrium runs, and taken from the last fifth of the segment. Errors
    come from resampling the disk-steps; the equilibrium asymptote is redrawn
    from its own error in each resample. Where no seed visits the minority
    state that error is zero or undefined, so the asymptote is held fixed.
    """
    n, length = units.shape
    tail = slice(length - length // 5, length)

    # all numbers of one read-mean curve, for the measured curve and for each resample
    def reduce(curve, s_inf_eq):
        s0, s_tail = curve[0], curve[tail].mean()
        tau_free, s_free = ms.step_fit_free(curve, dt)
        out = {"s0": s0, "s_inf_tail": s_tail, "s_inf_free": s_free, "tau_free": tau_free,
               "tau_eq": ms.step_response(curve, dt, s_inf_eq, s0),
               "tau_tail": ms.step_response(curve, dt, s_tail, s0)}
        # How much longer the fitted-asymptote time is than each fixed-asymptote time.
        out["free_over_eq"] = tau_free / out["tau_eq"]
        out["free_over_tail"] = tau_free / out["tau_tail"]
        for x in MEMORY_GRID:
            j = round(x * tau0 / dt)
            if j < length:
                out[f"mem_eq_{x}"] = (curve[j] - s_inf_eq) / (s0 - s_inf_eq)
                # Inside the tail window the tail asymptote would be compared with itself.
                if j < tail.start:
                    out[f"mem_tail_{x}"] = (curve[j] - s_tail) / (s0 - s_tail)
        return out

    x = units.astype(np.float32)
    best = reduce(x.mean(axis=0).astype(np.float64), s_eq[0])
    weights = rng.multinomial(n, np.full(n, 1 / n), size=N_BOOT).astype(np.float32) / n
    curves = (weights @ x).astype(np.float64)
    eq_draws = s_eq[0] + (s_eq[1] if np.isfinite(s_eq[1]) else 0.0) * rng.standard_normal(N_BOOT)
    boots = [reduce(c, e) for c, e in zip(curves, eq_draws)]

    # [value on the full sample, bootstrap SE]
    def pair(key):
        return [float(best[key]), boot_se([b.get(key, np.nan) for b in boots])]

    rec = {"steps": n, "s0": pair("s0"), "s_inf_eq": s_eq, "s_inf_tail": pair("s_inf_tail"),
           "s_inf_free": pair("s_inf_free"),
           "tau_step": {"free": pair("tau_free"), "eq_asymptote": pair("tau_eq"),
                        "tail_asymptote": pair("tau_tail")},
           "tau_step_free_over_fixed": {"eq_asymptote": pair("free_over_eq"),
                                        "tail_asymptote": pair("free_over_tail")},
           # How far the tail window is into the predicted relaxation (in units of tau_pred).
           "tail_start_over_pred_tau": float(tail.start * dt / tau_pred), "memory": []}
    for x_ in MEMORY_GRID:
        if f"mem_eq_{x_}" in best:
            t = round(x_ * tau0 / dt) * dt
            row = {"t_over_tau0": x_, "t_ns": t, "eq_asymptote": pair(f"mem_eq_{x_}"),
                   "telegraph": float(np.exp(-t / tau_pred))}
            if f"mem_tail_{x_}" in best:
                row["tail_asymptote"] = pair(f"mem_tail_{x_}")
            rec["memory"].append(row)
    curve = x.mean(axis=0)
    se = x.std(axis=0, ddof=1) / np.sqrt(n)
    return rec, np.stack([np.arange(length) * dt, curve, se])


# Slowdown of the step against the cosh law, for each estimate of I.
def step_over_cosh(rec, tau0, i_options):
    """Step time over the cosh-law time, tau_step cosh(I) / tau_c(0), for each I in i_options.

    tau0 is the [value, error] of tau_c(0); i_options maps a label to an
    [I, error] pair, and the error of I is carried into the ratio. Given for
    the equilibrium and the fitted asymptote.
    """
    return {kind: {label: cosh_ratio(rec["tau_step"][kind], tau0, i) for label, i in i_options.items()}
            for kind in ("eq_asymptote", "free")}


# Step responses grouped by (low field, high field, half period).
def step_set(stems, eq_points, fit, delta, rng):
    tau0_pair = eq_points[0]["tau_efold"]
    tau0 = tau0_pair[0]
    groups = {}
    for stem in stems:
        meta = meta_of(stem)
        groups.setdefault((meta["low_mT"], meta["high_mT"], meta["half_ns"]), []).append(stem)
    out, curves = [], {}
    for (lo, hi, half_ns), members in sorted(groups.items()):
        segs = {"up": [], "down": []}
        for stem in members:
            meta, dt, _, m = load(stem)
            s = ms.sign_reads(m)
            half = round(half_ns / dt)
            for j in range(2 * meta["periods"]):
                seg = s[j * half:(j + 1) * half + 1]
                if len(seg) == half + 1:
                    segs["up" if j % 2 == 0 else "down"].append(seg.T)
        for direction, target in (("up", hi), ("down", lo)):
            i_target, source = model_I(eq_points, fit, target)
            # Stationary read mean at the target field: measured (or mirrored) where an
            # equilibrium run exists, else tanh of the model input.
            if target in eq_points:
                p_up = eq_points[target]["p_up"]
                s_eq = [2 * p_up[0] - 1, 2 * p_up[1]]
            elif -target in eq_points:
                p_up = eq_points[-target]["p_up"]
                s_eq = [1 - 2 * p_up[0], 2 * p_up[1]]
            else:
                s_eq = [float(np.tanh(i_target)), np.nan]
            tau_pred = tau0 / np.cosh(i_target)
            rec, curve = step_record(np.concatenate(segs[direction]), dt, s_eq, tau0, tau_pred, rng)
            if source == "field_fit":
                i_options = extrapolated_I(fit["extrapolation_maps"], target)
            else:
                i_se = eq_points[target if source == "measured" else -target]["I"][1]
                i_options = {source: [i_target, i_se]}
            rec = {"low_mT": lo, "high_mT": hi, "half_ns": half_ns, "direction": direction,
                   "target_mT": target, "I_target": i_target, "I_target_source": source,
                   "pred_tau_cosh": tau_pred,
                   "pred_tau_brown": tau0 / float(ms.brown_relax_ratio(i_target, delta)),
                   "I_target_options": i_options,
                   "tau_step_over_cosh": step_over_cosh(rec, tau0_pair, i_options)} | rec
            curves[f"{lo}_{hi}_h{half_ns}_{direction}"] = curve
            out.append(rec)
    return out, curves


# [min, max] of the finite values.
def spread(values):
    v = [x for x in values if x is not None and np.isfinite(x)]
    return [float(min(v)), float(max(v))] if v else [np.nan, np.nan]


# Ranges over bias points of the quantities that are quoted as ranges.
def ranges(eq, tau_rows, ep, steps=()):
    """Extremes over bias points of the quantities quoted as ranges, so the ranges are on record.

    The cosh-law ranges leave out zero bias, where the ratio is 1 by definition.
    """
    weak = [pt for pt in eq.values() if abs(pt["I"][0]) <= 1.1]
    mid = [pt for pt in eq.values() if abs(pt["I"][0]) <= 1.6]
    # Minority well against the uniaxial macrospin form, up to abs(I) = 2.1.
    well = []
    for row in tau_rows:
        i = row["I"][0]
        if 0.1 < abs(i) <= 2.1:
            side = "minus" if i > 0 else "plus"
            well.append(row[f"well_{side}_kT"][0] / row[f"pred_well_{side}_kT"] - 1)
    return {"hyst_short_excess_absI_le_1p1": spread(
                [pt[f"dwell_hyst_{st}_short_excess"][0] for pt in weak for st in ("up", "down")]),
            "hyst_cv_absI_le_1p6": spread(
                [pt[f"dwell_hyst_{st}_cv"][0] for pt in mid for st in ("up", "down")]),
            "minority_well_rel_dev_absI_le_2p1": spread(well),
            "tau_efold_over_cosh_absI_le_2p1": spread(
                [r.get("tau_efold_over_cosh") for r in tau_rows if 0.1 < abs(r["I"][0]) <= 2.1]),
            "tau_dwell_over_cosh_absI_le_2p1": spread(
                [r.get("tau_dwell_over_cosh") for r in tau_rows if 0.1 < abs(r["I"][0]) <= 2.1]),
            "neighbour_corr": spread([pt["neighbour_corr"][0] for pt in eq.values()]),
            "rotor_rel_dev": spread([v["rotor_rel_dev"] for v in ep.values()]),
            # Strongly driven steps: fitted-asymptote time over each fixed-asymptote time.
            "step_tau_free_over_fixed_absB_ge_72": spread(
                [v[0] for st in steps if abs(st["target_mT"]) >= 72
                 for v in st["tau_step_free_over_fixed"].values()])}


# Time-step comparison at every field, each step against its own tau_c(0).
def time_step_table(sets):
    """Per bias and time step (fs): the dt-sensitive numbers, the cosh-law ratio and step-to-step gaps.

    The cosh-law ratio tau_c(I) cosh(I) / tau_c(0) uses tau_c(0) and I measured at the same time step.
    """
    fields = ("I", "tau_efold", "tau_dwell", "tau_tail", "tail_amp", "acf_0p1ns", "barrier_kT")
    out = {}
    for bias in sorted({k for pts in sets.values() for k in pts}):
        row = {}
        for dt, pts in sets.items():
            if bias in pts and 0 in pts:
                pt, (i, i_se) = pts[bias], pts[bias]["I"]
                row[dt] = {f: pt[f] for f in fields} | {"seeds": pt["seeds"], "flips": pt["flips"]}
                for f in ("tau_efold", "tau_dwell"):
                    row[dt][f"{f}_over_cosh"] = cosh_ratio(pt[f], pts[0][f], [i, i_se])
        for fine, coarse in (("25", "50"), ("12.5", "25"), ("12.5", "50")):
            if fine in row and coarse in row:
                row[f"{fine}_minus_{coarse}"] = {f: difference(row[fine][f], row[coarse][f])
                                                 for f in fields}
        out[bias] = row
    return out


# Same comparison with +B and -B pooled.
def time_step_pooled(sets):
    """The time-step table with the seeds of +B and -B pooled (the two fields are mirror images).

    sets maps a time step to points that still hold their seed records.
    """
    fields = ("tau_efold", "tau_dwell")
    out = {}
    for b in sorted({abs(k) for pts in sets.values() for k in pts} - {0}):
        row = {}
        for dt, pts in sets.items():
            recs = [(sign, r) for sign in (1, -1) for r in pts.get(sign * b, {}).get("seed_records", [])]
            if not recs or 0 not in pts:
                continue
            i = mean_se([sign * r["I"] for sign, r in recs])
            row[dt] = {"I": i, "seeds": len(recs), "seeds_minus_B": sum(s < 0 for s, _ in recs)}
            for f in fields:
                row[dt][f] = mean_se([r[f] for _, r in recs])
                row[dt][f"{f}_over_cosh"] = cosh_ratio(row[dt][f], pts[0][f], i)
        for fine, coarse in (("25", "50"), ("12.5", "25"), ("12.5", "50")):
            if fine in row and coarse in row:
                row[f"{fine}_minus_{coarse}"] = {
                    f: difference(row[fine][f], row[coarse][f])
                    for f in fields + tuple(f"{x}_over_cosh" for x in fields)}
        out[b] = row
    return out


# a / b for two [value, error] pairs, errors combined in quadrature.
def ratio(a, b):
    r = a[0] / b[0]
    return [float(r), float(r * np.hypot(a[1] / a[0], b[1] / b[0]))]


# a - b for two [value, error] pairs, with its error and the gap in units of that error.
def difference(a, b):
    d, e = a[0] - b[0], float(np.hypot(a[1], b[1]))
    return {"diff": float(d), "se": e, "sigma": float(d / e) if e > 0 else np.nan}


# Device E: angle distribution, m_z spread, and read decorrelation against in-plane
# field.
def easy_plane_set(stems, rng):
    groups = {}
    for stem in stems:
        groups.setdefault(meta_of(stem)["bx_mT"], []).append(stem)
    out = {}
    for b, members in sorted(groups.items()):
        recs = []
        for stem in sorted(members):
            meta, dt, _, m = load(stem)
            m = m.reshape(len(m), -1, 3)
            mx, my, mz = m[..., 0], m[..., 1], m[..., 2]
            phi = np.arctan2(my, mx)
            hist, _ = np.histogram(phi.ravel(), bins=16, range=(-np.pi, np.pi), density=True)
            flat = 1 / (2 * np.pi)
            s = ms.sign_reads(mx)
            max_lag = min(round(2.0 / dt), len(s) // 2)
            c = ms.autocorr(s, max_lag)
            cx = ms.autocorr(mx, max_lag)
            tau_tail, amp = ms.fit_exp_tail(c, dt)
            dw = dwell_block(s, dt, rng, null=False)
            p = float((s == 1).mean())
            switches = (np.diff(s, axis=0) != 0).sum()
            recs.append({"name": meta["name"], "angle_hist": list(map(float, hist / flat)),
                         "angle_hist_maxdev": float(np.abs(hist / flat - 1).max()),
                         "neglogP_range_kT": float(np.log(hist.max() / hist.min())),
                         "mz_rms": float(np.sqrt((mz ** 2).mean())),
                         "inplane_mod": float(np.sqrt(mx ** 2 + my ** 2).mean()),
                         "mean_mx": float(mx.mean()), "p_up": p, "I": float(np.arctanh(2 * p - 1)),
                         "tau_efold_sign": ms.efold_time(c, dt), "tau_efold_mx": ms.efold_time(cx, dt),
                         "tau_tail_sign": tau_tail, "tail_amp_sign": amp,
                         "dwell_cv_up": dw["up"]["cv"], "dwell_D_up": dw["up"]["D"],
                         "dwell_short_excess_up": dw["up"]["short_excess"],
                         "switches_per_ns": float(switches / (len(s) * dt * s.shape[1]))})
        agg = {k: mean_se([r[k] for r in recs]) for k in recs[0]
               if k not in ("name", "angle_hist")}
        agg["angle_hist"] = np.mean([r["angle_hist"] for r in recs], axis=0).tolist()
        # 2D rotor (uniform in-plane angle, energy -mu B cos phi): <cos phi> = I1(x) / I0(x).
        mu = 1.0e6 * 80 * CELL_VOL * agg["inplane_mod"][0]
        x = mu * b * 1e-3 / KT
        agg["rotor_mean_mx"] = float(i1(x) / i0(x) * agg["inplane_mod"][0]) if b else 0.0
        agg["rotor_rel_dev"] = agg["mean_mx"][0] / agg["rotor_mean_mx"] - 1 if b else np.nan
        agg["seed_records"] = recs
        out[b] = agg
    return out


# Measured read statistics on the r0 grid with the telegraph predictions named in
# labels.
def stale_summary(rows, labels):
    return [{"r0": r["r0"], "lag_ns": r["lag_ns"], "measured": r["measured"]["all"],
             "changed": [1 - r["measured"]["all"][0], r["measured"]["all"][1]],
             "same_up": r["measured"]["up"], "same_down": r["measured"]["down"],
             "acf": r["acf"]} | {lb: r[lb] for lb in labels if lb in r} for r in rows]


# Headline numbers, each a [value, standard error] pair where an error exists.
def summary(eq, fit, tau_rows, stale, tau0, checks, steps, stt, ep, steps50=()):
    keep = ("I", "I_pooled", "minority_frac", "flips", "seeds", "well_minus_kT", "well_plus_kT",
            "tau_efold", "tau_dwell", "tau_tail", "tail_amp", "acf_0p1ns", "p_up_blocks",
            "dwell_sign_up_cv", "dwell_sign_up_short_excess", "neighbour_corr")
    keep += tuple(f"dwell_hyst_{st}_{f}" for st in ("up", "down")
                  for f in ("rate", "n_flips", "n_dwells", "cv", "cv_null_mean", "short_excess",
                            "short_excess_null_mean", "short_excess_null_sd", "D",
                            "seedmean_D_null_p95", "seedmean_D_p_value"))
    by_bias = {r["bias"]: r for r in tau_rows}
    preds = ("I_model", "I_model_source", "few_flips", "pred_cosh", "pred_brown",
             "pred_well_minus_kT", "pred_well_plus_kT")
    labels = ("telegraph_dwell", "telegraph_efold", "telegraph_tail", "telegraph_cosh")
    whole = ("time_step", "time_step_pooled", "alpha03_over_alpha0p1", "field_slope_by_dt", "rerun_vs_earlier_25fs",
             "pitch128_over_pitch64_50fs")
    return {"dt_fs": PRODUCTION_FS, "tau_c0_ns": eq[0]["tau_efold"], "tau_c0_dwell_ns": eq[0]["tau_dwell"],
            "tau_c0_tail_fit_ns": eq[0]["tau_tail"],
            "barrier_kT": eq[0]["barrier_kT"], "peaks_mz": eq[0]["seed_records"][0]["peaks"],
            "frac_abs_mz_below_0p3": eq[0]["frac_abs_below_0p3"],
            "field_slope_I_per_mT": fit["slope_per_mT"], "field_fit_residual_rms": fit["residual_rms"],
            "field_fit_cubic_per_mT3": fit["cubic_coef_per_mT3"],
            "field_symmetry": fit["symmetry"],
            "ranges": ranges(eq, tau_rows, ep, steps),
            "points": {k: {f: pt.get(f) for f in keep} | {f: by_bias[k][f] for f in preds}
                       for k, pt in eq.items()},
            # telegraph_efold is tied to the measurement at r0 = 1; telegraph_dwell is not.
            "stale_zero_bias": stale_summary(stale.get(0, []), labels),
            "stale_18mT": stale_summary(stale.get(18, []), labels),
            "steps": steps,
            "steps_dt50": list(steps50),
            "checks": {name: pts if name in whole else
                       {k: {f: v[f] for f in ("tau_efold", "tau_dwell", "barrier_kT", "acf_0p1ns",
                                              "I", "flips", "seeds")} for k, v in pts.items()}
                       for name, pts in checks.items()},
            "spin_torque": {name: [{f: r.get(f) for f in ("bias", "I", "tau_efold", "tau_dwell",
                                                           "pred_cosh", "pred_brown", "flips",
                                                           "tau_efold_over_cosh",
                                                           "tau_efold_over_cosh_se",
                                                           "tau_dwell_over_cosh",
                                                           "tau_dwell_over_cosh_se")}
                                   | {"dt_fs": v["dt_fs"], "seeds": v["points"][r["bias"]]["seeds"]}
                                   for r in v["tau_vs_I"]] for name, v in stt.items()},
            "easy_plane_dt_fs": PRODUCTION_FS if ep else None,
            "easy_plane": {b: {f: v[f] for f in ("I", "tau_efold_sign", "tau_efold_mx", "mean_mx",
                                                 "rotor_mean_mx", "rotor_rel_dev", "angle_hist_maxdev",
                                                 "neglogP_range_kT", "mz_rms", "inplane_mod",
                                                 "dwell_cv_up", "switches_per_ns")}
                           for b, v in ep.items()}}


# Convert numpy scalars and arrays, and non-finite floats, for json.dump.
def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


# Write obj as indented json.
def save_json(path, obj):
    path.write_text(json.dumps(jsonable(obj), indent=1))


# Reduce every run directory into the records in OUTDIR (see the module docstring).
def main():
    rundirs, outdir = [Path(a) for a in sys.argv[1:-1]], Path(sys.argv[-1])
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    # Equilibrium runs by time step; the production step gives every record under bias.
    sets = {fs: two_state_set(stems) for fs, stems in sorted(by_time_step(rundirs, EQ_SETS).items())}
    # The 12.5 fs check has three fields only, too few for the fit.
    fits = {fs: field_fit(pts) for fs, (pts, _, _) in sets.items() if len(pts) >= 6}
    eq, eq_curves, eq_series = sets[PRODUCTION_FS]
    tau0 = eq[0]["tau_efold"][0]
    delta = eq[0]["barrier_kT"][0]
    fit = fits[PRODUCTION_FS]
    tau_rows = tau_vs_I(eq, eq[0], delta, fit)
    stale = {}
    for k, (dt, ss) in eq_series.items():
        ch = np.cosh(model_I(eq, fit, k)[0])
        stale[k] = stale_block(ss, dt, eq[k]["p_up"][0], tau0, {
            "telegraph_efold": eq[k]["tau_efold"], "telegraph_dwell": eq[k]["tau_dwell"],
            "telegraph_tail": eq[k]["tau_tail"],
            "telegraph_cosh": [x / ch for x in eq[0]["tau_efold"]],
            "telegraph_cosh_dwell": [x / ch for x in eq[0]["tau_dwell"]]})
    save_json(outdir / "device_p_equilibrium.json", {
        "device": "perpendicular disk 20 nm x 1 nm, Ms 1.0 MA/m, A 15 pJ/m, Ku 6.0e5 J/m^3, "
                  f"alpha 0.1, 300 K, Heun dt {PRODUCTION_FS:g} fs, 8x8 array at 64 nm pitch",
        "dt_fs": PRODUCTION_FS,
        "tau_c0_ns": eq[0]["tau_efold"], "tau_c0_tail_fit_ns": eq[0]["tau_tail"],
        "tau_c0_dwell_ns": eq[0]["tau_dwell"], "barrier_kT": eq[0]["barrier_kT"],
        "field_to_I": fit, "tau_vs_I": tau_rows, "stale_reads": stale,
        "points": eq})
    np.savez_compressed(outdir / "device_p_curves.npz", **{
        f"b{k:+d}_{name}": v for k, c in eq_curves.items() for name, v in c.items()})

    # Time-step check: every bias point at 50, 25 and 12.5 fs, each against its own tau_c(0).
    ref50, delta50 = sets[50.0][0][0], sets[50.0][0][0]["barrier_kT"][0]
    checks = {f"dt{fs:g}": strip(pts) for fs, (pts, _, _) in sets.items() if fs != PRODUCTION_FS}
    checks["time_step"] = time_step_table({f"{fs:g}": strip(pts) for fs, (pts, _, _) in sets.items()})
    checks["time_step_pooled"] = time_step_pooled({f"{fs:g}": pts for fs, (pts, _, _) in sets.items()})
    checks["field_slope_by_dt"] = {f"{fs:g}": f["slope_per_mT"] for fs, f in fits.items()}
    checks["rerun_vs_earlier_25fs"] = rerun_vs_check(eq)
    for prefix in ("alpha03_", "pitch128_"):
        stems = done_stems(rundirs, prefix)
        if stems:
            checks[prefix.rstrip("_")] = strip(two_state_set(stems)[0])
    if "alpha03" in checks:
        # Both at dt = 25 fs; tau_c proportional to 1/alpha would give 0.1 / 0.03.
        checks["alpha03_over_alpha0p1"] = {
            "seeds": [checks["alpha03"][0]["seeds"], eq[0]["seeds"]],
            "tau_efold_ratio": ratio(checks["alpha03"][0]["tau_efold"], eq[0]["tau_efold"]),
            "inverse_alpha_ratio": 0.1 / 0.03}
    if "pitch128" in checks:
        # Both at 50 fs; crosstalk would make the two pitches differ.
        checks["pitch128_over_pitch64_50fs"] = {
            f: ratio(checks["pitch128"][0][f], ref50[f]) for f in ("tau_efold", "tau_dwell")}
    save_json(outdir / "checks.json", checks)

    steps, steps50 = [], []
    by_dt = by_time_step(rundirs, ("step",))
    if PRODUCTION_FS in by_dt:
        steps, step_curves = step_set(by_dt[PRODUCTION_FS], eq, fit, delta, rng)
        rec = {"dt_fs": PRODUCTION_FS, "tau_c0_ns": tau0, "bootstrap_resamples": N_BOOT,
               "steps": steps}
        if 50.0 in by_dt:
            # The same steps at 50 fs, each against the 50 fs equilibrium runs.
            steps50, curves50 = step_set(by_dt[50.0], sets[50.0][0], fits[50.0], delta50, rng)
            rec["steps_dt50"] = steps50
            step_curves |= {f"dt50_{k}": v for k, v in curves50.items()}
        save_json(outdir / "step_response.json", rec)
        np.savez_compressed(outdir / "step_curves.npz", **step_curves)

    # Controls by time step: the production step gives the record, 50 fs sits beside it.
    ep = {}
    for fs, stems in sorted(by_time_step(rundirs, ("easyplane", "easyplane25")).items()):
        rec = easy_plane_set(stems, rng)
        if fs == PRODUCTION_FS:
            ep = rec
        save_json(outdir / ("easy_plane.json" if fs == PRODUCTION_FS else f"easy_plane_dt{fs:g}.json"),
                  rec)

    stt = {}
    for fs, stems in sorted(by_time_step(rundirs, ("stt", "sttprobe")).items()):
        # Cosh-law ratios against the zero-bias point of the same time step.
        ref = sets[fs][0][0]
        for name in ("sttprobe", "stt"):
            members = [x for x in stems if meta_of(x)["set"] == name]
            if members:
                pts, _, _ = two_state_set(members)
                key = name if fs == PRODUCTION_FS else f"{name}_dt{fs:g}"
                stt[key] = {"dt_fs": fs, "tau_vs_I": tau_vs_I(pts, ref, ref["barrier_kT"][0]),
                            "points": pts}
    if stt:
        save_json(outdir / "spin_torque.json", stt)

    save_json(outdir / "summary.json", summary(eq, fit, tau_rows, stale, tau0, checks, steps,
                                               stt, ep, steps50))

    manifest = []
    for p in sorted(p for d in rundirs for p in d.glob("*.done")):
        manifest.append(dict(meta_of(p.parent / p.name[:-5]), wall=p.read_text().strip()))
    save_json(outdir / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
