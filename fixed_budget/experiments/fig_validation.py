# Author: Manan Gupta <mnn@yogins.com>
"""Validation figure: the telegraph model against mumax3 and the sLLG macrospin.

Three steps, each reading only stored records:

    python experiments/fig_validation.py trace RUN.npz [DISK] [T0_NS] [LEN_NS]
        cut a short m_z(t) trace of one disk out of a packed mumax3 run
        (results/validation/mumax_trace.npz); run where the packed runs are
    python experiments/fig_validation.py build [MUMAX_DIR] [SLLG_DIR] [NAME]
        put both devices on common axes (results/validation/NAME.json,
        default validation.json from results/micromagnetics and
        results/sllg_matched, the macrospin matched to the mumax3 disk)
    python experiments/fig_validation.py draw OUT.pdf [NAME]
        single-column figure: (a) m_z(t) with the thresholded read,
        (b) relaxation time against input, (c) read-change probability
        against read ratio

Common axes. Each device is measured in units of its own zero-bias
correlation time tau_c(0), taken as the 1/e time of the read autocorrelation
C, and r0 = lag / tau_c(0). The input I is the logit of the read,
atanh<sign m_z>, at the applied field. For any stationary two-valued read
P(read differs from the one before) = 2 p (1 - p) (1 - C), and
P(same | minority) = 1 - p_majority (1 - C), so the stale-read curves follow
from the stored autocorrelations. The step-response time is an exponential
fit of the remaining memory g(t) between 0.9 and 0.1, the definition used in
micromagnetics/mmstats.py, applied to both devices.

Two things in the record are not tests of the model. At zero bias and r0 = 1
the read autocorrelation is 1/e for either device by the definition of
tau_c(0); those rows are marked by_construction. And where the mumax3 disk
never visits the minority state (108 mT) its I is not measured: the step
record keeps I under each extrapolation of the field-to-I map, and the figure
shows that range as a horizontal bar.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MUMAX = ROOT / "results" / "micromagnetics"
SLLG = ROOT / "results" / "sllg_matched"
OUT = ROOT / "results" / "validation"

R0_GRID = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 4]
MEMORY_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
# macrospin bias targets (I) run at nearly the same fields as the mumax3 points (mT)
PAIRS = {9: 0.5, 18: 1, 27: 1.5, 36: 2, 54: 3, 72: 4, 108: 6}


def load(path):
    return json.loads(Path(path).read_text())


def rel(path):
    """Path relative to the package root when it lies inside it."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def efold(t, c):
    """First time c(t) falls below 1/e (linear interpolation)."""
    k = int(np.flatnonzero(np.asarray(c) < np.exp(-1))[0])
    return float(t[k - 1] + (c[k - 1] - np.exp(-1)) / (c[k - 1] - c[k]) * (t[k] - t[k - 1]))


def at(t, y, x):
    """y(x) by linear interpolation in log t (curves are stored on log-spaced lags)."""
    return np.interp(np.log(x), np.log(t), y)


def fit_memory(t, g):
    """Exponential time of g(t) fitted where 0.1 < g < 0.9."""
    t, g = np.asarray(t), np.asarray(g)
    sel = (g > 0.1) & (g < 0.9)
    return float(-1 / np.polyfit(t[sel], np.log(g[sel]), 1)[0])


def ratio(a, b):
    """a / b for two [value, error] pairs, relative errors added in quadrature."""
    r = a[0] / b[0]
    return [float(r), float(r * np.hypot((a[1] or 0) / a[0], (b[1] or 0) / b[0]))]


def cosh_ratio(tau, tau0, i):
    """tau(I) cosh(I) / tau(0) with its error; tau, tau0, i are [value, error]."""
    r = tau[0] * np.cosh(i[0]) / tau0[0]
    rel = [(tau[1] or 0) / tau[0], (tau0[1] or 0) / tau0[0], np.tanh(i[0]) * (i[1] or 0)]
    return [float(r), float(r * np.sqrt(sum(x * x for x in rel)))]


# ------------------------------------------------------------------ trace
def trace(run, disk=0, t0=0.0, length=150.0):
    """Save m_z(t) of one disk of a packed mumax3 run (float16, ns)."""
    z = np.load(run)
    t = z["t"] * 1e9
    keep = (t >= t0) & (t <= t0 + length)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "mumax_trace.npz", t_ns=(t[keep] - t0).astype(np.float32),
                        mz=z["m"][keep, disk], run=Path(run).name, disk=disk, t0_ns=t0)
    print("wrote mumax_trace.npz:", int(keep.sum()), "samples of", Path(run).name, "disk", disk)


# ------------------------------------------------------------------ build
def mumax_part(mm_dir):
    """mumax3 disk on the common axes."""
    eq = load(mm_dir / "device_p_equilibrium.json")
    curves = np.load(mm_dir / "device_p_curves.npz")
    checks = load(mm_dir / "checks.json")
    steps = load(mm_dir / "step_response.json")["steps"]
    pts, tau0 = eq["points"], eq["tau_c0_ns"]
    p0 = pts["0"]
    out = {"dt_fs": eq.get("dt_fs", p0["sim_dt_fs"]), "tau_c0_ns": tau0,
           "tau_dwell_over_tau_c0": p0["tau_dwell"][0] / tau0[0],
           "tau_tail_over_tau_c0": p0["tau_tail"][0] / tau0[0], "tail_amp": p0["tail_amp"],
           "barrier_kT": eq["barrier_kT"], "frac_abs_mz_below_0p3": p0["frac_abs_below_0p3"],
           "slope_per_mT": eq["field_to_I"]["slope_per_mT"]}
    slope = out["slope_per_mT"][0]
    out["tau_vs_I"] = []
    for b, p in sorted(pts.items(), key=lambda kv: float(kv[0])):
        # a point whose minority state was never visited has no measured I (null in json)
        if p["I"][0] is None or not p["tau_efold"][0] or p["flips"] == 0:
            continue
        out["tau_vs_I"].append({
            "b_mT": float(b), "I": p["I"], "flips": p["flips"], "few_flips": p["flips"] < 30,
            "I_over_linear": p["I"][0] / (slope * float(b)) if float(b) else None,
            # the zero-bias point is tau_c(0) itself, so it carries no error of the ratio
            "tau_over_tau0": ratio(p["tau_efold"], tau0) if float(b) else [1.0, 0.0],
            "cosh_ratio": cosh_ratio(p["tau_efold"], tau0, p["I"]),
            "cosh_ratio_dwell": cosh_ratio(p["tau_dwell"], eq["tau_c0_dwell_ns"], p["I"])})
    # other time steps of the same points, each normalised by its own tau_c(0)
    out["time_step"] = []
    for b, row in checks.get("time_step", {}).items():
        for dt, r in row.items():
            if "minus" in dt or not r or float(dt) == out["dt_fs"] or None in r["tau_efold"]:
                continue
            t0 = checks["time_step"]["0"][dt]["tau_efold"]
            out["time_step"].append({
                "b_mT": float(b), "dt_fs": float(dt), "I": r["I"], "seeds": r["seeds"],
                "tau_over_tau0": ratio(r["tau_efold"], t0) if float(b) else [1.0, 0.0],
                "cosh_ratio": r["tau_efold_over_cosh"],
                "cosh_ratio_dwell": r["tau_dwell_over_cosh"]})
    # read autocorrelation on the r0 axis, zero bias and the I = 1 point
    out["acf"] = {}
    for b in ("0", "18"):
        key = f"b{int(b):+d}"
        lag, c = curves[f"{key}_acf_lag_ns"], curves[f"{key}_acf"]
        ok = lag <= 4.2 * tau0[0]
        out["acf"][b] = {"r0": (lag[ok] / tau0[0]).tolist(), "acf": c[ok].tolist(),
                         "p_up": pts[b]["p_up"][0], "I": pts[b]["I"][0],
                         "grid": [{"r0": s["r0"], "acf": s["acf"],
                                   "same_minority": s["measured"]["down"],
                                   "same_all": s["measured"]["all"]}
                                  for s in eq["stale_reads"][b]]}
    out["steps"] = []
    for s in steps:
        if s["low_mT"] == 0 and s["direction"] == "down":
            continue                      # relaxation back to zero field is not a driven step
        mem = {m["t_over_tau0"]: m["eq_asymptote"] for m in s["memory"]}
        out["steps"].append({
            "from_mT": s["low_mT"] if s["direction"] == "up" else s["high_mT"],
            "to_mT": s["target_mT"], "half_ns": s["half_ns"], "I": abs(s["I_target"]),
            "I_source": s["I_target_source"],
            # every estimate of I at the target field (one entry where I is measured)
            "I_options": {k: abs(v[0]) for k, v in s["I_target_options"].items()},
            "tau_over_tau0": ratio(s["tau_step"]["eq_asymptote"], tau0),
            "tau_over_cosh": s["tau_step_over_cosh"]["eq_asymptote"],
            "tau_free_over_tau0": ratio(s["tau_step"]["free"], tau0),
            "memory": {str(x): mem.get(x) for x in MEMORY_GRID}})
    return out


def macrospin_part(ms_dir):
    """sLLG macrospin on the common axes."""
    dev = load(ms_dir / "device.json")
    z = load(ms_dir / "device_bias_I+0.json")
    t0 = z["acf"]["tau_1e_s"]
    reps = dev["tau_c0_replicates"]
    hyst = z["dwell_hyst"]
    tau_dwell = 1 / (1 / hyst["plus"]["mean_s"] + 1 / hyst["minus"]["mean_s"])
    out = {"device": dev["device"], "tau_c0_ns": [reps["tau_1e_mean_s"] * 1e9, reps["tau_1e_sd_s"] * 1e9],
           "tau_c0_used_by_classifier_ns": dev["tau_c0_s"] * 1e9,
           "tau_dwell_over_tau_c0": tau_dwell / t0,
           "tau_tail_over_tau_c0": z["acf"]["tau_tail_s"] / t0,
           "tail_amp": [z["acf"]["tail_amplitude"], z["acf"]["tail_amplitude_se"]],
           "barrier_kT": z["hist"]["barrier_kT"],
           "frac_abs_mz_below_0p3": z["hist"]["frac_abs_below_0.3"],
           "slope_per_mT": [dev["slope_per_T"] * 1e-3, dev["slope_se_per_T"] * 1e-3]}
    slope = dev["slope_per_T"]
    out["tau_vs_I"] = []
    for p in sorted(dev["points"], key=lambda p: p["b_T"]):
        tau = [p["tau_1e_s"], p["tau_1e_se_s"]]
        i = [abs(p["i_exact"]), p["i_se"]]
        out["tau_vs_I"].append({
            "b_mT": p["b_T"] * 1e3, "i_target": p["i_target"], "I": [p["i_exact"], p["i_se"]],
            "I_over_linear": p["i_exact"] / (slope * p["b_T"]) if p["b_T"] else None,
            "tau_over_tau0": [tau[0] / t0, tau[1] / t0],
            "cosh_ratio": cosh_ratio(tau, [t0, z["acf"]["tau_1e_se_s"]], i),
            "tail_over_tau0": p["tau_tail_s"] / t0, "tail_amp": p["tail_amplitude"]})
    out["acf"] = {}
    for b, name in (("0", "I+0"), ("18", "I+1")):
        a = load(ms_dir / f"device_bias_{name}.json")
        lag, c = np.array(a["acf"]["acf_lag_s"]), np.array(a["acf"]["acf"])
        ok = lag <= 4.2 * t0
        out["acf"][b] = {"r0": (lag[ok] / t0).tolist(), "acf": c[ok].tolist(),
                         "acf_se": np.array(a["acf"]["acf_se"])[ok].tolist(),
                         "p_up": a["p_plus_exact"], "I": a["i_exact"]}
    out["steps"] = []
    for s in load(ms_dir / "device_step.json")["rows"]:
        t, g = np.array(s["t_s"]), np.array(s["g"])
        out["steps"].append({
            "from_mT": s["b_from_T"] * 1e3, "to_mT": s["b_to_T"] * 1e3, "I": s["i_to_exact"],
            "tau_over_tau0": [fit_memory(t, g) / t0, None],
            "tau_1e_over_tau0": s["tau_1e_s"] / t0,
            "memory": {str(x): [float(at(t, g, x * t0)), None] if x * t0 <= t[-1] else None
                       for x in MEMORY_GRID}})
    return out


def compare(mm, ms):
    """Side-by-side rows of the two devices and the telegraph model."""
    rows = {"static": {
        "barrier_kT": [mm["barrier_kT"], ms["barrier_kT"]],
        "slope_per_mT": [mm["slope_per_mT"], ms["slope_per_mT"]],
        "frac_abs_mz_below_0p3": [mm["frac_abs_mz_below_0p3"], ms["frac_abs_mz_below_0p3"]],
        "tau_c0_ns": [mm["tau_c0_ns"], ms["tau_c0_ns"]],
        "tau_dwell_over_tau_c0": [mm["tau_dwell_over_tau_c0"], ms["tau_dwell_over_tau_c0"]],
        "tau_tail_over_tau_c0": [mm["tau_tail_over_tau_c0"], ms["tau_tail_over_tau_c0"]],
        "tail_amp": [mm["tail_amp"], ms["tail_amp"]]}}
    ms_tau = {p["i_target"]: p for p in ms["tau_vs_I"]}
    rows["tau_vs_I"] = []
    for p in mm["tau_vs_I"]:
        q = ms_tau.get(PAIRS.get(abs(p["b_mT"]), -1) * np.sign(p["b_mT"]))
        if q:
            rows["tau_vs_I"].append({
                "b_mT": p["b_mT"], "mumax_I": p["I"], "macrospin_I": q["I"][0],
                "mumax_I_over_linear": p["I_over_linear"],
                "macrospin_I_over_linear": q["I_over_linear"],
                "mumax_cosh_ratio": p["cosh_ratio"], "macrospin_cosh_ratio": q["cosh_ratio"],
                "mumax_few_flips": p["few_flips"]})
    rows["reads"] = {}
    for b in ("0", "18"):
        a, m = mm["acf"][b], ms["acf"][b]
        p_maj = [max(x["p_up"], 1 - x["p_up"]) for x in (a, m)]
        ch = [np.cosh(x["I"]) for x in (a, m)]
        rows["reads"][b] = []
        for g in a["grid"]:
            r0 = g["r0"]
            c_ms = float(at(np.array(m["r0"]), np.array(m["acf"]), r0))
            rows["reads"][b].append({
                # at zero bias C(r0 = 1) = 1/e defines tau_c(0) for either device
                "by_construction": b == "0" and r0 == 1,
                "r0": r0, "mumax_acf": g["acf"], "macrospin_acf": c_ms,
                "telegraph_acf": [float(np.exp(-r0 * c)) for c in ch],
                "mumax_same_minority": g["same_minority"],
                "macrospin_same_minority": 1 - p_maj[1] * (1 - c_ms),
                "telegraph_same_minority": [float(1 - p * (1 - np.exp(-r0 * c)))
                                            for p, c in zip(p_maj, ch)]})
    rows["steps"] = []
    for s in mm["steps"]:
        # same field step in the macrospin run, whose fields differ by a few percent
        q = [x for x in ms["steps"] if abs(x["to_mT"] / abs(s["to_mT"]) - 1) < 0.05
             and abs(abs(x["from_mT"]) - abs(s["from_mT"])) < 0.05 * abs(s["to_mT"])]
        if not q:
            continue
        rows["steps"].append({
            "from_mT": s["from_mT"], "to_mT": s["to_mT"], "half_ns": s["half_ns"],
            "mumax_I": s["I"], "macrospin_I": q[0]["I"],
            "mumax_tau_over_tau0": s["tau_over_tau0"],
            "macrospin_tau_over_tau0": q[0]["tau_over_tau0"][0],
            "cosh_law": [1 / np.cosh(s["I"]), 1 / np.cosh(q[0]["I"])],
            # mumax3 step time over the cosh-law time, for each estimate of its I and
            # for the macrospin's I at the same field
            "mumax_tau_over_cosh": s["tau_over_cosh"] | {"macrospin_I": [
                s["tau_over_tau0"][0] * np.cosh(q[0]["I"]),
                s["tau_over_tau0"][1] * np.cosh(q[0]["I"])]},
            "macrospin_tau_over_cosh": q[0]["tau_over_tau0"][0] * np.cosh(q[0]["I"]),
            "memory": {x: {"mumax": s["memory"][x],
                           "macrospin": (q[0]["memory"][x] or [None])[0],
                           "telegraph": [float(np.exp(-float(x) * np.cosh(i)))
                                         for i in (s["I"], q[0]["I"])]}
                       for x in s["memory"] if s["memory"][x]}})
    return rows


def build(mm_dir=MUMAX, ms_dir=SLLG, name="validation"):
    mm, ms = mumax_part(Path(mm_dir)), macrospin_part(Path(ms_dir))
    rec = {"note": "times in units of each device's own 1/e correlation time at zero bias; "
                   "pairs in 'compare' are [mumax3, macrospin]; errors are seed standard errors "
                   "(mumax3) and spin-block standard errors (macrospin); rows marked "
                   "by_construction agree by the definition of tau_c(0) and test nothing; "
                   "where I_source is field_fit the mumax3 I is extrapolated (I_options)",
           "mumax_records": rel(mm_dir), "macrospin_records": rel(ms_dir),
           "mumax": mm, "macrospin": ms, "compare": compare(mm, ms)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(rec, indent=1))
    print(f"wrote {name}.json")


# ------------------------------------------------------------------- draw
def extrapolation_span(step):
    """Centre and half-width of the I estimates of a step (the lower bound is left out)."""
    v = [x for k, x in step["I_options"].items() if k != "largest_measured"]
    return (max(v) + min(v)) / 2, (max(v) - min(v)) / 2


def draw(pdf, name="validation"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6,
        "lines.linewidth": 1.2, "pdf.fonttype": 42, "figure.dpi": 300})
    blue, verm, green = "#0072B2", "#D55E00", "#009E73"
    v = load(OUT / f"{name}.json")
    mm, ms = v["mumax"], v["macrospin"]
    tr = np.load(OUT / "mumax_trace.npz")

    fig = plt.figure(figsize=(3.4, 3.3))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.5, 1.25, 0.95], hspace=0.78,
                          left=0.165, right=0.97, top=0.955, bottom=0.135)
    ax_a, ax_b, ax_c = (fig.add_subplot(g) for g in gs)

    # (a) micromagnetic trace and its thresholded read
    t, mz = tr["t_ns"], tr["mz"].astype(float)
    ax_a.plot(t, mz, color="0.6", lw=0.4)
    ax_a.step(t, np.where(mz >= 0, 1, -1), where="post", color=blue, lw=0.8)
    ax_a.set(xlim=(0, t[-1]), ylim=(-1.25, 1.25), yticks=[-1, 0, 1],
             xlabel="Time (ns)", ylabel=r"$m_z$")
    ax_a.xaxis.labelpad = 1

    # (b) relaxation time against input; open mumax3 markers: fewer than 30 flips
    # (equilibrium) or I not measured (steps), drawn at the middle of the
    # extrapolated range with a bar across it
    i = np.linspace(0, 7, 200)
    ax_b.plot(i, 1 / np.cosh(i), color="k", lw=0.9)
    ax_b.axhline(1, color="k", ls="--", lw=0.7)
    ax_b.text(6.9, 1.25, "bias-independent rate", ha="right", va="bottom", fontsize=7)
    ax_b.text(5.6, 5e-3, r"$1/\cosh I$", ha="right", va="top", fontsize=7)
    q = [p for p in ms["tau_vs_I"] if p["b_mT"] >= 0]
    ax_b.plot([p["I"][0] for p in q], [p["tau_over_tau0"][0] for p in q], color=verm,
              marker="s", ms=2.5, lw=0.8, label="macrospin, equil.")
    q = [s for s in ms["steps"] if s["from_mT"] < 0]          # steps from -B to +B
    ax_b.plot([s["I"] for s in q], [s["tau_over_tau0"][0] for s in q], color=verm, ls=":",
              marker="^", ms=3, lw=0.8, label="macrospin, step")
    kw = {"ls": "none", "capsize": 1.5, "elinewidth": 0.7, "zorder": 3}
    for is_open in (False, True):
        q = [p for p in mm["tau_vs_I"] if p["few_flips"] == is_open]
        ax_b.errorbar([abs(p["I"][0]) for p in q], [p["tau_over_tau0"][0] for p in q],
                      yerr=[p["tau_over_tau0"][1] for p in q], color=blue, marker="o", ms=3,
                      mfc="white" if is_open else blue, **kw,
                      label=None if is_open else "mumax3, equil.")
        q = [s for s in mm["steps"] if s["from_mT"] != 0 and s["half_ns"] != 50
             and (s["I_source"] == "field_fit") == is_open]
        span = [extrapolation_span(s) for s in q]
        ax_b.errorbar([c for c, _ in span], [s["tau_over_tau0"][0] for s in q],
                      yerr=[s["tau_over_tau0"][1] for s in q],
                      xerr=[h for _, h in span] if is_open else None,
                      color=green, marker="v", ms=3.5,
                      mfc="white" if is_open else green, **kw,
                      label=None if is_open else "mumax3, step")
    ax_b.set(yscale="log", xlim=(-0.2, 7), ylim=(1e-3, 4), yticks=[1e-3, 1e-2, 1e-1, 1],
             xlabel=r"Input, $|I|$", ylabel=r"$\tau/\tau_{\mathrm{corr}}(0)$")
    ax_b.legend(frameon=False, loc="lower left", handlelength=1.6, labelspacing=0.2,
                borderaxespad=0.2)

    # (c) probability that a zero-bias read differs from the one before
    r = np.geomspace(0.015, 5, 200)
    ax_c.plot(r, 0.5 * (1 - np.exp(-r)), color="k", lw=0.9, label="telegraph model")
    a = ms["acf"]["0"]
    ax_c.plot(a["r0"], 0.5 * (1 - np.array(a["acf"])), color=verm, lw=0.9, label="macrospin")
    g = mm["acf"]["0"]["grid"]
    ax_c.errorbar([x["r0"] for x in g], [0.5 * (1 - x["acf"][0]) for x in g],
                  yerr=[0.5 * x["acf"][1] for x in g], color=blue, marker="o", ms=3, ls="none",
                  capsize=1.5, elinewidth=0.7, zorder=3, label="mumax3")
    ax_c.axvline(4, color="0.5", ls="--", lw=0.7)
    ax_c.set(xscale="log", yscale="log", xlim=(0.015, 5), ylim=(6e-3, 0.7),
             xlabel=r"Sampling ratio, $r_0=\Delta t/\tau_{\mathrm{corr}}(0)$",
             ylabel="P(change)")
    ax_c.text(3.7, 8e-3, r"$2\tau$ rule", ha="right", va="bottom", fontsize=7, rotation=90)
    ax_c.legend(frameon=False, loc="upper left", handlelength=1.6, labelspacing=0.2,
                borderaxespad=0.2)

    for ax, s in zip((ax_a, ax_b, ax_c), "abc"):
        ax.grid(True, lw=0.3, alpha=0.4)
        # panel letters at a common distance from the figure edge
        x = fig.transFigure.inverted().transform(ax.transAxes.transform((0, 1)))[1]
        fig.text(0.01, x + 0.008, f"({s})", fontsize=9, fontweight="bold", va="bottom")
    fig.savefig(pdf)
    print("wrote", pdf)


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "trace":
        trace(args[0], *[float(x) if k else int(x) for k, x in enumerate(args[1:])])
    elif cmd == "build":
        build(*args)
    else:
        draw(*args)
