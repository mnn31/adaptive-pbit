# Author: Manan Gupta <mnn@yogins.com>
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "micromagnetics"))
import analyze
import analyze_checks
import mmstats as ms
from mx3gen import disk_array_script, square_wave


def _telegraph(k_up=0.3, k_down=0.7, n_steps=20000, n_disks=64, dt=0.1, seed=0):
    return ms.telegraph_series(k_up, k_down, n_steps, n_disks, dt, np.random.default_rng(seed))


def test_telegraph_series_stationary_and_acf():
    k_up, k_down, dt = 0.3, 0.7, 0.1
    s = _telegraph(k_up, k_down, dt=dt)
    p = k_up / (k_up + k_down)
    assert abs((s == 1).mean() - p) < 0.01
    c = ms.autocorr(s, 50)
    lags = np.arange(51) * dt
    assert np.allclose(c, np.exp(-lags * (k_up + k_down)), atol=0.02)
    tau, amp = ms.fit_exp_tail(c, dt)
    assert abs(tau - 1.0) < 0.05 and abs(amp - 1) < 0.03
    assert abs(ms.efold_time(c, dt) - 1.0) < 0.05


def test_dwell_rates_with_censoring():
    # Sampling at dt shifts the survival curve by about rate x dt; keep that small here.
    s = _telegraph(n_steps=20000, dt=0.01)
    dw = ms.dwell_times(s, 0.01)
    # Escape rate from the up state is k_down, from the down state k_up.
    rate_up, rate_dn = ms.exp_rate_mle(*dw[1][:2]), ms.exp_rate_mle(*dw[-1][:2])
    assert abs(rate_up - 0.7) < 0.04 and abs(rate_dn - 0.3) < 0.02
    d, c, first = dw[1]
    assert ms.exp_deviation(d[~first], c[~first], rate_up) < 0.03
    assert abs(ms.short_dwell_excess(d[~first], c[~first], rate_up, frac=0.2) - 1) < 0.15


def test_dwell_rate_unbiased_in_short_windows():
    # About seven runs per disk: dropping the flip that ends each first run would
    # give a rate about 15% low (tau 16 instead of 14).
    k, dt = 1 / 28.0, 0.05
    rates = []
    for seed in range(8):
        s = _telegraph(k, k, n_steps=4000, n_disks=64, dt=dt, seed=seed)
        dw = ms.dwell_times(s, dt)
        rates.append(ms.exp_rate_mle(*dw[1][:2]) + ms.exp_rate_mle(*dw[-1][:2]))
    assert abs(1 / np.mean(rates) - 14.0) < 0.5


def test_dwell_times_bookkeeping():
    s = np.array([1, 1, -1, -1, -1, 1], dtype=np.int8)[:, None]
    dw = ms.dwell_times(s, 1.0)
    assert dw[1][0].tolist() == [2.0, 1.0]
    assert dw[1][1].tolist() == [False, True] and dw[1][2].tolist() == [True, False]
    assert dw[-1][0].tolist() == [3.0]
    assert dw[-1][1].tolist() == [False] and dw[-1][2].tolist() == [False]
    # Two flips in six samples: one out of each state.
    assert (~dw[1][1]).sum() == 1 and (~dw[-1][1]).sum() == 1


def test_km_matches_empirical_without_censoring():
    d = np.array([1.0, 2.0, 3.0, 4.0])
    _t, s = ms.km_survival(d, np.zeros(4, dtype=bool))
    assert np.allclose(s, [0.75, 0.5, 0.25, 0.0])


def test_hysteretic_states_ignore_recrossings():
    m = np.array([0.8, 0.1, -0.1, 0.2, -0.5, -0.1, 0.3, 0.5])[:, None]
    s = ms.hysteretic_states(m, thr=0.4)[:, 0]
    assert s.tolist() == [1, 1, 1, 1, -1, -1, -1, 1]
    assert ms.sign_reads(m)[:, 0].tolist() == [1, 1, -1, 1, -1, -1, 1, 1]


def test_stale_read_matches_propagator():
    k_up, k_down, dt = 0.3, 0.7, 0.1
    s = _telegraph(k_up, k_down, dt=dt, seed=3)
    p = k_up / (k_up + k_down)
    meas = ms.stale_read(s, [3, 10])
    pred = np.array([ms.telegraph_stale(p, lag * dt * (k_up + k_down)) for lag in (3, 10)])
    assert np.allclose(meas, pred, atol=0.01)


def test_histogram_barrier_of_bimodal_sample():
    rng = np.random.default_rng(1)
    m = np.clip(np.concatenate([rng.normal(0.8, 0.1, 50000), rng.normal(-0.8, 0.1, 50000),
                                rng.uniform(-0.1, 0.1, 500)]), -1, 1)
    _, _, barrier, peaks = ms.mz_histogram(m)
    assert barrier > 3 and peaks[0] < -0.6 and peaks[1] > 0.6


def test_well_depths_of_tilted_double_well():
    centres = np.linspace(-0.99, 0.99, 100)
    energy = -2.0 * centres**2 + centres**4 - 0.5 * centres      # deeper well at m > 0
    dens = np.exp(-energy)
    lo, hi, saddle = ms.well_depths(dens, centres)
    assert hi > lo > 0 and abs(saddle) < 0.3
    i_min = np.argmin(dens[np.abs(centres) < 0.6]) + np.flatnonzero(np.abs(centres) < 0.6)[0]
    assert np.isclose(hi, np.log(dens.max() / dens[i_min]))


def test_brown_ratio_limits():
    assert np.isclose(ms.brown_relax_ratio(0.0, 4.0), 1.0)
    # Exact form: (1 - h^2) exp(-delta h^2) (cosh I - h sinh I), h = I / (2 delta); cosh I at large delta.
    i, d = 0.2, 40.0
    assert abs(ms.brown_relax_ratio(i, d) - np.cosh(i)) < 1e-3
    assert ms.brown_relax_ratio(2.0, 3.5) < np.cosh(2.0)


def test_step_response_recovers_tau():
    t = np.arange(400) * 0.05
    curve = 0.6 + (-0.6 - 0.6) * np.exp(-t / 2.5)
    assert abs(ms.step_response(curve, 0.05, 0.6, -0.6) - 2.5) < 1e-6


def test_step_fit_free_recovers_tau_and_asymptote():
    t = np.arange(2000) * 0.05
    curve = 0.6 - 1.2 * np.exp(-t / 2.5) + 0.002 * np.random.default_rng(0).standard_normal(len(t))
    tau, s_inf = ms.step_fit_free(curve, 0.05)
    assert abs(tau - 2.5) < 0.05 and abs(s_inf - 0.6) < 0.01
    # A wrong fixed asymptote biases the log-linear fit; the free fit does not need one.
    assert ms.step_response(curve, 0.05, 0.7, curve[0]) > 2.6


def test_stale_read_at_efold_time_is_fixed_by_the_acf():
    # At p = 1/2 the stale probability is (1 + C) / 2, so at the 1/e time it is
    # (1 + 1/e) / 2 whatever the process; agreement there is not a test.
    s = _telegraph(0.5, 0.5, dt=0.1, seed=5)
    tau = ms.efold_time(ms.autocorr(s, 50), 0.1)
    lag = round(tau / 0.1)
    assert abs(ms.stale_read(s, [lag])[0][0] - (1 + np.exp(-1)) / 2) < 0.01


def test_script_generator():
    txt = disk_array_script(nx=2, ny=3, seed=4, schedule=square_wave(0.018, -0.018, 5e-9, 2))
    assert txt.count("DefRegion(") == 6 and txt.count("TableAdd(m.Comp(2)") == 6
    assert txt.count("Run(5e-09)") == 4 and "B_ext = vector(0, 0, -0.018)" in txt
    assert "SetGridSize(64, 96, 1)" in txt


def test_escape_rates_match_run_estimator():
    s = _telegraph(n_steps=3000, n_disks=16, dt=0.05, seed=9)
    dw = ms.dwell_times(s, 0.05)
    k_up, k_dn, n_up, n_dn = ms.escape_rates(s, 0.05)
    assert np.isclose(k_up, ms.exp_rate_mle(*dw[1][:2]))
    assert np.isclose(k_dn, ms.exp_rate_mle(*dw[-1][:2]))
    assert n_up == (~dw[1][1]).sum() and n_dn == (~dw[-1][1]).sum()


@pytest.mark.parametrize("window", [0.5, 1.0, 2.0, 8.0])
def test_escape_rates_unbiased_at_any_window(window):
    # Telegraph process with tau_c = 14 and asymmetric rates; window in units of tau_c.
    k_up, k_dn, dt, tau = 1 / 21.0, 1 / 42.0, 0.05, 14.0
    n_steps = round(window * tau / dt)
    s = ms.telegraph_series(k_dn, k_up, n_steps, round(4e6 / n_steps), dt, np.random.default_rng(11))
    r_up, r_dn, n_up, n_dn = ms.escape_rates(s, dt)
    assert abs(r_up / k_up - 1) < 4 / np.sqrt(n_up)
    assert abs(r_dn / k_dn - 1) < 4 / np.sqrt(n_dn)
    assert abs(1 / (r_up + r_dn) / tau - 1) < 4 / np.sqrt(n_up + n_dn)
    # Mean whole dwells (first and last run of each disk dropped) are biased short:
    # a window of a few dwells keeps mostly the short ones.
    dw = ms.dwell_times(s, dt)
    whole = [dw[st][0][~dw[st][1] & ~dw[st][2]].mean() for st in (1, -1)]
    assert 1 / (1 / whole[0] + 1 / whole[1]) / tau < 0.8


def _points(fields, slope=0.05, cubic=0.0, seeds=3):
    """Bias points with I = slope B + cubic B^3 and tau = 10 / cosh I; a False entry is a field with no logit."""
    out = {}
    for b, measured in fields.items():
        i = slope * b + cubic * b ** 3
        recs = [{"I": i if measured else np.inf, "tau_efold": 10 / np.cosh(i) * (1 + 0.01 * k),
                 "tau_dwell": 8 / np.cosh(i) * (1 + 0.01 * k)} for k in (-1, 0, 1)[:seeds]]
        out[b] = {"I": analyze.mean_se([r["I"] for r in recs]), "seed_records": recs,
                  "tau_efold": analyze.mean_se([r["tau_efold"] for r in recs]),
                  "tau_dwell": analyze.mean_se([r["tau_dwell"] for r in recs])}
    return out


def test_field_fit_extrapolations_bracket_a_bent_map():
    fields = {b: True for b in (-36, -27, -18, -9, 0, 9, 18, 27, 36, 54, 72)} | {108: False}
    fit = analyze.field_fit(_points(fields, cubic=-5e-7))
    opts = fit["extrapolated_I"][108.0]
    truth = 0.05 * 108 - 5e-7 * 108 ** 3
    # The odd cubics recover the bend; the line over weak bias overshoots it.
    assert abs(opts["odd_cubic_all_measured"][0] - truth) < 1e-6
    assert abs(opts["odd_cubic_absI_le_2p1"][0] - truth) < 1e-6
    assert opts["linear"][0] > truth + 0.3 and opts["linear"][1] > 0
    assert np.isclose(opts["largest_measured"][0], 0.05 * 72 - 5e-7 * 72 ** 3)
    # The maps are odd in the field, so -B mirrors +B.
    mirror = analyze.extrapolated_I(fit["extrapolation_maps"], -108)
    assert np.isclose(mirror["odd_cubic_all_measured"][0], -truth)
    assert mirror["largest_measured"][0] < 0


def test_field_fit_ignores_points_with_unvisited_seeds():
    pts = _points({b: True for b in (-36, -18, 0, 18, 36, 72)})
    pts[72]["I"] = [9.9, 0.1]              # mean over the seeds that did visit: not a measurement
    pts[72]["I_all_seeds_finite"] = False
    fit = analyze.field_fit(pts)
    assert np.isclose(fit["extrapolation_maps"]["largest_measured"][0], 1.8)
    assert np.isclose(fit["extrapolated_I"][72.0]["odd_cubic_all_measured"][0], 3.6)


def test_step_over_cosh_per_estimate_of_I():
    rec = {"tau_step": {"eq_asymptote": [2.0, 0.1], "free": [2.5, 0.1]}}
    out = analyze.step_over_cosh(rec, [10.0, 0.0], {"a": [2.0, np.nan], "b": [3.0, np.nan]})
    assert np.isclose(out["eq_asymptote"]["a"][0], 0.2 * np.cosh(2.0))
    assert np.isclose(out["eq_asymptote"]["b"][1], 0.01 * np.cosh(3.0))
    assert np.isclose(out["free"]["b"][0], 0.25 * np.cosh(3.0))


def test_time_step_pooled_mirrors_negative_fields():
    pts = _points({0: True, 18: True, -18: True})
    row = analyze.time_step_pooled({"25": pts, "50": pts})[18]
    assert row["25"]["seeds"] == 6 and row["25"]["seeds_minus_B"] == 3
    assert np.isclose(row["25"]["I"][0], 0.9)
    assert np.isclose(row["25"]["tau_efold_over_cosh"][0], 1.0)
    assert row["25_minus_50"]["tau_efold"]["diff"] == 0


def test_stale_block_marks_the_definitional_row():
    s = _telegraph(0.5, 0.5, n_steps=4000, dt=0.05)
    tau = ms.efold_time(ms.autocorr(s, 200), 0.05)
    rows = analyze.stale_block([s], 0.05, 0.5, tau, {"telegraph_efold": [tau, np.nan],
                                                    "telegraph_dwell": [0.9 * tau, np.nan]})
    flagged = [(r["r0"], k) for r in rows for k in ("telegraph_efold", "telegraph_dwell")
               if r[k]["by_construction"]]
    assert flagged == [(1, "telegraph_efold")]


def test_script_generator_finer_mesh_and_current_steps():
    base = disk_array_script(nx=2, ny=2, seed=1)
    assert "SetCellSize(2e-09, 2e-09, 1e-09)" in base and "J = " not in base
    fine = disk_array_script(nx=2, ny=2, seed=1, cell=1e-9, pitch=48)
    assert "SetGridSize(96, 96, 1)" in fine and "SetCellSize(1e-09, 1e-09, 1e-09)" in fine
    wave = disk_array_script(nx=2, ny=2, seed=1, schedule=[(0.0, 1e-9, -4e10), (0.0, 2e-9, 4e10)])
    lines = wave.splitlines()
    assert "FixedLayer = vector(0, 0, 1)" in lines
    # Each segment sets its current before it runs.
    assert lines[lines.index("J = vector(0, 0, -4e+10)") + 2] == "Run(1e-09)"
    assert lines[lines.index("J = vector(0, 0, 4e+10)") + 2] == "Run(2e-09)"


def _packed_run(path, s, dt, meta):
    np.savez_compressed(f"{path}.npz", t=np.arange(len(s)) * dt * 1e-9, m=0.9 * s.astype(np.float16))
    Path(f"{path}.json").write_text(json.dumps(meta))


def test_current_steps_recover_the_telegraph_step_time(tmp_path):
    # A telegraph process driven by a square wave between I = -1.5 and +1.5: its step
    # time is tau_c(0) / cosh I, so the ratio to the cosh law is 1.
    rng = np.random.default_rng(3)
    f, i_step, dt, half, periods, n = 0.05, 1.5, 0.01, 20.0, 6, 256
    k_up, k_dn = f * np.exp(i_step), f * np.exp(-i_step)
    p = np.tanh(i_step) / 2 + 0.5
    burn = ms.telegraph_series(k_dn, k_up, round(5 / dt), n, dt, rng)
    parts, last = [burn], burn[-1]
    for j in range(2 * periods):
        up, dn = (k_up, k_dn) if j % 2 == 0 else (k_dn, k_up)
        seg = ms.telegraph_series(up, dn, round(half / dt) + 1, n, dt, rng, p0=(last == 1))
        parts.append(seg[1:])
        last = seg[-1]
    _packed_run(tmp_path / "sttstep_s0", np.concatenate(parts), dt,
                {"low_j": -1e10, "high_j": 1e10, "half_ns": half, "periods": periods,
                 "burn_ns": 5.0})
    tau0 = 1 / (2 * f)
    points = {1e10: {"I": [i_step, 0.01], "p_up": [p, 0.001], "I_all_seeds_finite": True,
                     "flips": 1000}}
    ref = {"tau_efold": [tau0, 0.0]}
    steps, _ = analyze_checks.current_steps([tmp_path / "sttstep_s0"], points, ref,
                                            np.random.default_rng(0))
    assert [st["direction"] for st in steps] == ["up", "down"]
    assert np.isclose(steps[1]["I_target"], -i_step) and "mirrored" in steps[1]["I_target_options"]
    for st in steps:
        ratio = st["tau_step_over_cosh"]["eq_asymptote"]
        assert abs(next(iter(ratio.values()))[0] - 1) < 0.15


def test_read_change_is_measured_over_telegraph():
    rows = [{"r0": 0.02, "measured": {"all": [0.98, 0.001]},
             "telegraph_efold": {"all": 0.99, "by_construction": False}}]
    out = analyze_checks.read_change(rows)["0.02"]
    assert np.isclose(out["over_telegraph"], 2.0) and np.isclose(out["changed"][0], 0.02)
