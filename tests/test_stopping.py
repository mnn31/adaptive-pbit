# Author: Manan Gupta <mnn@yogins.com>
"""Theory-validation tests for the stopping rules and the frozen cost proxy.

Everything here runs against closed-form or synthetic ground truth — no
network, no dataset — per docs/design.md "Unit tests before any experiment".
The point is to prove the rule implementations before any p-bit result is
attributed to them:

  * a Wald SPRT on i.i.d. Bernoulli votes must respect its own error bound;
  * the Dirichlet rule must respect alpha on i.i.d. multinomial votes;
  * that same Wald SPRT must BREAK on positively autocorrelated votes (the
    paper's premise — if this test ever passes silently, there is no paper);
  * MarkovSPRT with the true rho must repair it, and must collapse exactly to
    WaldSPRT at rho = 0;
  * the e-process must hold alpha under its documented assumption class,
    which includes strongly dependent streams.

Error-bound convention: Wald's inequalities with A=(1-beta)/alpha,
B=beta/(1-alpha) bound the realized errors by alpha/(1-beta) and
beta/(1-alpha), not by alpha and beta. Assertions use those.

ASSERTION DIRECTIONS (audit r1 item 15, reviewer C D1). A one-sided Wilson
LOWER bound answers "is the error significantly ABOVE the bound?" — the right
question for the violation tests. It is too weak for a RESTORATION claim,
because a rule with a huge error and a wide interval passes it. Restoration
tests therefore also assert the Wilson UPPER bound, within a documented slack
factor RESTORE_SLACK: with reps = 3000 and a realized error sitting at the
bound (which is what a well-calibrated SPRT does), binomial noise alone moves
the 95% upper limit by about +/-17%, so demanding upper <= nominal exactly
would be a coin flip, not a test. The slack is set so the assertion still
FAILS at a 1.5x inflation — and the uncorrected rule on the same streams is
above 2.5x, which the premise test asserts separately.
"""

RESTORE_SLACK = 1.3

from math import comb, log, sqrt

import numpy as np
import pytest
import torch

from apbit import cost
from apbit.infer_adaptive import evaluate_lanes_adaptive
from apbit.model import SBNN
from apbit.smtj import IdealSource, TelegraphSource
from apbit.stopping import (DirichletStop, EProcessStop, FixedN, MarkovSPRT,
                            WaldSPRT, dirichlet_plurality_prob_beta,
                            dirichlet_plurality_prob_mc, markov_rho_bounds,
                            markov_transition, regularized_incomplete_beta)


# ---------------------------------------------------------------- helpers --

def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    """Lower end of the Wilson interval for k/n — the CI slack in assertions."""
    if n == 0:
        return 0.0
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d)


def wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 1.0
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return min(1.0, (c + h) / d)


def iid_streams(p, reps, n, rng):
    return (rng.random((reps, n)) < p).astype(np.int64)


def markov_streams(p, rho, reps, n, rng):
    """Stationary two-state chains with P(1)=p and lag-1 autocorrelation rho."""
    p11, p01 = markov_transition(p, rho)
    u = rng.random((reps, n))
    z = np.empty((reps, n), dtype=np.int64)
    z[:, 0] = u[:, 0] < p
    for t in range(1, n):
        z[:, t] = u[:, t] < np.where(z[:, t - 1] == 1, p11, p01)
    return z


def multinomial_streams(probs, reps, n, rng):
    return rng.choice(len(probs), size=(reps, n), p=probs)


def error_rate(rule_fn, streams, truth):
    """(#wrong decisions, #streams, mean samples used)."""
    wrong, total = 0, 0
    for s in streams:
        r = rule_fn()
        wrong += int(r.run(s) != truth)
        total += r.samples_used
    return wrong, len(streams), total / len(streams)


def early_error_rate(rule_fn, streams, truth):
    """(#streams whose rule CROSSED early onto the wrong class, #early, #runs).

    Ville bounds P(the wrong class's e-value ever crosses 1/alpha) over all
    runs, so that — not a rate conditioned on having stopped — is the quantity
    to compare against alpha. The t_max plurality fallback is a separate,
    budget-limited estimator whose errors are not the e-process's to answer
    for, so it is excluded.
    """
    wrong, early = 0, 0
    for s in streams:
        r = rule_fn()
        d = r.run(s)
        if r.samples_used < r.t_max:
            early += 1
            wrong += int(d != truth)
    return wrong, early, len(streams)


# ------------------------------------------------ Beta/Dirichlet machinery --

def test_incomplete_beta_matches_exact_binomial_sum():
    # For integer shapes, 1 - I_{1/2}(c1+1, c2+1) = P(Bin(c1+c2+1, 1/2) <= c1)
    for c1, c2 in [(0, 0), (3, 1), (7, 7), (10, 4), (25, 12), (60, 41)]:
        n = c1 + c2 + 1
        exact = sum(comb(n, j) for j in range(c1 + 1)) / 2.0 ** n
        got = 1.0 - regularized_incomplete_beta(c1 + 1.0, c2 + 1.0, 0.5)
        assert abs(got - exact) < 1e-12, (c1, c2, got, exact)


def test_top2_beta_upper_bounds_the_full_dirichlet_probability():
    # the Adaptive-Consistency simplification ignores classes 3..k, so it can
    # only be optimistic; with k=2 the two agree exactly
    rng = np.random.default_rng(0)
    assert abs(dirichlet_plurality_prob_beta([10, 4])
               - dirichlet_plurality_prob_mc([10, 4], 60_000, rng)) < 0.01
    for c in ([10, 4, 2], [20, 15, 10, 5], [8, 7, 7, 6, 5]):
        b = dirichlet_plurality_prob_beta(c)
        m = dirichlet_plurality_prob_mc(c, 40_000, rng)
        assert b >= m - 0.01, (c, b, m)
    assert dirichlet_plurality_prob_beta([7, 7]) == pytest.approx(0.5, abs=1e-9)


def test_fixed_n_baseline_reads_exactly_n_and_takes_the_plurality():
    r = FixedN(7, k_classes=3)
    votes = [0, 1, 1, 2, 1, 0, 1, 0, 0, 0]
    assert r.run(votes) == 1
    assert r.samples_used == 7 and r.stopped


# ------------------------------------------------------ i.i.d. calibration --

@pytest.mark.parametrize("p0,p1,alpha,beta,t_max", [
    (0.30, 0.70, 0.05, 0.05, 300),
    (0.40, 0.60, 0.05, 0.05, 400),
    (0.40, 0.60, 0.02, 0.10, 400),
])
def test_wald_sprt_respects_its_error_bound_on_iid_votes(p0, p1, alpha, beta,
                                                         t_max):
    reps = 6000
    for truth_p, cls, nominal in ((p1, 1, beta / (1.0 - alpha)),
                                  (p0, 0, alpha / (1.0 - beta))):
        rng = np.random.default_rng(101)
        s = iid_streams(truth_p, reps, t_max, rng)
        wrong, n, _ = error_rate(
            lambda: WaldSPRT(alpha, beta, p0, p1, 1, t_max), s, cls)
        assert wilson_lower(wrong, n) <= nominal, (
            p0, p1, truth_p, wrong / n, nominal)


@pytest.mark.parametrize("probs,alpha", [
    ((0.5, 0.3, 0.2), 0.05),
    ((0.5, 0.3, 0.2), 0.10),
    ((0.6, 0.2, 0.1, 0.1), 0.05),
])
def test_dirichlet_stop_respects_alpha_on_iid_multinomial_votes(probs, alpha):
    # The Adaptive-Consistency guarantee is about plurality STABILITY, so it
    # is only meaningful where a real plurality exists; near-ties (0.4 vs
    # 0.35) are dominated by intrinsic estimation error, not rule error.
    probs = np.asarray(probs)
    truth = int(probs.argmax())
    rng = np.random.default_rng(202)
    s = multinomial_streams(probs, 2500, 300, rng)
    wrong, n, _ = error_rate(
        lambda: DirichletStop(alpha, len(probs), t_min=3, t_max=300), s, truth)
    assert wilson_lower(wrong, n) <= alpha, (probs, alpha, wrong / n)


# --------------------------------------- the premise: correlation breaks it --

@pytest.mark.parametrize("rho", [0.5, 0.7])
def test_wald_sprt_violates_its_nominal_error_under_positive_correlation(rho):
    """RQ1's premise, in miniature. Positive autocorrelation inflates the
    apparent evidence rate, the i.i.d. product statistic crosses the boundary
    on fewer real bits of information, and the guarantee fails."""
    p0, p1, alpha, beta, t_max = 0.3, 0.7, 0.05, 0.05, 300
    bound = alpha / (1.0 - beta)
    for truth_p, cls in ((p0, 0), (p1, 1)):
        rng = np.random.default_rng(303)
        s = markov_streams(truth_p, rho, 3000, t_max, rng)
        wrong, n, mean_t = error_rate(
            lambda: WaldSPRT(alpha, beta, p0, p1, 1, t_max), s, cls)
        realized = wrong / n
        assert wilson_lower(wrong, n) > 2.5 * bound, (rho, truth_p, realized)
        assert mean_t < 9.5   # and it stops EARLIER than it should


@pytest.mark.parametrize("rho", [0.3, 0.5, 0.7])
def test_markov_sprt_with_true_rho_restores_error_control(rho):
    """The contribution: same Wald thresholds, true joint likelihood.

    Both directions are asserted: not significantly above the bound (Wilson
    lower), AND the 95% upper limit inside RESTORE_SLACK x bound, so a 1.5x
    inflation would fail this test instead of hiding in a wide interval.
    """
    p0, p1, alpha, beta, t_max = 0.3, 0.7, 0.05, 0.05, 300
    for truth_p, cls, nominal in ((p0, 0, alpha / (1.0 - beta)),
                                  (p1, 1, beta / (1.0 - alpha))):
        rng = np.random.default_rng(303)
        s = markov_streams(truth_p, rho, 3000, t_max, rng)
        wrong, n, _ = error_rate(
            lambda: MarkovSPRT(alpha, beta, p0, p1, 2, t_max, rho=rho),
            s, cls)
        assert wilson_lower(wrong, n) <= nominal, (rho, truth_p, wrong / n)
        assert wilson_upper(wrong, n) <= RESTORE_SLACK * nominal, (
            rho, truth_p, wrong / n, wilson_upper(wrong, n))


@pytest.mark.parametrize("rho", [0.5, 0.7])
def test_markov_sprt_with_measured_rho_restores_error_control(rho):
    """MISSPECIFIED rho: the deployment story, not the oracle.

    The chip does not know rho; it measures the pooled lag-1 vote
    autocorrelation on an INDEPENDENT calibration block (here 200 synthetic
    streams from the same chain, mirroring phase 2a of the grid, which measures
    on the validation split) and feeds that number to the rule. The estimate is
    off by a percent or so, and the guarantee has to survive that — otherwise
    the correction is an oracle result and the deployment story is empty.
    """
    p0, p1, alpha, beta, t_max = 0.3, 0.7, 0.05, 0.05, 300
    cal = markov_streams(0.5, rho, 200, 300,
                         np.random.default_rng(999)).astype(float)
    rho_hat = float(np.mean([np.corrcoef(r[:-1], r[1:])[0, 1]
                             for r in cal if r.std() > 0]))
    assert abs(rho_hat - rho) < 0.05, rho_hat        # a real measurement
    for truth_p, cls, nominal in ((p0, 0, alpha / (1.0 - beta)),
                                  (p1, 1, beta / (1.0 - alpha))):
        rng = np.random.default_rng(404)
        s = markov_streams(truth_p, rho, 2000, t_max, rng)
        wrong, n, _ = error_rate(
            lambda: MarkovSPRT(alpha, beta, p0, p1, 2, t_max, rho=rho_hat),
            s, cls)
        assert wilson_lower(wrong, n) <= nominal, (rho, truth_p, wrong / n)
        assert wilson_upper(wrong, n) <= RESTORE_SLACK * nominal, (
            rho, truth_p, rho_hat, wrong / n, wilson_upper(wrong, n))


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.5, 0.7])
def test_markov_sprt_with_online_rho_stays_within_documented_tolerance(rho):
    """Online estimation from a ~10-vote stream cannot pin rho, so the rule is
    only approximately calibrated. Documented tolerance (and the reason the
    deployment story is a POOLED, chip-measured rho): realized error at or
    below nominal for rho <= 0.5, and no worse than 2.5x nominal at rho = 0.7
    (measured ~1.7-1.9x across seeds, vs ~4.5x for the uncorrected rule).
    """
    p0, p1, alpha, beta, t_max = 0.3, 0.7, 0.05, 0.05, 300
    tol_factor = 1.0 if rho <= 0.5 else 2.5
    for truth_p, cls, nominal in ((p0, 0, alpha / (1.0 - beta)),
                                  (p1, 1, beta / (1.0 - alpha))):
        rng = np.random.default_rng(404)
        s = markov_streams(truth_p, rho, 2000, t_max, rng)
        wrong, n, _ = error_rate(
            lambda: MarkovSPRT(alpha, beta, p0, p1, 2, t_max, rho=None),
            s, cls)
        assert wilson_lower(wrong, n) <= tol_factor * nominal, (
            rho, truth_p, wrong / n)


def test_markov_sprt_reduces_exactly_to_wald_sprt_at_rho_zero():
    rng = np.random.default_rng(505)
    huge = 10 ** 6           # t_min = t_max = huge: never stops, just accrues
    for _ in range(20):
        w = WaldSPRT(0.05, 0.05, 0.3, 0.7, huge, huge)
        m = MarkovSPRT(0.05, 0.05, 0.3, 0.7, huge, huge, rho=0.0)
        for v in (rng.random(200) < 0.6).astype(int):
            w.update(int(v))
            m.update(int(v))
            assert m.llr() == pytest.approx(w.llr(), abs=1e-11, rel=1e-12)


# -------------------------------------------- two-state chain construction --

def test_markov_transition_reproduces_p_and_rho():
    for p, rho in [(0.3, 0.6), (0.7, 0.2), (0.5, 0.8), (0.4, -0.3)]:
        rng = np.random.default_rng(606)
        z = markov_streams(p, rho, 1, 400_000, rng)[0].astype(float)
        assert abs(z.mean() - p) < 0.005, (p, rho, z.mean())
        x = z - z.mean()
        acf = (x[:-1] * x[1:]).mean() / x.var()
        assert abs(acf - rho) < 0.01, (p, rho, acf)


def test_multiclass_top2_reduction_paths_agree_with_the_binary_case():
    """k > 2 reduces to the current top-1-vs-top-2 sub-stream. When only two
    classes ever appear, the reduction must reproduce the binary rules
    exactly, including MarkovSPRT's sub-stream rebuild on a leader swap."""
    rng = np.random.default_rng(1010)
    huge = 10 ** 6
    for _ in range(15):
        votes = rng.choice([3, 7], size=60, p=[0.4, 0.6])
        # class 7 leads, so the k=10 reduction codes 7 as the "1" state
        wb = WaldSPRT(0.05, 0.05, 0.3, 0.7, huge, huge, k_classes=2)
        mb = MarkovSPRT(0.05, 0.05, 0.3, 0.7, huge, huge, k_classes=2,
                        rho=0.4)
        wm = WaldSPRT(0.05, 0.05, 0.3, 0.7, huge, huge, k_classes=10)
        mm = MarkovSPRT(0.05, 0.05, 0.3, 0.7, huge, huge, k_classes=10,
                        rho=0.4)
        for v in votes:
            b = 1 if v == 7 else 0
            wb.update(b)
            mb.update(b)
            wm.update(int(v))
            mm.update(int(v))
        assert wm.llr() == pytest.approx(wb.llr(), rel=1e-12)
        assert mm.llr() == pytest.approx(mb.llr(), rel=1e-12)
        assert mm._top2()[0] == 7


def test_multiclass_rules_run_and_stop_sensibly():
    rng = np.random.default_rng(1111)
    probs = np.array([0.55, 0.15, 0.10, 0.10, 0.10])
    s = multinomial_streams(probs, 400, 200, rng)
    for factory in (
        lambda: WaldSPRT(0.05, 0.05, 0.5, 0.75, 2, 200, k_classes=5),
        lambda: MarkovSPRT(0.05, 0.05, 0.5, 0.75, 2, 200, k_classes=5,
                           rho=0.3),
        lambda: MarkovSPRT(0.05, 0.05, 0.5, 0.75, 2, 200, k_classes=5),
        lambda: EProcessStop(0.05, 5, 1, 200),
    ):
        wrong, n, mean_t = error_rate(factory, s, 0)
        assert wrong / n < 0.10 and 2 <= mean_t <= 200, (factory, wrong / n,
                                                         mean_t)


def test_markov_rho_bounds_are_enforced():
    lo, hi = markov_rho_bounds(0.2)
    assert lo == pytest.approx(-0.25) and hi == 1.0     # max(-p/(1-p), -(1-p)/p)
    with pytest.raises(ValueError):
        markov_transition(0.2, -0.5)
    with pytest.raises(ValueError):
        markov_transition(0.2, 1.5)
    with pytest.raises(ValueError):
        MarkovSPRT(0.05, 0.05, 0.3, 0.7, rho=-0.9)


# ------------------------------------------------------ anytime-valid rule --

@pytest.mark.parametrize("q", [0.3, 0.5, 0.55])
def test_eprocess_holds_alpha_on_iid_streams(q):
    alpha, t_max = 0.05, 300
    truth = 1 if q > 0.5 else 0
    rng = np.random.default_rng(707)
    s = iid_streams(q, 800, t_max, rng)
    wrong, early, runs = early_error_rate(
        lambda: EProcessStop(alpha, 2, 1, t_max), s, truth)
    if q == 0.5:
        # no correct class exists; both nulls hold, so Ville caps EITHER
        # crossing at alpha each -> at most 2 alpha for any crossing at all
        assert wilson_lower(early, runs) <= 2 * alpha, early / runs
    else:
        assert wilson_lower(wrong, runs) <= alpha, (wrong, early, runs)


def test_eprocess_holds_alpha_on_dependent_streams_in_its_assumption_class():
    """Strongly dependent, non-identically-distributed, history-adaptive
    stream whose conditional mean never exceeds the null (1/2). This is the
    documented assumption class — a martingale-difference condition, NOT
    arbitrary dependence (see EProcessStop's docstring)."""
    alpha, t_max, reps = 0.05, 300, 800
    rng = np.random.default_rng(808)
    z = np.zeros((reps, t_max), dtype=np.int64)
    regime = np.ones(reps, dtype=bool)       # True: q=0.5, False: q=0.0
    for t in range(t_max):
        u = rng.random(reps)
        z[:, t] = np.where(regime, u < 0.5, False)
        # regime flips on the observed history -> heavy positive dependence
        flip = rng.random(reps) < np.where(z[:, t] == 1, 0.05, 0.20)
        regime = np.where(flip, ~regime, regime)
    lag1 = np.mean([np.corrcoef(r[:-1], r[1:])[0, 1] for r in z.astype(float)
                    if r.std() > 0])
    assert lag1 > 0.15, lag1          # the stream really is correlated

    # class 1 is the null-true class here (conditional rate <= 1/2 always),
    # so an early crossing onto class 1 is exactly the event Ville bounds
    crossings, _, runs = early_error_rate(
        lambda: EProcessStop(alpha, 2, 1, t_max), z, 0)
    assert wilson_lower(crossings, runs) <= alpha, crossings / runs


def test_eprocess_is_powered_when_the_signal_is_real():
    rng = np.random.default_rng(909)
    s = iid_streams(0.8, 300, 200, rng)
    stops = [EProcessStop(0.05, 2, 1, 200) for _ in range(300)]
    used = []
    for r, row in zip(stops, s):
        assert r.run(row) == 1
        used.append(r.samples_used)
    assert np.mean(used) < 40 and max(used) < 200


# --------------------------------------------------------- frozen cost proxy --

def test_cost_proxy_arithmetic():
    dims = (784, 256, 128, 10)
    assert cost.hidden_units(dims) == 384
    assert cost.macs_per_pass(dims) == 784 * 256 + 256 * 128 + 128 * 10
    assert cost.random_bits(16, dims) == 16 * 384
    assert cost.macs(16, dims) == 16 * cost.macs_per_pass(dims)
    assert cost.hidden_units((20, 16, 10)) == 16
    assert cost.macs_per_pass((20, 16, 10)) == 20 * 16 + 16 * 10


def test_latency_equivalent_cost():
    assert cost.latency_equivalent_passes(10, 4.0, 0.5) == pytest.approx(80.0)
    assert cost.latency_equivalent_passes(10, 4.0, 4.0) == pytest.approx(10.0)
    assert cost.throttled_equivalent_passes is cost.latency_equivalent_passes
    with pytest.raises(ValueError):
        cost.latency_equivalent_passes(10, 4.0, 0.0)


def test_settle_latency_is_charged_per_decision():
    # t_settle = settle_ratio * tau_corr(0); one pass slot = r0 * tau_corr(0)
    assert cost.settle_latency_passes(0.5, 4.0) == pytest.approx(8.0)
    assert cost.settle_latency_passes(4.0, 4.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cost.settle_latency_passes(0.0, 4.0)


def test_cost_report_charges_throttling_in_latency_only():
    """AUDIT R1 ITEM 8: bits and MACs must NOT scale with the slowdown."""
    dims = (784, 256, 128, 10)
    rep = cost.cost_report([4, 8, 12], dims, r0_operating=1.0,
                           r0_throttled=4.0, settle_ratio=4.0)
    assert rep["passes_mean"] == pytest.approx(8.0)
    assert rep["random_bits_mean"] == pytest.approx(8.0 * 384)
    assert rep["macs_mean"] == pytest.approx(8.0 * cost.macs_per_pass(dims))
    assert rep["throttled_latency_passes_mean"] == pytest.approx(32.0)
    assert rep["settle_latency_passes"] == pytest.approx(4.0)
    assert rep["latency_passes_with_settle_mean"] == pytest.approx(12.0)
    for gone in ("throttled_random_bits_mean", "throttled_macs_mean",
                 "throttled_passes_mean"):
        assert gone not in rep, f"{gone} must not come back"
    assert "throttled_latency_passes_mean" not in cost.cost_report([4], dims)


# ---------------------------------------------------------- tie-breaking --

def test_plurality_ties_are_broken_at_random_and_counted():
    """AUDIT R1 ITEM 1. A 3-3 tie must not always name the lower class id:
    numpy's argmax (which builds the reference) does exactly that, so a
    deterministic tie-break manufactures agreement with the reference."""
    picks = []
    for lane in range(400):
        r = FixedN(6, k_classes=3, tie_seed=1234 + 7919 * lane)
        assert r.run([0, 0, 0, 1, 1, 1]) in (0, 1)
        picks.append(r.decision())
        assert r.plurality_tie and r.tie_events == 1
    frac0 = np.mean([p == 0 for p in picks])
    assert 0.4 < frac0 < 0.6, frac0
    # untied streams never touch the tie machinery
    clean = FixedN(6, k_classes=3, tie_seed=5)
    clean.run([0, 0, 0, 0, 1, 2])
    assert clean.decision() == 0
    assert not clean.plurality_tie and clean.tie_events == 0


def test_tie_break_is_seeded_reproducible_and_idempotent():
    a = FixedN(4, k_classes=2, tie_seed=77)
    b = FixedN(4, k_classes=2, tie_seed=77)
    votes = [0, 1, 0, 1]
    assert a.run(votes) == b.run(votes)
    # asking twice must not redraw or double-count
    first = a.decision()
    assert a.decision() == first and a.tie_events == 1
    # tie_events accumulates across images, plurality_tie is per image
    a.reset()
    assert not a.plurality_tie
    a.run([0, 0, 1, 1])
    assert a.tie_events == 2


def test_tie_events_accumulate_but_reset_clears_the_flag():
    r = DirichletStop(0.05, 4, t_min=4, t_max=4, tie_seed=3)
    for _ in range(3):
        r.reset()
        r.run([0, 1, 2, 3])          # four-way tie every time
    assert r.tie_events == 3 and r.plurality_tie


# ------------------------------------------------------- smoke integration --

@pytest.mark.fast
@pytest.mark.parametrize("source_fn", [
    lambda: IdealSource(seed=0),
    lambda: TelegraphSource(r0=1.0, streaming=False, settle_ratio=4.0,
                            noise_seed=0),
    lambda: TelegraphSource(r0=0.5, streaming=True, noise_seed=0),
])
def test_adaptive_pipeline_runs_end_to_end(source_fn):
    """Untrained net, random data: accuracy is meaningless, plumbing is not."""
    torch.manual_seed(0)
    model = SBNN(dims=(20, 16, 10))
    xs, ys = torch.randn(40, 20), torch.randint(0, 10, (40,))
    t_max, lanes = 12, 10
    out = evaluate_lanes_adaptive(
        model, xs, ys, source_fn(),
        lambda: DirichletStop(0.05, 10, t_min=2, t_max=t_max),
        t_max=t_max, lanes=lanes, warmup_x=torch.randn(20, 20),
        warmup_per_lane=1, collect_votes=True)

    assert out["decisions"].shape == (40,) and out["samples_used"].shape == (40,)
    assert out["votes"].shape == (40, t_max)
    assert set(np.unique(out["decisions"])) <= set(range(10))
    assert out["samples_used"].min() >= 2 and out["samples_used"].max() <= t_max
    assert 0.0 <= out["accuracy"] <= 1.0
    assert out["accuracy"] + out["realized_error"] == pytest.approx(1.0)
    assert out["mean_samples"] == pytest.approx(out["samples_used"].mean())
    assert np.all(out["stopped_early"] == (out["samples_used"] < t_max))
    rep = cost.cost_report(out["samples_used"], model.dims)
    assert rep["random_bits_mean"] == pytest.approx(out["mean_samples"] * 16)


@pytest.mark.fast
def test_adaptive_pipeline_rejects_bad_lane_split():
    model = SBNN(dims=(20, 16, 10))
    with pytest.raises(ValueError, match="not divisible"):
        evaluate_lanes_adaptive(model, torch.randn(7, 20),
                                torch.zeros(7, dtype=torch.long),
                                IdealSource(0), lambda: FixedN(3, 10),
                                t_max=3, lanes=2)
