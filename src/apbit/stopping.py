# Author: Manan Gupta <mnn@yogins.com>
"""Sequential stopping rules for p-bit vote streams.

Every rule consumes a stream of per-pass class votes (int class ids, one per
stochastic pass of a single input; the vote is the argmax of that pass's
real-valued readout) and answers one question after each vote: stop now, or
draw another pass? All rules share the API

    r.reset(); stopped = r.update(vote); r.decision(); r.samples_used

`update` returns True the moment the rule commits; further updates are
ignored, so a stopped lane's decision and sample count are frozen. The flag
`r.fired` distinguishes the two ways a rule can commit: True when its own
criterion crossed (an evidence-driven stop), False when it ran out of budget
and t_max forced the plurality. Truncated decisions carry NO guarantee — the
error they contribute is budget error, not rule error — so the grid reports
the two channels separately (PROJECT_STATE.md protocol lesson 2).

Rules
-----
FixedN         baseline: read exactly n passes, decide the plurality.
DirichletStop  i.i.d.-assumption rule (THE VICTIM). Adaptive-Consistency of
               Aggarwal, Madaan, Yang & Mausam, EMNLP 2023: stop when the
               posterior probability that today's plurality class stays the
               plurality exceeds 1-alpha under Dir(c+1). Their released
               implementation uses the top-2 Beta simplification, so that is
               our default; an exact-ish Monte-Carlo Dirichlet path exists for
               validation. The multinomial posterior implicitly assumes the
               votes are i.i.d. — that assumption is what this project breaks.
WaldSPRT       binary SPRT on the top-1-vs-top-2 margin (Wald 1945), Wald
               thresholds A=(1-beta)/alpha, B=beta/(1-alpha). Secondary
               victim, present for theory contact: its error bound is a
               closed-form target we can measure against.
MarkovSPRT     THE CONTRIBUTION. Same decision structure and the same Wald
               thresholds, but the log-likelihood ratio is the TRUE JOINT
               likelihood ratio of the vote-indicator sequence under a
               first-order two-state Markov model (Schmitz & Suselbeck 1983;
               Fuh 2003; Dragalin, Tartakovsky & Veeravalli 1999 for the
               multiclass MSPRT context). The thresholds need no correction —
               a likelihood ratio of the correct joint laws is a nonnegative
               martingale under H0 whatever the dependence, so Ville/Wald
               still apply. This is deliberately NOT an N_eff rescale: no
               theorem licenses substituting N(1-rho)/(1+rho) into an SPRT
               boundary (see docs/prior_art.md, KEY STATS FRAMING).
EProcessStop   anytime-valid bracket via a nonnegative supermartingale and
               Ville's inequality at threshold 1/alpha (Howard et al. 2020;
               Ramdas, Grunwald, Vovk & Shafer 2023). See the guarantee note
               in its docstring — the assumption class is stated honestly and
               is NOT "arbitrary dependence".

Error-bound bookkeeping
-----------------------
Wald's inequalities with A=(1-beta)/alpha, B=beta/(1-alpha) give realized
errors alpha' <= alpha/(1-beta) and beta' <= beta/(1-alpha), not alpha and
beta exactly; the gap is the overshoot slack. Tests target the honest
alpha/(1-beta) form.

Plurality ties (audit r1 item 1)
-------------------------------
When a rule is forced to name a class from tied vote counts — the t_max
truncation fallback, and `decision()` on a lane that has not stopped — the
tie is broken UNIFORMLY AT RANDOM from the tied classes with a per-rule
seeded generator, not by lowest class id. Lowest-id tie-breaking is a
systematic bias that correlates with nothing physical and, because the
reference decision is computed by the same argmax convention, it silently
manufactures agreement (numpy's argmax also takes the lowest index). Every
rule counts its ties: `tie_events` accumulates over the lifetime of the rule
object (i.e. over all images the lane processed, never cleared by `reset`)
and `plurality_tie` flags the current image, so the drivers can report a
tie rate alongside every error number.

Multiclass limitation
---------------------
WaldSPRT and MarkovSPRT are binary tests. For k > 2 we reduce to the current
top-1-vs-top-2 pair: the evidence statistic is computed on the sub-stream of
votes falling in that pair, coded 1 for the leader. Because the pair and its
orientation are selected from the same data, the reduction is a practical
heuristic, not a k-class guarantee; the rigorous object for k > 2 is the
MSPRT of Dragalin et al. 1999, which we do not implement here. In the
k == 2 case (all theory-validation tests) the reduction is exact and the Wald
bounds apply directly.
"""

from math import exp, lgamma, log, log1p

import numpy as np

__all__ = [
    "StoppingRule", "FixedN", "DirichletStop", "WaldSPRT", "MarkovSPRT",
    "EProcessStop", "markov_transition", "markov_rho_bounds",
    "dirichlet_plurality_prob_beta", "dirichlet_plurality_prob_mc",
    "regularized_incomplete_beta",
]

_TINY = 1e-300
_CLIP = 1e-12


# --------------------------------------------------------------------------
# Beta / Dirichlet machinery (no scipy dependency)
# --------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float, itmax: int = 300,
            eps: float = 3e-16) -> float:
    """Continued fraction for the incomplete beta (Lentz's method)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b). Accurate to ~1e-14 for the (small integer) shapes we use."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = lgamma(a + b) - lgamma(a) - lgamma(b)
    if x < (a + 1.0) / (a + b + 2.0):
        front = exp(lbeta + a * log(x) + b * log1p(-x))
        return front * _betacf(a, b, x) / a
    front = exp(lbeta + b * log1p(-x) + a * log(x))
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def dirichlet_plurality_prob_beta(counts) -> float:
    """P(p_top1 > p_top2) under Dir(c+1), the Adaptive-Consistency statistic.

    Marginalising Dir(c+1) to its top two coordinates gives
    p1/(p1+p2) ~ Beta(c1+1, c2+1), so the answer is 1 - I_{1/2}(c1+1, c2+1).
    This is an UPPER bound on the true k-way plurality-stability probability
    (it ignores the other k-2 classes); Aggarwal et al. 2023 ship exactly this
    simplification, so it is what we reproduce.
    """
    c = sorted(counts)
    c1, c2 = c[-1], (c[-2] if len(c) > 1 else 0)
    return 1.0 - regularized_incomplete_beta(c1 + 1.0, c2 + 1.0, 0.5)


def dirichlet_plurality_prob_mc(counts, n_mc: int = 20000, rng=None) -> float:
    """Monte-Carlo P(argmax p == argmax c) under Dir(c+1). Validation path."""
    rng = np.random.default_rng(0) if rng is None else rng
    c = np.asarray(counts, dtype=np.float64)
    top = int(np.argmax(c))
    draws = rng.dirichlet(c + 1.0, size=n_mc)
    return float((np.argmax(draws, axis=1) == top).mean())


# --------------------------------------------------------------------------
# Two-state Markov parameterisation
# --------------------------------------------------------------------------

def markov_rho_bounds(p: float) -> tuple[float, float]:
    """Admissible lag-1 autocorrelation for a two-state chain with P(1)=p."""
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0,1), got {p}")
    return max(-p / (1.0 - p), -(1.0 - p) / p), 1.0


def markov_transition(p: float, rho: float) -> tuple[float, float]:
    """Return (P(1->1), P(0->1)) for stationary P(1)=p, lag-1 autocorr rho.

    Two free parameters pin a two-state chain. Solving stationarity and
    Corr(z_t, z_{t+1}) = rho gives

        P(1->1) = p + rho (1 - p),     P(0->1) = p (1 - rho)

    and hence P(1->0) = (1-rho)(1-p), P(0->0) = 1 - p + rho p. Both rows stay
    in [0,1] only for rho in [max(-p/(1-p), -(1-p)/p), 1]; rho outside that
    window does not correspond to any chain and is rejected.
    """
    lo, hi = markov_rho_bounds(p)
    if not (lo - 1e-12 <= rho <= hi):
        raise ValueError(f"rho={rho} outside [{lo:.6g}, {hi:.6g}] for p={p}")
    return p + rho * (1.0 - p), p * (1.0 - rho)


def _markov_logliks(p: float, rho: float) -> tuple[float, float, float, float]:
    """(log P00, log P01, log P10, log P11) with clipping for degeneracy."""
    p11, p01 = markov_transition(p, rho)
    p10, p00 = 1.0 - p11, 1.0 - p01
    return (log(min(max(p00, _CLIP), 1.0)), log(min(max(p01, _CLIP), 1.0)),
            log(min(max(p10, _CLIP), 1.0)), log(min(max(p11, _CLIP), 1.0)))


# --------------------------------------------------------------------------
# Base
# --------------------------------------------------------------------------

class StoppingRule:
    """Stateful sequential rule over per-pass class votes."""

    def __init__(self, k_classes: int = 2, t_min: int = 1, t_max: int = 64,
                 tie_seed: int = 0):
        if k_classes < 2:
            raise ValueError("k_classes must be >= 2")
        if t_max < 1 or t_min < 1 or t_min > t_max:
            raise ValueError(f"need 1 <= t_min <= t_max, got {t_min}, {t_max}")
        self.k = int(k_classes)
        self.t_min = int(t_min)
        self.t_max = int(t_max)
        # tie-break stream: created ONCE per rule object and deliberately not
        # reset per image, so a lane's tie-breaks are a fixed, reproducible
        # sequence given the construction order.
        self.tie_seed = int(tie_seed)
        self._tie_rng = np.random.default_rng(self.tie_seed)
        self.tie_events = 0
        self.reset()

    # -- lifecycle ---------------------------------------------------------
    def reset(self):
        self.counts = [0] * self.k
        self.t = 0
        self.stopped = False
        self.fired = False
        self._decision = 0
        self.plurality_tie = False       # tie broken at random on this image
        self._tie_pick = None
        self._tie_pick_t = -1
        self._reset_state()
        return self

    def _reset_state(self):
        pass

    @property
    def samples_used(self) -> int:
        """Passes actually READ (the frozen cost proxy's unit)."""
        return self.t

    def decision(self) -> int:
        return self._decision if self.stopped else self._plurality()

    # -- stream ------------------------------------------------------------
    def update(self, vote: int) -> bool:
        if self.stopped:
            return True
        v = int(vote)
        if not (0 <= v < self.k):
            raise ValueError(f"vote {v} outside 0..{self.k - 1}")
        self.counts[v] += 1
        self.t += 1
        self._observe(v)
        if self.t >= self.t_min:
            verdict = self._verdict()
            if verdict is not None:
                self.stopped = True
                self.fired = True
                self._decision = verdict
                return True
        if self.t >= self.t_max:
            self.stopped = True
            self._decision = self._plurality()
        return self.stopped

    def run(self, votes) -> int:
        """Feed a whole stream; return the decision. Convenience for tests."""
        for v in votes:
            if self.update(v):
                break
        return self.decision()

    # -- hooks -------------------------------------------------------------
    def _observe(self, vote: int):
        pass

    def _verdict(self):
        """Return the decided class to stop now, or None to continue."""
        return None

    # -- helpers -----------------------------------------------------------
    def _plurality(self) -> int:
        """Plurality class, TIES BROKEN UNIFORMLY AT RANDOM (seeded).

        Idempotent within one t: the draw is memoised against the current
        vote count, so calling `decision()` twice cannot change the answer or
        double-count the tie. A later vote that re-creates a tie draws again.
        """
        best = max(self.counts)
        top = [j for j in range(self.k) if self.counts[j] == best]
        if len(top) == 1:
            return top[0]
        if self._tie_pick_t != self.t:
            self._tie_pick = int(top[int(self._tie_rng.integers(len(top)))])
            self._tie_pick_t = self.t
            self.tie_events += 1
            self.plurality_tie = True
        return self._tie_pick

    def _top2(self) -> tuple[int, int]:
        """(leader, runner-up) indices; ties broken by lower class id."""
        if self.k == 2:
            return (0, 1) if self.counts[0] >= self.counts[1] else (1, 0)
        order = sorted(range(self.k), key=lambda j: (-self.counts[j], j))
        return order[0], order[1]


class FixedN(StoppingRule):
    """Baseline: always read exactly n passes, then take the plurality."""

    def __init__(self, n: int, k_classes: int = 2, tie_seed: int = 0):
        super().__init__(k_classes=k_classes, t_min=n, t_max=n,
                         tie_seed=tie_seed)
        self.n = int(n)


# --------------------------------------------------------------------------
# The victim: Dirichlet plurality-stability (Adaptive-Consistency, EMNLP 2023)
# --------------------------------------------------------------------------

class DirichletStop(StoppingRule):
    """Stop when P(plurality stays the plurality | Dir(c+1)) >= 1 - alpha.

    method='beta' (default, matches the released Adaptive-Consistency code):
        the top-2 Beta simplification, closed form.
    method='mc':
        Monte-Carlo over the full Dirichlet posterior — exact-ish, slower,
        and strictly more conservative. Validation path only.

    The multinomial-Dirichlet model treats the vote stream as i.i.d., which is
    exactly the assumption a correlated p-bit substrate violates.
    """

    def __init__(self, alpha: float, k_classes: int, t_min: int = 2,
                 t_max: int = 64, method: str = "beta", n_mc: int = 4000,
                 rng=None, tie_seed: int = 0):
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0,1)")
        if method not in ("beta", "mc"):
            raise ValueError(method)
        self.alpha = float(alpha)
        self.method = method
        self.n_mc = int(n_mc)
        self.rng = np.random.default_rng(0) if rng is None else rng
        super().__init__(k_classes=k_classes, t_min=t_min, t_max=t_max,
                         tie_seed=tie_seed)

    def confidence(self) -> float:
        if self.method == "beta":
            return dirichlet_plurality_prob_beta(self.counts)
        return dirichlet_plurality_prob_mc(self.counts, self.n_mc, self.rng)

    def _verdict(self):
        if self.confidence() >= 1.0 - self.alpha:
            return self._plurality()
        return None


# --------------------------------------------------------------------------
# Wald SPRT on the top-1-vs-top-2 margin
# --------------------------------------------------------------------------

class WaldSPRT(StoppingRule):
    """Binary SPRT, H0: rate = p0 vs H1: rate = p1 (p0 < p1).

    k_classes == 2 (theory-contact mode): the indicator is z_t = 1[vote == 1].
    Crossing log A decides class 1 (accept H1); crossing log B decides class 0
    (accept H0). Requires p0 < p1; interpret "accept H0" as "class 0 wins",
    which is only meaningful when p0 < 1/2 < p1 (the symmetric design).

    k_classes > 2: leader-oriented reduction on the current top-2 pair (a, b),
        LLR_t = c_a log(p1/p0) + c_b log((1-p1)/(1-p0)),
    the sufficient statistic of an i.i.d. Bernoulli SPRT on the sub-stream.
    Only the upper boundary fires (accepting H0 would name no class); a lane
    that never crosses runs to t_max and takes the plurality. Selecting the
    pair from the same data makes this a heuristic, not a guarantee — see the
    module docstring.
    """

    def __init__(self, alpha: float = 0.05, beta: float = 0.05,
                 p0: float = 0.5, p1: float = 0.7, t_min: int = 1,
                 t_max: int = 64, k_classes: int = 2, tie_seed: int = 0):
        if not (0.0 < alpha < 1.0 and 0.0 < beta < 1.0):
            raise ValueError("alpha, beta must be in (0,1)")
        if not (0.0 < p0 < p1 < 1.0):
            raise ValueError("need 0 < p0 < p1 < 1")
        self.alpha, self.beta = float(alpha), float(beta)
        self.p0, self.p1 = float(p0), float(p1)
        self.log_a = log((1.0 - self.beta) / self.alpha)
        self.log_b = log(self.beta / (1.0 - self.alpha))
        self._w1 = log(self.p1 / self.p0)
        self._w0 = log((1.0 - self.p1) / (1.0 - self.p0))
        super().__init__(k_classes=k_classes, t_min=t_min, t_max=t_max,
                         tie_seed=tie_seed)

    def llr(self) -> float:
        """Log joint likelihood ratio, i.i.d. product form."""
        if self.k == 2:
            return self.counts[1] * self._w1 + self.counts[0] * self._w0
        a, b = self._top2()
        return self.counts[a] * self._w1 + self.counts[b] * self._w0

    def _verdict(self):
        z = self.llr()
        if self.k == 2:
            if z >= self.log_a:
                return 1
            if z <= self.log_b:
                return 0
            return None
        if z >= self.log_a:
            return self._top2()[0]
        return None


# --------------------------------------------------------------------------
# The contribution: SPRT under a first-order Markov vote model
# --------------------------------------------------------------------------

class MarkovSPRT(WaldSPRT):
    """SPRT whose evidence is the TRUE JOINT likelihood ratio of a two-state
    Markov vote-indicator sequence.

    Model: z_1, ..., z_t is a stationary first-order chain with P(z=1) = p and
    lag-1 autocorrelation rho shared by both hypotheses; p = p0 under H0 and
    p = p1 under H1 (see `markov_transition` for the parameterisation and its
    admissibility window). The joint log-likelihood is

        log pi_p(z_1) + sum_{ij} n_ij log P_p(i -> j)

    with n_ij the transition counts, so the statistic is O(1) to maintain.
    Thresholds are Wald's, UNCHANGED: with the correct joint laws the
    likelihood ratio is a nonnegative martingale under H0 with unit mean, so
    Ville/Wald bound the crossing probability exactly as in the i.i.d. case.
    The repair is the likelihood, not the boundary.

    rho:
      float  — supplied and held fixed. The deployment story: a p-bit chip can
               measure the lag-1 autocorrelation of its own vote indicators
               from the same draws it is already taking.
      None   — estimated online from the stream (see `rho_point` / `rho_hat`).

    Online estimation is genuinely hard here: a lane stops after ~10 votes and
    a lag-1 autocorrelation from 10 samples is both noisy and biased low, and
    the stopping time selects the draws where it came out low. Three defences,
    all in `rho_hat`: the classical -1/m small-sample bias correction; a
    shrinkage factor m/(m+kappa) toward 0 while the stream is short; and a
    one-sided confidence inflation +rho_ucb/sqrt(m). The inflation is what
    makes the estimator safe rather than merely unbiased — over-stating rho
    only makes the rule read more samples. rho_hat is clipped to be
    nonnegative because the physical vote correlation of a persistent
    two-state device is nonnegative; a negative estimate is a small-sample
    artifact. Measured behaviour (synthetic chains, alpha=beta=0.05,
    p0=0.3/p1=0.7): realized error stays at or below nominal for rho <= 0.5
    and reaches about 1.8-1.9x nominal at rho = 0.7 — against ~4.5x for the
    uncorrected Wald rule on the same streams. If a deployment can afford it,
    POOL the estimate across inputs and pass it as `rho`: that is the on-chip
    story and it recovers the exact guarantee.

    At rho = 0 every transition probability collapses to the marginal and the
    statistic equals WaldSPRT's i.i.d. LLR exactly (unit-tested).
    """

    def __init__(self, alpha: float = 0.05, beta: float = 0.05,
                 p0: float = 0.5, p1: float = 0.7, t_min: int = 2,
                 t_max: int = 64, k_classes: int = 2, rho=None,
                 rho_shrinkage: float = 5.0, rho_ucb: float = 1.0,
                 rho_cap: float = 0.995, tie_seed: int = 0):
        if rho is not None:
            for p in (p0, p1):
                lo, hi = markov_rho_bounds(p)
                if not (lo - 1e-12 <= rho <= hi):
                    raise ValueError(
                        f"rho={rho} inadmissible for p={p} (bounds "
                        f"{lo:.6g}..{hi:.6g})")
        self.rho_fixed = None if rho is None else float(rho)
        self.rho_shrinkage = float(rho_shrinkage)
        self.rho_ucb = float(rho_ucb)
        lo0, _ = markov_rho_bounds(p0)
        lo1, _ = markov_rho_bounds(p1)
        self._rho_lo = max(lo0, lo1) + 1e-9
        self._rho_hi = float(rho_cap)
        super().__init__(alpha=alpha, beta=beta, p0=p0, p1=p1, t_min=t_min,
                         t_max=t_max, k_classes=k_classes, tie_seed=tie_seed)

    def _reset_state(self):
        self._z1 = None
        self._prev = None
        self._n = [[0, 0], [0, 0]]      # n[i][j] = transitions i -> j
        self._ones = 0                  # count of z == 1
        self._nz = 0                    # length of the indicator sub-stream
        self._votes: list[int] = []
        self._pair = None

    # -- indicator bookkeeping --------------------------------------------
    def _observe(self, vote: int):
        if self.k == 2:
            self._push(vote)
            return
        self._votes.append(vote)
        pair = self._top2()
        if pair != self._pair:
            self._pair = pair
            self._rebuild(pair)
        elif vote in pair:
            self._push(1 if vote == pair[0] else 0)

    def _push(self, z: int):
        if self._z1 is None:
            self._z1 = z
        else:
            self._n[self._prev][z] += 1
        self._prev = z
        self._ones += z
        self._nz += 1

    def _rebuild(self, pair):
        a, b = pair
        self._z1 = None
        self._prev = None
        self._n = [[0, 0], [0, 0]]
        self._ones = 0
        self._nz = 0
        for v in self._votes:
            if v == a:
                self._push(1)
            elif v == b:
                self._push(0)

    # -- rho ---------------------------------------------------------------
    def rho_point(self) -> float:
        """Bias-corrected, shrunk lag-1 sample autocorrelation of the vote
        indicator. m = n_z - 1 pairs, zbar the indicator mean:

            r1 = (n11/m - zbar^2) / (zbar (1 - zbar)) + 1/m,
            rho_point = r1 * m / (m + kappa)

        The +1/m removes the classical negative bias of the sample ACF; the
        shrinkage stops a two-sample stream from claiming rho = 1. Clipped to
        the window admissible for both p0 and p1.
        """
        m = self._nz - 1
        if m < 1:
            return 0.0
        zbar = self._ones / self._nz
        if zbar <= 1e-9 or zbar >= 1.0 - 1e-9:
            return float(self._rho_hi)   # a constant run: maximal persistence
        r1 = (self._n[1][1] / m - zbar * zbar) / (zbar * (1.0 - zbar))
        r1 = (r1 + 1.0 / m) * m / (m + self.rho_shrinkage)
        return float(min(max(r1, self._rho_lo), self._rho_hi))

    def rho_hat(self) -> float:
        """The value actually used: rho_point inflated by rho_ucb/sqrt(m) and
        floored at 0. Over-stating rho only costs samples, so the one-sided
        inflation converts a noisy estimate into a conservative one."""
        m = self._nz - 1
        if m < 1:
            return 0.0
        r = self.rho_point()
        if self.rho_ucb:
            r += self.rho_ucb / np.sqrt(m)
        return float(min(max(r, 0.0), self._rho_hi))

    @property
    def rho(self) -> float:
        return self.rho_fixed if self.rho_fixed is not None else self.rho_hat()

    # -- statistic ---------------------------------------------------------
    def llr(self) -> float:
        if self._z1 is None:
            return 0.0
        rho = self.rho
        n, z1 = self._n, self._z1
        out = 0.0
        for p, sign in ((self.p1, 1.0), (self.p0, -1.0)):
            l00, l01, l10, l11 = _markov_logliks(p, rho)
            ll = log(p if z1 == 1 else 1.0 - p)
            ll += (n[0][0] * l00 + n[0][1] * l01
                   + n[1][0] * l10 + n[1][1] * l11)
            out += sign * ll
        return out

    def _verdict(self):
        z = self.llr()
        if self.k == 2:
            if z >= self.log_a:
                return 1
            if z <= self.log_b:
                return 0
            return None
        if z >= self.log_a:
            return self._top2()[0]
        return None


# --------------------------------------------------------------------------
# Anytime-valid bracket: e-process / Ville
# --------------------------------------------------------------------------

class EProcessStop(StoppingRule):
    """Anytime-valid stop via a nonnegative supermartingale and Ville.

    For each class k we run a betting capital process on the indicator
    x_t = 1[vote_t == k] against the null H0_k: the class-k rate is at most
    `null_mean` (default 1/2):

        M_t(lambda) = prod_{s<=t} (1 + lambda (x_s - null_mean)),
        M_t         = (1/L) sum_l M_t(lambda_l),   lambda_l in (0, 1/null_mean)

    Each factor is nonnegative for lambda <= 1/null_mean, and a uniform
    mixture of nonnegative supermartingales starting at 1 is one too. Ville's
    inequality then gives P(sup_t M_t >= 1/alpha_eff) <= alpha_eff under H0_k.
    We stop and decide class k the first time its capital crosses 1/alpha_eff.
    With bonferroni=True, alpha_eff = alpha / (k_classes - 1), so the union
    bound over the wrong classes caps the realized error at alpha.

    GUARANTEE CLASS — stated precisely, deliberately NOT overclaimed.
    Validity needs the supermartingale property, i.e.

        E[x_t | F_{t-1}] <= null_mean   for every t under H0_k,

    a CONDITIONAL-MEAN (martingale-difference) condition. That is much weaker
    than i.i.d. — it permits arbitrary, adaptive, non-identically-distributed
    dependence on the past, including an adversary who picks each conditional
    rate after seeing the whole history — but it is NOT literally "valid under
    arbitrary dependence", which is how the design doc loosely phrased it. A
    concrete failure inside our own physics: a positively autocorrelated
    two-state vote chain with stationary rate p slightly below 1/2 has
    P(1 -> 1) = p + rho(1 - p) > 1/2, so the conditional mean exceeds the null
    after a 1 and the supermartingale property is broken. For such streams
    either raise `null_mean` above the worst-case conditional rate
    (conservative, costs samples) or use MarkovSPRT, whose joint likelihood
    models the dependence instead of assuming it away.

    Multiclass note: the per-class processes run on the FULL vote stream, not
    on a top-2 sub-stream — a data-dependent inclusion rule would destroy the
    martingale property. The price is conservatism: with k = 10 a plurality
    class at rate 0.4 never crosses a null_mean = 0.5 boundary and the lane
    runs to t_max.
    """

    def __init__(self, alpha: float = 0.05, k_classes: int = 2,
                 t_min: int = 1, t_max: int = 64, null_mean: float = 0.5,
                 n_lambda: int = 20, bonferroni: bool = True,
                 tie_seed: int = 0):
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0,1)")
        if not (0.0 < null_mean < 1.0):
            raise ValueError("null_mean must be in (0,1)")
        self.alpha = float(alpha)
        self.null_mean = float(null_mean)
        self.bonferroni = bool(bonferroni)
        eff = self.alpha / max(1, k_classes - 1) if bonferroni else self.alpha
        self.alpha_eff = eff
        self.log_threshold = log(1.0 / eff)
        self.lambdas = np.linspace(0.05, 0.95 / self.null_mean, int(n_lambda))
        super().__init__(k_classes=k_classes, t_min=t_min, t_max=t_max,
                         tie_seed=tie_seed)

    def _reset_state(self):
        self._log_cap = np.zeros((self.k, self.lambdas.size))
        self._x = np.zeros(self.k)

    def _observe(self, vote: int):
        self._x[:] = -self.null_mean
        self._x[vote] += 1.0
        self._log_cap += np.log1p(np.outer(self._x, self.lambdas))

    def log_e_values(self) -> np.ndarray:
        """log of the mixture e-value for each class."""
        m = self._log_cap.max(axis=1, keepdims=True)
        return (m[:, 0] + np.log(np.exp(self._log_cap - m).mean(axis=1)))

    def e_values(self) -> np.ndarray:
        return np.exp(self.log_e_values())

    def _verdict(self):
        le = self.log_e_values()
        j = int(np.argmax(le))
        if le[j] >= self.log_threshold:
            return j
        return None
