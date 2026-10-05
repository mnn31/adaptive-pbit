# Author: Manan Gupta <mnn@yogins.com>
"""FROZEN cost proxy (design.md, week 1 — never renegotiated).

Per decision, for a network with layer widths `dims` read for `passes`
stochastic passes:

    random bits = passes x (number of stochastic hidden units)
    MACs        = passes x sum_l dims[l] * dims[l+1]

Both are exact counts of the two things the substrate actually spends: one
Bernoulli draw per hidden unit per pass, and one full forward evaluation
(including the real-valued readout, which runs every pass) per pass. Biases
are adds, not multiply-accumulates, and are excluded — stated so the number
is reproducible, not because they are free.

The throttled baseline (Daniels et al. 2020's remedy: slow the clock until
the device decorrelates) buys independence with WALL-CLOCK ONLY:

    latency-equivalent passes = passes x (r0_throttled / r0_operating)

AUDIT R1 ITEM 8 — WHAT THE SLOWDOWN DOES AND DOES NOT SCALE. Throttling
lengthens the query interval; it does not add draws or arithmetic. A
throttled decision spends exactly the same number of random bits and the
same number of MACs as an unthrottled one with the same pass count, so the
earlier `throttled_random_bits_mean` / `throttled_macs_mean` fields were
wrong in kind, not just in size, and are GONE. The slowdown is reported in
one unit only — time — under the name `latency_equivalent_passes`, i.e.
"how many operating-clock pass slots this decision occupies". Every
comparison in the paper must state which unit it is in: random bits (a
count) or latency-equivalent passes (a duration).

SETTLE OVERHEAD. In settle mode each decision additionally waits
t_settle = settle_ratio x tau_corr(0) before its first read. In the same
operating-clock units that is settle_ratio / r0_operating extra pass slots
per DECISION (not per pass), reported as `settle_latency_passes` and added
into `latency_passes_with_settle_mean`. It is charged per decision because
the settle happens once per input presentation. Any throughput claim that
ignores it is overstated, so the report carries both.

NO JOULES ARE CLAIMED ANYWHERE. Hassan/Datta/Camsari PRApplied 15, 064046
(2021) is cited for per-bit energy context only; converting these counts to
energy would require device numbers this project does not measure.
"""

import numpy as np

__all__ = ["hidden_units", "macs_per_pass", "random_bits", "macs",
           "throttled_equivalent_passes", "latency_equivalent_passes",
           "settle_latency_passes", "cost_report"]


def hidden_units(dims) -> int:
    """Stochastic units per pass = every layer except input and readout."""
    dims = tuple(dims)
    if len(dims) < 2:
        raise ValueError("dims needs at least an input and an output layer")
    return int(sum(dims[1:-1]))


def macs_per_pass(dims) -> int:
    """Multiply-accumulates for one full forward pass, readout included."""
    dims = tuple(dims)
    if len(dims) < 2:
        raise ValueError("dims needs at least an input and an output layer")
    return int(sum(a * b for a, b in zip(dims[:-1], dims[1:])))


def random_bits(passes, dims):
    """Random bits drawn per decision. `passes` may be scalar or array."""
    return np.asarray(passes) * hidden_units(dims)


def macs(passes, dims):
    """MACs per decision. `passes` may be scalar or array."""
    return np.asarray(passes) * macs_per_pass(dims)


def latency_equivalent_passes(passes, r0_throttled: float,
                              r0_operating: float):
    """LATENCY-equivalent pass count for a clock-throttled baseline.

    r0 = query interval / correlation time, so a rule-compliant r0_throttled
    (>= 4, per the mean-dwell-time form of Daniels' 2-tau rule) takes
    r0_throttled / r0_operating times as long per pass as the operating point.
    This is a duration in operating-clock pass slots. It is NOT a bit count
    and NOT a MAC count: throttling buys independence with time alone
    (audit r1 item 8).
    """
    if not (r0_throttled > 0 and r0_operating > 0):
        raise ValueError("r0 values must be positive")
    return np.asarray(passes) * (r0_throttled / r0_operating)


# the old name, kept so nothing silently breaks; same latency semantics
throttled_equivalent_passes = latency_equivalent_passes


def settle_latency_passes(r0_operating: float, settle_ratio: float) -> float:
    """Settle wait per DECISION, in operating-clock pass slots.

    t_settle = settle_ratio x tau_corr(0) and one pass slot is
    dt = r0_operating x tau_corr(0), so the wait is settle_ratio / r0_operating
    pass slots, charged once per input presentation (not per pass).
    """
    if not (r0_operating > 0):
        raise ValueError("r0_operating must be positive")
    if settle_ratio is None or not np.isfinite(settle_ratio):
        return float("inf") if settle_ratio is not None else 0.0
    if settle_ratio < 0:
        raise ValueError("settle_ratio must be nonnegative")
    return float(settle_ratio) / float(r0_operating)


def cost_report(passes, dims, r0_operating: float = None,
                r0_throttled: float = None, settle_ratio: float = None) -> dict:
    """Frozen-proxy summary for a scalar or array of per-decision pass counts.

    Always returns mean passes, random bits and MACs. When both r0 values are
    given it adds the throttled baseline's LATENCY in operating-clock pass
    slots (`throttled_latency_passes_mean`) — no bit or MAC variant, see the
    module docstring. When `settle_ratio` is given it also adds the settle
    wait per decision and the settle-charged latency of the run itself, so a
    throughput ratio can be quoted with and without the overhead.
    """
    p = np.asarray(passes, dtype=np.float64)
    out = {
        "passes_mean": float(p.mean()),
        "random_bits_mean": float(np.mean(random_bits(p, dims))),
        "macs_mean": float(np.mean(macs(p, dims))),
        "hidden_units": hidden_units(dims),
        "macs_per_pass": macs_per_pass(dims),
    }
    if r0_operating is not None and r0_throttled is not None:
        eq = latency_equivalent_passes(p, r0_throttled, r0_operating)
        out["throttled_latency_passes_mean"] = float(np.mean(eq))
        out["latency_units"] = ("operating-clock pass slots; bits and MACs do "
                               "NOT scale with the slowdown")
    if settle_ratio is not None and r0_operating is not None:
        s = settle_latency_passes(r0_operating, settle_ratio)
        out["settle_latency_passes"] = s
        out["latency_passes_with_settle_mean"] = float(p.mean() + s)
        out["settle_ratio"] = (None if settle_ratio is None
                               else float(settle_ratio))
    return out
