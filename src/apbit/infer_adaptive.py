# Author: Manan Gupta <mnn@yogins.com>
"""Per-lane adaptive inference protocol.

Extends `infer.evaluate_lanes`: instead of a uniform T passes per image, each
lane runs its own sequential stopping rule over that image's per-pass votes
(vote = argmax of the pass's real-valued readout) and commits as soon as the
rule fires, or at t_max.

MODELLING CHOICE — what "stopping" means physically (corrected, audit r1
item 13). The randomness sources expose a batched `sample(i, layer)` over a
[B, N] state array and draw one uniform per (b, n) element per call, so a lane
cannot be masked out of a call without changing the rng stream every other lane
sees. Rather than fake per-lane isolation, we model this: a stopped lane stops
READING, but its devices keep being DRIVEN by the same input and keep
fluctuating under it. That is NOT an idle device — an idle sMTJ relaxes toward
p = 1/2, whereas here the pre-activation I stays applied, so the stopped lane's
devices continue to fluctuate about p_eq(I). The earlier "which is exactly what
an idle sMTJ does" wording was wrong and is retracted.

Why it does not matter numerically: what the continued driving could contaminate
is the NEXT image's initial state, and the next image begins with a settle
interval of settle_ratio x tau_corr(0). At the default settle_ratio = 4 the
retained memory of the prior state is exp(-4) = 1.8% at zero bias and less under
bias, so the difference between "kept being driven" and "powered down" survives
into the next decision at the ~1% level at most. In streaming mode there is no
settle and the state is shared by construction, which is the point of that arm.
The device-state trajectory is identical for every lane, stopped or not, which
keeps the lane-count-invariance check from paper 1 meaningful. Cost is charged
as passes actually READ — `samples_used` per image — which is what cost.py
consumes.

Consequences to state in Methods:
  * this protocol does not model a chip that powers down early lanes; a variant
    that does would change the settle physics and this sampling loop with it;
  * every rule sees the SAME recorded vote stream for a given cell, because the
    full t_max passes are always drawn. Rule comparisons within a cell are
    therefore OFFLINE REPLAYS on a common stream, not independent experiments,
    and the fixed-N ladder of run_grid.ladder_from_votes is decoded from the
    same prefixes.

Works with TelegraphSource (streaming and settle modes) and IdealSource;
warmup handling matches infer.evaluate_lanes.
"""

import inspect

import numpy as np
import torch

__all__ = ["evaluate_lanes_adaptive", "reference_decisions",
           "split_reference"]


@torch.no_grad()
def reference_decisions(model, xs, source, t_ref: int = 1024,
                        batch: int = 1000, return_margin: bool = False,
                        return_counts: bool = False):
    """Per-image asymptotic majority decision under an ideal (stateless)
    source: the reference against which STOPPING ERROR is measured.

    The stopping rules' nominal alpha bounds disagreement with the network's
    own asymptotic answer, not with the true label; the stationary law is
    correlation-invariant, so the ideal-source majority is the correct
    reference at every r0. Returns (decisions [N], tie_mask [N]) where
    tie_mask flags images whose top-two vote counts are equal at t_ref
    (excluded from stopping-error aggregates by the caller, count reported),
    optionally followed by the per-image margin (top - second, /t_ref) and the
    full [N, k] vote counts.

    One block of t_ref passes is one REFERENCE BLOCK. The grid draws two
    independent blocks and uses them for different jobs — see `split_reference`
    for why that matters.
    """
    model.eval()
    n = xs.shape[0]
    k = model.dims[-1]
    counts = np.zeros((n, k), dtype=np.int64)
    for a in range(0, n, batch):
        x = xs[a:a + batch]
        for _ in range(t_ref):
            v = model(x, source=source).argmax(1).cpu().numpy()
            counts[a + np.arange(x.shape[0]), v] += 1
    order = np.sort(counts, axis=1)
    tie = order[:, -1] == order[:, -2]
    out = [counts.argmax(1), tie]
    if return_margin:
        out.append((order[:, -1] - order[:, -2]) / float(t_ref))
    if return_counts:
        out.append(counts)
    return tuple(out)


def split_reference(counts_a: np.ndarray, counts_b: np.ndarray,
                    t_ref: int, margin_cut: float) -> dict:
    """SPLIT-SAMPLE reference convention (audit r1 item 3).

    Two independent reference blocks are drawn. Block A defines the STRATUM
    (which images count as marginal, from A's own top-minus-second margin);
    block B defines the REFERENCE DECISION each rule is scored against. Using
    one block for both jobs is what produced the measured error floor: an image
    lands in the marginal stratum partly because that block's noise pushed its
    margin down, and the same noise then makes the block's argmax more likely
    to be the wrong asymptotic answer, so selection and target share a noise
    term and the stopping error inherits a positive bias. Splitting the blocks
    makes stratum membership independent of the reference it is scored against
    and removes that term by construction.

    Returns the arrays the driver needs plus the cross-block agreement rate,
    which is the honest, directly measured upper bound on how much reference
    noise is left in the reported numbers.
    """
    order_a = np.sort(counts_a, axis=1)
    order_b = np.sort(counts_b, axis=1)
    margin_a = (order_a[:, -1] - order_a[:, -2]) / float(t_ref)
    margin_b = (order_b[:, -1] - order_b[:, -2]) / float(t_ref)
    dec_a, dec_b = counts_a.argmax(1), counts_b.argmax(1)
    marginal = margin_a < margin_cut
    tie_b = order_b[:, -1] == order_b[:, -2]
    keep = ~tie_b
    agree = dec_a == dec_b
    return {
        "reference": dec_b,                 # block B: the scoring target
        "reference_tie": tie_b,             # undefined target -> excluded
        "marginal": marginal,               # block A: the stratum
        "margin_a": margin_a,
        "margin_b": margin_b,
        "decision_a": dec_a,
        "top_counts_a": order_a[:, -1],     # for post-hoc q = top/t_ref cuts
        "top_counts_b": order_b[:, -1],
        "blocks_agree_frac": float(agree[keep].mean()) if keep.any() else None,
        "blocks_agree_frac_marginal": (
            float(agree[keep & marginal].mean())
            if (keep & marginal).any() else None),
    }


@torch.no_grad()
def evaluate_lanes_adaptive(model, xs, ys, source, rule_factory,
                            t_max: int = 64, lanes: int = 100,
                            warmup_x=None, warmup_per_lane: int = 0,
                            collect_votes: bool = False,
                            reference=None, reference_ties=None) -> dict:
    """Run one stopping rule per lane over a lane-major image stream.

    Args:
      model: SBNN (eval mode is forced).
      xs, ys: [N, D] images and [N] labels, laid out lane-major exactly as in
        infer.evaluate_lanes — lane b sees xs[b], xs[B+b], xs[2B+b], ...
      source: TelegraphSource or IdealSource.
      rule_factory: callable returning a fresh StoppingRule. Called `lanes`
        times up front, with the lane index if it accepts one (so each lane can
        get its own tie-break seed); each rule is reset() before every image.
      t_max: hard pass budget per image. A rule's own t_max should not exceed
        it (rules are reset per image, so their t_max governs; this is the
        loop bound and the physical read count).
      collect_votes: also return the [N, t_max] per-pass vote array (feeds the
        vote-autocorrelation mechanism figure and offline rule replays).
      reference: optional [N] reference decisions (xs order) — block B of the
        split-sample reference. When given, the output includes
        `stopping_error` = P(decision != reference) over non-tie images — the
        quantity the rules' nominal alpha actually bounds. `realized_error`
        (1 - label accuracy) is TASK error and is NOT alpha-bounded; never
        conflate.
      reference_ties: optional [N] bool tie mask; tied images are excluded from
        `stopping_error` (count reported as `reference_tie_count`).

    Returns a dict with per-image arrays in the original xs order
    (`decisions`, `labels`, `samples_used`, `correct`, `stopped_early`,
    `fired`, `plurality_tie`) and aggregates (`accuracy`, `mean_samples`,
    `realized_error`, `early_stop_frac`, `truncation_rate`,
    `plurality_tie_rate`, `t_max`, `lanes`; plus `stopping_error`,
    `reference_tie_count` when `reference` is given).

    `fired` is True for images whose rule committed on its own criterion and
    False for images truncated at t_max — the two error channels must be
    reported separately, since only the fired one is alpha-bounded.
    `plurality_tie` flags images whose named class came out of a random
    tie-break (audit r1 item 1); the rate is reported so no error number is
    quoted without it.
    """
    model.eval()
    n = int(xs.shape[0])
    if n % lanes != 0:
        raise ValueError(f"{n} images not divisible into {lanes} lanes")
    steps = n // lanes

    if warmup_per_lane > 0:
        if warmup_x is None or warmup_x.shape[0] < warmup_per_lane * lanes:
            raise ValueError("insufficient warmup pool")
        for t in range(warmup_per_lane):
            batch = warmup_x[t * lanes:(t + 1) * lanes]
            source.new_input()
            for _ in range(t_max):
                model(batch, source=source)

    rules = [_make_rule(rule_factory, b) for b in range(lanes)]
    decisions = np.empty(n, dtype=np.int64)
    samples = np.empty(n, dtype=np.int64)
    early = np.zeros(n, dtype=bool)
    fired = np.zeros(n, dtype=bool)
    ties = np.zeros(n, dtype=bool)
    votes_all = np.empty((n, t_max), dtype=np.int64) if collect_votes else None

    for step in range(steps):
        idx = np.arange(step * lanes, (step + 1) * lanes)
        batch = xs[step * lanes:(step + 1) * lanes]
        for r in rules:
            r.reset()
        live = list(range(lanes))
        source.new_input()
        for t in range(t_max):
            votes = model(batch, source=source).argmax(1).cpu().numpy()
            if collect_votes:
                votes_all[idx, t] = votes
            if live:
                still = []
                for b in live:
                    if not rules[b].update(int(votes[b])):
                        still.append(b)
                    else:
                        early[idx[b]] = rules[b].samples_used < rules[b].t_max
                        fired[idx[b]] = rules[b].fired
                live = still
        for b in range(lanes):
            decisions[idx[b]] = rules[b].decision()
            samples[idx[b]] = rules[b].samples_used
            ties[idx[b]] = getattr(rules[b], "plurality_tie", False)

    labels = ys.detach().cpu().numpy().astype(np.int64)
    correct = decisions == labels
    out = {
        "decisions": decisions,
        "labels": labels,
        "samples_used": samples,
        "correct": correct,
        "stopped_early": early,
        "fired": fired,
        "plurality_tie": ties,
        "accuracy": float(correct.mean()),
        "realized_error": float(1.0 - correct.mean()),
        "mean_samples": float(samples.mean()),
        "early_stop_frac": float(early.mean()),
        "truncation_rate": float(1.0 - fired.mean()),
        "plurality_tie_rate": float(ties.mean()),
        "t_max": int(t_max),
        "lanes": int(lanes),
    }
    if reference is not None:
        ref = np.asarray(reference, dtype=np.int64)
        if ref.shape != decisions.shape:
            raise ValueError("reference shape mismatch")
        keep = np.ones(n, dtype=bool) if reference_ties is None \
            else ~np.asarray(reference_ties, dtype=bool)
        out["stopping_error"] = float((decisions[keep] != ref[keep]).mean())
        out["reference_tie_count"] = int(n - keep.sum())
    if collect_votes:
        out["votes"] = votes_all
    return out


def _make_rule(rule_factory, lane: int):
    """Call `rule_factory` with the lane index when it wants one.

    Zero-arg factories are still supported (every test and the pre-checks use
    them); a one-arg factory gets the lane index so the driver can give each
    lane an independent tie-break stream. Arity is read from the signature
    rather than caught from a TypeError, so a TypeError raised INSIDE the
    factory is never mistaken for the wrong arity.
    """
    try:
        n_args = len(inspect.signature(rule_factory).parameters)
    except (TypeError, ValueError):
        n_args = 0
    return rule_factory(lane) if n_args else rule_factory()
