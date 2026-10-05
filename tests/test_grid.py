# Author: Manan Gupta <mnn@yogins.com>
"""Plumbing tests for the grid driver (experiments/run_grid.py).

These are NOT science tests — the theory validation lives in test_stopping.py.
What is checked here is that the driver cannot silently lie about its own
bookkeeping: that the strata partition the kept images exactly, that the
fired/truncated split partitions them exactly, that the cost proxy and the
autocorrelation diagnostic actually appear in the payload, that a result lands
on disk atomically, and that the pruned grid is the size it claims to be.

Everything runs on a tiny untrained network over synthetic images so the whole
file stays well inside a minute.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from apbit.diagnostics import vote_acf_curve
from apbit.model import SBNN
from apbit.infer_adaptive import (evaluate_lanes_adaptive, reference_decisions,
                                  split_reference)
from apbit.smtj import IdealSource, TelegraphSource
from apbit.stopping import FixedN

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import run_grid as G  # noqa: E402

K = 4
DIMS = (16, 8, K)
N_IMAGES = 200
LANES = 100
T_MAX = 16
T_REF = 64


@pytest.fixture(scope="module")
def tiny():
    """(model, xs, ys, ref, tie, marginal) — an untrained 16-8-4 SBNN on
    synthetic images. Untrained is deliberate: it makes the reference margins
    small, so the marginal stratum is populated and the strata bookkeeping is
    actually exercised instead of being trivially all-easy."""
    torch.manual_seed(0)
    model = SBNN(DIMS)
    model.eval()
    g = torch.Generator().manual_seed(1)
    xs = torch.rand(N_IMAGES, DIMS[0], generator=g)
    ys = torch.randint(0, K, (N_IMAGES,), generator=g)
    ref, tie, margin = reference_decisions(model, xs, IdealSource(seed=7),
                                           t_ref=T_REF, batch=N_IMAGES,
                                           return_margin=True)
    marginal = margin < G.MARGIN_CUT
    return model, xs, ys, ref, tie, marginal


@pytest.mark.fast
@pytest.mark.parametrize("rule", list(G.RULES))
def test_tiny_cell_end_to_end(tiny, tmp_path, rule):
    """One full per-cell pipeline: source -> rule -> strata -> termination ->
    cost -> acf -> atomic write -> read back."""
    model, xs, ys, ref, tie, marginal = tiny
    r0 = 0.5
    spec = G.rule_spec(rule, K, T_MAX, rho=0.3)
    src = TelegraphSource(r0=r0, chip_seed=0, noise_seed=1000,
                          bias_independent_tau=True)
    res = G.evaluate_cell(model, xs, ys, src, spec, ref, tie, marginal, r0,
                          t_max=T_MAX, lanes=LANES)
    res.pop("_out")

    keep = int((~tie).sum())
    se = res["stopping_error"]
    assert se["all"]["n"] == keep
    assert se["marginal"]["n"] + se["easy"]["n"] == keep, \
        "strata must partition"
    assert se["marginal"]["n"] > 0, \
        "fixture should populate the marginal stratum"

    bt = res["by_termination"]
    assert bt["fired"]["n"] + bt["truncated"]["n"] == keep, \
        "fired/truncated must partition"
    assert bt["fired_marginal"]["n"] + bt["truncated_marginal"]["n"] == \
        se["marginal"]["n"]
    assert 0.0 <= res["truncation_rate"] <= 1.0
    if bt["truncated"]["n"]:
        assert bt["truncated"]["mean_samples"] == pytest.approx(T_MAX)

    assert 0.0 <= res["task_accuracy"] <= 1.0
    assert G.T_MIN <= res["mean_samples"] <= T_MAX
    assert res["cost"]["passes_mean"] == pytest.approx(res["mean_samples"])
    assert res["cost"]["hidden_units"] == DIMS[1]
    assert "throttled_latency_passes_mean" in res["cost"], \
        "finite r0 must carry a throttled LATENCY equivalent"
    assert res["cost"]["throttled_latency_passes_mean"] == pytest.approx(
        res["mean_samples"] * G.R0_THROTTLED / r0)
    # audit r1 item 8: the slowdown is time, not bits or MACs
    for gone in ("throttled_random_bits_mean", "throttled_macs_mean"):
        assert gone not in res["cost"]
    assert res["cost"]["settle_latency_passes"] == pytest.approx(
        G.SETTLE_RATIO / r0)

    # audit r1 item 2: per-image arrays, and they must REPRODUCE the aggregates
    pi = res["per_image"]
    d_arr = np.asarray(pi["decisions"])
    s_arr = np.asarray(pi["samples_used"])
    f_arr = np.asarray(pi["fired"], dtype=bool)
    assert d_arr.shape == s_arr.shape == f_arr.shape == (N_IMAGES,)
    assert s_arr.mean() == pytest.approx(res["mean_samples"])
    assert (1.0 - f_arr.mean()) == pytest.approx(res["truncation_rate"])
    kp = ~np.asarray(tie, dtype=bool)
    mg = np.asarray(marginal, dtype=bool)
    assert ((d_arr[kp & mg] != np.asarray(ref)[kp & mg]).mean()
            == pytest.approx(se["marginal"]["stopping_error"]))
    assert 0.0 <= res["plurality_tie_rate"] <= 1.0
    assert (len(pi["plurality_tie_index"]) / N_IMAGES
            == pytest.approx(res["plurality_tie_rate"]))
    assert sorted(set(pi["plurality_tie_index"])) == pi["plurality_tie_index"]
    # the ACF is a DEVICE measurement over the marginal mask (reference ties
    # included — precheck2 convention), so it is bounded by the mask, not by
    # the tie-filtered stratum the stopping error is reported on
    acf = res["vote_lag1_acf"]
    assert acf["marginal_n_images"] <= int(marginal.sum())
    assert np.isnan(acf["marginal"]) or -1.0 <= acf["marginal"] <= 1.0

    payload = {"git_sha": "deadbeef", "phase": 2, "cell_id": "unit",
               "config": {"rule": rule, "rule_spec": spec}, **res}
    out = tmp_path / "unit.json"
    G.write_atomic(out, payload)
    assert out.exists()
    assert not (tmp_path / "unit.json.tmp").exists(), \
        "tmp must be renamed away"
    assert json.loads(out.read_text())["config"]["rule"] == rule


@pytest.mark.fast
def test_ideal_source_has_no_throttled_cost(tiny):
    """r0=inf has no clock to slow, so the throttled baseline is absent."""
    model, xs, ys, ref, tie, marginal = tiny
    spec = G.rule_spec("dirichlet", K, T_MAX)
    res = G.evaluate_cell(model, xs, ys, IdealSource(seed=1000), spec, ref,
                          tie, marginal, np.inf, t_max=T_MAX, lanes=LANES)
    res.pop("_out")
    assert "throttled_passes_mean" not in res["cost"]


@pytest.mark.fast
def test_fixed_n_never_fires(tiny):
    """FixedN has no verdict, so every decision is a truncation — the ladder's
    cells must report truncation_rate 1 and an empty fired channel."""
    model, xs, ys, ref, tie, marginal = tiny
    spec = G.rule_spec("fixed", K, T_MAX)
    res = G.evaluate_cell(model, xs, ys, IdealSource(seed=1000), spec, ref,
                          tie, marginal, np.inf, t_max=T_MAX, lanes=LANES)
    out = res.pop("_out")
    assert res["truncation_rate"] == 1.0
    assert res["by_termination"]["fired"]["n"] == 0
    assert res["mean_samples"] == pytest.approx(T_MAX)

    rungs = G.ladder_from_votes(out["votes"], out["labels"], ref, ~tie,
                                marginal, DIMS, np.inf, ns=(1, 4, T_MAX))
    assert [r["n_passes"] for r in rungs] == [1, 4, T_MAX]
    for r in rungs:
        assert r["cost"]["passes_mean"] == pytest.approx(r["n_passes"])
        assert (r["stopping_error"]["marginal"]["n"]
                + r["stopping_error"]["easy"]["n"]
                == r["stopping_error"]["all"]["n"])
    # The full-budget rung and the FixedN run it was decoded from see the SAME
    # vote counts, so they must agree on the tie rate exactly. They need not
    # agree on the tied images themselves: both break ties at random (audit r1
    # item 1) from independent seeded streams, so the accuracies can differ by
    # at most that tie rate.
    tie_rate = res["plurality_tie_rate"]
    assert rungs[-1]["plurality_tie_rate"] == pytest.approx(tie_rate)
    assert abs(rungs[-1]["task_accuracy"] - res["task_accuracy"]) <= tie_rate
    # ... and at N=1 every image is "tied" in the sense of a single vote? no:
    # a single vote gives a unique argmax, so N=1 must have no ties at all
    assert rungs[0]["plurality_tie_rate"] == 0.0


@pytest.mark.fast
def test_grid_coverage_and_size():
    """The v2 phase-2 grid: no rule-by-arm pruning, plus the new axes."""
    cells = G.phase2_cells()
    cids = [c["cid"] for c in cells]
    assert len(cids) == len(set(cids)), "cell ids must be unique"
    assert len(cells) < 1100, f"phase 2 has {len(cells)} cells, budget is 1100"
    assert len(cells) > 600, \
        "suspiciously small: a coverage rule is too greedy"
    # every main rule, plus the on-chip-rho variant of convergence review 1
    # item 8 (which runs on the primary arm only — see the dedicated test)
    assert {c["config"]["rule"] for c in cells} == set(G.ALL_RULES)
    assert {G.r0_tag(c["r0"]) for c in cells} == {G.r0_tag(r) for r in G.R0S}
    # ideal cells are shared across arms, never duplicated per device class
    ideal = [c for c in cells if np.isinf(c["r0"])]
    assert {c["config"]["device_class"] for c in ideal} == {"ideal"}
    # AUDIT R1 ITEM 6: arrhenius+streaming now carries EVERY rule
    arr_stream = [c for c in cells
                  if c["config"]["device_class"] == "arrhenius"
                  and c["config"]["mode"] == "streaming"]
    assert {c["config"]["rule"] for c in arr_stream} == set(G.RULES)
    # three train seeds only on the primary arm
    prim = {c["config"]["train_seed"] for c in cells
            if (c["config"]["device_class"], c["config"]["mode"])
            == G.PRIMARY_ARM and c["config"]["dataset"] == "mnist"}
    assert prim == set(G.TRAIN_SEEDS)
    assert {c["config"]["train_seed"] for c in cells
            if c["config"]["dataset"] == "fashion"} == {0}


@pytest.mark.fast
def test_audit_r1_new_axes_are_enumerated_and_named_apart():
    """Items 4, 5 and 7: the rate-ceiling arm, the sigma_delta column and the
    t_min / settle_ratio sensitivity rows exist, and their cell ids never
    collide with the default rows (a variant that overwrote a default would be
    invisible in the figures)."""
    cells = {c["cid"]: c for c in G.phase2_cells()}
    cap = [c for c in cells.values()
           if c["config"]["device_class"] == G.CAP_ARM[0]]
    assert len(cap) == len(G.CAP_RULES) * len(G.CAP_R0S) * 2
    assert {G.r0_tag(c["r0"]) for c in cap} == {G.r0_tag(r)
                                               for r in G.CAP_R0S}
    sd = [c for c in cells.values() if c["config"]["sigma_delta"] > 0]
    assert len(sd) == (len(G.SIGMA_DELTAS) * len(G.SIGMA_CHIP_SEEDS)
                       * len(G.SIGMA_RULES) * len(G.SIGMA_R0S))
    assert {c["config"]["chip_seed"] for c in sd} == set(G.SIGMA_CHIP_SEEDS)
    assert {c["config"]["centering"] for c in sd} == {"mean-tau"}
    tm = [c for c in cells.values() if c["config"]["t_min"] != G.T_MIN]
    assert {c["config"]["t_min"] for c in tm} == set(G.T_MIN_SENS)
    sr = [c for c in cells.values()
          if c["config"]["settle_ratio"] != G.SETTLE_RATIO]
    assert {c["config"]["settle_ratio"] for c in sr} == set(G.SETTLE_RATIO_SENS)
    for group in (sd, tm, sr):
        for c in group:
            assert G.variant_tag(
                c["config"]["sigma_delta"], c["config"]["chip_seed"],
                c["config"]["t_min"], c["config"]["settle_ratio"]) in c["cid"]
    # every cell records every knob, default or not (item 7)
    for c in cells.values():
        for key in ("sigma_delta", "centering", "t_min", "settle_ratio",
                    "chip_seed"):
            assert key in c["config"], (c["cid"], key)


@pytest.mark.fast
def test_onchip_rho_variant_is_the_same_rule_on_a_different_measurement():
    """Convergence review 1 item 8.

    The variant must (a) be enumerated exactly where the note says — primary arm,
    r0 in {1, 0.5}, 3 train seeds x 2 noise seeds = 12 cells — (b) build the
    IDENTICAL rule spec as markov_fixedrho given the same rho, so the comparison
    isolates the measurement rather than the rule, and (c) read the
    all-fluctuating-inputs ACF instead of the marginal-mask one.
    """
    cells = [c for c in G.phase2_cells()
             if c["config"]["rule"] == G.ONCHIP_RULE]
    assert len(cells) == 12
    assert {(c["config"]["device_class"], c["config"]["mode"])
            for c in cells} == {G.PRIMARY_ARM}
    assert {G.r0_tag(c["r0"]) for c in cells} == {G.r0_tag(r)
                                                 for r in G.ONCHIP_R0S}
    assert {c["config"]["train_seed"] for c in cells} == set(G.TRAIN_SEEDS)
    assert {c["config"]["noise_seed"] for c in cells} == set(
        G.ONCHIP_NOISE_SEEDS)
    assert all(c["config"]["dataset"] == "mnist" for c in cells)

    spec_a = G.rule_spec("markov_fixedrho", K, T_MAX, rho=0.3)
    spec_b = G.rule_spec(G.ONCHIP_RULE, K, T_MAX, rho=0.3)
    assert spec_a == spec_b, "same rule, different measurement"
    assert G.rho_field("markov_fixedrho") == "marginal_vote_lag1_acf"
    assert G.rho_field(G.ONCHIP_RULE) == "all_vote_lag1_acf"
    # the clamp is shared, so a difference between the two flavours is the ACF's
    assert G.clamp_rho(-0.2) == 0.0
    assert G.clamp_rho(float("nan")) == 0.0
    assert G.clamp_rho(0.999) == pytest.approx(0.995)
    assert G.clamp_rho(0.3) == pytest.approx(0.3)


@pytest.mark.fast
def test_accepted_code_hashes_keep_protocol_identical_cells_fresh(tmp_path,
                                                                 monkeypatch):
    """Convergence review 1 item 8: an additive driver edit must not invalidate
    a grid that took a day to run, but an UNLISTED hash still must."""
    monkeypatch.setattr(G, "GRID", tmp_path)
    legacy = G.ACCEPTED_CODE_HASHES[0]
    assert legacy != G.code_hash(), \
        "the listed hash is a PAST hash; the current one is accepted anyway"
    assert G.accepted_code_hashes() == {G.code_hash(), *G.ACCEPTED_CODE_HASHES}
    G.write_atomic(G.cell_path("unit_legacy"),
                   {"cell_id": "unit_legacy", "code_hash": legacy})
    assert G.fresh_result("unit_legacy", G.code_hash()) is not None
    G.write_atomic(G.cell_path("unit_unlisted"),
                   {"cell_id": "unit_unlisted", "code_hash": "f" * 64})
    assert G.fresh_result("unit_unlisted", G.code_hash()) is None
    assert all(len(h) == 64 for h in G.ACCEPTED_CODE_HASHES)


@pytest.mark.fast
def test_other_phase_sizes():
    for phase in ("0", "1", "2", "3", "4"):
        tasks = G.tasks_for_phase(phase)
        assert tasks, f"phase {phase} enumerated nothing"
        cids = [t["cid"] for t in tasks]
        assert len(cids) == len(set(cids))
    assert len(G.phase3_cells()) < 120
    assert len(G.phase4_cells()) == 6
    # phase 2a covers exactly the device conditions fixedrho needs, including
    # the sigma_delta / chip-seed / settle_ratio variants (audit r1 items 5, 7)
    need = {G.rho_key(c["config"]["device_class"], c["config"]["mode"],
                      c["r0"], c["config"]["sigma_delta"],
                      c["config"]["chip_seed"], c["config"]["settle_ratio"])
            for c in G.phase2_cells()
            if c["config"]["rule"] in G.FIXEDRHO_RULES}
    have = {G.rho_key(t["device_class"], t["mode"], t["r0"],
                      t["sigma_delta"], t["chip_seed"], t["settle_ratio"])
            for t in G.phase2a_tasks()}
    assert need == have


@pytest.mark.fast
def test_dry_run_lists_without_running(capsys):
    assert G.main(["--phase", "2", "--dry-run", "--only", "dirichlet"]) == 0
    text = capsys.readouterr().out
    assert "planned cells" in text
    assert "p2_mnist_ts0_easyplane-settle_r1_dirichlet_ns1000" in text
    assert "markov" not in text, "--only must filter"


# ---------------------------------------------------------------------------
# audit r1 item 3 — split-sample reference
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_split_reference_separates_the_stratum_from_the_target(tiny):
    """Block A must define the stratum and block B the scoring target, and the
    two must be independent draws — that independence is the whole point of the
    fix, so the test checks the wiring, not just the shapes."""
    model, xs, ys, *_ = tiny
    _, _, ca = reference_decisions(model, xs, IdealSource(seed=G.REF_SEED_A),
                                  t_ref=T_REF, batch=N_IMAGES,
                                  return_counts=True)
    _, _, cb = reference_decisions(model, xs, IdealSource(seed=G.REF_SEED_B),
                                   t_ref=T_REF, batch=N_IMAGES,
                                   return_counts=True)
    assert not np.array_equal(ca, cb), "the two blocks must be independent"
    sp = split_reference(ca, cb, T_REF, G.MARGIN_CUT)

    assert np.array_equal(sp["reference"], cb.argmax(1)), "target = block B"
    order_a = np.sort(ca, axis=1)
    assert np.array_equal(
        sp["marginal"],
        ((order_a[:, -1] - order_a[:, -2]) / T_REF) < G.MARGIN_CUT), \
        "stratum = block A"
    order_b = np.sort(cb, axis=1)
    assert np.array_equal(sp["reference_tie"],
                          order_b[:, -1] == order_b[:, -2])
    assert np.array_equal(sp["top_counts_a"], order_a[:, -1])
    # the cross-block agreement rate is the measured residual reference noise
    assert 0.0 <= sp["blocks_agree_frac"] <= 1.0
    assert sp["blocks_agree_frac_marginal"] <= sp["blocks_agree_frac"] + 1e-12


# ---------------------------------------------------------------------------
# audit r1 item 10 — content-hash freshness and --verify
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_code_hash_covers_the_numerics_sources_only(tmp_path):
    files = G.code_files()
    names = {f.name for f in files}
    assert "stopping.py" in names and "smtj.py" in names
    assert "run_grid.py" in names
    assert not any(f.suffix != ".py" for f in files)
    h = G.code_hash()
    assert len(h) == 64 and h == G.code_hash(), "hash must be deterministic"


@pytest.mark.fast
def test_fresh_result_rejects_v1_cells_and_wrong_hashes(tmp_path,
                                                       monkeypatch):
    monkeypatch.setattr(G, "GRID", tmp_path)
    chash = G.code_hash()
    # a v1 cell: git-SHA-stamped, no code_hash -> INCOMPATIBLE, never reused
    G.write_atomic(G.cell_path("unit_v1"),
                   {"cell_id": "unit_v1", "git_sha": "a2e635a" * 5})
    assert G.fresh_result("unit_v1", chash) is None
    G.write_atomic(G.cell_path("unit_stale"),
                   {"cell_id": "unit_stale", "code_hash": "0" * 64})
    assert G.fresh_result("unit_stale", chash) is None
    G.write_atomic(G.cell_path("unit_fresh"),
                   {"cell_id": "unit_fresh", "code_hash": chash})
    assert G.fresh_result("unit_fresh", chash) is not None
    # unreadable file is not a cache hit either
    (tmp_path / "unit_broken.json").write_text('{"cell_id": "unit_b')
    assert G.fresh_result("unit_broken", chash) is None


@pytest.mark.fast
def test_diff_payload_ignores_volatile_stamps_but_catches_numbers():
    a = {"code_hash": "x", "git_sha": "1", "wall_seconds": 1.0,
         "stopping_error": {"marginal": {"n": 700, "stopping_error": 0.04}},
         "per_image": {"decisions": [1, 2, 3]}}
    b = json.loads(json.dumps(a))
    b["code_hash"], b["git_sha"], b["wall_seconds"] = "y", "2", 99.0
    assert G.diff_payload(a, b) == []
    b["stopping_error"]["marginal"]["stopping_error"] = 0.041
    assert any("stopping_error" in d for d in G.diff_payload(a, b))
    b = json.loads(json.dumps(a))
    b["per_image"]["decisions"] = [1, 2, 4]
    assert any("decisions[2]" in d for d in G.diff_payload(a, b))
    b = json.loads(json.dumps(a))
    del b["per_image"]
    assert any("per_image" in d for d in G.diff_payload(a, b))


@pytest.mark.fast
def test_verify_reports_a_missing_or_unknown_cell(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "GRID", tmp_path)
    msgs = []
    assert G.verify_cell("p2_mnist_ts0_easyplane-settle_r1_dirichlet_ns1000",
                         G.code_hash(), log=msgs.append) == 2
    assert "NOT ON DISK" in msgs[-1]
    G.write_atomic(G.cell_path("not_a_grid_cell"), {"cell_id": "x"})
    assert G.verify_cell("not_a_grid_cell", G.code_hash(),
                         log=msgs.append) == 2
    assert "not enumerated" in msgs[-1]


# ---------------------------------------------------------------------------
# audit r1 item 15 — the vote stream's correlation beyond lag 1
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_network_vote_stream_is_correlated_beyond_lag_one():
    """The Markov rule models lag-1 only, so the honest question is whether the
    NETWORK's vote stream (not a synthetic chain) has correlation at k > 1.
    Under a correlated device it must, and it must decay with k; under the
    ideal source it must vanish at every lag."""
    torch.manual_seed(0)
    model = SBNN((16, 12, 3))
    model.eval()
    g = torch.Generator().manual_seed(5)
    xs = torch.rand(60, 16, generator=g)
    ys = torch.randint(0, 3, (60,), generator=g)

    def votes_for(source):
        out = evaluate_lanes_adaptive(
            model, xs, ys, source, lambda: FixedN(96, k_classes=3),
            t_max=96, lanes=30, collect_votes=True)
        return out["votes"]

    corr = vote_acf_curve(votes_for(TelegraphSource(
        r0=0.25, streaming=True, noise_seed=3)), max_lag=8)
    ideal = vote_acf_curve(votes_for(IdealSource(seed=3)), max_lag=8)
    c = corr["mean_acf"]
    assert corr["n_images_fluctuating"] > 10
    assert c[0] > 0.2, c[:4]
    assert c[1] > 0.05, c[:4]                  # memory beyond lag 1
    assert c[0] > c[1] > c[3], c[:4]           # and it decays
    i = ideal["mean_acf"]
    assert abs(i[0]) < 0.12 and abs(i[1]) < 0.12, i[:4]
