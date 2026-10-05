# Author: Manan Gupta <mnn@yogins.com>
"""Plumbing tests for the figure script (experiments/make_figures.py).

Same spirit as test_grid.py: not science, bookkeeping. What is checked is that
the one script that makes every paper figure (a) runs to completion against a
still-filling grid without inventing data, (b) emits the full PDF set plus a
manifest naming every expected-but-absent cell, (c) refuses to treat a
half-written or corrupt cell file as data, (d) computes ECE the way the frozen
convention says (equal-mass bins), and (e) implements the audit-mandated
statistics: calibration-referenced normalization, alpha/(1-beta) for the SPRTs,
cluster-robust CIs over train seeds, and the default-knob filter that keeps the
sensitivity rows out of the main-text curves.

Every panel is exercised against a SYNTHETIC grid directory built from
run_grid's own cell ids and the v2 payload schema, so the code paths are tested
before the real cells land. The v1 grid (results/grid, older schema) is also fed
in as a robustness case: an older cell must degrade to "pending", never crash
the script.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

import make_figures as MF  # noqa: E402
import run_grid as G  # noqa: E402

PDFS = ("F1_headline_stopping_error.pdf", "F2_matched_guarantee_cost.pdf",
        "F3_accuracy_cost_pareto.pdf", "F4_calibration.pdf",
        "F5_mechanism.pdf", "F6_matched_budget_excess.pdf",
        "F7_fashion_replication.pdf", "F8a_ablations_rule.pdf",
        "F8b_ablations_device.pdf")
FIG_KEYS = {"F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"}
N_SYNTH = 400


# ---------------------------------------------------------------------------
# unit checks on the statistics the audit mandated
# ---------------------------------------------------------------------------

def test_reliability_is_equal_mass_and_scores_known_cases():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, size=3000)
    correct = (rng.uniform(size=conf.size) < conf).astype(float)
    bc, ba, bw, ece = MF.reliability(conf, correct)
    assert len(bc) == MF.ECE_BINS
    assert np.allclose(bw.sum(), 1.0)
    assert bw.max() - bw.min() <= 1.0 / conf.size + 1e-9  # equal mass
    assert ece < 0.05                                     # calibrated by build

    # maximally overconfident: always sure, never right.
    _, _, _, bad = MF.reliability(np.ones(200), np.zeros(200))
    assert bad == pytest.approx(1.0)
    assert np.isnan(MF.reliability([], [])[3])


def test_summarize_ignores_missing_and_reports_n():
    assert MF.summarize([]) is None
    s = MF.summarize([1.0, None, 3.0, float("nan")])
    assert (s["n"], s["mean"], s["min"], s["max"]) == (2, 2.0, 1.0, 3.0)


def test_guarantee_filter_uses_the_wilson_upper_bound():
    hi_ok = {"stopping_error": {"marginal":
             {"stopping_error_wilson95": [0.01, MF.ALPHA - 1e-6]}}}
    hi_bad = {"stopping_error": {"marginal":
              {"stopping_error_wilson95": [0.01, MF.ALPHA + 1e-6]}}}
    assert MF.guarantee_holds(hi_ok)
    assert not MF.guarantee_holds(hi_bad)


def test_sprt_rules_are_scored_against_alpha_over_one_minus_beta():
    """AUDIT R1: the SPRTs never had an alpha bound; theirs is a/(1-beta)."""
    assert MF.nominal_denominator("dirichlet") == pytest.approx(MF.ALPHA)
    assert MF.nominal_denominator("eprocess") == pytest.approx(MF.ALPHA)
    for rule in ("wald", "markov_online", "markov_fixedrho"):
        assert MF.nominal_denominator(rule) == pytest.approx(
            MF.ALPHA / (1.0 - MF.BETA))


def test_cluster_summarize_uses_t_of_2_on_three_train_seed_means():
    """The replicate unit is the model. With 3 clusters the interval is the
    cluster-mean t(2) interval; with 1 it must fall back to min-max and SAY so.
    """
    def cell(seed, v):
        return {"config": {"train_seed": seed}, "v": v}
    payloads = [cell(0, 1.0), cell(0, 3.0), cell(1, 2.0), cell(2, 6.0)]
    s = MF.cluster_summarize(payloads, lambda p: p["v"])
    means = np.array([2.0, 2.0, 6.0])          # cluster means
    se = means.std(ddof=1) / np.sqrt(3)
    assert s["n_clusters"] == 3 and s["n"] == 4
    assert s["mean"] == pytest.approx(means.mean())
    assert s["hi"] - s["mean"] == pytest.approx(MF.T_DF2 * se)
    assert "cluster-t(2)" in s["ci_kind"]
    # the interval must be WIDER than a naive per-cell min-max here
    assert s["lo"] < 1.0 and s["hi"] > 6.0

    one = MF.cluster_summarize([cell(0, 1.0), cell(0, 3.0)],
                               lambda p: p["v"])
    assert one["n_clusters"] == 1 and one["lo"] == 1.0 and one["hi"] == 3.0
    assert "min-max" in one["ci_kind"] and "1 train seed" in one["ci_kind"]
    assert MF.cluster_summarize([], lambda p: p["v"]) is None
    assert MF.cluster_summarize([cell(0, None)], lambda p: p["v"]) is None


def test_default_knob_filter_excludes_the_sensitivity_rows():
    """A t_min=2 or sigma_delta=1.0 cell shares arm, rule and r0 with a default
    cell; if `is_default` let it through, F1 would average the two."""
    base = {"phase": 2, "dataset": "mnist", "device_class": "easyplane",
            "mode": "settle", "r0": "0.5", "rule": "dirichlet",
            "sigma_delta": 0.0, "t_min": G.T_MIN,
            "settle_ratio": G.SETTLE_RATIO}
    pred = MF.primary_pred("dirichlet", 0.5)
    assert MF.is_default(base) and pred(base)
    for knob, value in (("sigma_delta", 1.0), ("t_min", 2),
                        ("settle_ratio", 8.0)):
        variant = {**base, knob: value}
        assert not MF.is_default(variant)
        assert not pred(variant)
        # ... but an explicit filter on that knob finds it
        assert MF.cell_pred(phase=2, arm=MF.PRIMARY_ARM, r0=0.5,
                            rule="dirichlet", defaults=False,
                            **{**MF.DEFAULT_KNOBS, knob: value})(variant)


def test_frontier_interpolation_is_log_linear_and_clamped():
    frontier = [(4, 0.10), (16, 0.05), (64, 0.02)]
    assert MF.frontier_at(frontier, 4) == pytest.approx(0.10)
    assert MF.frontier_at(frontier, 8) == pytest.approx(0.075)   # log midpoint
    assert MF.frontier_at(frontier, 1) == pytest.approx(0.10)    # clamped low
    assert MF.frontier_at(frontier, 256) == pytest.approx(0.02)  # clamped high
    assert MF.frontier_at([], 8) is None
    assert MF.frontier_at(frontier, None) is None


# ---------------------------------------------------------------------------
# synthetic v2 grid
# ---------------------------------------------------------------------------

def _p1_payload(dataset, ts, rng):
    """A split-sample reference cell: block A margins, block B decisions."""
    t_ref = G.T_REF
    ref = rng.integers(0, 10, size=N_SYNTH)
    margin_a = rng.uniform(0.0, 1.0, size=N_SYNTH)
    top_a = np.clip(margin_a + rng.uniform(0.0, 0.2, size=N_SYNTH), 0, 1)
    tie = np.zeros(N_SYNTH, dtype=int)
    tie[:3] = 1
    return {
        "code_hash": G.code_hash(), "git_sha": "deadbeef",
        "cell_id": G.ref_cid(dataset, ts, "test"), "phase": 1,
        "config": {"dataset": dataset, "train_seed": ts, "split": "test",
                   "t_ref": t_ref, "margin_cut": G.MARGIN_CUT},
        "reference": [int(v) for v in ref],
        "reference_tie": [int(v) for v in tie],
        "margin_counts": [int(v) for v in np.rint(margin_a * t_ref)],
        "top_counts_block_a": [int(v) for v in np.rint(top_a * t_ref)],
        # block A's own decision: needed by the floor-free (agree-substratum)
        # statistics of convergence review 1 item 3. Agrees with block B on most
        # images, disagrees on a few, exactly like the real cells.
        "decision_block_a": [int(v) if i % 20 else int((v + 1) % 10)
                             for i, v in enumerate(ref)],
        "blocks_agree_frac": 0.99, "blocks_agree_frac_marginal": 0.95,
        "reference_label_accuracy": 0.98, "reference_tie_count": 3,
        "margin_summary": {"marginal_count": int((margin_a
                                                  < G.MARGIN_CUT).sum())},
    }


def _p2_payload(cid, cfg, r0, rng):
    """A v2 phase-2 cell: aggregates, diagnostics AND per-image arrays.

    The marginal error is built from the fired + truncated wrong counts exactly
    as a real cell reports them, and the per-image decisions are made to
    reproduce that same error against the synthetic reference, so the tests can
    check the post-hoc path against the aggregate path.
    """
    n_m = 700
    err = (650 * 0.02 + 50 * 0.2) / n_m
    trunc = {"markov_online": 0.83}.get(cfg["rule"], 0.01)
    lag1 = 0.0 if r0 == "inf" else 0.45
    acf = [lag1 * (0.55 ** k) for k in range(G.ACF_MAX_LAG)]
    rho = ({"rho_used": 0.19 if cfg["rule"] == G.ONCHIP_RULE else 0.27,
            "rho_field": G.rho_field(cfg["rule"])}
           if cfg["rule"] in G.FIXEDRHO_RULES else {})
    return {
        "code_hash": G.code_hash(), "git_sha": "deadbeef",
        "cell_id": cid, "phase": 2,
        "config": {**cfg, "r0": r0, **rho},
        "n_images": N_SYNTH, "mean_samples": 12.5, "task_accuracy": 0.97,
        "truncation_rate": trunc, "plurality_tie_rate": 0.004,
        "stopping_error": {
            "all": {"n": 990, "stopping_error": err / 2,
                    "stopping_error_over_alpha": err / 2 / MF.ALPHA,
                    "task_accuracy": 0.97, "mean_samples": 12.5},
            "marginal": {"n": n_m, "stopping_error": err,
                         "stopping_error_over_alpha": err / MF.ALPHA,
                         "stopping_error_wilson95": [0.02, 0.045],
                         "task_accuracy": 0.8, "mean_samples": 30.0},
            "easy": {"n": 990 - n_m, "stopping_error": 0.0,
                     "stopping_error_over_alpha": 0.0,
                     "task_accuracy": 0.99, "mean_samples": 6.0}},
        "by_termination": {
            "fired_marginal": {"n": 650, "stopping_error": 0.02},
            "truncated_marginal": {"n": 50, "stopping_error": 0.2}},
        "vote_lag1_acf": {"marginal": lag1, "marginal_n_images": n_m},
        "vote_diagnostics_marginal": {
            "acf_curve": {"lags": list(range(1, G.ACF_MAX_LAG + 1)),
                          "mean_acf": acf, "n_images_fluctuating": 300},
            "inflation": {"if_true": 1 + 2 * sum(a for a in acf if a > 0),
                          "if_ar1": (1 + lag1) / (1 - lag1),
                          "ratio_true_over_ar1": 1.02, "lag1": lag1},
            "entropy_rate": {"entropy_rate_bits": 0.8,
                             "iid_entropy_bits": 0.95},
            "rho_histogram": {"mean": lag1, "n_images": 300}},
        "cost": {"passes_mean": 12.5, "random_bits_mean": 4800.0,
                 "throttled_latency_passes_mean": 100.0,
                 "settle_latency_passes": 8.0},
        "per_image": {
            "decisions": [int(v) for v in rng.integers(0, 10, size=N_SYNTH)],
            "samples_used": [12] * N_SYNTH,
            "fired": [1] * N_SYNTH,
            "plurality_tie_index": []},
    }


def _p3_payload(cid, cfg, r0):
    return {
        "code_hash": G.code_hash(), "git_sha": "deadbeef",
        "cell_id": cid, "phase": 3, "config": {**cfg, "r0": r0},
        "ladder": [{"n_passes": n, "task_accuracy": 0.90 + 0.02 * i / 8,
                    "plurality_tie_rate": 0.3 / (i + 1),
                    "stopping_error": {"marginal": {
                        "n": 700, "stopping_error": 0.12 / (i + 1),
                        "task_accuracy": 0.70 + 0.03 * i / 8,
                        "mean_samples": float(n)}}}
                   for i, n in enumerate(G.LADDER)],
    }


def _p4_payload(cid, cfg, r0, rng):
    n = N_SYNTH
    panels = []
    for n_passes in G.CALIB_NS:
        top = rng.integers(n_passes // 2 + 1, n_passes + 1, size=n)
        correct = (rng.uniform(size=n) < top / n_passes).astype(int)
        panels.append({"n_passes": n_passes,
                       "top_counts": [int(v) for v in top],
                       "correct": [int(v) for v in correct],
                       "decision": [0] * n, "agrees_reference": [1] * n,
                       "plurality_tie_rate": 0.01})
    return {"code_hash": G.code_hash(), "git_sha": "deadbeef",
            "cell_id": cid, "phase": 4, "config": {**cfg, "r0": r0},
            "n_images": n, "panels": panels}


@pytest.fixture(scope="module")
def synthetic_grid(tmp_path_factory):
    """REAL cell ids with plausible v2 payloads, plus one corrupt file, so every
    figure path (including the ablations and the post-hoc panel) executes."""
    d = tmp_path_factory.mktemp("grid2")
    rng = np.random.default_rng(1)
    for ds in ("mnist", "fashion"):
        for ts in G.TRAIN_SEEDS:
            (d / f"{G.ref_cid(ds, ts, 'test')}.json").write_text(
                json.dumps(_p1_payload(ds, ts, rng)))
    for cell in G.phase2_cells():
        cfg = cell["config"]
        arms = (MF.PRIMARY_ARM, ("ideal", "ideal"), G.CAP_ARM)
        if ((cfg["device_class"], cfg["mode"]) in arms
                and cfg["noise_seed"] == 1000):
            (d / f"{cell['cid']}.json").write_text(json.dumps(
                _p2_payload(cell["cid"], cfg, G.r0_tag(cell["r0"]), rng)))
    for cell in G.phase3_cells():
        cfg = cell["config"]
        if cfg["dataset"] == "mnist" and cfg["noise_seed"] == 1000:
            (d / f"{cell['cid']}.json").write_text(json.dumps(
                _p3_payload(cell["cid"], cfg, G.r0_tag(cell["r0"]))))
    for cell in G.phase4_cells():
        (d / f"{cell['cid']}.json").write_text(json.dumps(
            _p4_payload(cell["cid"], cell["config"], G.r0_tag(cell["r0"]),
                        rng)))
    # a cell caught mid-write / truncated by a kill -9
    (d / "p2_mnist_ts9_broken_r1_dirichlet_ns1000.json").write_text(
        '{"cell_id": "p2_mnist_ts9_broken_r1_dirichl')
    return d


def test_synthetic_grid_exercises_every_panel(synthetic_grid, tmp_path):
    out = tmp_path / "figs"
    assert MF.main(["--grid-dir", str(synthetic_grid),
                    "--out-dir", str(out)]) == 0
    manifest = json.loads((out / "figures_manifest.json").read_text())

    # the truncated file is pending, never data
    assert manifest["cells_unreadable_or_pending"] == [
        "p2_mnist_ts9_broken_r1_dirichlet_ns1000.json"]
    assert not manifest["grid_complete"]
    assert manifest["n_cells_missing_union"] > 0
    assert set(manifest["figures"]) == FIG_KEYS
    assert manifest["code_hash"] == G.code_hash()
    assert manifest["grid_dir"] == str(synthetic_grid)   # outside the repo

    # F1(a): every peer rule normalized against its own ideal calibration, so
    # the r0=inf point is exactly 1.0 by construction
    f1 = manifest["figures"]["F1"]["panels"]
    for rule, row in f1["a_calibration_referenced"]["series"].items():
        assert row["inf"]["mean"] == pytest.approx(1.0)
        assert f1["a_calibration_referenced"]["calibration"][rule][
            "ideal_marginal_error"] > 0
    # F1(b): SPRT denominators
    raw = f1["b_raw_over_own_nominal"]["series"]
    assert raw["wald"]["nominal_denominator"] == pytest.approx(
        MF.ALPHA / (1 - MF.BETA))
    assert raw["dirichlet"]["nominal_denominator"] == pytest.approx(MF.ALPHA)
    # F1(c): marginal-stratum accuracy, not all-image accuracy
    acc = f1["c_marginal_fixed_n_accuracy"]["series"]
    assert set(acc) == {str(n) for n in MF.OVERLAY_NS}
    assert acc["4"]["inf"]["mean"] < 0.95

    # F3 got its fixed-N reference curve and truncation annotations
    f3 = manifest["figures"]["F3"]["panels"]["pareto"]["panels"]
    assert len(f3["0.5"]["fixed_n_ladder"]) == len(G.LADDER)
    assert f3["0.5"]["adaptive_rules"]["dirichlet"]["truncation_rate"]["mean"] \
        == pytest.approx(0.01)

    # F4 computed an ECE per fixed-N readout at both r0 slices
    for tag in ("inf", "0.5"):
        by_n = manifest["figures"]["F4"]["panels"]["reliability"]["panels"][
            tag]["by_n"]
        assert set(by_n) == {str(n) for n in G.CALIB_NS}
        for entry in by_n.values():
            assert 0.0 <= entry["pooled_ece"] <= 1.0
            assert len(entry["bin_confidence"]) == MF.ECE_BINS

    # F5: the ACF curve panel carries the full lag range, and the stacked
    # decomposition adds up to the total it decomposes
    acf = manifest["figures"]["F5"]["panels"]["a_acf_curve"]["series"]
    assert len(acf["0.5"]["mean_acf"]) == G.ACF_MAX_LAG
    # the per-arm lag-1 panel (kept from v1) must cover every arm plus the
    # shared ideal control
    by_arm = manifest["figures"]["F5"]["panels"]["b_vote_acf_by_arm"]["series"]
    for arm in G.ARMS + (G.CAP_ARM,):
        assert "-".join(arm) in by_arm
    assert any("ideal source" in k for k in by_arm)
    dec = manifest["figures"]["F5"]["panels"][
        "d_error_decomposition"]["series"]
    for rule, d in dec.items():
        if not d["marginal_stopping_error"]:
            continue
        total = (d["fired_contribution"]["mean"]
                 + d["truncated_contribution"]["mean"])
        assert total == pytest.approx(d["marginal_stopping_error"]["mean"],
                                      abs=1e-9)

    # F6: the matched-budget frontier came from the IDEAL ladder
    f6 = manifest["figures"]["F6"]["panels"]["matched_budget_excess"]
    assert f6["ideal_frontier"]
    assert {e["n_passes"] for e in f6["ideal_frontier"]} == set(G.LADDER)
    assert f6["series"]["dirichlet"]["0.5"]["mean"] > 0

    # F8: markov_online lives here, and the post-hoc panel used the per-image
    # arrays plus the phase-1 split reference (no extra cells)
    f8 = manifest["figures"]["F8"]["panels"]["ablations"]["panels"]
    assert f8["markov_online"]["markov_online"]["truncation_rate"]["0.5"][
        "mean"] == pytest.approx(0.83)
    assert f8["posthoc"]["available"]
    cuts = f8["posthoc"]["margin_cuts"]["dirichlet"]["0.5"]
    assert set(cuts) == {"0.8", "0.9", "0.95"}
    assert all(v is None or 0.0 <= v["mean"] <= 1.0 for v in cuts.values())
    assert f8["posthoc"]["q_restriction"]["dirichlet"]["0.5"]["error"]
    assert f8["rate_ceiling_arm"]

    for name in PDFS:
        assert (out / name).read_bytes()[:4] == b"%PDF"

    captions = (out / "captions_draft.md").read_text()
    for key in FIG_KEYS:
        assert f"## {key}" in captions
    assert "split-sample" in captions
    assert "cluster-t(2)" in captions


def test_only_subset_still_writes_a_consistent_manifest_and_captions(
        synthetic_grid, tmp_path):
    """--only must not leave the manifest and the caption file disagreeing
    (the caption writer once assumed all figures were always rebuilt)."""
    out = tmp_path / "figs"
    assert MF.main(["--grid-dir", str(synthetic_grid), "--out-dir", str(out),
                    "--only", "F2", "F4"]) == 0
    manifest = json.loads((out / "figures_manifest.json").read_text())
    assert set(manifest["figures"]) == {"F2", "F4"}
    captions = (out / "captions_draft.md").read_text()
    assert "## F2" in captions and "## F4" in captions
    assert "## F1" not in captions
    assert (out / "F2_matched_guarantee_cost.pdf").exists()
    assert not (out / "F1_headline_stopping_error.pdf").exists()


# ---------------------------------------------------------------------------
# the real grids, through the command line
# ---------------------------------------------------------------------------

@pytest.mark.fast
@pytest.mark.parametrize("grid", ["results/grid2", "results/grid"])
def test_runs_clean_on_a_real_grid_directory(tmp_path, grid):
    """results/grid2 is the v2 grid (possibly empty while the re-run is
    pending); results/grid is the audited v1 record, whose cells predate the
    per-image arrays and the split reference. Both must produce the full figure
    set with the missing pieces declared, and neither may crash the script."""
    out = tmp_path / "figs"
    # snapshot before and after: the grid may legitimately be filling while the
    # figures are drawn (that is the scenario the defensive reads exist for), so
    # only cells present on BOTH sides can be held against the manifest
    before = {p.stem for p in (ROOT / grid).glob("*.json")}
    proc = subprocess.run(
        [sys.executable, str(EXPERIMENTS / "make_figures.py"),
         "--grid-dir", grid, "--out-dir", str(out)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr

    for name in PDFS:
        pdf = out / name
        assert pdf.exists(), f"{name} not written"
        assert pdf.read_bytes()[:4] == b"%PDF"
        assert pdf.stat().st_size > 2000
    assert (out / "captions_draft.md").exists()

    manifest = json.loads((out / "figures_manifest.json").read_text())
    assert manifest["git_sha"]
    assert set(manifest["figures"]) == FIG_KEYS
    assert manifest["grid_dir"] == grid, "grid_dir must be repo-relative"
    for key, rec in manifest["figures"].items():
        assert rec["file"] in PDFS
        # a split figure (F8) lists every PDF it wrote, not just the first
        assert set(rec.get("files", [rec["file"]])) <= set(PDFS)
        assert rec["n_cells_used"] == len(rec["cells_used"])
        assert rec["n_cells_missing"] == len(rec["cells_missing"])
        assert rec["complete"] == (not rec["cells_missing"])
        assert set(rec["cells_used"]).isdisjoint(rec["cells_missing"])

    # the manifest's missing list must agree with what is actually on disk
    on_disk = before & {p.stem for p in (ROOT / grid).glob("*.json")}
    for rec in manifest["figures"].values():
        assert not (set(rec["cells_missing"]) & on_disk), \
            "a cell listed as missing is present on disk"

    # while the grid is still filling the manifest MUST say so, loudly
    expected = set(MF.expected_index())
    if expected - on_disk:
        assert manifest["n_cells_missing_union"] > 0
        assert not manifest["grid_complete"]
        assert "PARTIAL" in proc.stderr
    else:
        assert manifest["n_cells_missing_union"] == 0

    # captions must carry the cell accounting, not just prose
    captions = (out / "captions_draft.md").read_text()
    for key in FIG_KEYS:
        assert f"## {key}" in captions
    assert "cells consumed" in captions


@pytest.mark.fast
def test_cells_from_another_code_hash_are_flagged_not_hidden(tmp_path):
    """A grid half-recomputed after a source change would otherwise average two
    protocols together silently. The cells are still plotted (excluding them
    would turn a partial re-run into unexplained empty panels), but they must be
    counted in the manifest and shouted about on stderr."""
    grid = tmp_path / "grid2"
    grid.mkdir()
    rng = np.random.default_rng(2)
    cell = G.phase2_cells()[0]
    payload = _p2_payload(cell["cid"], cell["config"], G.r0_tag(cell["r0"]),
                          rng)
    payload["code_hash"] = "0" * 64                     # another protocol
    (grid / f"{cell['cid']}.json").write_text(json.dumps(payload))
    out = tmp_path / "figs"
    proc = subprocess.run(
        [sys.executable, str(EXPERIMENTS / "make_figures.py"),
         "--grid-dir", str(grid), "--out-dir", str(out), "--only", "F1"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert "DIFFERENT code_hash" in proc.stderr
    manifest = json.loads((out / "figures_manifest.json").read_text())
    assert manifest["provenance"]["n_cells_stale_code_hash"] == 1
    assert cell["cid"] in manifest["provenance"]["cells_stale_code_hash"]
