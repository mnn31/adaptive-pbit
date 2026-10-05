# Author: Manan Gupta <mnn@yogins.com>
"""Single-script figure regeneration for the paper (paper-1 discipline).

EVERY figure in the manuscript is produced by ONE run of this script, straight
from the per-cell JSONs in results/grid2/, into paper/figs/*.pdf. Vector output,
no rasterized text, no manual touch-ups, ever: if a figure is wrong the fix
goes here and the whole set is re-emitted.

Two guardrails, because the grid is long-running and the figures are written
while it is still filling in:

  * DEFENSIVE READS. A cell being written concurrently (tmp file + os.replace
    in run_grid.write_atomic) can never be seen half-written, but a truncated
    or unreadable file is still treated as "pending", listed by name, and
    excluded — never silently averaged over.
  * MISSING CELLS ARE LOUD. The expected cell set for each figure is derived
    from run_grid's own enumeration functions (single source of truth, no
    duplicated grid definition here). Any expected-but-absent cell is recorded
    in paper/figs/figures_manifest.json AND warned about on stderr. A partial
    aggregate is never plotted without being flagged; a panel with no data at
    all is drawn as an explicit "DATA PENDING" placeholder rather than an empty
    axis that could be mistaken for a result.

CONVENTIONS SET BY AUDIT ROUND 1 (docs/audit_r1_triage.md item 11) — these are
binding, and the manuscript may not contradict them:

  * CALIBRATION-REFERENCED NORMALIZATION is the primary quantity. Each rule's
    curve is its marginal-stratum stopping error at r0 DIVIDED BY THE SAME
    RULE'S ERROR UNDER IDEAL RANDOMNESS (r0=inf), so the plotted number is the
    degradation the entropy quality caused, with each rule's own conservatism,
    overshoot slack and stratum interaction divided out. The raw error/nominal
    ratio is kept as a secondary panel because the absolute level still matters
    for a deployment claim — but it is not the headline, because a rule that is
    2x conservative to begin with is not "compliant" for the same reason a rule
    that is 2x anti-conservative is not "violating".
  * SPRT RAW RATIOS ARE DIVIDED BY alpha/(1-beta), not alpha. Wald's
    inequalities bound the realized error by alpha/(1-beta), so alpha is not
    the guarantee those rules ever had.
  * CLUSTER-ROBUST CIs over TRAIN SEEDS replace min-max bands wherever three
    train seeds exist: the replicate unit is the model, noise seeds within a
    model are not independent replicates of it. With 3 clusters the interval is
    mean +/- t(2) x SE of the three cluster means, t(2) = 4.303. Where only one
    train seed exists the band is min-max ACROSS CELLS and is LABELLED as such,
    everywhere it appears.
  * TRUNCATION RATE IS ANNOTATED on every rule marker (marker area encodes it,
    with a size legend, and the exact numbers are in the caption file). A
    stopping error at 90% truncation is a fixed-N result wearing a rule's name.
  * markov_online is an ABLATION, not a peer rule (it truncates 80-89% of
    images, whole-test-set rate, even under ideal randomness), so it lives in
    its own subplot in F8.
  * SCOPE LABELS ARE PART OF THE CONVENTION. Errors, compliance and F6's budget
    axis are MARGINAL-STRATUM quantities; the pass counts and truncation rates
    in F2/F3/F8(a) are WHOLE-TEST-SET means, because cost per decision is a
    deployment quantity over all traffic. Both bases are coherent; the caption
    file names the basis at every use and the manuscript may not silently mix
    them inside one ratio.
  * The e-process is labelled "e-process (Ville, martingale-difference null)".

Outputs (paper/figs/):
  F1_headline_stopping_error.pdf   (a) calibration-referenced degradation vs r0,
                                   (b) raw error / each rule's own nominal
                                   bound, (c) marginal-stratum fixed-N accuracy
                                   at N in {4, 8, 64}, all train seeds.
  F2_matched_guarantee_cost.pdf    mean passes per decision vs r0 with the
                                   guarantee filter, against the throttling
                                   baseline charged in LATENCY only.
  F3_accuracy_cost_pareto.pdf      task accuracy vs mean passes, the fixed-N
                                   ladder AT THE SAME r0 as each panel's
                                   same-entropy reference (co-headline). The
                                   IDEAL ladder as a cross-r0 frontier is F6's
                                   denominator, not this grey curve.
  F4_calibration.pdf               reliability diagrams + ECE (15 equal-mass
                                   bins) at fixed N, ideal vs degraded entropy.
                                   NOT a manuscript figure since convergence
                                   review 1 item 12: the panel is uninformative
                                   (all curves sit in the top-right corner) and
                                   the manuscript keeps only the ECE numbers, so
                                   the printed figure numbers are F1-F3 = Figs.
                                   1-3, F5-F7 = Figs. 4-6 and F8a/F8b = Figs.
                                   7-8. Still built, and still the source of
                                   those ECE numbers (and of the binning
                                   -convention sensitivity, review 2 item 7).
  F5_mechanism.pdf                 vote ACF(k) out to k=48, the integrated
                                   inflation factor it implies, and the
                                   fired-vs-truncated error decomposition.
  F6_matched_budget_excess.pdf     each rule's marginal error divided by the
                                   IDEAL fixed-N frontier interpolated at that
                                   rule's own mean budget (the effect size that
                                   survives budget control).
  F7_fashion_replication.pdf       F1(a) layout on Fashion-MNIST.
  F8a_ablations_rule.pdf           markov_online; t_min and settle_ratio
                                   sensitivity (the rule/protocol row).
  F8b_ablations_device.pdf         sigma_delta and the rate-ceiling arm (the
                                   device row). The post-hoc stratum-cut
                                   sensitivity is still COMPUTED and written to
                                   captions_draft.md, but is no longer a panel
                                   (convergence review 2 item 2): six panels
                                   could not be drawn at print size.
  figures_manifest.json            provenance, cells consumed, cells missing.
  captions_draft.md                caption stubs stating exactly what was
                                   plotted, so the manuscript cannot drift,
                                   plus a "Manuscript statistics" section for
                                   the numbers the text quotes in a sentence
                                   rather than on an axis (stratum share, the
                                   aggregate null, the tie-excluded
                                   denominator, the cross-block reference noise
                                   floor, the streaming first-read transient).

Usage:  python experiments/make_figures.py [--grid-dir DIR] [--out-dir DIR]
                                           [--only F1 F3]
"""

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_grid as G  # noqa: E402  (grid definitions = single source of truth)

from apbit.cost import latency_equivalent_passes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GRID_DIR = ROOT / "results" / "grid2"
OUT_DIR = ROOT / "paper" / "figs"

ALPHA = G.ALPHA
BETA = G.BETA
R0S = G.R0S                      # (inf, 4, 2, 1, 0.5, 0.25) — ideal to worst
R0_TAGS = [G.r0_tag(r) for r in R0S]
FINITE_R0 = [r for r in R0S if np.isfinite(r)]
PRIMARY_ARM = G.PRIMARY_ARM      # ("easyplane", "settle")
DATASET = "mnist"                # the headline dataset; fashion is replication
PARETO_R0 = ("inf", "1", "0.5")  # r0 slices shown in F3
ECE_BINS = 15                    # frozen convention (design.md)
OVERLAY_NS = (4, 8, 64)          # F1(c): the rules' own budget range, plus 64
T_DF2 = 4.302652729911275        # t_{0.975, df=2} — 3 train-seed clusters
PEER_RULES = ("wald", "markov_fixedrho", "dirichlet", "eprocess")
ABLATION_RULES = ("markov_online",)
SENS_RULES = G.SENS_RULES

# Okabe-Ito colourblind-safe palette. (colour, marker, linestyle, label, short)
RULE_STYLE = {
    "dirichlet":       ("#D55E00", "o", "-",  "Dirichlet (naive)",
                        "Dirichlet\n(naive)"),
    "wald":            ("#E69F00", "s", "-",  "Wald SPRT (naive)",
                        "Wald\n(naive)"),
    "markov_online":   ("#0072B2", "^", "--", r"Markov SPRT (online $\rho$)",
                        "Markov\n(online)"),
    "markov_fixedrho": ("#009E73", "D", "-",  r"Markov SPRT (measured $\rho$)",
                        "Markov\n(meas. $\\rho$)"),
    "eprocess":        ("#CC79A7", "v", "-.",
                        "e-process (Ville, mart.-diff. null)", "e-process"),
    # convergence review 1 item 8: same rule, on-chip-measurable rho. Not in
    # RULE_ORDER — it is a check reported in prose, not a curve in any figure.
    "markov_onchiprho": ("#117733", "X", ":",
                         r"Markov SPRT (on-chip $\rho$)",
                         "Markov\n(on-chip $\\rho$)"),
}
RULE_ORDER = ("dirichlet", "wald", "markov_online", "markov_fixedrho",
              "eprocess")
ONCHIP_RULE = G.ONCHIP_RULE
# One-line legend names, so no panel ever shows a raw grid key like
# "markov_fixedrho" (convergence review 1 item 21: one spelling per rule).
RULE_LABEL_SHORT = {
    "dirichlet": "Dirichlet", "wald": "Wald SPRT",
    "markov_online": r"Markov SPRT (online $\rho$)",
    "markov_fixedrho": r"Markov SPRT (meas. $\rho$)",
    "markov_onchiprho": r"Markov SPRT (on-chip $\rho$)",
    "eprocess": "e-process",
}
ARM_STYLE = {
    ("easyplane", "settle"):     ("#0072B2", "o", "-",  "easy-plane, settle"),
    ("easyplane", "streaming"):  ("#56B4E9", "s", "--",
                                 "easy-plane, streaming"),
    ("arrhenius", "settle"):     ("#D55E00", "^", "-",  "Arrhenius, settle"),
    ("arrhenius", "streaming"):  ("#E69F00", "v", "--",
                                 "Arrhenius, streaming"),
    ("arrhcap", "settle"):       ("#882255", "P", "-",
                                 "Arrhenius + rate ceiling, settle"),
}
# short forms for the small F8 panels, where the full arm names run off the axes
ARM_LABEL_SHORT = {
    ("easyplane", "settle"): "easy-plane",
    ("easyplane", "streaming"): "easy-plane, stream",
    ("arrhenius", "settle"): "Arrhenius",
    ("arrhenius", "streaming"): "Arrhenius, stream",
    ("arrhcap", "settle"): "Arrhenius + ceiling",
}
N_STYLE = {4: "#D55E00", 8: "#CC79A7", 16: "#E69F00", 64: "#0072B2",
           256: "#009E73"}
THROTTLE_COLOUR = "#000000"

# marker area encodes the truncation rate (audit r1 item 11)
TRUNC_SIZES = (3.2, 9.0)         # markersize at truncation 0 and 1
TRUNC_LEGEND = (0.0, 0.5, 1.0)

# ---------------------------------------------------------------------------
# PRINT GEOMETRY (convergence review 2 item 2 — the binding constraint)
# ---------------------------------------------------------------------------
# ieeeaccess.cls reports \columnwidth = 242.674 pt and \textwidth = 505.122 pt
# in TeX points (1 in = 72.27 pt), i.e. 3.358 in and 6.988 in. Every figure is
# DRAWN AT THE WIDTH IT IS PRINTED AT and included with `width=\columnwidth` or
# `width=\textwidth`, so \includegraphics applies a scale of ~1 and a native
# font point is a printed font point. Review 1 left the figures drawn at 3.5 or
# 7.16 in and included at 0.55-0.74 of the target width, which scaled every
# label down to 2.2-3.7 pt on the page. Nothing here may be included at a
# fraction of its drawing width again without shrinking the fonts to match.
COL_W = 3.358                    # in — \columnwidth
TEXT_W = 6.988                   # in — \textwidth
F8B_W = 0.72 * TEXT_W            # in — F8b is a two-panel double-column float
FONT_MIN = 7.0                   # pt — no text in any figure is smaller


# ---------------------------------------------------------------------------
# style
# ---------------------------------------------------------------------------

def setup_style():
    """Two-column IEEE defaults, drawn at print size with a 7 pt floor.

    Sizes are chosen so that the SMALLEST text in any figure (minor tick
    labels, legend bodies) is FONT_MIN, and the figures are drawn at COL_W /
    TEXT_W so that no rescaling happens at \\includegraphics time.
    """
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "pdf.compression": 6, "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 7.6, "axes.labelsize": 7.6, "axes.titlesize": 7.8,
        "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
        "legend.fontsize": FONT_MIN,
        "axes.linewidth": 0.6, "lines.linewidth": 1.1, "lines.markersize": 3.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "legend.frameon": False, "legend.handlelength": 2.6,
        "legend.labelspacing": 0.25, "legend.borderaxespad": 0.2,
        "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.35,
        "grid.color": "0.7", "axes.axisbelow": True,
        "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


# ---------------------------------------------------------------------------
# provenance + defensive grid loading
# ---------------------------------------------------------------------------

def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        return False
    return bool(out)


def expected_index() -> dict:
    """cid -> flat metadata dict, from run_grid's own cell enumeration.

    Phases 2/3/4 carry (dataset, train_seed, device_class, mode, r0, rule?,
    noise_seed, sigma_delta, t_min, settle_ratio); phases 1 and 2a are indexed
    too so the manifest can flag a missing reference or rho measurement.
    """
    idx = {}
    for cell in G.phase2_cells():
        idx[cell["cid"]] = dict(cid=cell["cid"], phase=2,
                                r0=G.r0_tag(cell["r0"]), **cell["config"])
    for cell in G.phase3_cells():
        idx[cell["cid"]] = dict(cid=cell["cid"], phase=3,
                                r0=G.r0_tag(cell["r0"]), rule="fixed_ladder",
                                **cell["config"])
    for cell in G.phase4_cells():
        idx[cell["cid"]] = dict(cid=cell["cid"], phase=4,
                                r0=G.r0_tag(cell["r0"]), rule="fixed_calib",
                                **cell["config"])
    for task in G.phase2a_tasks():
        idx[task["cid"]] = dict(cid=task["cid"], phase="2a",
                                device_class=task["device_class"],
                                mode=task["mode"], r0=G.r0_tag(task["r0"]),
                                sigma_delta=task["sigma_delta"],
                                chip_seed=task["chip_seed"],
                                settle_ratio=task["settle_ratio"])
    for task in G.phase1_tasks():
        idx[task["cid"]] = dict(cid=task["cid"], phase=1,
                                dataset=task["dataset"],
                                train_seed=task["train_seed"],
                                split=task["split"])
    return idx


class GridStore:
    """Everything readable in the grid directory, plus what should be there."""

    def __init__(self, grid_dir: Path):
        self.dir = Path(grid_dir)
        self.expected = expected_index()
        self.loaded, self.unreadable = {}, []
        for path in sorted(self.dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError, UnicodeDecodeError,
                    ValueError):
                # concurrent writer, truncated file, or a stray non-JSON file:
                # pending, not data.
                self.unreadable.append(path.name)
                continue
            if not isinstance(payload, dict):
                self.unreadable.append(path.name)
                continue
            self.loaded[payload.get("cell_id", path.stem)] = payload
        self.unexpected = sorted(set(self.loaded) - set(self.expected))

    def sha_summary(self) -> dict:
        shas, hashes, dirty = {}, {}, 0
        for payload in self.loaded.values():
            sha = str(payload.get("git_sha", "unknown"))[:8]
            ch = str(payload.get("code_hash", "MISSING(v1 cell)"))[:12]
            shas[sha] = shas.get(sha, 0) + 1
            hashes[ch] = hashes.get(ch, 0) + 1
            dirty += bool(payload.get("git_dirty"))
        return {"cells_by_git_sha": shas, "cells_by_code_hash": hashes,
                "cells_written_dirty": dirty,
                "n_cells_stale_code_hash": len(self.stale),
                "cells_stale_code_hash": sorted(self.stale)[:20],
                "current_code_hash": G.code_hash()[:12]}

    @property
    def stale(self) -> set:
        """Cells whose code_hash is neither the current one nor an ACCEPTED one.

        They are still PLOTTED — excluding them would turn a partially
        recomputed grid into empty panels with no explanation — but they are
        counted in the manifest and warned about on stderr, because mixing two
        protocols inside one aggregate is exactly the failure the content hash
        exists to catch. The accepted set is run_grid's
        ACCEPTED_CODE_HASHES + the current hash: earlier hashes whose difference
        from now provably cannot move a stored number (convergence review 1
        item 8), each carrying its reason in the driver.
        """
        ok = G.accepted_code_hashes()
        return {cid for cid, p in self.loaded.items()
                if p.get("code_hash") not in ok}

    def reference_cell(self, dataset: str, train_seed: int, split="test"):
        return self.loaded.get(G.ref_cid(dataset, train_seed, split))


class Collector:
    """Pulls cells for one figure, recording used/missing for the manifest."""

    def __init__(self, store: GridStore, name: str):
        self.store, self.name = store, name
        self.used, self.missing = set(), set()
        self.panels = {}

    def pull(self, pred):
        cids = sorted(c for c, e in self.store.expected.items() if pred(e))
        got = [c for c in cids if c in self.store.loaded]
        miss = [c for c in cids if c not in self.store.loaded]
        self.used.update(got)
        self.missing.update(miss)
        return [self.store.loaded[c] for c in got], got, miss

    def note(self, panel: str, **kw):
        self.panels.setdefault(panel, {}).update(kw)

    def record(self, filename: str, *more: str) -> dict:
        """One manifest record; `more` names the extra PDFs of a split figure.

        F8 is printed as two figures (convergence review 2 item 2 — six panels
        at print width were illegible), but it is ONE cell pull and ONE panel
        dictionary, so it stays one record and lists both files.
        """
        return {
            "file": filename,
            "files": [filename, *more],
            "complete": not self.missing,
            "n_cells_used": len(self.used),
            "n_cells_missing": len(self.missing),
            "cells_used": sorted(self.used),
            "cells_missing": sorted(self.missing),
            "panels": self.panels,
        }


# ---------------------------------------------------------------------------
# predicates
# ---------------------------------------------------------------------------

DEFAULT_KNOBS = {"sigma_delta": 0.0, "t_min": G.T_MIN,
                 "settle_ratio": G.SETTLE_RATIO}


def is_default(e) -> bool:
    """True for cells at the frozen protocol settings.

    Every main-text curve must filter on this: the sensitivity and sigma_delta
    rows share device class, mode, rule and r0 with the default rows and would
    otherwise be silently averaged into them.
    """
    return all(float(e.get(k, v)) == float(v)
               for k, v in DEFAULT_KNOBS.items())


def arm_at(arm, r0):
    """Device/mode key for `arm` at `r0` (r0=inf collapses to the shared
    ideal-source cells — run_grid coverage note (e))."""
    return G.arm_key(arm[0], arm[1], r0)


def cell_pred(phase=2, dataset=DATASET, arm=None, r0=None, rule=None,
              defaults=True, **extra):
    """Predicate over the expected-cell metadata.

    `rule=None` means "every PEER-OR-ABLATION rule", i.e. the five rules of
    RULE_ORDER — NOT literally every rule in the grid. The on-chip-rho variant
    (convergence review 1 item 8) is a rule-level re-run of an already recorded
    device condition: within a cell every rule reads the SAME recorded vote
    stream, so its cells duplicate the vote streams of the fixedrho cells at the
    same (train seed, noise seed, r0) and would double-count them in every
    device-measurement average (F5's ACF, inflation and entropy panels). It is
    therefore only ever pulled by naming it.
    """
    dev = mode = None
    if arm is not None and r0 is not None:
        dev, mode = arm_at(arm, r0)
    tag = None if r0 is None else G.r0_tag(r0)

    def pred(e):
        if e.get("phase") != phase:
            return False
        if dataset is not None and e.get("dataset") != dataset:
            return False
        if dev is not None and (e.get("device_class") != dev
                                or e.get("mode") != mode):
            return False
        if tag is not None and e.get("r0") != tag:
            return False
        if rule is not None and e.get("rule") != rule:
            return False
        if (rule is None and phase == 2
                and e.get("rule") not in RULE_ORDER):
            return False
        if defaults and not is_default(e):
            return False
        for k, v in extra.items():
            if k not in e or float(e[k]) != float(v):
                return False
        return True
    return pred


def primary_pred(rule, r0, dataset=DATASET, phase=2, **extra):
    return cell_pred(phase=phase, dataset=dataset, arm=PRIMARY_ARM, r0=r0,
                     rule=rule, **extra)


def warn(msg: str):
    print(f"WARNING  {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# aggregation: cluster-robust over train seeds, min-max otherwise
# ---------------------------------------------------------------------------

def summarize(values) -> dict:
    """mean / min / max / sd over per-cell point estimates (None if empty).

    Per-cell Wilson intervals are NOT pooled: they describe within-cell
    binomial noise on the same 10k test images, so averaging them would
    understate seed-to-seed spread and overstate precision.
    """
    arr = np.asarray([v for v in values
                      if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return None
    return {"n": int(arr.size), "mean": float(arr.mean()),
            "min": float(arr.min()), "max": float(arr.max()),
            "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0}


def cluster_from_pairs(pairs) -> dict:
    """cluster_summarize's arithmetic on ready-made (train_seed, value) pairs.

    Same interval convention, same `ci_kind` labels; used where the value is not
    a per-cell payload field but a derived per-cell quantity (the post-hoc
    stratum-cut ratios in F8(f)), so those numbers cannot drift from F1(a)'s.
    """
    by = {}
    for seed, v in pairs:
        if v is None or not np.isfinite(v):
            continue
        by.setdefault(int(seed), []).append(float(v))
    if not by:
        return None
    cluster_means = {s: float(np.mean(v)) for s, v in sorted(by.items())}
    flat = [v for lst in by.values() for v in lst]
    if len(cluster_means) == 3:
        arr = np.asarray(list(cluster_means.values()), dtype=float)
        se = float(arr.std(ddof=1) / np.sqrt(arr.size))
        mean = float(arr.mean())
        return {"mean": mean, "lo": mean - T_DF2 * se, "hi": mean + T_DF2 * se,
                "n": len(flat), "n_clusters": 3, "se": se,
                "ci_kind": "cluster-t(2) over 3 train seeds",
                "cluster_means": cluster_means}
    mean = float(np.mean(flat))
    return {"mean": mean, "lo": float(np.min(flat)), "hi": float(np.max(flat)),
            "n": len(flat), "n_clusters": len(cluster_means), "se": None,
            "ci_kind": f"min-max over cells ({len(cluster_means)} train seed"
                       f"{'s' if len(cluster_means) != 1 else ''})",
            "cluster_means": cluster_means}


def cluster_summarize(payloads, value_fn) -> dict:
    """Cluster-robust aggregate over TRAIN SEEDS (audit r1 item 11).

    The replicate unit is the trained model: noise seeds inside one model share
    that model's idiosyncrasies, so treating them as independent replicates
    understates the interval. With three train-seed clusters the reported
    interval is (mean of the three cluster means) +/- t(2) x SE, t(2) = 4.303,
    and `ci_kind` says so. With fewer clusters no such interval exists and the
    band falls back to MIN-MAX over cells, labelled `min-max (1 train seed)` —
    a range, not a confidence interval, and the caption must say that.
    """
    by = {}
    for p in payloads:
        v = value_fn(p)
        if v is None or not np.isfinite(v):
            continue
        seed = p.get("config", {}).get("train_seed", 0)
        by.setdefault(int(seed), []).append(float(v))
    if not by:
        return None
    cluster_means = {s: float(np.mean(v)) for s, v in sorted(by.items())}
    flat = [v for lst in by.values() for v in lst]
    if len(cluster_means) == 3:
        arr = np.asarray(list(cluster_means.values()), dtype=float)
        se = float(arr.std(ddof=1) / np.sqrt(arr.size))
        mean = float(arr.mean())
        return {"mean": mean, "lo": mean - T_DF2 * se, "hi": mean + T_DF2 * se,
                "n": len(flat), "n_clusters": 3, "se": se,
                "ci_kind": "cluster-t(2) over 3 train seeds",
                "cluster_means": cluster_means}
    mean = float(np.mean(flat))
    return {"mean": mean, "lo": float(np.min(flat)), "hi": float(np.max(flat)),
            "n": len(flat), "n_clusters": len(cluster_means), "se": None,
            "ci_kind": f"min-max over cells ({len(cluster_means)} train seed"
                       f"{'s' if len(cluster_means) != 1 else ''})",
            "cluster_means": cluster_means}


def save_print(fig, path, target_w):
    """Write `fig` so the PDF really is `target_w` inches wide.

    `savefig(bbox_inches="tight")` crops to the drawn content, so the saved
    width is NOT the figsize: it comes out up to an inch narrower (unused
    subplot margin) or a little wider (a legend hanging past the axes). Either
    way `\\includegraphics[width=\\columnwidth]` then rescales the file, and
    every font in it, by whatever ratio that happens to be — which is how the
    previous revision's figures ended up lettered at 2.2-3.7 pt on the page
    (convergence review 2 item 2).

    So the figure size is iterated until the width savefig will actually
    produce equals the width it is included at. The scale factor at
    \\includegraphics time is then 1 and a 7 pt label is 7 pt on paper, which
    is a property of the build rather than of anyone's arithmetic. Heights are
    NOT held fixed in the ratio; only the width is pinned, because the width is
    what the page geometry fixes.
    """
    def tight():
        fig.canvas.draw()
        return (fig.get_tightbbox(fig.canvas.get_renderer()).width
                + 2.0 * plt.rcParams["savefig.pad_inches"])

    for _ in range(8):
        got = tight()
        if abs(got - target_w) <= 0.004 * target_w:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * target_w / got, h)
    fig.savefig(path)
    got = tight()                      # what was actually written
    if abs(got - target_w) > 0.01 * target_w:
        warn(f"{path.name}: written {got:.3f} in wide against a printed "
             f"{target_w:.3f} in — every font in it will be scaled by "
             f"{target_w / got:.3f}")
    return path


def band(ax, xs, stats, colour):
    xs2 = [x for x, s in zip(xs, stats) if s]
    if len(xs2) < 2:
        return
    lo = [s.get("lo", s.get("min")) for s in stats if s]
    hi = [s.get("hi", s.get("max")) for s in stats if s]
    ax.fill_between(xs2, lo, hi, color=colour, alpha=0.16, linewidth=0)


def line(ax, xs, stats, colour, marker, ls, label, **kw):
    xs2 = [x for x, s in zip(xs, stats) if s]
    ys = [s["mean"] for s in stats if s]
    if not xs2:
        return
    ax.plot(xs2, ys, color=colour, marker=marker, linestyle=ls, label=label,
            markeredgewidth=0.6, **kw)


def trunc_marker_size(trunc) -> float:
    if trunc is None or not np.isfinite(trunc):
        return TRUNC_SIZES[0]
    t = min(max(float(trunc), 0.0), 1.0)
    return TRUNC_SIZES[0] + t * (TRUNC_SIZES[1] - TRUNC_SIZES[0])


def trunc_markers(ax, xs, stats, truncs, colour, marker):
    """Overplot the line's markers with area encoding the truncation rate."""
    for x, s, t in zip(xs, stats, truncs):
        if not s:
            continue
        ax.plot([x], [s["mean"]], marker=marker, color=colour,
                markerfacecolor=colour, markeredgecolor=colour,
                linestyle="none",
                markersize=trunc_marker_size(t["mean"] if t else None),
                zorder=3)


def _keep(ax, leg):
    """Keep a second legend on the axes without letting it be clipped.

    `Axes.add_artist` is what makes two legends coexist, but it also clips the
    artist to the axes patch — which silently DELETES any legend placed outside
    the axes (matplotlib 3.11). Turning clipping off is the whole fix.
    """
    ax.add_artist(leg)
    leg.set_clip_on(False)
    return leg


def trunc_size_legend(ax, loc="lower right", outside=False):
    """The marker-area key.

    `outside=True` puts it in a horizontal strip BELOW the axes, for panels whose
    data fills the frame — an in-axes key there overprints the curves it is
    supposed to explain (convergence review 1 item 11).
    """
    # the outside strip carries three sizes, not five: a five-entry row is wider
    # than the axes, and a figure scaled to a column width pays for that with
    # smaller tick labels everywhere else.
    ticks = (0.0, 0.5, 1.0) if outside else TRUNC_LEGEND
    handles = [plt.Line2D([], [], color="0.45", marker="o", linestyle="none",
                          markerfacecolor="0.45",
                          markersize=trunc_marker_size(t),
                          label=(f"{t:.0%}" if outside
                                 else f"truncated {t:.0%}"))
               for t in ticks]
    kw = (dict(loc="upper center", bbox_to_anchor=(0.5, -0.24),
               ncol=len(handles), columnspacing=0.9)
          if outside else dict(loc=loc, ncol=1, labelspacing=0.35))
    leg = ax.legend(handles=handles, fontsize=FONT_MIN, handletextpad=0.5,
                    title="marker area = truncation rate" if outside
                    else "marker area", title_fontsize=FONT_MIN, **kw)
    _keep(ax, leg)
    return leg


def rule_legend(ax, rules, loc="upper left", ncol=1, extra=()):
    """Legend built by hand: the plotted markers are area-coded for truncation,
    so they must not be reused as legend keys at whatever size they happen to
    be. Keys carry the rule colour, marker and linestyle at a fixed size."""
    handles = [plt.Line2D([], [], color=RULE_STYLE[r][0],
                          marker=RULE_STYLE[r][1],
                          linestyle=RULE_STYLE[r][2], markersize=3.4,
                          label=RULE_STYLE[r][3])
               for r in rules] + list(extra)
    leg = ax.legend(handles=handles, loc=loc, ncol=ncol)
    _keep(ax, leg)
    return leg


def r0_axis(ax, tags=None, compact=False):
    """The shared r0 category axis.

    `compact` shortens the label to `$r_0$`: the long form is a whole line of
    text and, on a row of panels that is not the bottom row, it lands on the next
    row's titles (convergence review 1 item 2).
    """
    tags = tags or R0_TAGS
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(["ideal" if t == "inf" else t for t in tags])
    ax.set_xlabel(r"$r_0$" if compact else
                  r"sampling ratio $r_0$  (query interval / correlation time)")
    ax.set_xlim(-0.35, len(tags) - 0.65)


def log_tick_labels(ax, axis="y", mantissas=(2, 3, 5)):
    """Label the named minor ticks on a log axis (convergence review 1 item 11).

    A log axis spanning less than a decade shows exactly one labelled tick, so
    the reader cannot read a headline value off it at all. Decades keep the plain
    `%g` major label and the 2/3/5 minor ticks get labels too, one point smaller,
    which is enough to read a factor of 6 off the axis without turning it into a
    grid of numbers.
    """
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))

    def minor(v, _):
        if v <= 0 or not np.isfinite(v):
            return ""
        m = v / 10.0 ** np.floor(np.log10(v) + 1e-9)
        return f"{v:g}" if any(abs(m - k) < 0.05 for k in mantissas) else ""
    a.set_minor_formatter(matplotlib.ticker.FuncFormatter(minor))
    ax.tick_params(axis=axis, which="minor", labelsize=FONT_MIN)


def pad_limits(ax, values, axis="y", lo_pad=0.10, hi_pad=0.08):
    """Explicit limits with room, from the values actually plotted.

    Autoscale can leave a marker sitting on the frame (F1(c)'s N=4 point at
    r0=0.25 did exactly that, convergence review 1 item 9), and it never leaves
    room for an in-axes legend. Both are fixed by setting the limits from the
    data instead of hoping.
    """
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return
    lo, hi = min(vals), max(vals)
    span = hi - lo or abs(hi) or 1.0
    if axis == "y":
        ax.set_ylim(lo - lo_pad * span, hi + hi_pad * span)
    else:
        ax.set_xlim(lo - lo_pad * span, hi + hi_pad * span)


def pending(ax, message: str):
    """Explicit placeholder — never an empty axis that reads as a result."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linestyle((0, (3, 3)))
        spine.set_color("0.55")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=7.5,
            color="0.35", transform=ax.transAxes, wrap=True)


# ---------------------------------------------------------------------------
# per-cell accessors
# ---------------------------------------------------------------------------

def marginal(payload) -> dict:
    return payload["stopping_error"]["marginal"]


def marginal_error(payload):
    return marginal(payload).get("stopping_error")


def marginal_budget(payload):
    m = marginal(payload)
    return m.get("mean_samples")


def truncation(payload):
    return payload.get("truncation_rate")


def nominal_denominator(rule: str) -> float:
    """The bound the rule actually has (claims that die, item 1 and 11).

    Wald's inequalities give realized error <= alpha/(1-beta) for both SPRTs, so
    dividing their error by alpha would compare them against a bound they never
    had. Dirichlet's alpha is a posterior-probability threshold, not a
    frequentist bound at all — it is divided by alpha only as the nominal level
    its authors state, and the caption says so.
    """
    return ALPHA / (1.0 - BETA) if rule in (
        "wald", "markov_online", "markov_fixedrho") else ALPHA


def guarantee_holds(payload) -> bool:
    """Wilson UPPER bound on the marginal-stratum stopping error <= alpha.

    Upper bound, not the point estimate: 'the guarantee is met' has to survive
    the sampling noise of the cell, otherwise F2 would compare costs at
    unmatched error levels.
    """
    hi = marginal(payload).get("stopping_error_wilson95")
    return bool(hi and hi[1] is not None and hi[1] <= ALPHA)


def calibration_error(col, rule, dataset=DATASET, stratum="marginal", **extra):
    """The rule's OWN ideal-randomness (r0=inf) stopping error on `stratum`.

    This is the denominator of the headline normalization. It is the pooled mean
    over the ideal cells of that rule (which are shared by every arm, so there
    is one set of them per rule x train seed x noise seed), not a per-cell
    pairing: pairing cell-to-cell would put the calibration's own binomial
    noise into every point twice.
    """
    payloads, _, _ = col.pull(primary_pred(rule, np.inf, dataset=dataset,
                                          **extra))
    s = summarize([p["stopping_error"][stratum].get("stopping_error")
                   for p in payloads])
    return (s["mean"] if s and s["mean"] and s["mean"] > 0 else None), s


# ---------------------------------------------------------------------------
# F1 — headline
# ---------------------------------------------------------------------------

def rule_curve(col, rule, r0s, dataset=DATASET, value=None, **extra):
    """(stats, trunc_stats) per r0 for one rule, cluster-robust."""
    stats, truncs = [], []
    for r0 in r0s:
        payloads, _, _ = col.pull(primary_pred(rule, r0, dataset=dataset,
                                              **extra))
        stats.append(cluster_summarize(payloads, value or marginal_error))
        truncs.append(cluster_summarize(payloads, truncation))
    return stats, truncs


def normalized_curve(col, rule, r0s, dataset=DATASET, **extra):
    """Calibration-referenced degradation curve for one rule."""
    calib, calib_stat = calibration_error(col, rule, dataset=dataset, **extra)
    if calib is None:
        return [None] * len(r0s), [None] * len(r0s), None, calib_stat
    stats, truncs = rule_curve(
        col, rule, r0s, dataset=dataset,
        value=lambda p: (None if marginal_error(p) is None
                         else marginal_error(p) / calib), **extra)
    return stats, truncs, calib, calib_stat


def figure_f1(store, out_dir):
    col = Collector(store, "F1")
    # ONE ROW at \textwidth, not three stacked panels at two thirds of a
    # column: the stacked version printed at 2.2 in wide and its labels came
    # out at 2.6-3.7 pt (convergence review 2 item 2). Side by side at print
    # width every panel is 2.2 in of axes and the type is drawn at its final
    # size. The rule legend and the marker-area key move to one shared strip
    # below the row, which is also the only way three panels this size can
    # carry six legend entries at >= 7 pt.
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 2.36),
                             gridspec_kw={"wspace": 0.37})
    ax, ax_raw, ax_acc = axes
    xs = list(range(len(R0S)))

    # (a) calibration-referenced normalization — the headline
    norm_series, calib = {}, {}
    for rule in PEER_RULES:
        stats, truncs, c, cstat = normalized_curve(col, rule, R0S)
        colour, marker, ls, label, _ = RULE_STYLE[rule]
        band(ax, xs, stats, colour)
        line(ax, xs, stats, colour, "none", ls, label)
        trunc_markers(ax, xs, stats, truncs, colour, marker)
        norm_series[rule] = {t: s for t, s in zip(R0_TAGS, stats)}
        calib[rule] = {"ideal_marginal_error": c, "ideal_cells": cstat}
    ax.axhline(1.0, color="0.25", linestyle=":", linewidth=0.9, zorder=1)
    ax.set_yscale("log")
    log_tick_labels(ax)
    ax.set_ylabel("stopping error / same rule's\nideal-randomness error")
    # HEADROOM (convergence review 2 item 3): the headline Dirichlet point at
    # 6.03 and the top of its band were clipped by an autoscaled frame that
    # topped out near 6. The limit is set from the plotted band, never guessed.
    _norm_hi = [st.get("hi", st.get("max"))
                for s in norm_series.values() for st in s.values() if st]
    if _norm_hi:
        ax.set_ylim(top=max(6.8, 1.12 * max(_norm_hi)))
    # short: the arm, alpha and t_max used to ride in this title and at print
    # size the string ran into panel (b)'s. They are in the caption and in
    # Methods, where a reader looks for them anyway.
    ax.set_title("(a) calibration-referenced", pad=3)
    r0_axis(ax, compact=True)

    # (b) raw ratio against each rule's OWN nominal bound
    raw_series = {}
    for rule in PEER_RULES:
        den = nominal_denominator(rule)
        stats, truncs = rule_curve(
            col, rule, R0S,
            value=lambda p, den=den: (None if marginal_error(p) is None
                                      else marginal_error(p) / den))
        colour, marker, ls, label, _ = RULE_STYLE[rule]
        band(ax_raw, xs, stats, colour)
        line(ax_raw, xs, stats, colour, "none", ls, label)
        trunc_markers(ax_raw, xs, stats, truncs, colour, marker)
        raw_series[rule] = {"nominal_denominator": den,
                            "series": {t: s for t, s in zip(R0_TAGS, stats)}}
    if any(any(v for v in e["series"].values()) for e in raw_series.values()):
        ax_raw.axhline(1.0, color="0.25", linestyle=":", linewidth=0.9)
        ax_raw.set_yscale("log")
        log_tick_labels(ax_raw)
        ax_raw.set_ylabel("stopping error / own nominal\n"
                          r"bound ($\alpha$; SPRTs: $\alpha/(1-\beta)$)")
    else:
        pending(ax_raw, "raw-ratio panel: no phase-2 cells on disk yet")
    ax_raw.set_title("(b) raw level, secondary", pad=3)
    r0_axis(ax_raw, compact=True)

    # (c) marginal-stratum fixed-N accuracy, all train seeds
    acc_series = {}
    for n_passes in OVERLAY_NS:
        stats = []
        for r0 in R0S:
            payloads, _, _ = col.pull(primary_pred(None, r0, phase=3))
            stats.append(cluster_summarize(
                payloads,
                lambda p, n=n_passes: _rung_value(p, n)))
        colour = N_STYLE.get(n_passes, "0.3")
        band(ax_acc, xs, stats, colour)
        line(ax_acc, xs, stats, colour, "o", "-", f"fixed $N$={n_passes}")
        acc_series[str(n_passes)] = {t: s for t, s in zip(R0_TAGS, stats)}
    if any(any(v for v in s.values()) for s in acc_series.values()):
        ax_acc.set_ylabel("task accuracy,\nmarginal stratum")
        # explicit limits from the plotted values, with room below for the
        # legend: autoscale left the N=4, r0=0.25 point on the frame and the
        # legend on top of the data (convergence review 1 item 9).
        pad_limits(ax_acc,
                   [v for s in acc_series.values() for st in s.values() if st
                    for v in (st["mean"], st.get("lo", st.get("min")),
                              st.get("hi", st.get("max")))],
                   lo_pad=0.30, hi_pad=0.10)
        # ncol=1 keeps every entry inside the panel frame (convergence
        # review 3 item 4: ncol=3 overflowed the right spine).
        ax_acc.legend(loc="lower left", ncol=1, fontsize=FONT_MIN,
                      columnspacing=0.8, handlelength=1.6, handletextpad=0.4)
    else:
        pending(ax_acc, "fixed-N accuracy panel: phase-3 ladder cells not yet "
                        "on disk")
    ax_acc.set_title("(c) matched stratum and budget", pad=3)
    r0_axis(ax_acc, compact=True)

    # one shared key below the row: rule colours/markers and the marker-area
    # scale. In-axes legends at this panel width would either overprint the
    # data or have to shrink below the 7 pt floor.
    handles = [plt.Line2D([], [], color=RULE_STYLE[r][0],
                          marker=RULE_STYLE[r][1],
                          linestyle=RULE_STYLE[r][2], markersize=3.6,
                          label=RULE_STYLE[r][3])
               for r in PEER_RULES]
    handles += [plt.Line2D([], [], color="0.45", marker="o", linestyle="none",
                           markerfacecolor="0.45",
                           markersize=trunc_marker_size(t),
                           label=f"truncation {t:.0%}" if t == 0.0
                           else f"{t:.0%}")
                for t in TRUNC_LEGEND]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, -0.10), ncol=4, fontsize=FONT_MIN,
               handletextpad=0.5, columnspacing=1.1)

    path = save_print(fig, out_dir / "F1_headline_stopping_error.pdf",
                      TEXT_W)
    plt.close(fig)

    col.note("a_calibration_referenced",
             filters={"dataset": DATASET, "arm": list(PRIMARY_ARM),
                      "split": "test", "stratum": "marginal (block-A ref "
                      f"margin < {G.MARGIN_CUT})", "phase": 2,
                      "knobs": DEFAULT_KNOBS},
             normalization="each rule's marginal stopping error divided by the "
                           "SAME rule's pooled r0=inf (ideal randomness) "
                           "marginal stopping error",
             aggregation="cluster-robust over train seeds where 3 exist "
                         "(mean of cluster means +/- t(2)=4.303 x SE); "
                         "min-max over cells otherwise, labelled per point",
             marker_encoding="marker area = mean truncation rate",
             calibration=calib, series=norm_series)
    col.note("b_raw_over_own_nominal",
             note="SPRT rules divided by alpha/(1-beta), not alpha",
             series=raw_series)
    col.note("c_marginal_fixed_n_accuracy",
             filters={"phase": 3, "n_passes": list(OVERLAY_NS),
                      "stratum": "marginal", "train_seeds": "all available"},
             series=acc_series)
    return col.record(path.name), norm_series, raw_series, acc_series, calib


def _rung_value(payload, n_passes, key="task_accuracy",
                stratum="marginal"):
    """Marginal-stratum value at one fixed-N rung of a phase-3 ladder."""
    for rung in payload.get("ladder", []):
        if rung.get("n_passes") != n_passes:
            continue
        block = rung.get("stopping_error", {}).get(stratum, {})
        return block.get("task_accuracy" if key == "task_accuracy" else key)
    return None


# ---------------------------------------------------------------------------
# F2 — matched-guarantee cost
# ---------------------------------------------------------------------------

def figure_f2(store, out_dir):
    col = Collector(store, "F2")
    fig, ax = plt.subplots(figsize=(COL_W, 2.46))
    tags = [G.r0_tag(r) for r in FINITE_R0]
    xs = list(range(len(FINITE_R0)))

    series = {}
    for rule in RULE_ORDER:
        stats, frac_ok, truncs = [], [], []
        for r0 in FINITE_R0:
            payloads, _, _ = col.pull(primary_pred(rule, r0))
            stats.append(cluster_summarize(payloads,
                                           lambda p: p.get("mean_samples")))
            truncs.append(cluster_summarize(payloads, truncation))
            ok = [guarantee_holds(p) for p in payloads]
            frac_ok.append(float(np.mean(ok)) if ok else None)
        colour, marker, ls, label, _short = RULE_STYLE[rule]
        line(ax, xs, stats, colour, "none", ls, label, alpha=0.9)
        for x, s, f, t in zip(xs, stats, frac_ok, truncs):
            if not s:
                continue
            style = ({"markerfacecolor": colour} if f == 1.0 else
                     {"markerfacecolor": "white"} if f == 0.0 else
                     {"markerfacecolor": colour, "fillstyle": "left"})
            ax.plot([x], [s["mean"]], marker=marker, color=colour,
                    markeredgecolor=colour, linestyle="none",
                    markersize=trunc_marker_size(t["mean"] if t else None),
                    **style)
            # PARTIAL compliance is the interesting case and a half-filled
            # marker of 4 pt cannot carry it (convergence review 2 item 16):
            # print the percentage next to the point instead of asking the
            # reader to resolve a fill.
            if f is not None and 0.0 < f < 1.0:
                ax.annotate(f"{f:.0%}", (x, s["mean"]),
                            textcoords="offset points", xytext=(4.5, -1.0),
                            fontsize=FONT_MIN, color=colour,
                            ha="left", va="top", zorder=5)
        series[rule] = {t: {"mean_samples": s,
                            "frac_cells_meeting_guarantee": f,
                            "truncation_rate": tr}
                        for t, s, f, tr in zip(tags, stats, frac_ok, truncs)}

    # throttling baseline: the naive rule run at the rule-compliant r0=4 and
    # charged in LATENCY only (audit r1 item 8: bits never scale with slowdown)
    base_payloads, base_used, _ = col.pull(
        primary_pred("dirichlet", G.R0_THROTTLED))
    base = cluster_summarize(base_payloads, lambda p: p.get("mean_samples"))
    base_ok = ([guarantee_holds(p) for p in base_payloads]
               if base_payloads else [])
    throttle = None
    if base:
        eq = [float(latency_equivalent_passes(base["mean"], G.R0_THROTTLED,
                                              r0)) for r0 in FINITE_R0]
        ax.plot(xs, eq, color=THROTTLE_COLOUR, linestyle=(0, (4, 2)),
                marker="*", markersize=5,
                label=rf"throttled naive ($r_0$={G.R0_THROTTLED:g}, "
                      "latency-charged)")
        settle = [G.SETTLE_RATIO / r0 for r0 in FINITE_R0]
        throttle = {
            "base_cells": base_used,
            "base_mean_samples": base,
            "base_frac_cells_meeting_guarantee":
                (float(np.mean(base_ok)) if base_ok else None),
            "latency_equivalent_passes": dict(zip(tags, eq)),
            "settle_latency_passes_per_decision": dict(zip(tags, settle)),
            "unit_note": ("latency-equivalent passes are a DURATION in "
                          "operating-clock pass slots; the same comparison in "
                          "random BITS is just the pass counts, unscaled. "
                          "Report both, always."),
        }
        # LIKE-FOR-LIKE baseline (convergence review 1 item 18): the plotted
        # baseline is the CHEAPEST naive rule, which is a different rule family
        # from the corrected one and therefore favourable to throttling. The
        # same-family comparison — throttled Wald against the corrected Markov
        # SPRT — is recorded so the manuscript can disclose the choice with a
        # number instead of a hedge.
        wald_payloads, _, _ = col.pull(
            primary_pred("wald", G.R0_THROTTLED))
        wald_base = cluster_summarize(wald_payloads,
                                      lambda p: p.get("mean_samples"))
        if wald_base:
            throttle["like_for_like_wald"] = {
                "base_mean_samples": wald_base,
                "base_frac_cells_meeting_guarantee": float(np.mean(
                    [guarantee_holds(p) for p in wald_payloads])),
                "latency_equivalent_passes": {
                    t: float(latency_equivalent_passes(
                        wald_base["mean"], G.R0_THROTTLED, r0))
                    for t, r0 in zip(tags, FINITE_R0)},
            }
    else:
        warn("F2: throttling baseline (dirichlet @ r0=4) has no cells yet")

    ax.set_yscale("log")
    log_tick_labels(ax)
    ax.set_ylabel("mean stochastic passes per decision")
    # NOT "at its calibrated level": the filter is Wilson-95 upper <= alpha,
    # which is looser than each rule's own calibration (convergence review 1
    # item 10).
    ax.set_title(r"cost of holding the marginal-stratum"
                 "\n" r"error at or below $\alpha$", pad=3)
    r0_axis(ax, tags)
    means = [s["mean"] for r in series.values()
             for s in (e["mean_samples"] for e in r.values()) if s]
    if means:
        dmin, dmax = min(means), max(means)
        if throttle:
            dmax = max(dmax, max(throttle["latency_equivalent_passes"]
                                 .values()))
        # tight now that the legend lives outside the axes: the old headroom was
        # a decade of blank paper reserved for it (item 10).
        ax.set_ylim(dmin * 0.75, dmax * 10 ** 0.18)
    else:
        pending(ax, "no phase-2 cells on disk yet")
    handles, labels = ax.get_legend_handles_labels()
    fill_key = [
        plt.Line2D([], [], color="0.35", marker="o", linestyle="none",
                   markerfacecolor="0.35",
                   label="filled: every cell Wilson-95 $\\leq\\alpha$"),
        plt.Line2D([], [], color="0.35", marker="o", linestyle="none",
                   markerfacecolor="white",
                   label="open: none; half: some, labelled %"),
        plt.Line2D([], [], color="0.35", marker="o", linestyle="none",
                   markerfacecolor="0.35",
                   markersize=TRUNC_SIZES[1],
                   label="marker area = truncation rate"),
    ]
    # BELOW the axes: the online-rho markers sit at 220-240 passes, right where
    # an upper-left legend goes (convergence review 1 item 10).
    ax.legend(handles + fill_key, labels + [h.get_label() for h in fill_key],
              loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=1,
              fontsize=FONT_MIN, handletextpad=0.5, columnspacing=1.0)

    path = save_print(fig, out_dir / "F2_matched_guarantee_cost.pdf", COL_W)
    plt.close(fig)

    col.note("cost_vs_r0",
             filters={"dataset": DATASET, "arm": list(PRIMARY_ARM),
                      "phase": 2, "split": "test", "knobs": DEFAULT_KNOBS,
                      "guarantee_filter": "marginal-stratum stopping-error "
                                          f"Wilson95 upper <= alpha={ALPHA}"},
             aggregation="cluster-robust over train seeds where 3 exist, "
                         "min-max otherwise; marker fill encodes the fraction "
                         "of cells meeting the guarantee, marker area the "
                         "truncation rate",
             series=series, throttling_baseline=throttle,
             cost_proxy="apbit.cost.latency_equivalent_passes "
                        "(passes x r0_throttled/r0_operating) — LATENCY only; "
                        "settle overhead reported separately")
    return col.record(path.name), series, throttle


# ---------------------------------------------------------------------------
# F3 — accuracy vs cost Pareto (co-headline)
# ---------------------------------------------------------------------------

def figure_f3(store, out_dir):
    col = Collector(store, "F3")
    fig, axes = plt.subplots(1, len(PARETO_R0), figsize=(TEXT_W, 2.24),
                             sharey=True, gridspec_kw={"wspace": 0.10})
    panels = {}
    for ax, tag in zip(axes, PARETO_R0):
        r0 = float("inf") if tag == "inf" else float(tag)
        ladder_payloads, _, _ = col.pull(primary_pred(None, r0, phase=3))
        rungs = {}
        for p in ladder_payloads:
            for rung in p.get("ladder", []):
                rungs.setdefault(rung["n_passes"], []).append(
                    rung["task_accuracy"])
        ladder = {n: summarize(v) for n, v in sorted(rungs.items())}
        if ladder:
            ns = sorted(ladder)
            ax.plot(ns, [ladder[n]["mean"] for n in ns], color="0.25",
                    marker=".", linestyle="-", linewidth=1.0, zorder=2,
                    label="fixed-$N$ ladder (same $r_0$ as the panel)")
            ax.fill_between(ns, [ladder[n]["min"] for n in ns],
                            [ladder[n]["max"] for n in ns], color="0.25",
                            alpha=0.15, linewidth=0)

        pts = {}
        for rule in RULE_ORDER:
            payloads, _, _ = col.pull(primary_pred(rule, r0))
            if not payloads:
                continue
            x = cluster_summarize(payloads, lambda p: p.get("mean_samples"))
            y = cluster_summarize(payloads, lambda p: p.get("task_accuracy"))
            t = cluster_summarize(payloads, truncation)
            colour, marker, _, label, _short = RULE_STYLE[rule]
            ax.errorbar([x["mean"]], [y["mean"]],
                        xerr=[[max(0.0, x["mean"] - x["lo"])],
                              [max(0.0, x["hi"] - x["mean"])]],
                        yerr=[[max(0.0, y["mean"] - y["lo"])],
                              [max(0.0, y["hi"] - y["mean"])]],
                        color=colour, marker=marker,
                        markersize=trunc_marker_size(t["mean"] if t else None),
                        elinewidth=0.7, capsize=1.5, linestyle="none",
                        label=label, zorder=3)
            pts[rule] = {"mean_samples": x, "task_accuracy": y,
                         "truncation_rate": t}

        if not ladder and not pts:
            pending(ax, "no cells on disk yet")
        else:
            ax.set_xscale("log")
            if not ladder:
                ax.text(0.03, 0.97, "fixed-$N$ reference curve pending\n"
                                    "(phase-3 cells not on disk)",
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=FONT_MIN, color="0.35")
        ax.set_title("ideal randomness" if tag == "inf" else rf"$r_0$={tag}",
                     pad=3)
        panels[tag] = {"fixed_n_ladder": {str(k): v
                                          for k, v in ladder.items()},
                       "adaptive_rules": pts}

    axes[0].set_ylabel("task accuracy (all test images)")
    axes[len(axes) // 2].set_xlabel("mean passes per decision")
    handles, labels_ = [], []
    for ax in axes:                      # one shared legend, below the panels
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels_:
                handles.append(h)
                labels_.append(lab)
    if handles:
        fig.legend(handles, labels_, loc="upper center",
                   bbox_to_anchor=(0.5, -0.02), ncol=min(3, len(handles)),
                   fontsize=FONT_MIN, columnspacing=1.4, handletextpad=0.4)
    path = save_print(fig, out_dir / "F3_accuracy_cost_pareto.pdf", TEXT_W)
    plt.close(fig)

    col.note("pareto",
             filters={"dataset": DATASET, "arm": list(PRIMARY_ARM),
                      "split": "test", "r0_panels": list(PARETO_R0),
                      "ladder_phase": 3, "adaptive_phase": 2,
                      "knobs": DEFAULT_KNOBS},
             aggregation="cluster-robust over train seeds where 3 exist, "
                         "min-max otherwise; marker area = truncation rate; "
                         "ladder band = min-max over ladder cells",
             panels=panels)
    return col.record(path.name), panels


# ---------------------------------------------------------------------------
# F4 — calibration (RQ3)
# ---------------------------------------------------------------------------

def reliability(conf, correct, n_bins=ECE_BINS):
    """Equal-MASS binning (frozen convention): sort by confidence, split into
    n_bins near-equal groups. Returns (bin_conf, bin_acc, bin_w, ECE)."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if conf.size == 0:
        return np.array([]), np.array([]), np.array([]), float("nan")
    order = np.argsort(conf, kind="stable")
    ece, bc, ba, bw = 0.0, [], [], []
    total = conf.size
    for chunk in np.array_split(order, min(n_bins, total)):
        if chunk.size == 0:
            continue
        c, a = conf[chunk].mean(), correct[chunk].mean()
        w = chunk.size / total
        bc.append(c)
        ba.append(a)
        bw.append(w)
        ece += w * abs(a - c)
    return np.array(bc), np.array(ba), np.array(bw), float(ece)


def figure_f4(store, out_dir):
    col = Collector(store, "F4")
    tags = ("inf", "0.5")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0), sharey=True)
    panels = {}
    for ax, tag in zip(axes, tags):
        r0 = float("inf") if tag == "inf" else float(tag)
        payloads, used, _ = col.pull(primary_pred(None, r0, phase=4))
        entry = {"cells": used, "n_cells": len(payloads), "by_n": {}}
        if not payloads:
            pending(ax, "calibration panel pending\n"
                        "(phase-4 cells not on disk)")
            ax.set_title("ideal randomness" if tag == "inf"
                         else rf"$r_0$={tag}", pad=3)
            panels[tag] = entry
            continue

        ax.plot([0, 1], [0, 1], color="0.6", linestyle=":", linewidth=0.8,
                zorder=1)
        for n_passes in G.CALIB_NS:
            conf, correct, per_cell_ece = [], [], []
            for p in payloads:
                for panel in p.get("panels", []):
                    if panel["n_passes"] != n_passes:
                        continue
                    c = np.asarray(panel["top_counts"], float) / n_passes
                    y = np.asarray(panel["correct"], float)
                    conf.append(c)
                    correct.append(y)
                    per_cell_ece.append(reliability(c, y)[3])
            if not conf:
                continue
            c = np.concatenate(conf)
            y = np.concatenate(correct)
            bc, ba, bw, ece = reliability(c, y)
            colour = N_STYLE.get(n_passes, "0.3")
            ax.plot(bc, ba, color=colour, marker="o", markersize=3,
                    linestyle="-", label=f"$N$={n_passes} (ECE={ece:.3f})")
            entry["by_n"][str(n_passes)] = {
                "pooled_ece": ece,
                "per_cell_ece": summarize(per_cell_ece),
                "n_images_pooled": int(c.size),
                "bin_confidence": [float(v) for v in bc],
                "bin_accuracy": [float(v) for v in ba],
                "bin_weight": [float(v) for v in bw],
                "mean_confidence": float(c.mean()),
                "accuracy": float(y.mean()),
            }
        ax.set_xlabel("confidence (vote share of decided class)")
        ax.set_title(("ideal randomness" if tag == "inf" else rf"$r_0$={tag}")
                     + f"  ({len(payloads)} noise seeds pooled)", pad=3)
        ax.legend(loc="upper left")
        panels[tag] = entry

    axes[0].set_ylabel("empirical accuracy")
    # CROP to the occupied region (convergence review 1 item 12). Full [0,1]^2
    # axes put every curve in the top-right eighth of the panel and the
    # deviations the figure exists to show were sub-pixel. The diagonal stays a
    # diagonal because both axes get the same limits.
    occupied = [v for panel in panels.values()
                for ent in panel["by_n"].values()
                for v in ent["bin_confidence"] + ent["bin_accuracy"]]
    if occupied:
        lo = max(0.0, min(occupied) - 0.02)
        for ax in axes:
            ax.set_xlim(lo, 1.005)
            ax.set_ylim(lo, 1.005)
            ax.set_aspect("equal", adjustable="box")
    fig.suptitle(f"reliability, {ECE_BINS} equal-mass bins, MNIST test, "
                 f"{PRIMARY_ARM[0]}+{PRIMARY_ARM[1]} "
                 f"(axes cropped to the occupied range)", fontsize=8, y=1.02)
    path = save_print(fig, out_dir / "F4_calibration.pdf", TEXT_W)
    plt.close(fig)

    col.note("reliability",
             filters={"dataset": DATASET, "arm": list(PRIMARY_ARM),
                      "phase": 4, "split": "test", "fixed_n": list(G.CALIB_NS),
                      "r0": list(tags)},
             aggregation=f"(confidence, correct) pairs pooled across noise "
                         f"seeds; ECE over {ECE_BINS} equal-mass bins from "
                         "top_counts / n_passes; per-cell ECE spread "
                         "reported alongside",
             panels=panels)
    return col.record(path.name), panels


# ---------------------------------------------------------------------------
# F5 — mechanism: ACF(k), inflation factor, error decomposition
# ---------------------------------------------------------------------------

def figure_f5(store, out_dir):
    col = Collector(store, "F5")
    # 2x2 at the FULL text width, not 0.58 of it: four panels each get 3.1 in
    # of paper and the labels are drawn at their printed size (convergence
    # review 2 item 2).
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_W, 3.50),
                             gridspec_kw={"wspace": 0.22, "hspace": 0.48})
    ax, ax_arm = axes[0]
    ax2, ax3 = axes[1]

    # (a) ACF(k) out to k = 48, primary arm, one curve per r0
    acf_curves = {}
    for r0 in R0S:
        payloads, _, _ = col.pull(primary_pred(None, r0))
        curves = [p.get("vote_diagnostics_marginal", {})
                   .get("acf_curve", {}) for p in payloads]
        lag_lists = [c.get("mean_acf") for c in curves if c.get("mean_acf")]
        if not lag_lists:
            continue
        arr = np.asarray([np.asarray(v, dtype=float) for v in lag_lists])
        mean = np.nanmean(arr, axis=0)
        lags = curves[0].get("lags") or list(range(1, arr.shape[1] + 1))
        colour = plt.cm.viridis(0.15 + 0.7 * R0S.index(r0) / max(1,
                                                                 len(R0S) - 1))
        ax.plot(lags, mean, color=colour, linewidth=1.0,
                label=("ideal" if np.isinf(r0) else rf"$r_0$={G.r0_tag(r0)}"))
        acf_curves[G.r0_tag(r0)] = {
            "lags": lags, "mean_acf": [float(v) for v in mean],
            "n_cells": int(arr.shape[0])}
    if acf_curves:
        ax.axhline(0.0, color="0.5", linewidth=0.6, linestyle=":")
        ax.set_xlabel("lag $k$ (stochastic passes)")
        ax.set_ylabel("vote-indicator ACF")
        ax.legend(loc="upper right", ncol=2, fontsize=FONT_MIN,
                  columnspacing=1.0, handlelength=1.8, handletextpad=0.4)
    else:
        pending(ax, "ACF(k) panel: no phase-2 diagnostics on disk yet")
    ax.set_title("(a) images whose votes fluctuate", pad=3)

    # (b) lag-1 vote ACF vs r0, per ARM, with the phase-2a rho fed to the rule.
    #     Kept from v1: the headline is reported as an arm RANGE, so the arms
    #     have to be shown side by side, and the open squares are the deployment
    #     story (rho measured on validation, never on test).
    xs_all = list(range(len(R0S)))
    acf_series = {}
    for arm in G.ARMS + (G.CAP_ARM,):
        stats = []
        for r0 in R0S:
            if np.isinf(r0):
                stats.append(None)   # shown once as the shared ideal point
                continue
            payloads, _, _ = col.pull(cell_pred(phase=2, arm=arm, r0=r0))
            stats.append(cluster_summarize(
                payloads, lambda p: p["vote_lag1_acf"]["marginal"]))
        colour, marker, ls, label = ARM_STYLE[tuple(arm)]
        band(ax_arm, xs_all, stats, colour)
        line(ax_arm, xs_all, stats, colour, marker, ls, label)
        acf_series["-".join(arm)] = {t: s for t, s in zip(R0_TAGS, stats)}

    ideal_payloads, _, _ = col.pull(cell_pred(phase=2, arm=PRIMARY_ARM,
                                              r0=np.inf))
    ideal = cluster_summarize(ideal_payloads,
                              lambda p: p["vote_lag1_acf"]["marginal"])
    if ideal:
        ax_arm.plot([0], [ideal["mean"]], marker="*", color="0.2",
                    markersize=6, linestyle="none",
                    label="ideal source (i.i.d. control)")
        ax_arm.axhline(ideal["mean"], color="0.5", linestyle=":",
                       linewidth=0.7)
    acf_series["ideal source (shared by all arms)"] = {
        t: (ideal if t == "inf" else None) for t in R0_TAGS}

    rho_used = {}
    for arm in G.ARMS + (G.CAP_ARM,):
        for i, r0 in enumerate(R0S):
            if np.isinf(r0):
                continue
            dev, mode = arm_at(arm, r0)
            payloads, _, _ = col.pull(
                lambda e, dev=dev, mode=mode, tag=G.r0_tag(r0):
                    e.get("phase") == "2a" and e.get("device_class") == dev
                    and e.get("mode") == mode and e.get("r0") == tag
                    and float(e.get("sigma_delta", 0.0)) == 0.0
                    and float(e.get("settle_ratio",
                                    G.SETTLE_RATIO)) == G.SETTLE_RATIO)
            for pl in payloads:
                rho_used[f"{dev}-{mode}_r{G.r0_tag(r0)}"] = pl["rho_used"]
                ax_arm.plot([i], [pl["rho_used"]], marker="s",
                            markerfacecolor="none", markeredgecolor="0.35",
                            markersize=4, linestyle="none", zorder=4)
    # The ideal source is NOT plotted as an open square (there is no device arm
    # at r0=inf and the panel must stay as it was), but the rule is still fed a
    # number there, and the manuscript quotes it: at ideal randomness the
    # VALIDATION-measured rho is what MarkovSPRT consumes, not the test-set ACF
    # that the ideal control curve shows. Recorded so the caption carries both.
    ideal_rho_payloads, _, _ = col.pull(
        lambda e: (e.get("phase") == "2a"
                   and e.get("device_class") == "ideal"
                   and e.get("r0") == G.r0_tag(np.inf)))
    for pl in ideal_rho_payloads:
        rho_used["ideal-ideal_rinf"] = pl["rho_used"]
    if rho_used:
        # SHORT (convergence review 2 item 10): the long form ran across the
        # panel and over the markers it was describing. The caption carries
        # "fed to Markov SPRT".
        ax_arm.plot([], [], marker="s", markerfacecolor="none",
                    markeredgecolor="0.35", linestyle="none",
                    label=r"$\rho$ measured on val")
    if any(any(v for v in s.values()) for s in acf_series.values()):
        ax_arm.set_ylabel("lag-1 vote ACF")
        # headroom first, so the key sits above the curves rather than on them
        ax_arm.set_ylim(top=ax_arm.get_ylim()[1] + 0.42 * (
            ax_arm.get_ylim()[1] - ax_arm.get_ylim()[0]))
        ax_arm.legend(loc="upper left", ncol=2, fontsize=FONT_MIN,
                      columnspacing=0.9, handlelength=1.8,
                      handletextpad=0.4, labelspacing=0.2)
    else:
        pending(ax_arm, "per-arm ACF panel: no phase-2 cells on disk yet")
    r0_axis(ax_arm, compact=True)
    ax_arm.set_title("(b) every arm, and the $\\rho$ the rule is fed", pad=3)

    # (c) integrated inflation factor and the AR(1) prediction
    infl = {}
    xs = list(range(len(R0S)))
    for key, colour, label in (
            ("if_true", "#0072B2", "integrated (initial-positive-sequence)"),
            ("if_ar1", "#D55E00", r"AR(1) prediction $(1+\rho)/(1-\rho)$")):
        stats = []
        for r0 in R0S:
            payloads, _, _ = col.pull(primary_pred(None, r0))
            stats.append(cluster_summarize(
                payloads,
                lambda p, key=key: (p.get("vote_diagnostics_marginal", {})
                                    .get("inflation", {}).get(key))))
        band(ax2, xs, stats, colour)
        line(ax2, xs, stats, colour, "o", "-", label)
        infl[key] = {t: s for t, s in zip(R0_TAGS, stats)}
    ratio_stats = []
    for r0 in R0S:
        payloads, _, _ = col.pull(primary_pred(None, r0))
        ratio_stats.append(cluster_summarize(
            payloads,
            lambda p: (p.get("vote_diagnostics_marginal", {})
                       .get("inflation", {}).get("ratio_true_over_ar1"))))
    infl["ratio_true_over_ar1"] = {t: s for t, s in zip(R0_TAGS, ratio_stats)}
    if any(any(v for v in s.values()) for s in infl.values()):
        ax2.set_ylabel("variance inflation factor")
        ax2.legend(loc="upper left", fontsize=FONT_MIN, handlelength=1.8,
                   handletextpad=0.4)
    else:
        pending(ax2, "inflation panel: no phase-2 diagnostics on disk yet")
    r0_axis(ax2, compact=True)
    ax2.set_title("(c) how much memory the stream carries", pad=3)

    # (d) fired vs truncated error decomposition at r0 = 0.5
    r0_dec = 0.5
    decomp, labels_, fired_v, trunc_v = {}, [], [], []
    for rule in RULE_ORDER:
        payloads, _, _ = col.pull(primary_pred(rule, r0_dec))
        f_rate, t_rate, tot = [], [], []
        for p in payloads:
            m = marginal(p)
            n_m = m["n"]
            if not n_m:
                continue
            bt = p["by_termination"]
            fm, tm = bt["fired_marginal"], bt["truncated_marginal"]
            fw = (fm["stopping_error"] or 0.0) * fm["n"]
            tw = (tm["stopping_error"] or 0.0) * tm["n"]
            f_rate.append(fw / n_m)
            t_rate.append(tw / n_m)
            tot.append(m["stopping_error"])
        s_f, s_t, s_tot = summarize(f_rate), summarize(t_rate), summarize(tot)
        decomp[rule] = {"fired_contribution": s_f,
                        "truncated_contribution": s_t,
                        "marginal_stopping_error": s_tot}
        if s_f or s_t:
            labels_.append(rule)
            fired_v.append(s_f["mean"] if s_f else 0.0)
            trunc_v.append(s_t["mean"] if s_t else 0.0)

    if labels_:
        pos = np.arange(len(labels_))
        ax3.bar(pos, fired_v, width=0.62, color="#0072B2", label="fired "
                "(the bounded channel)")
        ax3.bar(pos, trunc_v, width=0.62, bottom=fired_v, color="#E69F00",
                label=rf"truncated at $t_{{\max}}$={G.T_MAX} (budget error)")
        ax3.axhline(ALPHA, color="0.25", linestyle=":", linewidth=0.9)
        ax3.text(len(labels_) - 0.45, ALPHA * 1.04, rf"$\alpha$={ALPHA:g}",
                 fontsize=FONT_MIN, color="0.25", ha="right", va="bottom")
        ax3.set_xticks(pos)
        ax3.set_xticklabels([RULE_STYLE[r][4] for r in labels_],
                            fontsize=FONT_MIN)
        ax3.set_ylabel("error contribution")
        ax3.legend(loc="upper right", fontsize=FONT_MIN, handlelength=1.6,
                   handletextpad=0.4)
    else:
        pending(ax3, f"no phase-2 cells at $r_0$={r0_dec} on disk yet")
    ax3.set_title(rf"(d) where the error comes from, $r_0$={r0_dec:g}", pad=3)

    path = save_print(fig, out_dir / "F5_mechanism.pdf", TEXT_W)
    plt.close(fig)

    col.note("a_acf_curve",
             filters={"dataset": DATASET, "arm": list(PRIMARY_ARM),
                      "phase": 2, "stratum": "images whose votes fluctuate",
                      "max_lag": G.ACF_MAX_LAG},
             aggregation="per-lag mean over the marginal-stratum per-image ACF "
                         "within a cell, then averaged over cells (the vote "
                         "stream is recorded for the full budget whatever the "
                         "rule does, so it measures the device)",
             series=acf_curves)
    # the entropy RATE of the fitted chain, recorded (not plotted) so the word
    # "entropy" in the framing is backed by the right quantity: bits per pass
    # of a dependent stream, not the marginal entropy of one draw
    entropy = {}
    for key in ("entropy_rate_bits", "iid_entropy_bits", "gap_bits"):
        stats = []
        for r0 in R0S:
            payloads, _, _ = col.pull(primary_pred(None, r0))
            stats.append(cluster_summarize(
                payloads,
                lambda p, key=key: (p.get("vote_diagnostics_marginal", {})
                                    .get("entropy_rate", {}).get(key))))
        entropy[key] = {t: s for t, s in zip(R0_TAGS, stats)}

    col.note("b_vote_acf_by_arm",
             filters={"dataset": DATASET, "phase": 2, "split": "test",
                      "stratum": "images whose votes fluctuate",
                      "arms": ["-".join(a) for a in G.ARMS + (G.CAP_ARM,)]},
             aggregation="vote_lag1_acf.marginal pooled over every rule and "
                         "seed in the arm (the vote stream is collected for the "
                         "full budget regardless of the rule, so it measures "
                         "the device); cluster-robust where 3 train seeds exist",
             series=acf_series, ideal_reference=ideal,
             rho_used_from_phase2a=rho_used)
    col.note("c_inflation_factor",
             note="if_true uses initial-positive-sequence truncation; "
                  "ratio > 1 means memory beyond lag 1, i.e. the first-order "
                  "correction under-corrects",
             series=infl,
             entropy_rate="bits per pass of the FITTED two-state chain, "
                          "H = -sum_i pi_i sum_j P_ij log2 P_ij, next to the "
                          "i.i.d. entropy of the same marginal; the gap is the "
                          "predictability a correlated stream hands the rule",
             entropy_series=entropy)
    col.note("d_error_decomposition",
             filters={"dataset": DATASET, "arm": list(PRIMARY_ARM),
                      "r0": G.r0_tag(r0_dec), "phase": 2,
                      "stratum": "marginal"},
             aggregation="per-cell wrong-count share of the marginal stratum, "
                         "split by fired vs truncated; mean over cells "
                         "(stacked bars sum to the marginal stopping error)",
             series=decomp)
    return (col.record(path.name), acf_curves, infl, decomp, entropy,
            acf_series, rho_used)


# ---------------------------------------------------------------------------
# F6 — matched-budget excess (the effect size that survives budget control)
# ---------------------------------------------------------------------------

def ideal_frontier(col, dataset=DATASET, stratum="marginal") -> list:
    """[(N, marginal stopping error)] under IDEAL randomness, from phase 3."""
    payloads, _, _ = col.pull(primary_pred(None, np.inf, dataset=dataset,
                                           phase=3))
    by_n = {}
    for p in payloads:
        for rung in p.get("ladder", []):
            e = rung.get("stopping_error", {}).get(stratum, {}) \
                    .get("stopping_error")
            if e is not None:
                by_n.setdefault(int(rung["n_passes"]), []).append(float(e))
    return [(n, float(np.mean(v))) for n, v in sorted(by_n.items())]


def frontier_at(frontier, budget):
    """Log-linear interpolation of the ideal frontier at a mean budget."""
    if not frontier or budget is None or not np.isfinite(budget):
        return None
    if budget <= frontier[0][0]:
        return frontier[0][1]
    for (n0, e0), (n1, e1) in zip(frontier, frontier[1:]):
        if n0 <= budget <= n1:
            w = (np.log(budget) - np.log(n0)) / (np.log(n1) - np.log(n0))
            return e0 + w * (e1 - e0)
    return frontier[-1][1]


def figure_f6(store, out_dir):
    col = Collector(store, "F6")
    fig, ax = plt.subplots(figsize=(COL_W, 2.44))
    frontier = ideal_frontier(col)
    xs = list(range(len(R0S)))
    series = {}
    if not frontier:
        pending(ax, "matched-budget panel: the ideal fixed-N ladder (phase 3, "
                    "r0=inf) is not on disk yet")
    else:
        for rule in PEER_RULES:
            def excess(p, frontier=frontier):
                e, b = marginal_error(p), marginal_budget(p)
                ref = frontier_at(frontier, b)
                return None if (e is None or not ref) else e / ref
            stats, truncs = rule_curve(col, rule, R0S, value=excess)
            colour, marker, ls, label, _ = RULE_STYLE[rule]
            band(ax, xs, stats, colour)
            line(ax, xs, stats, colour, "none", ls, label)
            trunc_markers(ax, xs, stats, truncs, colour, marker)
            series[rule] = {t: s for t, s in zip(R0_TAGS, stats)}
        ax.axhline(1.0, color="0.25", linestyle=":", linewidth=0.9)
        ax.text(-0.30, 1.03, "ideal randomness, same budget",
                fontsize=FONT_MIN, color="0.25", ha="left", va="bottom")
        ax.set_ylabel("marginal error / ideal fixed-$N$ frontier\n"
                      "at the rule's own mean budget")
        # headroom for the key: at print size the four-entry legend reached the
        # r0=0.25 markers (review 2 item 2's resize made the type bigger, so
        # the room it needs grew with it).
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.55 * (hi - lo))
        rule_legend(ax, PEER_RULES, loc="upper left")
        trunc_size_legend(ax, outside=True)        # item 11: out of the data
    r0_axis(ax)
    ax.set_title("budget-controlled excess error", pad=3)
    path = save_print(fig, out_dir / "F6_matched_budget_excess.pdf", COL_W)
    plt.close(fig)

    col.note("matched_budget_excess",
             filters={"dataset": DATASET, "arm": list(PRIMARY_ARM),
                      "phase": 2, "frontier_phase": 3, "stratum": "marginal",
                      "knobs": DEFAULT_KNOBS},
             method="each rule's marginal-stratum stopping error divided by "
                    "the IDEAL-randomness fixed-N marginal error, log-linearly "
                    "interpolated at that rule's own marginal mean sample "
                    "count — so the comparison is at matched stratum AND "
                    "matched budget",
             ideal_frontier=[{"n_passes": n, "marginal_error": e}
                             for n, e in frontier],
             series=series)
    return col.record(path.name), series, frontier


# ---------------------------------------------------------------------------
# F7 — Fashion-MNIST replication (F1a layout)
# ---------------------------------------------------------------------------

def figure_f7(store, out_dir):
    col = Collector(store, "F7")
    fig, ax = plt.subplots(figsize=(COL_W, 2.44))
    xs = list(range(len(R0S)))
    series, calib = {}, {}
    for rule in PEER_RULES:
        stats, truncs, c, cstat = normalized_curve(col, rule, R0S,
                                                   dataset="fashion")
        colour, marker, ls, label, _ = RULE_STYLE[rule]
        band(ax, xs, stats, colour)
        line(ax, xs, stats, colour, "none", ls, label)
        trunc_markers(ax, xs, stats, truncs, colour, marker)
        series[rule] = {t: s for t, s in zip(R0_TAGS, stats)}
        calib[rule] = {"ideal_marginal_error": c, "ideal_cells": cstat}
    if any(any(v for v in s.values()) for s in series.values()):
        ax.axhline(1.0, color="0.25", linestyle=":", linewidth=0.9)
        ax.set_yscale("log")
        log_tick_labels(ax)
        ax.set_ylabel("stopping error / same rule's\nideal-randomness error")
        lo, hi = ax.get_ylim()          # headroom, as in F6
        ax.set_ylim(lo, hi * 2.1)
        rule_legend(ax, PEER_RULES, loc="upper left")
        trunc_size_legend(ax, outside=True)        # item 11: out of the data
    else:
        pending(ax, "Fashion-MNIST replication: no phase-2 fashion cells on "
                    "disk yet")
    r0_axis(ax)
    ax.set_title("Fashion-MNIST replication (1 train seed)", pad=3)
    path = save_print(fig, out_dir / "F7_fashion_replication.pdf", COL_W)
    plt.close(fig)
    col.note("fashion_replication",
             filters={"dataset": "fashion", "arm": list(PRIMARY_ARM),
                      "phase": 2, "stratum": "marginal",
                      "knobs": DEFAULT_KNOBS},
             normalization="same as F1(a): divided by the same rule's r0=inf "
                           "marginal error on Fashion-MNIST",
             aggregation="min-max over noise-seed cells — Fashion runs ONE "
                         "train seed, so no cluster-robust interval exists "
                         "and none is claimed",
             calibration=calib, series=series)
    return col.record(path.name), series


# ---------------------------------------------------------------------------
# F8 — ablations: markov_online, sensitivities, sigma_delta, rate ceiling,
#      post-hoc stratum restrictions
# ---------------------------------------------------------------------------

def posthoc_strata(store, col, rules=PEER_RULES, r0s=(np.inf, 1.0, 0.5),
                   cuts=(0.8, 0.9, 0.95), q_cut=0.75) -> dict:
    """Post-hoc restrictions from the per-image arrays (audit r1 item 12).

    Needs nothing but what is already on disk: the phase-2 cells carry
    per-image decisions, the phase-1 cell carries the block-A margins (the
    stratum), the block-A top counts (for the q = top/T_ref restriction) and
    the block-B reference plus its ties (the target). No extra grid cells.

    `margin_cut_ratios` is the same quantity F1(a) plots, recomputed at each
    cut: per-cell error at that cut divided by the POOLED ideal-randomness
    (r0=inf) error at THE SAME cut, then aggregated cluster-robustly over train
    seeds. Ratios of the rounded absolute means in `margin_cuts` do NOT
    reproduce it and must not be quoted as if they did — at cut 0.9 this column
    is by construction identical to F1(a), which is the internal-consistency
    check the manuscript needs.
    """
    out = {"margin_cuts": {}, "margin_cut_ratios": {}, "q_restriction": {},
           "cut_values": list(cuts), "q_cut": q_cut, "available": False}
    refs = {}
    for ts in G.TRAIN_SEEDS:
        cell = store.reference_cell(DATASET, ts)
        if not cell or "top_counts_block_a" not in cell:
            continue
        t_ref = float(cell["config"]["t_ref"])
        refs[ts] = {
            "ref": np.asarray(cell["reference"], dtype=np.int64),
            "tie": np.asarray(cell["reference_tie"], dtype=bool),
            "margin_a": np.asarray(cell["margin_counts"], float) / t_ref,
            "q_a": np.asarray(cell["top_counts_block_a"], float) / t_ref,
        }
    if not refs:
        return out
    out["available"] = True
    for rule in rules:
        for r0 in r0s:
            tag = G.r0_tag(r0)
            payloads, _, _ = col.pull(primary_pred(rule, r0))
            per_cut = {f"{c:g}": [] for c in cuts}
            per_cut_seeded = {f"{c:g}": [] for c in cuts}
            q_vals = []
            for p in payloads:
                ts = p["config"]["train_seed"]
                if ts not in refs or "per_image" not in p:
                    continue
                r = refs[ts]
                d = np.asarray(p["per_image"]["decisions"], dtype=np.int64)
                if d.shape != r["ref"].shape:
                    continue
                keep = ~r["tie"]
                wrong = d != r["ref"]
                for c in cuts:
                    m = keep & (r["margin_a"] < c)
                    if m.any():
                        per_cut[f"{c:g}"].append(float(wrong[m].mean()))
                        per_cut_seeded[f"{c:g}"].append(
                            (int(ts), float(wrong[m].mean())))
                m = keep & (r["margin_a"] < G.MARGIN_CUT) & (r["q_a"] >= q_cut)
                if m.any():
                    q_vals.append((float(wrong[m].mean()), int(m.sum())))
            out["margin_cuts"].setdefault(rule, {})[tag] = {
                c: summarize(v) for c, v in per_cut.items()}
            out["_seeded"] = out.get("_seeded", {})
            out["_seeded"].setdefault(rule, {})[tag] = per_cut_seeded
            out["q_restriction"].setdefault(rule, {})[tag] = {
                "error": summarize([v for v, _ in q_vals]),
                "n_images": (int(np.mean([n for _, n in q_vals]))
                             if q_vals else None),
            }

    # calibration-referenced ratio at every cut, F1(a)'s convention exactly:
    # pooled ideal denominator at the SAME cut, cluster-robust over train seeds.
    seeded = out.pop("_seeded", {})
    ideal_tag = G.r0_tag(np.inf)
    for rule, byr0 in seeded.items():
        denom = {}
        for c in cuts:
            vals = [v for _, v in byr0.get(ideal_tag, {}).get(f"{c:g}", [])]
            denom[f"{c:g}"] = (float(np.mean(vals))
                               if vals and float(np.mean(vals)) > 0 else None)
        for tag, per_cut_seeded in byr0.items():
            row = {}
            for c in cuts:
                key = f"{c:g}"
                den = denom[key]
                pairs = per_cut_seeded.get(key, [])
                row[key] = (None if den is None else
                            cluster_from_pairs([(ts, v / den)
                                                for ts, v in pairs]))
            out["margin_cut_ratios"].setdefault(rule, {})[tag] = row
        out["margin_cut_ratios"].setdefault(rule, {})["_denominator"] = denom
    return out


def figure_f8(store, out_dir):
    col = Collector(store, "F8")
    # TWO PRINTED FIGURES, not one six-panel block (convergence review 2 item
    # 2). Six panels on a 3.8 in-wide float drove the lettering to 2.2 pt; at
    # print width a 2x3 grid would be 5.5 in tall and still cramped. The split
    # is by subject: F8a is the rule/protocol row (online rho, t_min, settle),
    # F8b the device row (dwell-time dispersion, rate ceiling). The old panel
    # (f), post-hoc stratum-cut sensitivity, is DROPPED as a panel — it plotted
    # three points per rule whose values the manuscript already tabulates — but
    # it is still COMPUTED here and written to the caption file, and the
    # manuscript carries it as a small table.
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 2.28),
                             gridspec_kw={"wspace": 0.30})
    a, b, c = axes
    fig2, axes2 = plt.subplots(1, 2, figsize=(0.72 * TEXT_W, 2.28),
                               gridspec_kw={"wspace": 0.34})
    d, e = axes2
    xs = list(range(len(R0S)))
    notes = {}

    # (a) markov_online, the degenerate ablation
    ab = {}
    for rule in ABLATION_RULES:
        stats, truncs, cal, cstat = normalized_curve(col, rule, R0S)
        colour, marker, ls, label, _ = RULE_STYLE[rule]
        band(a, xs, stats, colour)
        line(a, xs, stats, colour, "none", ls, label)
        trunc_markers(a, xs, stats, truncs, colour, marker)
        ab[rule] = {"normalized": {t: s for t, s in zip(R0_TAGS, stats)},
                    "truncation_rate": {t: s for t, s in zip(R0_TAGS, truncs)},
                    "calibration": {"ideal_marginal_error": cal,
                                    "ideal_cells": cstat}}
    if any(v for v in ab.get("markov_online", {}).get("normalized",
                                                      {}).values()):
        a.axhline(1.0, color="0.25", linestyle=":", linewidth=0.9)
        a.set_ylabel("error / own ideal calibration")
        rule_legend(a, ABLATION_RULES, loc="upper left")
    else:
        pending(a, "markov_online ablation: no cells on disk")
    r0_axis(a, compact=True)
    a.set_title("(a) online-$\\rho$ ablation (near-fixed-$N$)", pad=3)

    # (b) t_min sensitivity, (c) settle_ratio sensitivity.
    # KNOB_LABEL, not knob.replace("_", " "): the raw key printed a literal
    # "t min" on the axis and in every legend entry (convergence review 2
    # item 11).
    KNOB_LABEL = {"t_min": r"$t_\mathrm{min}$",
                  "settle_ratio": "settle ratio"}
    sens = {"t_min": {}, "settle_ratio": {}}
    for ax_, knob, values, r0s_, colours in (
            (b, "t_min", (2, G.T_MIN, 8), G.HEADLINE_R0,
             ("#D55E00", "#0072B2", "#009E73")),
            (c, "settle_ratio", (2.0, G.SETTLE_RATIO, 8.0),
             (1.0, 0.5), ("#D55E00", "#0072B2", "#009E73"))):
        tags = [G.r0_tag(r) for r in r0s_]
        pos = list(range(len(r0s_)))
        drawn = False
        for v, colour in zip(values, colours):
            # EVERY knob is pinned, not just the one being varied: a
            # sensitivity row shares arm, rule and r0 with the default row, so
            # a partial filter would average the two together.
            knobs = dict(DEFAULT_KNOBS)
            knobs[knob] = v
            for rule, ls in (("dirichlet", "-"), ("wald", "--")):
                stats = []
                for r0 in r0s_:
                    payloads, _, _ = col.pull(
                        cell_pred(phase=2, dataset=DATASET, arm=PRIMARY_ARM,
                                  r0=r0, rule=rule, defaults=False, **knobs))
                    stats.append(cluster_summarize(
                        payloads,
                        lambda p: (None if marginal_error(p) is None else
                                   marginal_error(p)
                                   / nominal_denominator(
                                       p["config"]["rule"]))))
                if any(stats):
                    drawn = True
                    # the rule is carried by the LINESTYLE and named once in
                    # the caption, so the legend only has to say which knob
                    # value a colour is: three entries, not six, which is what
                    # makes 7 pt fit (convergence review 2 items 2 and 4).
                    line(ax_, pos, stats, colour, "o", ls,
                         f"{KNOB_LABEL[knob]}={v:g}"
                         if rule == "dirichlet" else None)
                sens[knob].setdefault(f"{v:g}", {})[rule] = {
                    t: s for t, s in zip(tags, stats)}
        if drawn:
            ax_.axhline(1.0, color="0.25", linestyle=":", linewidth=0.9)
            ax_.set_xticks(pos)
            ax_.set_xticklabels(["ideal" if t == "inf" else t for t in tags])
            ax_.set_xlabel(r"$r_0$")
            ax_.set_ylabel("error / own nominal bound")
            handles, labels_k = ax_.get_legend_handles_labels()
            handles += [
                plt.Line2D([], [], color="0.35", linestyle="-",
                           label="solid: Dirichlet"),
                plt.Line2D([], [], color="0.35", linestyle="--",
                           label="dashed: Wald SPRT")]
            # handlelength 3.0: at 2.0 a dashed key showed one dash and was
            # indistinguishable from the solid one (convergence review 2
            # item 4).
            ax_.legend(handles=handles, loc="upper left", fontsize=FONT_MIN,
                       ncol=1, handlelength=3.0, handletextpad=0.4,
                       labelspacing=0.2)
        else:
            pending(ax_, f"{knob} sensitivity: no cells on disk")
        ax_.set_title(f"({'b' if knob == 't_min' else 'c'}) "
                      f"{KNOB_LABEL[knob]} sensitivity", pad=3)

    # (d) sigma_delta column
    sig = {}
    tags_sd = [G.r0_tag(r) for r in G.SIGMA_R0S]
    pos = list(range(len(G.SIGMA_R0S)))
    drawn = False
    for sd, colour in zip((0.0,) + G.SIGMA_DELTAS,
                          ("#0072B2", "#E69F00", "#D55E00")):
        for rule, ls in (("dirichlet", "-"), ("markov_fixedrho", "--")):
            stats = []
            for r0 in G.SIGMA_R0S:
                payloads, _, _ = col.pull(
                    cell_pred(phase=2, dataset=DATASET, arm=PRIMARY_ARM, r0=r0,
                              rule=rule, defaults=False, sigma_delta=sd,
                              t_min=G.T_MIN, settle_ratio=G.SETTLE_RATIO))
                stats.append(cluster_summarize(
                    payloads,
                    lambda p: (None if marginal_error(p) is None else
                               marginal_error(p)
                               / nominal_denominator(p["config"]["rule"]))))
            if any(stats):
                drawn = True
                line(d, pos, stats, colour, "o", ls,
                     rf"$\sigma_\Delta$={sd:g}"
                     if rule == "dirichlet" else None)
            sig.setdefault(f"{sd:g}", {})[rule] = {t: s for t, s
                                                   in zip(tags_sd, stats)}
    if drawn:
        d.axhline(1.0, color="0.25", linestyle=":", linewidth=0.9)
        d.set_xticks(pos)
        d.set_xticklabels(tags_sd)
        d.set_xlabel(r"$r_0$")
        d.set_ylabel("error / own nominal bound")
        # headroom, then a five-entry key: three dispersion colours and the two
        # linestyle keys that say which rule is which (items 2/4/11).
        d.set_ylim(top=d.get_ylim()[1] + 0.50 * (d.get_ylim()[1]
                                                 - d.get_ylim()[0]))
        handles, _ = d.get_legend_handles_labels()
        handles += [plt.Line2D([], [], color="0.35", linestyle="-",
                               label="solid: Dirichlet"),
                    plt.Line2D([], [], color="0.35", linestyle="--",
                               label="dashed: Markov SPRT")]
        d.legend(handles=handles, loc="upper left", fontsize=FONT_MIN, ncol=2,
                 columnspacing=0.8, handlelength=3.0, handletextpad=0.35,
                 labelspacing=0.2)
    else:
        pending(d, "sigma_delta column: no cells on disk")
    d.set_title(r"(a) device variation $\sigma_\Delta$", pad=3)

    # (e) the rate-ceiling arm against the arms that bracket it
    cap = {}
    tags_cap = [G.r0_tag(r) for r in G.CAP_R0S]
    pos = list(range(len(G.CAP_R0S)))
    drawn = False
    for arm in (G.CAP_ARM, ("arrhenius", "settle"), PRIMARY_ARM):
        colour, marker, ls, label = ARM_STYLE[tuple(arm)]
        for rule, lsr in (("dirichlet", ls),):
            stats = []
            for r0 in G.CAP_R0S:
                payloads, _, _ = col.pull(
                    cell_pred(phase=2, dataset=DATASET, arm=arm, r0=r0,
                              rule=rule))
                stats.append(cluster_summarize(
                    payloads,
                    lambda p: (None if marginal_error(p) is None else
                               marginal_error(p) / ALPHA)))
            if any(stats):
                drawn = True
                line(e, pos, stats, colour, marker, lsr,
                     ARM_LABEL_SHORT.get(tuple(arm), label))
            cap["-".join(arm)] = {t: s for t, s in zip(tags_cap, stats)}
    if drawn:
        e.axhline(1.0, color="0.25", linestyle=":", linewidth=0.9)
        e.set_xticks(pos)
        e.set_xticklabels(tags_cap)
        e.set_xlabel(r"$r_0$")
        e.set_ylabel(r"Dirichlet error / $\alpha$")
        e.set_ylim(top=e.get_ylim()[1] + 0.30 * (e.get_ylim()[1]
                                                 - e.get_ylim()[0]))
        e.legend(loc="upper left", fontsize=FONT_MIN, handlelength=2.2,
                 handletextpad=0.4, labelspacing=0.2)
    else:
        pending(e, "rate-ceiling arm: no cells on disk")
    e.set_title("(b) attempt-frequency rate ceiling", pad=3)

    # post-hoc stratum restrictions from the per-image arrays. COMPUTED, not
    # plotted (convergence review 2 item 2): the old panel (f) carried three
    # points per rule, all of which the manuscript tabulates to four decimals,
    # so it was the cheapest panel to give up for print size. The numbers keep
    # flowing to the caption file and from there into the manuscript's table.
    ph = posthoc_strata(store, col)

    path = save_print(fig, out_dir / "F8a_ablations_rule.pdf", TEXT_W)
    plt.close(fig)
    path2 = save_print(fig2, out_dir / "F8b_ablations_device.pdf",
                       F8B_W)
    plt.close(fig2)

    notes = {"markov_online": ab, "sensitivity": sens, "sigma_delta": sig,
             "rate_ceiling_arm": cap, "posthoc": ph}
    col.note("ablations",
             filters={"dataset": DATASET, "phase": 2},
             note="markov_online is here and NOT in F1 because it truncates "
                  "80-89% of images (whole-test-set rate) even under ideal "
                  "randomness, so its "
                  "'error' is a fixed-N=t_max error wearing a rule's name; "
                  "the truncation rates are in the caption file. Split into "
                  "TWO printed figures at print size: F8a = (a) online rho, "
                  "(b) t_min, (c) settle ratio; F8b = (a) sigma_delta, "
                  "(b) rate ceiling. The post-hoc stratum-cut sensitivity is "
                  "computed and tabulated below but no longer drawn.",
             panels=notes)
    return col.record(path.name, path2.name), notes


# ---------------------------------------------------------------------------
# manuscript statistics — quoted in the text, not plotted on any axis
# ---------------------------------------------------------------------------

def _p1_share(payload):
    """(n_marginal, n_total, frac) from a phase-1 cell's BLOCK-A margin.

    Block A is the stratum-defining block under the split-sample convention, so
    the share quoted in the text is block A's, never block B's.
    """
    ms = payload.get("margin_summary") or {}
    n_marg = ms.get("marginal_count")
    n_total = (payload.get("config", {}).get("n_images")
               or len(payload.get("reference") or []) or None)
    if n_marg is None or not n_total:
        return None
    frac = ms.get("marginal_frac")
    frac = float(n_marg) / float(n_total) if frac is None else float(frac)
    return int(n_marg), int(n_total), frac


def _p1_disagreement(payload, key):
    """1 - recorded cross-block agreement, i.e. the reference noise floor."""
    v = payload.get(key)
    return None if v is None else 1.0 - float(v)


def manuscript_stats(store) -> dict:
    """The text-only quantities the manuscript quotes, from the same cells.

    Nothing here draws a figure. It exists so the binding rule holds without an
    exception: every number in the manuscript comes out of this file, including
    the ones that live in a sentence rather than on an axis. Cells are pulled
    through a THROWAWAY collector, so the figure manifest is untouched, and the
    aggregation conventions are the figures' own (cluster-robust over train
    seeds where three exist, min-max otherwise, labelled either way).
    """
    col = Collector(store, "manuscript_stats")
    out = {}

    # (1) marginal-stratum share of the test set, per dataset x train seed.
    shares = {}
    for dataset in G.DATASETS:
        rows = {}
        for seed in G.TRAIN_SEEDS:
            payload = store.reference_cell(dataset, seed)
            if payload is None:
                continue
            got = _p1_share(payload)
            if got is None:
                continue
            rows[seed] = {"n_marginal": got[0], "n_total": got[1],
                          "frac": got[2]}
        shares[dataset] = {
            "per_train_seed": rows,
            "frac": summarize([r["frac"] for r in rows.values()]),
        }
    out["marginal_share"] = shares

    # (2) the aggregate null: whole-test-set ("all" stratum) stopping error of
    #     the naive Dirichlet rule as a multiple of alpha, primary arm. The MAX
    #     over r0 is the disclosure number, so it is named explicitly.
    by_r0 = {}
    for r0 in R0S:
        payloads, _, _ = col.pull(primary_pred("dirichlet", r0))
        by_r0[G.r0_tag(r0)] = cluster_summarize(
            payloads,
            lambda p: p["stopping_error"]["all"].get(
                "stopping_error_over_alpha"))
    have = {t: s for t, s in by_r0.items() if s}
    worst = max(have, key=lambda t: have[t]["mean"]) if have else None
    out["aggregate_null"] = {
        "rule": "dirichlet", "stratum": "all", "dataset": DATASET,
        "arm": list(PRIMARY_ARM), "over_alpha_by_r0": by_r0,
        "worst_r0": worst, "worst": by_r0.get(worst) if worst else None,
    }

    # (3) the scored denominator after block-B reference-tie exclusion.
    ties = {}
    for dataset in G.DATASETS:
        p1 = {}
        for seed in G.TRAIN_SEEDS:
            payload = store.reference_cell(dataset, seed)
            if payload is None:
                continue
            p1[seed] = payload.get("reference_tie_count")
        denoms = sorted({int(p["stopping_error"]["all"]["n"])
                         for p in store.loaded.values()
                         if p.get("phase") == 2
                         and p.get("config", {}).get("dataset") == dataset
                         and p.get("stopping_error", {}).get(
                             "all", {}).get("n") is not None})
        ties[dataset] = {
            "reference_tie_count_per_train_seed": p1,
            "scored_denominators_seen": denoms,
            "n_total": G.N_TEST,
        }
    out["tie_exclusion"] = ties

    # (4) reference noise floor: the RECORDED cross-block agreement, reported
    #     as a disagreement rate, overall and on the marginal stratum.
    floor = {}
    for dataset in G.DATASETS:
        rows = {}
        for seed in G.TRAIN_SEEDS:
            payload = store.reference_cell(dataset, seed)
            if payload is None:
                continue
            rows[seed] = {
                "all": _p1_disagreement(payload, "blocks_agree_frac"),
                "marginal": _p1_disagreement(payload,
                                             "blocks_agree_frac_marginal"),
            }
        floor[dataset] = {
            "per_train_seed": rows,
            "all": summarize([r["all"] for r in rows.values()]),
            "marginal": summarize([r["marginal"] for r in rows.values()]),
        }
    out["reference_noise_floor"] = floor

    # (5) the streaming first-read transient: the N=1 collapse and the N=256
    #     recovery on the phase-3 fixed-N ladder, streaming arms at r0=0.25.
    trans = {}
    for arm in (("easyplane", "streaming"), ("arrhenius", "streaming")):
        for dataset in G.DATASETS:
            payloads, cids, _ = col.pull(cell_pred(
                phase=3, dataset=dataset, arm=arm, r0=0.25,
                rule="fixed_ladder"))
            if not payloads:
                continue
            key = f"{arm[0]}-{arm[1]}/{dataset}"
            ent = {"n_cells": len(payloads), "cells": cids}
            for n in (1, G.T_MAX):
                ent[f"N={n}"] = {
                    "task_accuracy": summarize(
                        [_ladder_accuracy(p, n) for p in payloads]),
                    "task_accuracy_marginal": summarize(
                        [_rung_value(p, n) for p in payloads]),
                }
            trans[key] = ent
    out["streaming_transient"] = {"r0": "0.25", "recovery_n": G.T_MAX,
                                 "arms": trans}

    # (6) convergence review 1 item 1: the whole-test-set ("all" stratum) picture
    #     stated as the manuscript now has to state it. What is small over the
    #     whole test set is the LEVEL, not the calibration-referenced FACTOR, and
    #     the dilution is by easy inputs whose error is exactly zero rather than
    #     by inputs decided in few passes. All three quantities live here.
    whole = {"factor_by_rule": {}, "easy_error_by_rule": {},
             "marginal_error_mass_share_by_rule": {},
             "level_over_alpha_by_rule": {}}
    for rule in PEER_RULES:
        calib_all, calib_all_stat = calibration_error(col, rule,
                                                     stratum="all")
        f_row, e_row, s_row, l_row = {}, {}, {}, {}
        for r0 in R0S:
            payloads, _, _ = col.pull(primary_pred(rule, r0))
            f_row[G.r0_tag(r0)] = (None if calib_all is None else
                                   cluster_summarize(
                                       payloads,
                                       lambda p: (None if _stratum_error(p, "all")
                                                  is None else
                                                  _stratum_error(p, "all")
                                                  / calib_all)))
            e_row[G.r0_tag(r0)] = cluster_summarize(
                payloads, lambda p: _stratum_error(p, "easy"))
            s_row[G.r0_tag(r0)] = cluster_summarize(payloads, _error_mass_share)
            l_row[G.r0_tag(r0)] = cluster_summarize(
                payloads,
                lambda p: p["stopping_error"]["all"].get(
                    "stopping_error_over_alpha"))
        whole["factor_by_rule"][rule] = {
            "ideal_all_stratum_error": calib_all,
            "ideal_cells": calib_all_stat, "series": f_row}
        whole["easy_error_by_rule"][rule] = e_row
        whole["marginal_error_mass_share_by_rule"][rule] = s_row
        whole["level_over_alpha_by_rule"][rule] = l_row
    out["whole_test_set"] = whole

    # (7) convergence review 1 item 3: the additive reference-noise floor does
    #     NOT cancel in the calibration-referenced ratio — it attenuates the
    #     ratio toward unity. The floor-free version restricts to the substratum
    #     where the two reference blocks AGREE, which is where no floor is left
    #     to attenuate anything, and the factors there are LARGER: the reported
    #     ones are conservative.
    out["floor_free"] = _floor_free_factors(store, col)

    # (8) convergence review 1 item 8: the on-chip rho substitution. Same rule,
    #     same thresholds, fed the ACF over ALL fluctuating inputs (no reference
    #     block, no labels) instead of the block-A marginal-mask ACF.
    out["onchip_rho"] = _onchip_rho_check(store, col)

    # (9) convergence review 2 item 1: the corrected rule is NOT the only rule
    #     that keeps every replicate cell compliant down to r0=1 — the
    #     e-process does too, at essentially the same cost and with no rho
    #     measurement at all. The head-to-head that replaces the exclusivity
    #     claim is assembled here so the manuscript quotes one table rather
    #     than three scattered figure panels.
    out["corrected_vs_eprocess"] = _head_to_head(col)

    # (10) convergence review 2 item 5: "exactly 0 on the easy substratum" is
    #      true for the corrected rule only. Per-cell maxima and non-zero cell
    #      counts are what the honest sentence needs.
    out["easy_substratum_detail"] = _easy_substratum_detail(col)

    # (11) convergence review 2 item 6: the fixed-N ladder brackets quoted next
    #      to cluster-robust peer-rule accuracies must be cluster-robust too.
    out["ladder_accuracy_cluster"] = _ladder_cluster(col)

    # (12) convergence review 2 item 8: where the ideal fixed-budget frontier
    #      actually reaches the corrected rule's r0=1 error.
    out["frontier_crossing"] = _frontier_crossing(col)

    # (13) convergence review 2 item 7: how much of the reported ECE depends on
    #      the equal-mass binning convention.
    out["ece_convention"] = _ece_convention(col)
    return out


def _head_to_head(col, rules=("markov_fixedrho", "eprocess")) -> dict:
    """Corrected rule vs the e-process on every axis that separates them.

    Same cells and same conventions as F1/F2 — this is a re-tabulation, not a
    new measurement — but side by side, because the claim it replaces was a
    comparative one and a reader cannot check a comparison spread over two
    figures and a table.
    """
    out = {"note": ("both rules keep 100% of replicate cells inside alpha at "
                    "r0 >= 1; they separate on cost ABOVE the floor, on "
                    "partial compliance BELOW it, and on the level"),
           "by_rule": {}}
    for rule in rules:
        den = nominal_denominator(rule)
        row = {"nominal_denominator": den, "series": {}}
        for r0 in R0S:
            payloads, _, _ = col.pull(primary_pred(rule, r0))
            if not payloads:
                continue
            ok = [guarantee_holds(p) for p in payloads]
            row["series"][G.r0_tag(r0)] = {
                "mean_passes": cluster_summarize(
                    payloads, lambda p: p.get("mean_samples")),
                "marginal_error": cluster_summarize(payloads, marginal_error),
                "level_over_own_bound": cluster_summarize(
                    payloads,
                    lambda p, den=den: (None if marginal_error(p) is None
                                        else marginal_error(p) / den)),
                "n_cells_inside_alpha": int(sum(ok)),
                "n_cells": len(ok),
                "frac_cells_inside_alpha": float(np.mean(ok)),
            }
        out["by_rule"][rule] = row
    return out


def _easy_substratum_detail(col) -> dict:
    """Per-cell easy-substratum error: mean, MAX, and how many cells are 0.

    The cluster-robust mean of a quantity that is zero in most cells reads as
    "exactly 0" and is not (convergence review 2 item 5). The maximum over
    cells and the non-zero count are the numbers a sentence can stand on.
    """
    out = {"stratum": f"block-A margin >= {G.MARGIN_CUT}", "by_rule": {}}
    for rule in PEER_RULES:
        row = {}
        for r0 in R0S:
            payloads, _, _ = col.pull(primary_pred(rule, r0))
            vals = [_stratum_error(p, "easy") for p in payloads]
            vals = [v for v in vals if v is not None]
            ns = [p["stopping_error"]["easy"].get("n") for p in payloads]
            ns = [n for n in ns if n]
            if not vals:
                continue
            row[G.r0_tag(r0)] = {
                "n_cells": len(vals),
                "n_cells_nonzero": int(sum(v > 0 for v in vals)),
                "max": float(max(vals)),
                "mean": float(np.mean(vals)),
                "max_misdecisions": (int(round(max(vals) * max(ns)))
                                     if ns else None),
                "easy_n_images": (int(round(float(np.mean(ns))))
                                  if ns else None),
            }
        out["by_rule"][rule] = row
    return out


def _ladder_cluster(col, ns=(4, 8), r0=np.inf) -> dict:
    """Cluster-robust whole-test-set fixed-N ladder accuracy.

    F3's ladder band is min-max over ladder cells, which is NARROWER than a
    cluster interval and therefore flattering when quoted beside cluster-robust
    peer-rule accuracies (convergence review 2 item 6). This is the same rungs
    under the peer rules' own convention.
    """
    payloads, _, _ = col.pull(primary_pred(None, r0, phase=3))
    out = {"r0": G.r0_tag(r0), "n_cells": len(payloads), "by_n": {}}
    for n in ns:
        out["by_n"][str(n)] = cluster_summarize(
            payloads, lambda p, n=n: _ladder_accuracy(p, n))
    return out


def _frontier_crossing(col, rule="markov_fixedrho", r0=1.0) -> dict:
    """Where the IDEAL fixed-N frontier actually reaches the rule's error.

    The manuscript said the frontier "does not reach until N=128", but the
    N=128 rung is ABOVE the rule's error, so it has not reached it there
    (convergence review 2 item 8). The crossing is solved on the same
    log-linear interpolation F6 uses.
    """
    frontier = ideal_frontier(col)
    payloads, _, _ = col.pull(primary_pred(rule, r0))
    err = cluster_summarize(payloads, marginal_error)
    budget = cluster_summarize(payloads, lambda p: p.get("mean_samples"))
    out = {"rule": rule, "r0": G.r0_tag(r0), "marginal_error": err,
           "whole_test_mean_passes": budget,
           "frontier": [{"n_passes": n, "marginal_error": e}
                        for n, e in frontier]}
    if not frontier or not err:
        return out
    target = err["mean"]
    cross = None
    for (n0, e0), (n1, e1) in zip(frontier, frontier[1:]):
        if (e0 - target) * (e1 - target) <= 0 and e0 != e1:
            w = (e0 - target) / (e0 - e1)
            cross = float(np.exp(np.log(n0) + w * (np.log(n1) - np.log(n0))))
    out["bracketing_rungs"] = {
        str(n): e for n, e in frontier if n in (64, 128, 256)}
    out["crossing_n"] = cross
    if cross and budget:
        out["budget_ratio"] = float(cross / budget["mean"])
    return out


def _ece_convention(col, ns=G.CALIB_NS) -> dict:
    """ECE under the frozen convention and under one plausible alternative.

    The reported values use the frozen convention: a STABLE sort of the
    confidences split by `np.array_split` into 15 near-equal groups, so a tie
    mass larger than a bin is spread over several bins. A reader reimplementing
    "15 equal-mass bins" as quantile EDGES with duplicate edges collapsed puts
    all of that tie mass in ONE bin and gets a different number at large N,
    where the reported values are small. Both are computed here so the
    manuscript can state the convention AND its sensitivity instead of quoting
    a four-decimal number as if it were convention-free.
    """
    out = {"bins": ECE_BINS,
           "convention": "stable sort + np.array_split into 15 near-equal "
                         "groups (frozen; this is what the reported values "
                         "use)",
           "alternative": "quantile bin EDGES with duplicate edges collapsed, "
                          "so all tie mass at one confidence lands in one bin",
           "by_r0": {}}
    for tag in ("inf", "0.5"):
        r0 = float("inf") if tag == "inf" else float(tag)
        payloads, _, _ = col.pull(primary_pred(None, r0, phase=4))
        row = {}
        for n in ns:
            conf, corr = [], []
            for p in payloads:
                for panel in p.get("panels", []):
                    if panel["n_passes"] != n:
                        continue
                    conf.append(np.asarray(panel["top_counts"], float) / n)
                    corr.append(np.asarray(panel["correct"], float))
            if not conf:
                continue
            c, y = np.concatenate(conf), np.concatenate(corr)
            row[str(n)] = {
                "ece_frozen": reliability(c, y)[3],
                "ece_quantile_edges_collapsed": _ece_quantile_edges(c, y),
                "distinct_confidences": int(np.unique(c).size),
            }
        out["by_r0"][tag] = row
    # the degraded/ideal ratio the manuscript quotes, under both conventions
    ideal, deg = out["by_r0"].get("inf", {}), out["by_r0"].get("0.5", {})
    out["ratio_degraded_over_ideal"] = {
        n: {k.replace("ece_", "ratio_"): (deg[n][k] / ideal[n][k]
                                          if ideal.get(n, {}).get(k) else None)
            for k in ("ece_frozen", "ece_quantile_edges_collapsed")}
        for n in deg if n in ideal}
    return out


def _ece_quantile_edges(conf, correct, n_bins=ECE_BINS) -> float:
    """ECE under quantile EDGES with duplicate edges collapsed."""
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    edges = np.unique(np.quantile(conf, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        return float("nan")
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0,
                  edges.size - 2)
    ece, total = 0.0, conf.size
    for b in range(edges.size - 1):
        m = idx == b
        if not m.any():
            continue
        ece += m.sum() / total * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def _stratum_error(payload, stratum):
    return payload["stopping_error"][stratum].get("stopping_error")


def _error_mass_share(payload):
    """Marginal stratum's share of the cell's ABSOLUTE wrong-decision count.

    The stratum is ~8% of the test set, so "the whole-test-set level is small"
    and "the stratum carries the failure" are the same fact seen twice: this is
    the second one, wrong_marginal / wrong_all, and it is what licenses calling
    the whole-set null a dilution rather than an absence.
    """
    se = payload["stopping_error"]
    a, m = se["all"], se["marginal"]
    if not a["n"] or a.get("stopping_error") is None:
        return None
    wrong_all = a["stopping_error"] * a["n"]
    if wrong_all <= 0:
        return None
    return (m["stopping_error"] or 0.0) * m["n"] / wrong_all


def _agree_masks(store):
    """Per train seed: (keep, marginal, blocks_agree) boolean arrays.

    `blocks_agree` is block-A's decision == block-B's reference, i.e. the
    substratum on which the split-sample convention leaves NO residual
    reference-noise floor. Everything comes from the phase-1 cells already on
    disk.
    """
    masks = {}
    for ts in G.TRAIN_SEEDS:
        cell = store.reference_cell(DATASET, ts)
        if not cell or "decision_block_a" not in cell:
            continue
        t_ref = float(cell["config"]["t_ref"])
        ref = np.asarray(cell["reference"], dtype=np.int64)
        masks[ts] = {
            "ref": ref,
            "keep": ~np.asarray(cell["reference_tie"], dtype=bool),
            "marginal": (np.asarray(cell["margin_counts"], float) / t_ref
                         < G.MARGIN_CUT),
            "agree": np.asarray(cell["decision_block_a"],
                                dtype=np.int64) == ref,
        }
    return masks


def _floor_free_factors(store, col, rules=("wald", "dirichlet",
                                           "markov_fixedrho", "eprocess"),
                        r0s=R0S) -> dict:
    """Calibration-referenced factors on the AGREE substratum (item 3).

    Same normalization as F1(a) — per-cell error divided by the POOLED
    ideal-randomness error on the same substratum, then cluster-robust over train
    seeds — but restricted to images where the two independent reference blocks
    agree. The residual disagreement inflates numerator and denominator by the
    same ADDITIVE amount, which pulls a ratio toward 1 rather than cancelling;
    removing it is the check on how conservative the reported factors are.
    """
    out = {"available": False, "series": {}, "ideal_error": {},
           "n_images": {}}
    masks = _agree_masks(store)
    if not masks:
        return out
    out["available"] = True

    def per_cell(rule, r0):
        payloads, _, _ = col.pull(primary_pred(rule, r0))
        rows, ns = [], []
        for p in payloads:
            ts = p["config"]["train_seed"]
            if ts not in masks or "per_image" not in p:
                continue
            mk = masks[ts]
            d = np.asarray(p["per_image"]["decisions"], dtype=np.int64)
            if d.shape != mk["ref"].shape:
                continue
            m = mk["keep"] & mk["marginal"] & mk["agree"]
            if not m.any():
                continue
            rows.append((int(ts), float((d[m] != mk["ref"][m]).mean())))
            ns.append(int(m.sum()))
        return rows, ns

    ideal_tag = G.r0_tag(np.inf)
    for rule in rules:
        ideal_rows, ideal_ns = per_cell(rule, np.inf)
        den = (float(np.mean([v for _, v in ideal_rows]))
               if ideal_rows else None)
        out["ideal_error"][rule] = den
        out["n_images"][rule] = (int(np.mean(ideal_ns)) if ideal_ns else None)
        row = {}
        for r0 in r0s:
            tag = G.r0_tag(r0)
            rows, _ = per_cell(rule, r0)
            row[tag] = (None if not den or not rows else
                        cluster_from_pairs([(ts, v / den) for ts, v in rows]))
        out["series"][rule] = row
    out["note"] = (f"marginal stratum restricted to block-A decision == "
                   f"block-B reference; the ideal ({ideal_tag}) column is 1.0 "
                   "by construction")
    return out


def _onchip_rho_check(store, col) -> dict:
    """The item-8 substitution, cell by cell and summarized.

    Reports, per r0: how many replicate cells keep the marginal-stratum Wilson-95
    upper bound at or below alpha, the whole-test-set mean passes, and BOTH rho
    values (the marginal-mask one the headline rule is fed, and the
    all-fluctuating-inputs one a chip could measure) so the caption can state
    what was substituted for what.
    """
    out = {"available": False, "rule": ONCHIP_RULE, "by_r0": {},
           "rho": {}, "reference_rule": "markov_fixedrho"}
    for r0 in G.ONCHIP_R0S:
        tag = G.r0_tag(r0)
        payloads, used, _ = col.pull(primary_pred(ONCHIP_RULE, r0))
        base, _, _ = col.pull(primary_pred("markov_fixedrho", r0))
        if not payloads:
            continue
        out["available"] = True
        ok = [guarantee_holds(p) for p in payloads]
        out["by_r0"][tag] = {
            "n_cells": len(payloads), "cells": used,
            "n_cells_inside_alpha": int(sum(ok)),
            "frac_cells_inside_alpha": float(np.mean(ok)),
            "mean_passes": cluster_summarize(payloads,
                                             lambda p: p.get("mean_samples")),
            "marginal_error": cluster_summarize(payloads, marginal_error),
            "truncation_rate": cluster_summarize(payloads, truncation),
            "rho_used": sorted({round(float(p["config"]["rho_used"]), 6)
                                for p in payloads}),
            "baseline_marginal_rho_rule": {
                "n_cells": len(base),
                "n_cells_inside_alpha": int(sum(guarantee_holds(p)
                                                for p in base)),
                "mean_passes": cluster_summarize(
                    base, lambda p: p.get("mean_samples")),
                "rho_used": sorted({round(float(p["config"]["rho_used"]), 6)
                                    for p in base
                                    if p["config"].get("rho_used")}),
            },
        }
    # the two measurements, straight from the phase-2a cells they come from
    for r0 in G.ONCHIP_R0S:
        cid = G.rho_cid(PRIMARY_ARM[0], PRIMARY_ARM[1], r0)
        pl = store.loaded.get(cid)
        if pl:
            out["rho"][G.r0_tag(r0)] = {
                "marginal_vote_lag1_acf": pl.get("marginal_vote_lag1_acf"),
                "all_vote_lag1_acf": pl.get("all_vote_lag1_acf"),
                "marginal_acf_n_images": pl.get("marginal_acf_n_images"),
                "all_acf_n_images": pl.get("all_acf_n_images"),
                "cell": cid,
            }
    return out


def _ladder_accuracy(payload, n_passes):
    """Whole-test-set task accuracy at one rung of a phase-3 ladder."""
    for rung in payload.get("ladder", []):
        if rung.get("n_passes") == n_passes:
            return rung.get("task_accuracy")
    return None


# ---------------------------------------------------------------------------
# captions
# ---------------------------------------------------------------------------

def fmt(stat, digits=2):
    if not stat:
        return "n/a"
    lo = stat.get("lo", stat.get("min"))
    hi = stat.get("hi", stat.get("max"))
    kind = stat.get("ci_kind")
    body = (f"{stat['mean']:.{digits}f} "
            f"[{lo:.{digits}f}-{hi:.{digits}f}], n={stat['n']}")
    return body + (f", {kind}" if kind else "")


def _rule_table(A, series, tags=None, digits=2, rules=PEER_RULES):
    tags = tags or R0_TAGS
    A("| rule | " + " | ".join(f"r0={t}" for t in tags) + " |")
    A("|---|" + "---|" * len(tags))
    for rule in rules:
        row = series.get(rule) or {}
        A(f"| {RULE_STYLE[rule][3]} | "
          + " | ".join(fmt(row.get(t), digits) for t in tags) + " |")


def _pct(stat, digits=2):
    """A stat dict rendered as a percentage range (mean [min-max], n)."""
    if not stat:
        return "n/a"
    lo = stat.get("lo", stat.get("min"))
    hi = stat.get("hi", stat.get("max"))
    return (f"{100 * stat['mean']:.{digits}f}% "
            f"[{100 * lo:.{digits}f}-{100 * hi:.{digits}f}%], n={stat['n']}")


def write_manuscript_stats(A, stats):
    """The 'quoted in a sentence' numbers, appended after the figure sections.

    These are not plotted anywhere, which is exactly why they need to be here:
    the manuscript may not state a number this file does not.
    """
    A("## Manuscript statistics — quoted in the text, not on any axis")
    A("")
    A("Same cells, same conventions as the figures above (cluster-robust over "
      "train seeds where three exist, min-max otherwise; block A defines the "
      "stratum, block B the reference decision). These feed prose only, so "
      "they are NOT counted in any figure's cell accounting.")
    A("")

    A("### M1 — marginal-stratum share of the test set (block-A margin < "
      f"{G.MARGIN_CUT})")
    A("")
    A("| dataset | share | per train seed (n_marginal / n_total) |")
    A("|---|---|---|")
    for dataset, ent in stats["marginal_share"].items():
        rows = ent["per_train_seed"]
        detail = ", ".join(
            f"ts{s}: {r['n_marginal']}/{r['n_total']} = {100 * r['frac']:.2f}%"
            for s, r in rows.items()) or "n/a"
        A(f"| {dataset} | {_pct(ent['frac'])} | {detail} |")
    A("")

    A("### M2 — the aggregate null: whole-test-set stopping error over alpha")
    A("")
    agg = stats["aggregate_null"]
    A(f"- {RULE_STYLE[agg['rule']][3]} rule, **'{agg['stratum']}' stratum "
      f"(all {G.N_TEST} images, ties excluded)**, "
      f"{agg['arm'][0]}/{agg['arm'][1]} arm, {agg['dataset']}. This is the "
      "rule with the LARGEST whole-test-set error of the four peers, so it is "
      "the disclosure number; the max over r0 is what 'at most' refers to.")
    A("")
    A("| r0 | " + " | ".join(R0_TAGS) + " |")
    A("|---|" + "---|" * len(R0_TAGS))
    A("| all-stratum error / alpha | "
      + " | ".join(fmt(agg["over_alpha_by_r0"].get(t), 3)
                   for t in R0_TAGS) + " |")
    A("")
    if agg["worst"]:
        w = agg["worst"]
        hi = w.get("hi", w.get("max"))
        A(f"- WORST r0 = {agg['worst_r0']}: **{w['mean']:.3f}** x alpha "
          f"[{w.get('lo', w.get('min')):.3f}-{hi:.3f}] — i.e. the "
          "whole-test-set effect never reaches the nominal level, and the "
          "manuscript's 'null over the whole test set' means at most "
          f"{w['mean']:.2f} x alpha.")
    else:
        A("- PENDING: no phase-2 cells for the primary arm on disk.")
    A("")

    A("### M3 — scored denominator after block-B reference-tie exclusion")
    A("")
    A("| dataset | block-B ties per train seed | scored n seen in phase 2 |")
    A("|---|---|---|")
    for dataset, ent in stats["tie_exclusion"].items():
        ties = ", ".join(f"ts{s}: {v}" for s, v in
                         ent["reference_tie_count_per_train_seed"].items())
        seen = ", ".join(f"{v:,}" for v in ent["scored_denominators_seen"])
        A(f"| {dataset} | {ties or 'n/a'} | {seen or 'n/a'} |")
    A("")
    A(f"- the full test split is {G.N_TEST:,} images; a block-B plurality tie "
      "has no reference decision to score against, so it is excluded and "
      "counted rather than broken.")
    A("")

    A("### M4 — reference noise floor: recorded cross-block disagreement")
    A("")
    A("- 1 - blocks_agree_frac between the two INDEPENDENT 1024-pass "
      "ideal-source reference blocks, as recorded in the phase-1 cells. This "
      "is the residual reference noise that survives the split-sample "
      "convention. It inflates every reported error ADDITIVELY, and an additive "
      "term common to numerator and denominator does NOT cancel in a ratio — it "
      "pulls the ratio TOWARD UNITY, so the calibration-referenced factors are "
      "attenuated by it rather than freed of it (convergence review 1 item 3). "
      "M7 measures how much, by removing it.")
    A("")
    A("| dataset | disagreement, whole test set | disagreement, marginal "
      "stratum |")
    A("|---|---|---|")
    for dataset, ent in stats["reference_noise_floor"].items():
        A(f"| {dataset} | {_pct(ent['all'], 3)} | {_pct(ent['marginal'], 2)} |")
    A("")

    A("### M5 — streaming first-read transient (r0=0.25 fixed-N ladder)")
    A("")
    st = stats["streaming_transient"]
    A(f"- the N=1 readout in streaming mode returns the PREVIOUS input's "
      f"state, so single-pass accuracy sits at chance; it recovers by "
      f"N={st['recovery_n']}. Whole-test-set task accuracy on the phase-3 "
      f"ladder at r0={st['r0']} (marginal stratum in parentheses).")
    A("")
    A(f"| arm / dataset | N=1 | N={st['recovery_n']} | cells |")
    A("|---|---|---|---|")
    for key, ent in st["arms"].items():
        cells = []
        for n in (1, st["recovery_n"]):
            e = ent[f"N={n}"]
            cells.append(fmt(e["task_accuracy"], 4)
                         + " (" + fmt(e["task_accuracy_marginal"], 4) + ")")
        A(f"| {key} | " + " | ".join(cells) + f" | {ent['n_cells']} |")
    if not st["arms"]:
        A("| n/a | n/a | n/a | 0 |")
    A("")

    A("### M6 — the WHOLE-TEST-SET picture: small LEVEL, not a small FACTOR")
    A("")
    A("- The three quantities the whole-test-set claim needs, all on the 'all' "
      "stratum (every scored image, ties excluded), primary arm, MNIST. Read "
      "them together: the calibration-referenced FACTOR over the whole test set "
      "is as large as the marginal-stratum one, so the whole-set statement is a "
      "statement about the LEVEL (error / alpha), not about the effect. The "
      "dilution is by EASY inputs, whose stopping error is exactly zero — not "
      "by inputs decided in few passes.")
    A("")
    A("Whole-test-set calibration-referenced factor (each rule's 'all'-stratum "
      "error divided by the SAME rule's pooled r0=inf 'all'-stratum error, "
      "F1(a)'s convention on the whole test set):")
    A("")
    _rule_table(A, {k: v["series"]
                    for k, v in stats["whole_test_set"]["factor_by_rule"]
                    .items()})
    A("")
    A("| rule | ideal 'all'-stratum error (the denominator) | cells |")
    A("|---|---|---|")
    for rule, ent in stats["whole_test_set"]["factor_by_rule"].items():
        v = ent["ideal_all_stratum_error"]
        A(f"| {RULE_STYLE[rule][3]} | " + (f"{v:.6f}" if v else "n/a")
          + " | " + fmt(ent["ideal_cells"], 6) + " |")
    A("")
    A("Whole-test-set LEVEL (error / alpha) — this is the quantity that is "
      "small, and 'at most' means the max over r0 of the largest rule:")
    A("")
    _rule_table(A, stats["whole_test_set"]["level_over_alpha_by_rule"],
                digits=3)
    A("")
    A("EASY substratum (block-A margin >= "
      f"{G.MARGIN_CUT}) stopping error — the dilution, stated as its own "
      "number:")
    A("")
    _rule_table(A, stats["whole_test_set"]["easy_error_by_rule"], digits=6)
    A("")
    A("MARGINAL stratum's share of the cell's ABSOLUTE wrong-decision count "
      "(wrong_marginal / wrong_all) — the stratum is ~8% of the images and "
      "carries this much of the error mass:")
    A("")
    _rule_table(A, stats["whole_test_set"]
                ["marginal_error_mass_share_by_rule"], digits=4)
    A("")

    A("### M7 — floor-free factors on the AGREE substratum")
    A("")
    ff = stats["floor_free"]
    A("- The residual cross-block disagreement (M4) inflates the numerator AND "
      "the denominator of a calibration-referenced ratio by the same ADDITIVE "
      "amount, which pulls the ratio TOWARD UNITY — it does not cancel. This "
      "table removes it by restricting the marginal stratum to images where "
      "block A's decision equals block B's reference, then applies F1(a)'s "
      "normalization unchanged (per-cell error / pooled ideal error on the same "
      "substratum, cluster-robust over train seeds). Larger factors here mean "
      "the reported ones are CONSERVATIVE.")
    if ff.get("available"):
        A("")
        _rule_table(A, ff["series"], rules=tuple(ff["series"]))
        A("")
        A("| rule | ideal error on the agree substratum | images in it |")
        A("|---|---|---|")
        for rule in ff["series"]:
            v = ff["ideal_error"].get(rule)
            A(f"| {RULE_STYLE[rule][3]} | " + (f"{v:.6f}" if v else "n/a")
              + f" | {ff['n_images'].get(rule)} |")
        A("")
        A(f"- {ff.get('note', '')}")
    else:
        A("- PENDING: needs the phase-1 cells' `decision_block_a` array and "
          "phase-2 per-image arrays on disk.")
    A("")

    A("### M8 — on-chip rho substitution (convergence review 1 item 8)")
    A("")
    oc = stats["onchip_rho"]
    A("- The headline Markov rule is calibrated with the MARGINAL-stratum vote "
      "ACF, whose mask comes from the block-A ideal-source reference — a chip "
      "does not have that. This check feeds the same rule, same thresholds, the "
      "ACF over ALL FLUCTUATING INPUTS instead (phase-2a field "
      "`all_vote_lag1_acf`): no reference block, no labels, no test data. It is "
      "the SMALLER number, so it is the conservative substitution — less "
      "correction — and the question is whether compliance survives it.")
    if oc.get("available"):
        A("")
        A("| r0 | rho (marginal mask, headline) | rho (all fluctuating, "
          "on-chip) | images (marg / all) |")
        A("|---|---|---|---|")
        for tag, ent in oc["rho"].items():
            A(f"| {tag} | {ent['marginal_vote_lag1_acf']:.4f} | "
              f"{ent['all_vote_lag1_acf']:.4f} | "
              f"{ent['marginal_acf_n_images']} / {ent['all_acf_n_images']} |")
        A("")
        A("| r0 | cells inside alpha | mean passes (whole test set) | "
          "marginal error | truncation | headline rule, same r0 |")
        A("|---|---|---|---|---|---|")
        for tag, ent in oc["by_r0"].items():
            b = ent["baseline_marginal_rho_rule"]
            A(f"| {tag} | **{ent['n_cells_inside_alpha']}/{ent['n_cells']}** | "
              f"{fmt(ent['mean_passes'], 2)} | "
              f"{fmt(ent['marginal_error'], 4)} | "
              f"{fmt(ent['truncation_rate'], 3)} | "
              f"{b['n_cells_inside_alpha']}/{b['n_cells']} inside alpha at "
              f"{fmt(b['mean_passes'], 2)} passes |")
        A("")
        A("- the substitution is a rule-level re-run of an already recorded "
          "device condition, so it reuses the phase-2a measurement and the "
          "phase-1 reference unchanged and adds no new device cell. It is "
          "deliberately absent from every figure: within a cell all rules "
          "replay the SAME recorded vote stream, so plotting it beside the "
          "headline rule would show a paired replay as an independent point.")
    else:
        A("- PENDING: no markov_onchiprho cells on disk.")
    A("")

    A("### M9 — corrected rule vs the e-process, head to head "
      "(convergence review 2 item 1)")
    A("")
    A("- BOTH rules keep 100% of replicate cells inside alpha at every "
      "r0 >= 1, so no rule is 'the only rule' that does. They separate "
      "elsewhere: the corrected rule is CHEAPER above the floor, holds "
      "PARTIAL compliance below it where the e-process holds none, and sits "
      "lower in LEVEL at every sampling ratio; the e-process needs no rho "
      "measurement at all, which is its own advantage. Quote the comparison, "
      "not an exclusivity.")
    A("")
    hh = stats["corrected_vs_eprocess"]["by_rule"]
    A("| rule | quantity | " + " | ".join(R0_TAGS) + " |")
    A("|---|---|" + "---|" * len(R0_TAGS))
    for rule, ent in hh.items():
        s = ent["series"]
        for key, lab, dig in (("mean_passes", "mean passes (whole test)", 2),
                              ("marginal_error", "marginal error", 4),
                              ("level_over_own_bound",
                               "error / own nominal bound", 2)):
            A(f"| {RULE_STYLE[rule][3]} | {lab} | "
              + " | ".join(fmt(s.get(t, {}).get(key), dig)
                           for t in R0_TAGS) + " |")
        A(f"| {RULE_STYLE[rule][3]} | cells inside alpha | "
          + " | ".join(
              (f"{s[t]['n_cells_inside_alpha']}/{s[t]['n_cells']} "
               f"({100 * s[t]['frac_cells_inside_alpha']:.0f}%)")
              if t in s else "n/a" for t in R0_TAGS) + " |")
    A("")

    A("### M10 — easy substratum, per cell (convergence review 2 item 5)")
    A("")
    A("- The cluster-robust MEAN of the easy-substratum error rounds to zero "
      "for three of the four rules, but only the corrected rule is exactly "
      "zero in every cell. Per-cell maxima and non-zero cell counts below; "
      "`max misdec.` is the largest number of wrong decisions in any one cell "
      "out of the easy substratum's image count.")
    A("")
    ed = stats["easy_substratum_detail"]
    A(f"- easy substratum = {ed['stratum']}")
    A("")
    A("| rule | quantity | " + " | ".join(R0_TAGS) + " |")
    A("|---|---|" + "---|" * len(R0_TAGS))
    for rule, row in ed["by_rule"].items():
        for key, lab, form in (
                ("max", "max over cells", lambda v: f"{v:.6f}"),
                ("n_cells_nonzero", "non-zero cells", None),
                ("max_misdecisions", "max misdec. in a cell", None),
                ("easy_n_images", "easy images per cell", None)):
            vals = []
            for t in R0_TAGS:
                e = row.get(t)
                if not e:
                    vals.append("n/a")
                elif key == "n_cells_nonzero":
                    vals.append(f"{e[key]}/{e['n_cells']}")
                elif form:
                    vals.append(form(e[key]))
                else:
                    vals.append(str(e[key]))
            A(f"| {RULE_STYLE[rule][3]} | {lab} | " + " | ".join(vals) + " |")
    A("")

    A("### M11 — fixed-N ladder accuracy, CLUSTER-ROBUST "
      "(convergence review 2 item 6)")
    A("")
    lc = stats["ladder_accuracy_cluster"]
    A("- F3's ladder band is min-max over ladder cells. Quoted beside the "
      "peer rules' cluster-robust accuracies that is a convention mix, and "
      "the min-max bracket is the narrower one. These are the same rungs "
      f"(r0={lc['r0']}, whole test set, {lc['n_cells']} cells) under the "
      "cluster-robust convention; the manuscript uses these.")
    A("")
    A("| N | cluster-robust whole-test accuracy |")
    A("|---|---|")
    for n, s in lc["by_n"].items():
        A(f"| {n} | {fmt(s, 4)} |")
    A("")

    A("### M12 — where the ideal frontier reaches the corrected rule "
      "(convergence review 2 item 8)")
    A("")
    fc = stats["frontier_crossing"]
    A(f"- {RULE_STYLE[fc['rule']][3]} at r0={fc['r0']}: marginal error "
      f"{fmt(fc['marginal_error'], 4)} at "
      f"{fmt(fc['whole_test_mean_passes'], 2)} whole-test-set passes.")
    A("- bracketing ideal fixed-N rungs: "
      + ", ".join(f"N={n}: {v:.4f}"
                  for n, v in fc.get("bracketing_rungs", {}).items()))
    if fc.get("crossing_n"):
        A(f"- the ideal frontier is still ABOVE that error at N=128 and "
          f"reaches it only at N = {fc['crossing_n']:.0f} (log-linear "
          f"interpolation between rungs, F6's convention), i.e. "
          f"{fc['budget_ratio']:.1f}x the corrected rule's mean budget.")
    else:
        A("- PENDING: no ideal fixed-N ladder on disk.")
    A("")

    A("### M13 — ECE binning convention and its sensitivity "
      "(convergence review 2 item 7)")
    A("")
    ec = stats["ece_convention"]
    A(f"- FROZEN convention (what every reported ECE uses): {ec['convention']}")
    A(f"- ALTERNATIVE a reader may implement instead: {ec['alternative']}")
    A("- The two agree wherever the confidences are not heavily tied and "
      "diverge at large N under ideal randomness, where the ECE is smallest. "
      "The manuscript must name the convention and the sensitivity rather "
      "than quote the fourth decimal as if it were convention-free.")
    A("")
    A("| r0 | N | distinct confidences | ECE (frozen) | ECE (quantile edges, "
      "collapsed) |")
    A("|---|---|---|---|---|")
    for tag, row in ec["by_r0"].items():
        for n, e in row.items():
            A(f"| {tag} | {n} | {e['distinct_confidences']} | "
              f"{e['ece_frozen']:.4f} | "
              f"{e['ece_quantile_edges_collapsed']:.4f} |")
    A("")
    A("Degraded/ideal ratio (r0=0.5 over r0=inf) under each convention:")
    A("")
    A("| N | ratio (frozen) | ratio (quantile edges, collapsed) |")
    A("|---|---|---|")
    for n, r in stats["ece_convention"]["ratio_degraded_over_ideal"].items():
        A(f"| {n} | " + " | ".join(
            (f"{r[k]:.1f}" if r.get(k) else "n/a")
            for k in ("ratio_frozen", "ratio_quantile_edges_collapsed"))
          + " |")
    A("")


def write_captions(path, records, data, sha, stats=None):
    L = []
    A = L.append
    A("# Caption stubs — generated by experiments/make_figures.py")
    A("")
    A(f"_git {sha[:8]} · code_hash {G.code_hash()[:12]} · "
      f"{_dt.datetime.now(_dt.timezone.utc):%Y-%m-%d %H:%MZ} "
      "· regenerate, never hand-edit; the manuscript captions must not say "
      "anything this file does not._")
    A("")
    A("Conventions used everywhere below. **Stopping error** = P(adaptive "
      "decision != reference decision) where the reference is the majority "
      f"vote over T_ref={G.T_REF} ideal-source passes of REFERENCE BLOCK B, "
      "and the **marginal stratum** is the images whose INDEPENDENT BLOCK-A "
      f"margin is < {G.MARGIN_CUT} (split-sample convention, audit r1 item 3). "
      "Block-B reference ties are excluded and counted. Plurality ties inside "
      "a rule are broken uniformly at random with a seeded per-lane stream, "
      "and the tie rate is reported per cell. alpha = "
      f"{ALPHA}; the SPRT rules' own bound is alpha/(1-beta) = "
      f"{ALPHA / (1 - BETA):.4f}; t_max = {G.T_MAX}; MNIST test split "
      f"(n={G.N_TEST}); primary arm = {PRIMARY_ARM[0]} devices in "
      f"{PRIMARY_ARM[1]} mode; at r0=inf every arm shares the same "
      "ideal-source cells, which are also each rule's CALIBRATION cells. "
      "Aggregates are CLUSTER-ROBUST over train seeds where three exist "
      "(mean of the three cluster means +/- t(2)=4.303 x SE) and MIN-MAX over "
      "cells otherwise — every table cell states which. Marker area encodes "
      "the truncation rate. Every main-text curve is filtered to the frozen "
      f"protocol knobs {DEFAULT_KNOBS}; the sensitivity rows live in F8.")
    A("")

    for key in ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"):
        if key not in records:      # --only rebuilt a subset
            continue
        rec = records[key]
        A(f"## {key} — "
          + ", ".join(f"`{f}`" for f in rec.get("files", [rec["file"]])))
        A("")
        A(f"- cells consumed: **{rec['n_cells_used']}**; expected but "
          f"missing: **{rec['n_cells_missing']}**"
          + ("" if rec["complete"] else "  ← FIGURE IS ON PARTIAL DATA"))
        if key == "F1":
            norm, raw, acc, calib = data["F1"]
            A("- Panel (a) THE HEADLINE: marginal-stratum stopping error "
              "divided by the SAME RULE's ideal-randomness (r0=inf) error, "
              "versus r0; log y; the dotted line at 1.0 is that calibration. "
              "This is the degradation caused by entropy quality, with each "
              "rule's own conservatism divided out.")
            A("- Panel (b) secondary: the raw level, each rule over its OWN "
              "nominal bound (alpha for Dirichlet and the e-process, "
              "alpha/(1-beta) for the SPRTs).")
            A("- Panel (c): marginal-stratum task accuracy at fixed "
              f"N in {list(OVERLAY_NS)} — matched stratum AND matched budget, "
              "all available train seeds. The honest statement is that the "
              "rule degrades FASTER than accuracy, not before it.")
            A("")
            A("Ideal-randomness calibration (the denominator of panel a):")
            A("")
            A("| rule | ideal marginal error | cells |")
            A("|---|---|---|")
            for rule in PEER_RULES:
                cal = calib.get(rule, {})
                v = cal.get("ideal_marginal_error")
                A(f"| {RULE_STYLE[rule][3]} | "
                  + (f"{v:.4f}" if v else "n/a") + " | "
                  + fmt(cal.get("ideal_cells"), 4) + " |")
            A("")
            A("Panel (a) — calibration-referenced degradation:")
            A("")
            _rule_table(A, norm)
            A("")
            A("Panel (b) — raw error / own nominal bound:")
            A("")
            _rule_table(A, {k: v["series"] for k, v in raw.items()})
            A("")
            A("Panel (c) — marginal-stratum accuracy at fixed N:")
            A("")
            A("| N | " + " | ".join(f"r0={t}" for t in R0_TAGS) + " |")
            A("|---|" + "---|" * len(R0_TAGS))
            for n, row in acc.items():
                A(f"| {n} | " + " | ".join(fmt(row.get(t), 4)
                                           for t in R0_TAGS) + " |")
        elif key == "F2":
            f2_series, f2_throttle = data["F2"]
            tags = [G.r0_tag(r) for r in FINITE_R0]
            A("- Mean stochastic passes per decision versus r0, per rule, "
              "log y. Marker FILL encodes the guarantee filter (filled = "
              "every replicate cell has marginal-stratum stopping error with "
              "Wilson-95 upper bound <= alpha; half = some; open = none); "
              "marker AREA encodes the truncation rate.")
            A("- SCOPE, read this before quoting any number in this section: "
              "the mean passes per decision and the truncation rates below are "
              "WHOLE-TEST-SET means (cell field `mean_samples` / "
              "`truncation_rate`, all scored images), because cost per "
              "decision is a deployment quantity over the traffic the device "
              "actually sees. Only the guarantee filter ('ok %') is a "
              "MARGINAL-STRATUM quantity. F6's budget axis is the "
              "marginal-stratum mean instead, and says so there; the two bases "
              "are each internally coherent and must never be mixed inside one "
              "ratio.")
            A("- Dashed black: the field's standard remedy — the naive "
              f"Dirichlet rule at the compliant r0={G.R0_THROTTLED:g}, charged "
              "in LATENCY-equivalent passes only (passes x r0_throttled / "
              "r0_operating). Bits and MACs do NOT scale with the slowdown, so "
              "any throughput claim must state its unit; the settle overhead "
              "(settle_ratio / r0 pass slots per decision) is listed below and "
              "must be charged when quoting a throughput ratio.")
            A("")
            A("| rule | " + " | ".join(f"r0={t}" for t in tags) + " |")
            A("|---|" + "---|" * len(tags))
            for rule in RULE_ORDER:
                cells = []
                for t in tags:
                    ent = (f2_series.get(rule) or {}).get(t) or {}
                    fr = ent.get("frac_cells_meeting_guarantee")
                    tr = ent.get("truncation_rate")
                    cells.append(
                        fmt(ent.get("mean_samples"), 1)
                        + (f" (ok {fr:.0%})" if fr is not None else "")
                        # two decimals: at a few parts in a thousand a 0-dp
                        # percentage prints "0%", and Table 2 of the manuscript
                        # quotes this column (convergence review 1 item 17)
                        + (f" (trunc {100 * tr['mean']:.2f}%)" if tr else ""))
                A(f"| {RULE_STYLE[rule][3]} | " + " | ".join(cells) + " |")
            if f2_throttle:
                eq = f2_throttle["latency_equivalent_passes"]
                st = f2_throttle["settle_latency_passes_per_decision"]
                A("| throttled naive (latency-charged) | "
                  + " | ".join(f"{eq[t]:.1f}" for t in tags) + " |")
                A("| settle overhead, pass slots/decision | "
                  + " | ".join(f"{st[t]:.1f}" for t in tags) + " |")
                A("")
                lfl = f2_throttle.get("like_for_like_wald")
                if lfl:
                    A("| throttled WALD (like-for-like, latency-charged) | "
                      + " | ".join(
                          f"{lfl['latency_equivalent_passes'][t]:.1f}"
                          for t in tags) + " |")
                A("")
                if lfl:
                    A("- LIKE-FOR-LIKE baseline (item 18): the plotted baseline "
                      "is the cheapest naive rule (Dirichlet), a different rule "
                      "family from the corrected one, which favours throttling. "
                      "Throttled WALD costs "
                      + fmt(lfl["base_mean_samples"], 2)
                      + f" passes at r0={G.R0_THROTTLED:g} (guarantee met in "
                      f"{lfl['base_frac_cells_meeting_guarantee']:.0%} of its "
                      "cells), i.e. "
                      + f"{lfl['latency_equivalent_passes']['1']:.1f} "
                      "latency-equivalent slots against the corrected rule's "
                      "15.1 at r0=1 — a ratio near 3x rather than 1.4x. Quote "
                      "whichever, but name which.")
                A("- throttling baseline base cost: "
                  + fmt(f2_throttle["base_mean_samples"], 2)
                  + f" passes at r0={G.R0_THROTTLED:g}"
                  + (f"; guarantee met in "
                     f"{f2_throttle['base_frac_cells_meeting_guarantee']:.0%} "
                     "of its cells"
                     if f2_throttle["base_frac_cells_meeting_guarantee"]
                     is not None else ""))
        elif key == "F3":
            f3 = data["F3"]
            A("- Task accuracy versus mean passes per decision, one panel per "
              f"r0 in {list(PARETO_R0)}; markers = adaptive rules (phase 2), "
              "marker area = truncation rate, error bars = cluster-robust or "
              "min-max as labelled. Log x. Accuracy and passes are both "
              "WHOLE-TEST-SET means. Co-headline with F1.")
            A("- Grey curve = the fixed-N ladder "
              f"(N in {list(G.LADDER)}, phase 3) AT THE SAME r0 AS THE PANEL, "
              "not the ideal ladder: it is the same-entropy fixed-budget "
              "reference, so each panel compares adaptive against fixed at "
              "matched randomness quality. Only the r0=inf panel's grey curve "
              "is the ideal ladder. (The ideal ladder as a cross-r0 frontier "
              "is F6's denominator, and is labelled as such there.)")
            A("")
            A("Fixed-N ladder, WHOLE-TEST-SET task accuracy per rung (min-max "
              "over ladder cells) — the reference the adaptive markers are "
              "compared against inside each panel:")
            A("")
            ladder_ns = sorted({int(n) for panel in f3.values()
                                for n in panel["fixed_n_ladder"]})
            A("| r0 | " + " | ".join(f"N={n}" for n in ladder_ns) + " |")
            A("|---|" + "---|" * len(ladder_ns))
            for tag, panel in f3.items():
                A(f"| {tag} | " + " | ".join(
                    fmt(panel["fixed_n_ladder"].get(str(n)), 4)
                    for n in ladder_ns) + " |")
            A("")
            for tag, panel in f3.items():
                nlad = len(panel["fixed_n_ladder"])
                nrule = len(panel["adaptive_rules"])
                A(f"- r0={tag}: {nlad} ladder rungs, {nrule} adaptive rules "
                  "plotted")
                for rule, ent in panel["adaptive_rules"].items():
                    A(f"  - {RULE_STYLE[rule][3]}: passes "
                      f"{fmt(ent['mean_samples'], 1)}; accuracy "
                      f"{fmt(ent['task_accuracy'], 4)}; truncation "
                      f"{fmt(ent['truncation_rate'], 3)}")
        elif key == "F4":
            f4 = data["F4"]
            A("- **NOT A MANUSCRIPT FIGURE.** Convergence review 1 item 12 found "
              "the reliability panel uninformative and offered cropping or "
              "dropping it; the manuscript keeps the ECE numbers below in the "
              "text and omits the panel, so the printed figure numbers run "
              "F1, F2, F3 = Figs. 1-3 and F5..F8 = Figs. 4-7. The panel is "
              "still built here, and the numbers the text quotes are these.")
            A(f"- Reliability diagrams, {ECE_BINS} equal-mass bins, conf. "
              "= top_counts / n_passes for fixed-N readouts "
              f"N in {list(G.CALIB_NS)}; left = ideal randomness, right = "
              "r0=0.5. ECE in the legend is computed on the pooled "
              "(confidence, correct) pairs across noise seeds; no temperature "
              "scaling anywhere.")
            A("- Equal-mass binning sorts confidences with a STABLE sort "
              "(numpy kind='stable') and splits the sorted order into "
              f"{ECE_BINS} near-equal groups. Small-N readouts produce heavy "
              "confidence ties (conf = top_counts / N takes few distinct "
              "values), so the bin edges — and hence ECE at the fourth decimal "
              "— depend on the tie order; the stable sort fixes that order to "
              "the input order and makes the number reproducible rather than "
              "tie-order free.")
            for tag, panel in f4.items():
                if not panel["by_n"]:
                    A(f"- r0={tag}: PENDING (no phase-4 cells on disk)")
                    continue
                bits = ", ".join(f"N={n}: ECE={v['pooled_ece']:.4f}"
                                 for n, v in panel["by_n"].items())
                A(f"- r0={tag} ({panel['n_cells']} cells pooled): {bits}")
        elif key == "F5":
            acf_curves, infl, decomp, entropy, by_arm, rho_used = data["F5"]
            A("- Panel (a): mean lag-k autocorrelation of the vote indicator, "
              f"k = 1..{G.ACF_MAX_LAG}, one curve per r0, over IMAGES WHOSE "
              "VOTES FLUCTUATE (a saturated image has no defined "
              "autocorrelation and is excluded — the panel is not over all "
              "images).")
            A("- Panel (b): the lag-1 vote ACF versus r0 for EVERY arm "
              "(the headline is reported as a range across arms, so the arms "
              "are shown together), with the ideal-source i.i.d. control and "
              "the open squares marking the rho measured on the VALIDATION "
              "split and fed to the Markov rule — no labels, no test data. The "
              f"validation measurement reads the first {G.N_VAL_CALIB} images "
              "of the validation split, not all of it.")
            _rho_n = ((stats or {}).get("onchip_rho", {})
                      .get("rho", {}).get("1", {}))
            A("- SCOPE OF THE rho MEASUREMENT (convergence review 2 item 12): "
              "the phase-2a cells run on TRAIN SEED 0 only, and the single "
              "number they produce is transferred UNCHANGED to all three "
              "trained models and to the test split. Within that measurement "
              "the marginal-stratum average is taken over the images whose "
              "votes actually fluctuate, which is a small set — "
              + (f"{_rho_n['marginal_acf_n_images']} of the first "
                 f"{G.N_VAL_CALIB} validation images at r0=1, against "
                 f"{_rho_n['all_acf_n_images']} for the "
                 "all-fluctuating-inputs version"
                 if _rho_n else "see M8 for the counts")
              + " (M8's table gives both counts at both r0). Both facts make "
              "the calibration cheaper than it looks and both are stated "
              "rather than left implicit.")
            if rho_used:
                A("- rho HANDED TO MarkovSPRT (phase-2a, validation, no "
                  "labels). This is the number the rule consumes; the curves "
                  "in (b) are the TEST-set realization of the same quantity "
                  "and the two differ, most visibly at ideal randomness where "
                  "the rule is fed "
                  + (f"{rho_used['ideal-ideal_rinf']:+.4f}"
                     if "ideal-ideal_rinf" in rho_used else "n/a")
                  + " while the test ACF is "
                  + (f"{ideal_ref['mean']:+.4f}"
                     if (ideal_ref := (by_arm.get(
                         "ideal source (shared by all arms)", {})
                         .get("inf"))) else "n/a")
                  + ": " + ", ".join(f"{k}={v:+.4f}"
                                     for k, v in sorted(rho_used.items())))
            A("- Panel (c): the integrated variance inflation factor implied "
              "by that curve (initial-positive-sequence truncation) against "
              "the AR(1) prediction (1+rho)/(1-rho) from the same lag-1 number "
              "the Markov rule is fed. Their ratio crossing 1 is the "
              "predictor of the first-order correction's validity floor.")
            A("- Panel (d): the marginal-stratum stopping error at r0=0.5 "
              "split into the fired channel (the bounded one) and the "
              "truncated channel (budget error); bars sum to the total.")
            A("")
            A("| r0 | ACF(1) | ACF(2) | ACF(4) | ACF(8) | ACF(16) | cells |")
            A("|---|---|---|---|---|---|---|")
            for tag, ent in acf_curves.items():
                vals = ent["mean_acf"]

                def at(k, vals=vals):
                    return f"{vals[k - 1]:.3f}" if len(vals) >= k else "n/a"
                A(f"| {tag} | {at(1)} | {at(2)} | {at(4)} | {at(8)} | "
                  f"{at(16)} | {ent['n_cells']} |")
            A("")
            A("| arm | " + " | ".join(f"r0={t}" for t in R0_TAGS) + " |")
            A("|---|" + "---|" * len(R0_TAGS))
            for arm, row in by_arm.items():
                A(f"| {arm} | " + " | ".join(fmt(row.get(t), 3)
                                             for t in R0_TAGS) + " |")
            A("")
            A("| inflation | " + " | ".join(f"r0={t}" for t in R0_TAGS) + " |")
            A("|---|" + "---|" * len(R0_TAGS))
            for key2, row in infl.items():
                A(f"| {key2} | " + " | ".join(fmt(row.get(t), 3)
                                              for t in R0_TAGS) + " |")
            A("")
            A("- entropy RATE of the fitted two-state vote chain (bits per "
              "pass, images whose votes fluctuate) next to the i.i.d. entropy "
              "of the same marginal: the GAP is the predictability the "
              "correlated stream hands the stopping rule for free, and it is "
              "the quantity that justifies calling this an entropy-quality "
              "effect rather than merely a correlation.")
            A("")
            A("| entropy (bits/pass) | " + " | ".join(f"r0={t}"
                                                      for t in R0_TAGS) + " |")
            A("|---|" + "---|" * len(R0_TAGS))
            for key2, row in entropy.items():
                A(f"| {key2} | " + " | ".join(fmt(row.get(t), 4)
                                              for t in R0_TAGS) + " |")
            A("")
            A("| rule @ r0=0.5 | fired | truncated | total (marginal) |")
            A("|---|---|---|---|")
            for rule, dd in decomp.items():
                A(f"| {RULE_STYLE[rule][3]} | "
                  f"{fmt(dd['fired_contribution'], 4)} | "
                  f"{fmt(dd['truncated_contribution'], 4)} | "
                  f"{fmt(dd['marginal_stopping_error'], 4)} |")
        elif key == "F6":
            f6_series, frontier = data["F6"]
            A("- Each rule's marginal-stratum stopping error divided by the "
              "IDEAL-randomness fixed-N marginal error, log-linearly "
              "interpolated at that rule's OWN marginal mean budget: the "
              "effect size at matched stratum and matched budget. 1.0 means "
              "'no worse than ideal randomness would be with the same number "
              "of passes'.")
            A("- BASIS: the budget this interpolates at is the MARGINAL-STRATUM "
              "mean pass count, matching the stratum the error is measured on — "
              "not the whole-test-set mean that F2/F3 report as cost per "
              "decision. Both are correct on their own basis; any comparison "
              "that crosses them (e.g. a whole-set mean budget against a "
              "fixed-N rung) must name the basis it uses.")
            if frontier:
                A("- ideal frontier used: "
                  + ", ".join(f"N={n}: {e:.4f}" for n, e in frontier))
            A("")
            _rule_table(A, f6_series)
        elif key == "F7":
            A("- Fashion-MNIST replication of F1(a), same normalization. ONE "
              "train seed, so the band is min-max across noise-seed cells and "
              "no cluster-robust interval is claimed.")
            A("")
            _rule_table(A, data["F7"])
        elif key == "F8":
            n8 = data["F8"]
            A("- (a) markov_online, the DEGENERATE ABLATION. The PLOTTED "
              "quantity on the y-axis is the calibration-referenced error, the "
              "same normalization as F1(a) — error / this rule's own "
              "ideal-randomness marginal error — NOT a truncation percentage. "
              "Truncation is encoded by MARKER AREA only, and its values are "
              "listed next; they are what make this an ablation rather than a "
              "peer rule (near fixed-N=t_max at every r0, ideal randomness "
              "included).")
            A("- (a) truncation rates, WHOLE-TEST-SET means (cell field "
              "`truncation_rate`, all scored images — not a marginal-stratum "
              "rate): "
              + ", ".join(
                  f"r0={t}: " + (f"{s['mean']:.0%}" if s else "n/a")
                  for t, s in (n8["markov_online"].get("markov_online", {})
                               .get("truncation_rate", {}) or {}).items()))
            A("- SPLIT INTO TWO PRINTED FIGURES (convergence review 2 item 2): "
              "**F8a** = (a) online-rho ablation, (b) t_min, (c) settle ratio; "
              "**F8b** = (a) sigma_delta, (b) rate ceiling. Panel letters below "
              "are each file's own.")
            A("- F8a (b) t_min in {2, 4, 8} and (c) settle_ratio in {2, 4, 8} "
              "on the primary arm: the headline is not an artifact of either "
              "choice. In BOTH panels a SOLID line is the Dirichlet rule and a "
              "DASHED line the Wald SPRT; colour encodes the knob value. The "
              "caption must say so, because the legend only names the colours.")
            A("- F8b (a) device-to-device variation sigma_delta in "
              "{0, 0.5, 1.0}, mean-tau centred so dispersion is isolated from "
              "a mean slowdown, 3 chip seeds. SOLID = Dirichlet, DASHED = "
              "Markov SPRT (measured rho).")
            A("- F8b (b) the Arrhenius arm WITH the attempt-frequency rate "
              f"ceiling (C = tau_corr(0)/tau0 = {G.RATE_CEILING_R0:g}) against "
              "the uncapped Arrhenius and easy-plane arms: the ceiling removes "
              "the bias-driven self-decorrelation that made Arrhenius the "
              "weak-effect class.")
            A("- post-hoc stratum sensitivity from the per-image arrays, no "
              "extra grid cells. NO LONGER A PANEL (review 2 item 2): the "
              "margin cuts " + str(n8["posthoc"]["cut_values"])
              + " and the "
              + f"q >= {n8['posthoc']['q_cut']} restriction (reference "
                "top-class vote share) are COMPUTED here and TABULATED below, "
                "and the manuscript carries them as a small table and as "
                "prose. No figure may be cited for either.")
            if n8["posthoc"]["available"]:
                A("")
                A("Absolute marginal-stratum error at each cut (min-max over "
                  "cells), and the q-restricted error:")
                A("")
                A("| rule | r0 | cut 0.8 | cut 0.9 | cut 0.95 | "
                  "q>=0.75 (marginal) |")
                A("|---|---|---|---|---|---|")
                for rule, byr0 in n8["posthoc"]["margin_cuts"].items():
                    for tag, cuts in byr0.items():
                        q = (n8["posthoc"]["q_restriction"].get(rule, {})
                             .get(tag, {}) or {})
                        A(f"| {RULE_STYLE[rule][3]} | {tag} | "
                          + " | ".join(fmt(cuts.get(f"{c:g}"), 4)
                                       for c in n8["posthoc"]["cut_values"])
                          + " | " + fmt(q.get("error"), 4) + " |")
                ratios = n8["posthoc"].get("margin_cut_ratios") or {}
                if ratios:
                    A("")
                    A("CALIBRATION-REFERENCED ratio at each cut — the quantity "
                      "the manuscript quotes for cut invariance. Each per-cell "
                      "error is divided by the POOLED ideal-randomness error at "
                      "THE SAME cut and then aggregated over train seeds, "
                      "exactly F1(a)'s convention, so the cut-0.9 column "
                      "reproduces F1(a) by construction. Ratios formed by hand "
                      "from the rounded absolute means above differ in the "
                      "second decimal and are not this quantity.")
                    A("")
                    A("| rule | r0 | ratio @ cut 0.8 | ratio @ cut 0.9 | "
                      "ratio @ cut 0.95 |")
                    A("|---|---|---|---|---|")
                    for rule, byr0 in ratios.items():
                        for tag, row in byr0.items():
                            if tag in ("_denominator", G.r0_tag(np.inf)):
                                continue
                            A(f"| {RULE_STYLE[rule][3]} | {tag} | "
                              + " | ".join(
                                  fmt(row.get(f"{c:g}"), 2)
                                  for c in n8["posthoc"]["cut_values"])
                              + " |")
            else:
                A("- post-hoc table PENDING: needs the phase-1 split-reference "
                  "cells and phase-2 per-image arrays on disk.")
        if rec["n_cells_missing"]:
            A("")
            A(f"- MISSING CELLS ({rec['n_cells_missing']}), first 10: "
              + ", ".join(f"`{c}`" for c in rec["cells_missing"][:10])
              + (" ..." if rec["n_cells_missing"] > 10 else ""))
        A("")
    if stats:
        write_manuscript_stats(A, stats)
    path.write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

BUILDERS = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--grid-dir", default=str(GRID_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--only", nargs="+", default=list(BUILDERS),
                    choices=list(BUILDERS),
                    help="subset of figures to rebuild (the manifest then "
                         "covers only those)")
    args = ap.parse_args(argv)

    grid_dir, out_dir = Path(args.grid_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    store = GridStore(grid_dir)
    sha, dirty = git_sha(), git_dirty()
    print(f"figures @ {sha[:8]}{' (DIRTY TREE)' if dirty else ''} from "
          f"{grid_dir} ({len(store.loaded)} readable cells, "
          f"{len(store.expected)} expected by the grid def)", flush=True)
    if store.unreadable:
        warn(f"{len(store.unreadable)} unreadable/pending file(s) skipped: "
             + ", ".join(store.unreadable[:5])
             + (" ..." if len(store.unreadable) > 5 else ""))
    if store.stale:
        warn(f"{len(store.stale)} of {len(store.loaded)} cells were written at "
             f"a DIFFERENT code_hash than the current {G.code_hash()[:12]} — "
             "they are plotted but the aggregates MIX PROTOCOLS; re-run the "
             "grid before quoting these numbers (e.g. "
             + ", ".join(sorted(store.stale)[:3]) + ")")

    records, data = {}, {}
    if "F1" in args.only:
        records["F1"], norm, raw, acc, calib = figure_f1(store, out_dir)
        data["F1"] = (norm, raw, acc, calib)
    if "F2" in args.only:
        records["F2"], s, t = figure_f2(store, out_dir)
        data["F2"] = (s, t)
    if "F3" in args.only:
        records["F3"], data["F3"] = figure_f3(store, out_dir)
    if "F4" in args.only:
        records["F4"], data["F4"] = figure_f4(store, out_dir)
    if "F5" in args.only:
        records["F5"], a, i, dec, ent, byarm, rho_u = figure_f5(store, out_dir)
        data["F5"] = (a, i, dec, ent, byarm, rho_u)
    if "F6" in args.only:
        records["F6"], s, fr = figure_f6(store, out_dir)
        data["F6"] = (s, fr)
    if "F7" in args.only:
        records["F7"], data["F7"] = figure_f7(store, out_dir)
    if "F8" in args.only:
        records["F8"], data["F8"] = figure_f8(store, out_dir)

    all_missing = sorted({c for r in records.values()
                          for c in r["cells_missing"]})
    try:
        grid_rel = str(Path(grid_dir).resolve().relative_to(ROOT))
    except ValueError:
        grid_rel = str(grid_dir)          # outside the repo (tests use tmp)
    manifest = {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"),
        "script": "experiments/make_figures.py",
        "git_sha": sha,
        "git_dirty": dirty,
        "code_hash": G.code_hash(),
        "grid_dir": grid_rel,             # relative to the repo root (item 14)
        "grid_complete": not all_missing and not store.unreadable,
        "cells_readable": len(store.loaded),
        "cells_expected_by_grid": len(store.expected),
        "cells_unreadable_or_pending": store.unreadable,
        "cells_present_but_unexpected": store.unexpected,
        "provenance": store.sha_summary(),
        "conventions": {
            "alpha": ALPHA, "beta": BETA,
            "sprt_nominal_bound": ALPHA / (1.0 - BETA),
            "t_max": G.T_MAX, "t_ref": G.T_REF,
            "margin_cut": G.MARGIN_CUT, "ece_bins": ECE_BINS,
            "primary_arm": list(PRIMARY_ARM), "dataset": DATASET,
            "split": "test", "default_knobs": DEFAULT_KNOBS,
            "reference_convention": "split-sample: block A defines the "
                                    "stratum, block B the reference decision",
            "normalization": "primary = calibration-referenced (each rule "
                             "divided by its own r0=inf error); secondary = "
                             "raw over each rule's own nominal bound",
            "aggregation": "cluster-robust over train seeds where 3 exist "
                           "(t(2)=4.303 on the cluster means); min-max over "
                           "cells otherwise, labelled; per-cell Wilson "
                           "intervals are never pooled",
            "guarantee_filter": "marginal-stratum stopping error, Wilson-95 "
                                "upper bound <= alpha",
            "truncation_encoding": "marker area, with a size legend",
            "cost_proxy": "src/apbit/cost.py (frozen week 1); throttling "
                          "charged in LATENCY-equivalent passes only",
        },
        "figures": records,
        "cells_missing_union": all_missing,
        "n_cells_missing_union": len(all_missing),
    }
    (out_dir / "figures_manifest.json").write_text(
        json.dumps(manifest, indent=1))
    write_captions(out_dir / "captions_draft.md", records, data, sha,
                   stats=manuscript_stats(store))

    for key, rec in records.items():
        status = "complete" if rec["complete"] else \
            f"PARTIAL — {rec['n_cells_missing']} expected cells missing"
        print(f"  {key}: {rec['file']}  ({rec['n_cells_used']} cells, "
              f"{status})", flush=True)
        if not rec["complete"]:
            warn(f"{key} plotted on PARTIAL data: {rec['n_cells_missing']} "
                 f"expected cells missing, e.g. "
                 + ", ".join(rec["cells_missing"][:3]))
    print(f"manifest: {out_dir / 'figures_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
