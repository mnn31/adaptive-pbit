# Author: Manan Gupta <mnn@yogins.com>
"""Full experiment grid — phased, resumable, CONTENT-HASH-stamped.

One JSON per (config, seed) under results/grid2/. A cell is recomputed unless a
result file already exists AND records the CURRENT CODE HASH: source changed =>
results are stale, no exceptions. Every cell is written atomically (tmp file +
os.replace) the moment it finishes, so a kill -9 at hour four loses at most the
cell in flight. Execution is strictly sequential: the randomness sources are
stateful objects whose draw order IS the physics, and no multiprocessing pool
is allowed to interleave them.

FRESHNESS (audit r1 item 10). v1 stamped the HEAD git SHA, which invalidates
every cell on a commit that touched only the manuscript and validates cells
across a dirty tree. The stamp is now `code_hash` = sha256 over the
concatenated bytes of sorted(src/apbit/*.py) + this file: exactly the code that
can change a number, nothing that cannot. The git SHA is still recorded, as
provenance, but never used to decide freshness. Cells written before this
change carry no `code_hash` and are treated as INCOMPATIBLE — which they are,
because the protocol itself changed (split-sample reference, random tie-break,
per-image arrays). This driver therefore writes into results/grid2/ and leaves
results/grid/ untouched as the audited v1 record.

ACCEPTED_CODE_HASHES (convergence review 1 item 8). A content hash over this
file is strict by design: ANY edit to the driver invalidates all 1049 grid2
cells, including edits that cannot move a stored number. Item 8 asks for exactly
such an edit — a NEW rule variant (`markov_onchiprho`) on twelve NEW cells,
with no change to the frozen protocol, to the sources, to the seeds or to the
draw order of any existing cell. Re-running the whole grid to satisfy a hash
would destroy the audit trail the hash exists to protect, so the accepted set is
made explicit instead: `fresh_result` reuses a cell whose stamp is the CURRENT
hash or any hash listed in ACCEPTED_CODE_HASHES, and the list carries the reason
each entry is protocol-identical. The current hash is never hardcoded (it is a
hash OF this file, so writing it here would change it); it is always accepted,
which is what makes the new cells fresh. Any change that CAN move a stored
number must drop the superseded entry from this list rather than add to it.

    --verify CELL_ID   recompute one named cell into a scratch directory and
                       diff it against the stored payload. Never overwrites.

PHASES (each idempotent; `--phase 0 1 2 3 4` or any subset)
  0  Checkpoints. MNIST + Fashion-MNIST x train seeds {0,1,2}, the paper-1
     784-256-128-10 STE-trained SBNN. results/sbnn_mnist_seed0.pt already
     exists and is reused (checkpoints are tracked in git deliberately, and
     train.py has not changed since it was made; provenance is by tracked file,
     not by hash stamp — this is the one documented exception).
  1  SPLIT-SAMPLE reference: TWO independent 1024-pass ideal-source blocks per
     (dataset, train seed) on the TEST split. Block A defines the margin
     stratum, block B the reference decision every rule is scored against
     (audit r1 item 3; rationale in infer_adaptive.split_reference). The
     cross-block agreement rate is recorded as the measured residual reference
     noise. Also records the |I| distribution the rate ceiling acts on
     (audit r1 item 4). DISCIPLINE (design.md): the pre-checks used val[:2000]
     because the protocol was still being tuned; the protocol is now frozen
     (t_max, t_min, margin cut, alpha, rules), so the grid's reported numbers
     come from test. The one val reference that remains is the 2000-image MNIST
     val set used by phase 2a below, which measures a DEVICE property and never
     sees a decision boundary being tuned.
  2  Main grid (2a: rho calibration, then the cells). See COVERAGE below.
  3  Fixed-N ladder, N in {1,2,4,8,16,32,64,128,256}: the Pareto reference for
     F3, the matched-BUDGET frontier for the excess panel, and the
     marginal-stratum accuracy overlay for F1(b).
  4  Calibration panel data for RQ3 (demoted to one figure): per-image
     (vote-share confidence, correct) pairs at fixed N in {4,16,64,256}, dumped
     raw so ECE/Brier/NLL can be computed at writing time under the frozen
     15-equal-mass-bin convention. No temperature scaling here.

COVERAGE (v2, after the audit)
  (a) NO RULE-BY-ARM PRUNING. v1 ran only the naive Dirichlet rule on
      arrhenius+streaming, on the rationale that Arrhenius devices
      self-decorrelate under bias so that arm adds no regime the others do not
      bracket. THE DATA FALSIFIED THAT (audit r1 item 6): the effect is
      near-threshold-unit-driven and turned out to be robust ACROSS arms, which
      is exactly why the headline is now reported as an arm range — so the
      missing arm is a hole in the range, not a redundancy. All five rules now
      run on all four arms.
  (b) Full 5-noise-seed replication only at r0 in {inf, 1, 0.5} — the ideal
      control and the two operating points the headline claims are stated at.
      Two seeds elsewhere (those r0 are shape-of-curve points).
  (c) Fashion-MNIST is the replication dataset: train seed 0 only, 3 noise
      seeds at the headline r0 and 2 elsewhere. Its F1a-equivalent panel is
      now REPORTED, not just computed.
  (d) Three train seeds only on the PRIMARY arm, easyplane+settle — the arm the
      headline effect lives in, and therefore the one whose error bars have to
      carry model-replicate variance (they are cluster-robust over these three).
      Secondary arms run at train seed 0.
  (e) At r0=inf the source is IdealSource, so device class and mode are not
      merely similar but bit-identical (same object, same seed, same draw
      order). The ideal cells are keyed as device class "ideal", computed once
      per (dataset, train seed, rule, noise seed, t_min) and shared by every
      arm. These are also the per-rule CALIBRATION cells the headline
      normalizes against, so they carry the same replication as the headline r0.
  (f) NEW ARM `arrhcap` (audit r1 item 4): Arrhenius devices with the
      attempt-frequency rate ceiling k_tot = min(2 f cosh I, 1/tau0) at the
      worst-corner tau0 (C = tau_corr(0)/tau0 = 2). Settle mode, dirichlet +
      markov_fixedrho, r0 {1, 0.5, 0.25}, 2 noise seeds.
  (g) NEW COLUMN sigma_delta (audit r1 item 5): device-to-device variation
      {0.5, 1.0} with centering='mean-tau' (so dispersion is isolated from a
      mean slowdown), 3 chip seeds, primary arm, r0 {1, 0.5}, dirichlet +
      markov_fixedrho.
  (h) NEW SENSITIVITY ROWS (audit r1 item 7): t_min in {2, 8} and
      settle_ratio in {2, 8} on the primary arm at the headline r0, for the
      three rules the paper leads with. settle_ratio, sigma_delta, centering
      and t_min are recorded in EVERY cell config, default or not.

CHIP SEED. Outside the sigma_delta column chip_seed is tied to train_seed, so a
"replicate" varies the model AND the device-variation draw together. Rationale:
sigma_delta = 0 there, so the chip seed only sets the initial idle state of the
fluctuators, which the settle interval and the streaming warmup wash out. The
sigma_delta column is where chip seed becomes a real axis, and it gets three of
them.

  (i) ON-CHIP RHO VARIANT `markov_onchiprho` (convergence review 1 item 8). The
      fixedrho rule is calibrated with the MARGINAL-stratum vote ACF, and the
      marginal mask comes from the block-A ideal-source reference — a quantity a
      deployed chip does not have. The variant is the same rule fed the ACF over
      ALL FLUCTUATING INPUTS instead (phase-2a field `all_vote_lag1_acf`), which
      needs no reference block and no labels, and is therefore the number a chip
      could actually measure. It is smaller than the marginal-stratum ACF (0.186
      vs 0.271 at r0=1), so it is the CONSERVATIVE substitution: less correction,
      and the question is whether compliance survives it. Primary arm, r0 in
      {1, 0.5}, 2 noise seeds x 3 train seeds = 12 cells. It reuses the existing
      phase-2a measurements unchanged (both rho flavours are recorded in the same
      cell), so no new 2a cell is needed.

PHASE 2a — why measuring rho is not cheating. MarkovSPRT-fixedrho needs a
lag-1 vote autocorrelation. It is measured on the VALIDATION split, per
(device class, mode, r0, sigma_delta, chip seed, settle_ratio), as the mean
marginal-stratum vote ACF, and it uses NO label information and no test data —
it is the same on-chip measurement the deployment story describes
(diagnostics.vote_lag1_acf). The key is that wide because every one of those
knobs changes the measured ACF, and a rho measured under the wrong device
condition is just a mis-measurement: the settle interval materially changes it
(pre-check: easyplane r0=0.5 gives 0.42 settling vs 0.53 streaming), and with
sigma_delta > 0 the chips differ from each other, which is precisely the
per-device calibration the deployment story asks for.
"""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from apbit.cost import cost_report
from apbit.data import splits
from apbit.diagnostics import (abs_preactivation_summary, vote_diagnostics,
                               vote_lag1_acf, wilson_interval)
from apbit.infer_adaptive import (evaluate_lanes_adaptive, reference_decisions,
                                  split_reference)
from apbit.smtj import IdealSource, TelegraphSource
from apbit.stopping import (DirichletStop, EProcessStop, FixedN, MarkovSPRT,
                            WaldSPRT)
from apbit.train import load_checkpoint, train

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
GRID = RESULTS / "grid2"          # v2 protocol; results/grid/ is the v1 record

# ---- frozen protocol (do not tune; the pre-checks did that) ----------------
ALPHA = 0.05
BETA = 0.05
P0, P1 = 0.5, 0.75
T_MAX = 256
T_MIN = 4
T_REF = 1024
LANES = 100
MARGIN_CUT = 0.9          # block-A reference margin below this = "marginal"
N_TEST = 10000            # full test set, 100 lane-major steps of 100 lanes
N_VAL_CALIB = 2000        # val images for the phase-2a rho measurement
WARMUP_PER_LANE = 2       # streaming mode only, drawn from train images
DIMS = (784, 256, 128, 10)
R0_THROTTLED = 4.0        # Daniels' 2-tau remedy in mean-dwell form
SETTLE_RATIO = 4.0        # default settle interval, in tau_corr(0) units
CENTERING = "mean-tau"    # only matters when sigma_delta > 0
ACF_MAX_LAG = 48          # audit r1 item 9
TIE_SEED = 20260819       # base for the per-lane random tie-break streams
REF_SEED_A = 7            # block A: defines the margin stratum
REF_SEED_B = 4242         # block B: defines the reference decision
RATE_CEILING_R0 = 2.0     # arrhcap arm: C = tau_corr(0)/tau0, worst corner
ABS_I_IMAGES = 500        # |I| distribution sample (phase 1)
ABS_I_PASSES = 2

# ---- grid axes -------------------------------------------------------------
DATASETS = ("mnist", "fashion")
TRAIN_SEEDS = (0, 1, 2)
ARMS = (("arrhenius", "settle"), ("easyplane", "settle"),
        ("easyplane", "streaming"), ("arrhenius", "streaming"))
PRIMARY_ARM = ("easyplane", "settle")
R0S = (np.inf, 4.0, 2.0, 1.0, 0.5, 0.25)
HEADLINE_R0 = (np.inf, 1.0, 0.5)
RULES = ("dirichlet", "wald", "markov_online", "markov_fixedrho", "eprocess")
NOISE_SEEDS = (1000, 1001, 1002, 1003, 1004)
LADDER = (1, 2, 4, 8, 16, 32, 64, 128, 256)
CALIB_NS = (4, 16, 64, 256)

# ---- audit r1 additions ----------------------------------------------------
CAP_ARM = ("arrhcap", "settle")
CAP_RULES = ("dirichlet", "markov_fixedrho")
CAP_R0S = (1.0, 0.5, 0.25)
SIGMA_DELTAS = (0.5, 1.0)
SIGMA_CHIP_SEEDS = (0, 1, 2)
SIGMA_R0S = (1.0, 0.5)
SIGMA_RULES = ("dirichlet", "markov_fixedrho")
SENS_RULES = ("dirichlet", "wald", "markov_fixedrho")
T_MIN_SENS = (2, 8)
SETTLE_RATIO_SENS = (2.0, 8.0)
SENS_R0S_FINITE = (1.0, 0.5)

# ---- convergence review 1 item 8: the on-chip rho variant -------------------
ONCHIP_RULE = "markov_onchiprho"
ONCHIP_R0S = (1.0, 0.5)
ONCHIP_NOISE_SEEDS = NOISE_SEEDS[:2]      # 1000, 1001
FIXEDRHO_RULES = ("markov_fixedrho", ONCHIP_RULE)
ALL_RULES = RULES + (ONCHIP_RULE,)

# Code hashes whose cells stay valid at the CURRENT hash. Each entry is an
# assertion that nothing between it and now can move a stored number.
ACCEPTED_CODE_HASHES = (
    # grid2 as run at git 133e737 (1049 cells). The only driver change since is
    # convergence review 1 item 8: the markov_onchiprho rule variant, its twelve
    # new cell descriptors, and this list. Additive — no existing cell's config,
    # seed, source, rule spec or draw order is touched, and phases 1/2a/3/4 are
    # byte-identical in behaviour.
    "ab9049e94fd2735574ab2e03c4ef7908cfe10e22c20380b3bf705de1f6682d03",
)


# ---------------------------------------------------------------------------
# provenance + atomic io
# ---------------------------------------------------------------------------

def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def git_dirty() -> bool:
    out = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    return bool(out)


def code_files() -> list:
    """The files whose bytes can change a number, in a fixed order."""
    return sorted((ROOT / "src" / "apbit").glob("*.py")) + \
        [ROOT / "experiments" / "run_grid.py"]


def code_hash() -> str:
    """sha256 over the concatenated bytes of the numerics-bearing sources.

    Path names are hashed alongside the contents so a renamed module is a new
    hash. This is the freshness key: it ignores manuscript commits and it
    catches an uncommitted edit, which is exactly what a git SHA gets wrong in
    both directions.
    """
    h = hashlib.sha256()
    for p in code_files():
        h.update(p.relative_to(ROOT).as_posix().encode())
        h.update(b"\0")
        h.update(p.read_bytes())
    return h.hexdigest()


def write_atomic(path: Path, payload: dict, compact: bool = False):
    """Write JSON to a tmp file in the same directory, then rename over the
    target. os.replace is atomic on POSIX, so a reader never sees a half file
    and an interrupted run never leaves a corrupt result behind. `compact`
    drops the indentation AND the whitespace after the separators, for payloads
    that carry per-image arrays (phases 1, 2 and 4): at 10k elements per array
    the default ", " separator alone costs 10 KB per array and indent=1 costs
    ten times that. With it on, a phase-2 cell lands under 100 KB.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")) if compact
                   else json.dumps(payload, indent=1))
    os.replace(tmp, path)


def cell_path(cid: str) -> Path:
    return GRID / f"{cid}.json"


def stamp(cid: str, phase, sha: str, chash: str) -> dict:
    """The provenance block every cell carries."""
    return {"code_hash": chash, "git_sha": sha, "git_dirty": git_dirty(),
            "cell_id": cid, "phase": phase}


def accepted_code_hashes(chash: str = None) -> set:
    """The hashes a stored cell may carry and still be reused.

    The current hash plus ACCEPTED_CODE_HASHES (see the module docstring): the
    latter are earlier hashes whose difference from now is provably unable to
    move a stored number, so a protocol-identical additive edit to this driver
    does not invalidate a grid that took a day to compute.
    """
    return {chash or code_hash(), *ACCEPTED_CODE_HASHES}


def fresh_result(cid: str, chash: str):
    """Return the cached payload iff it exists and carries an accepted hash.

    A payload with no `code_hash` at all predates audit r1 and is INCOMPATIBLE
    (the protocol changed), so it is never reused — not even when its git_sha
    matches the v1 grid commit a2e635a. `None` and any unlisted hash are equally
    rejected; only the current hash and the explicitly reasoned entries in
    ACCEPTED_CODE_HASHES are accepted.
    """
    p = cell_path(cid)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    stored = payload.get("code_hash")
    return payload if stored in accepted_code_hashes(chash) else None


# ---------------------------------------------------------------------------
# naming
# ---------------------------------------------------------------------------

def r0_tag(r0) -> str:
    return "inf" if np.isinf(r0) else f"{r0:g}"


def arm_key(device_class: str, mode: str, r0):
    """At r0=inf the source is IdealSource regardless of arm — collapse the
    arms onto one shared cell (coverage note (e))."""
    return ("ideal", "ideal") if np.isinf(r0) else (device_class, mode)


def variant_tag(sigma_delta: float = 0.0, chip_seed: int = 0,
                t_min: int = T_MIN, settle_ratio: float = SETTLE_RATIO) -> str:
    """Suffix naming only the knobs that differ from the frozen protocol, so
    default cell ids are byte-identical to the v1 ones and a variant can never
    silently overwrite a default."""
    bits = ""
    if sigma_delta:
        bits += f"_sd{sigma_delta:g}_cs{chip_seed}"
    if int(t_min) != T_MIN:
        bits += f"_tmin{int(t_min)}"
    if float(settle_ratio) != SETTLE_RATIO:
        bits += f"_sr{settle_ratio:g}"
    return bits


# ---------------------------------------------------------------------------
# cached loaders (a grid run touches the same tensors hundreds of times)
# ---------------------------------------------------------------------------

_SPLITS: dict = {}
_MODELS: dict = {}
_REFS: dict = {}
_RHOS: dict = {}


def get_splits(dataset: str):
    if dataset not in _SPLITS:
        _SPLITS[dataset] = splits(dataset)
    return _SPLITS[dataset]


def ckpt_path(dataset: str, seed: int) -> Path:
    return RESULTS / f"sbnn_{dataset}_seed{seed}.pt"


def ensure_checkpoint(dataset: str, seed: int, log=print) -> Path:
    path = ckpt_path(dataset, seed)
    if path.exists():
        return path
    log(f"  training {path.name} (30 epochs, cpu)...")
    train(dataset=dataset, dims=DIMS, seed=seed, ckpt_name=path.name, log=log)
    return path


def get_model(dataset: str, seed: int):
    key = (dataset, seed)
    if key not in _MODELS:
        model = load_checkpoint(ensure_checkpoint(dataset, seed))
        model.eval()
        _MODELS[key] = model
    return _MODELS[key]


# ---------------------------------------------------------------------------
# phase 1 — split-sample reference and the margin distribution
# ---------------------------------------------------------------------------

def ref_cid(dataset: str, seed: int, split: str) -> str:
    return f"p1_ref_{dataset}_ts{seed}_{split}"


def split_images(dataset: str, split: str):
    """(xs, ys) for the named split. 'test' = the full 10k test set; 'val2000'
    = val[:2000], used only by the phase-2a device measurement."""
    trx, _, vx, vy, tx, ty = get_splits(dataset)
    if split == "test":
        return tx[:N_TEST], ty[:N_TEST]
    if split == "val2000":
        return vx[:N_VAL_CALIB], vy[:N_VAL_CALIB]
    raise ValueError(split)


def margin_summary(margin_counts: np.ndarray, t_ref: int) -> dict:
    m = margin_counts / float(t_ref)
    qs = [0, 1, 5, 10, 25, 50, 75, 90, 100]
    hist, edges = np.histogram(m, bins=20, range=(0.0, 1.0))
    return {
        "mean": float(m.mean()),
        "sd": float(m.std()),
        "quantiles": {f"q{q}": float(np.percentile(m, q)) for q in qs},
        "tie_count": int((margin_counts == 0).sum()),
        "marginal_cut": MARGIN_CUT,
        "marginal_count": int((m < MARGIN_CUT).sum()),
        "marginal_frac": float((m < MARGIN_CUT).mean()),
        "histogram_counts": [int(c) for c in hist],
        "histogram_edges": [float(e) for e in edges],
    }


def compute_reference(dataset: str, seed: int, split: str, sha: str,
                      chash: str, log=print) -> dict:
    t0 = time.time()
    model = get_model(dataset, seed)
    xs, ys = split_images(dataset, split)
    _, _, counts_a = reference_decisions(
        model, xs, IdealSource(seed=REF_SEED_A), t_ref=T_REF, batch=1000,
        return_counts=True)
    _, _, counts_b = reference_decisions(
        model, xs, IdealSource(seed=REF_SEED_B), t_ref=T_REF, batch=1000,
        return_counts=True)
    sp = split_reference(counts_a, counts_b, T_REF, MARGIN_CUT)
    labels = ys.numpy().astype(np.int64)
    ref = sp["reference"]
    counts_margin_a = np.rint(sp["margin_a"] * T_REF).astype(np.int64)
    counts_margin_b = np.rint(sp["margin_b"] * T_REF).astype(np.int64)
    payload = {
        **stamp(ref_cid(dataset, seed, split), 1, sha, chash),
        "config": {"dataset": dataset, "train_seed": seed, "split": split,
                   "n_images": int(xs.shape[0]), "t_ref": T_REF,
                   "reference_convention": "split-sample (audit r1 item 3)",
                   "stratum_block": f"A = IdealSource(seed={REF_SEED_A})",
                   "reference_block": f"B = IdealSource(seed={REF_SEED_B})",
                   "margin_cut": MARGIN_CUT, "dims": list(DIMS)},
        "reference_label_accuracy": float((ref == labels).mean()),
        "reference_label_accuracy_block_a": float(
            (sp["decision_a"] == labels).mean()),
        "reference_tie_count": int(sp["reference_tie"].sum()),
        "blocks_agree_frac": sp["blocks_agree_frac"],
        "blocks_agree_frac_marginal": sp["blocks_agree_frac_marginal"],
        "margin_summary": margin_summary(counts_margin_a, T_REF),
        "margin_summary_block_b": margin_summary(counts_margin_b, T_REF),
        "reference": [int(v) for v in ref],
        "reference_tie": [int(v) for v in sp["reference_tie"]],
        "margin_counts": [int(v) for v in counts_margin_a],
        "margin_counts_block_b": [int(v) for v in counts_margin_b],
        "top_counts_block_a": [int(v) for v in sp["top_counts_a"]],
        "decision_block_a": [int(v) for v in sp["decision_a"]],
        "note": ("margin_counts (block A) defines the marginal stratum; "
                 "reference (block B) is the scoring target; the two blocks "
                 "are independent 1024-pass ideal-source runs"),
        "wall_seconds": time.time() - t0,
    }
    if split == "test":
        payload["abs_preactivation"] = abs_preactivation_summary(
            model, xs[:ABS_I_IMAGES], IdealSource(seed=11),
            passes=ABS_I_PASSES, batch=250, ceilings=(RATE_CEILING_R0, 4.0))
    write_atomic(cell_path(payload["cell_id"]), payload, compact=True)
    log(f"  ref {dataset} ts{seed} {split}: "
        f"acc={payload['reference_label_accuracy']:.4f} "
        f"ties={payload['reference_tie_count']} "
        f"marginal={payload['margin_summary']['marginal_count']} "
        f"blocks_agree={payload['blocks_agree_frac']:.4f} "
        f"({payload['wall_seconds']:.0f}s)")
    return payload


def get_reference(dataset: str, seed: int, split: str, chash: str, log=print):
    """(ref, tie, marginal) arrays, computed on demand and cached on disk.

    ref/tie come from block B (the scoring target), marginal from block A (the
    stratum) — the split-sample convention.
    """
    key = (dataset, seed, split, chash)
    if key in _REFS:
        return _REFS[key]
    payload = fresh_result(ref_cid(dataset, seed, split), chash)
    if payload is None:
        payload = compute_reference(dataset, seed, split, git_sha(), chash,
                                    log=log)
    ref = np.asarray(payload["reference"], dtype=np.int64)
    tie = np.asarray(payload["reference_tie"], dtype=bool)
    counts_a = np.asarray(payload["margin_counts"], dtype=np.int64)
    marginal = (counts_a / float(payload["config"]["t_ref"])) < MARGIN_CUT
    _REFS[key] = (ref, tie, marginal)
    return _REFS[key]


# ---------------------------------------------------------------------------
# sources and rules
# ---------------------------------------------------------------------------

def make_source(device_class: str, mode: str, r0, chip_seed: int,
                noise_seed: int, sigma_delta: float = 0.0,
                settle_ratio: float = SETTLE_RATIO,
                centering: str = CENTERING):
    if np.isinf(r0) or device_class == "ideal":
        return IdealSource(seed=noise_seed)
    if device_class not in ("arrhenius", "easyplane", "arrhcap"):
        raise ValueError(device_class)
    if mode not in ("settle", "streaming"):
        raise ValueError(mode)
    return TelegraphSource(
        r0=float(r0), chip_seed=chip_seed, noise_seed=noise_seed,
        sigma_delta=float(sigma_delta), centering=centering,
        settle_ratio=float(settle_ratio),
        streaming=(mode == "streaming"),
        bias_independent_tau=(device_class == "easyplane"),
        rate_ceiling_r0=(RATE_CEILING_R0 if device_class == "arrhcap"
                         else None))


def rule_spec(rule: str, k: int, t_max: int, rho=None,
              t_min: int = T_MIN) -> dict:
    """Constructor kwargs for the rule, recorded verbatim in the result JSON
    so a cell can be replayed from its own config block."""
    common = {"k_classes": k, "t_min": int(t_min), "t_max": t_max}
    if rule == "dirichlet":
        return {"cls": "DirichletStop", "alpha": ALPHA, "method": "beta",
                **common}
    if rule == "eprocess":
        return {"cls": "EProcessStop", "alpha": ALPHA, "null_mean": 0.5,
                "bonferroni": True, **common}
    sprt = {"alpha": ALPHA, "beta": BETA, "p0": P0, "p1": P1, **common}
    if rule == "wald":
        return {"cls": "WaldSPRT", **sprt}
    if rule == "markov_online":
        return {"cls": "MarkovSPRT", "rho": None, **sprt}
    if rule in FIXEDRHO_RULES:
        # markov_fixedrho and markov_onchiprho are the SAME rule; they differ
        # only in which measured autocorrelation is handed to it (marginal-mask
        # vs all-fluctuating-inputs), so the spec must be identical or the
        # comparison would confound the rho with the rule.
        return {"cls": "MarkovSPRT", "rho": float(rho), **sprt}
    if rule == "fixed":
        return {"cls": "FixedN", "n": t_max, "k_classes": k}
    raise ValueError(rule)


def make_rule_factory(spec: dict, tie_seed_base: int = TIE_SEED):
    """Lane-indexed factory: each lane gets its own tie-break stream.

    The lane's tie seed is tie_seed_base + 7919 * lane (7919 prime, so the
    streams do not align across lanes). Ties are broken at random (audit r1
    item 1) and this is the only randomness in the driver that is not the
    device's.
    """
    kw = {k: v for k, v in spec.items() if k != "cls"}
    cls = {"DirichletStop": DirichletStop, "WaldSPRT": WaldSPRT,
           "MarkovSPRT": MarkovSPRT, "EProcessStop": EProcessStop,
           "FixedN": FixedN}[spec["cls"]]
    return lambda lane=0: cls(tie_seed=tie_seed_base + 7919 * int(lane), **kw)


def warmup_pool(dataset: str, mode: str, lanes: int, r0):
    """Streaming lanes get WARMUP_PER_LANE unscored TRAIN images each, so the
    scored stream starts in steady state (infer.evaluate_lanes convention).
    Settle mode and the ideal source have nothing to warm up."""
    if mode != "streaming" or np.isinf(r0):
        return None, 0
    trx, *_ = get_splits(dataset)
    return trx[:WARMUP_PER_LANE * lanes], WARMUP_PER_LANE


# ---------------------------------------------------------------------------
# phase 2a — rho calibration on the validation split
# ---------------------------------------------------------------------------

def rho_key(device_class: str, mode: str, r0, sigma_delta: float = 0.0,
            chip_seed: int = 0, settle_ratio: float = SETTLE_RATIO) -> tuple:
    """Device condition the rho measurement is keyed by.

    chip_seed only enters when sigma_delta > 0: with no device variation every
    chip has rate_factor == 1 exactly, so a per-chip rho would be the same
    number measured three times.
    """
    return (device_class, mode, r0_tag(r0), float(sigma_delta),
            int(chip_seed) if sigma_delta else 0, float(settle_ratio))


def rho_cid(device_class: str, mode: str, r0, sigma_delta: float = 0.0,
            chip_seed: int = 0, settle_ratio: float = SETTLE_RATIO) -> str:
    return (f"p2a_rho_{device_class}-{mode}_r{r0_tag(r0)}"
            + variant_tag(sigma_delta, chip_seed, T_MIN, settle_ratio))


def compute_rho(device_class: str, mode: str, r0, sha: str, chash: str,
                sigma_delta: float = 0.0, chip_seed: int = 0,
                settle_ratio: float = SETTLE_RATIO, log=print) -> dict:
    t0 = time.time()
    dataset, seed = "mnist", 0
    model = get_model(dataset, seed)
    xs, ys = split_images(dataset, "val2000")
    _, _, marginal = get_reference(dataset, seed, "val2000", chash, log=log)
    cs = int(chip_seed) if sigma_delta else seed
    src = make_source(device_class, mode, r0, chip_seed=cs, noise_seed=999,
                      sigma_delta=sigma_delta, settle_ratio=settle_ratio)
    warm, wpl = warmup_pool(dataset, mode, LANES, r0)
    spec = rule_spec("fixed", model.dims[-1], T_MAX)
    out = evaluate_lanes_adaptive(
        model, xs, ys, src, make_rule_factory(spec), t_max=T_MAX, lanes=LANES,
        warmup_x=warm, warmup_per_lane=wpl, collect_votes=True)
    acf_m, n_m = vote_lag1_acf(out["votes"], marginal)
    acf_a, n_a = vote_lag1_acf(out["votes"])
    used = clamp_rho(acf_m)
    payload = {
        **stamp(rho_cid(device_class, mode, r0, sigma_delta, cs,
                        settle_ratio), "2a", sha, chash),
        "config": {"device_class": device_class, "mode": mode,
                   "r0": r0_tag(r0), "dataset": dataset, "train_seed": seed,
                   "chip_seed": cs, "noise_seed": 999, "split": "val2000",
                   "n_images": int(xs.shape[0]), "t_max": T_MAX,
                   "lanes": LANES, "margin_cut": MARGIN_CUT,
                   "sigma_delta": float(sigma_delta), "centering": CENTERING,
                   "settle_ratio": float(settle_ratio),
                   "warmup_per_lane": wpl},
        "marginal_vote_lag1_acf": acf_m,
        "marginal_acf_n_images": n_m,
        "all_vote_lag1_acf": acf_a,
        "all_acf_n_images": n_a,
        "rho_used": used,
        "vote_diagnostics_marginal": vote_diagnostics(
            out["votes"], marginal, max_lag=ACF_MAX_LAG),
        "note": "rho is a device measurement on val; no labels, no test data",
        "wall_seconds": time.time() - t0,
    }
    write_atomic(cell_path(payload["cell_id"]), payload, compact=True)
    log(f"  rho {device_class}-{mode} r0={r0_tag(r0)}"
        f"{variant_tag(sigma_delta, cs, T_MIN, settle_ratio)}: "
        f"acf_marginal={acf_m:.4f} (n={n_m}) -> rho={used:.4f} "
        f"({payload['wall_seconds']:.0f}s)")
    return payload


def clamp_rho(acf) -> float:
    """The single place a measured ACF becomes a rule parameter.

    Identical to the clamp compute_rho applies to `rho_used`, so the on-chip
    flavour is handled exactly as the marginal one and any difference in a cell
    is the ACF's, not the clamp's.
    """
    return (0.0 if acf is None or not np.isfinite(acf)
            else float(min(max(float(acf), 0.0), 0.995)))


def rho_field(rule: str) -> str:
    """Which phase-2a measurement the rule consumes.

    markov_fixedrho takes the MARGINAL-stratum ACF, whose mask comes from the
    block-A ideal-source reference — not something a deployed chip has.
    markov_onchiprho takes the ACF over all fluctuating inputs, which needs no
    reference block, no labels and no test data: the truly on-chip quantity
    (convergence review 1 item 8).
    """
    return ("all_vote_lag1_acf" if rule == ONCHIP_RULE
            else "marginal_vote_lag1_acf")


def get_rho(device_class: str, mode: str, r0, chash: str,
            sigma_delta: float = 0.0, chip_seed: int = 0,
            settle_ratio: float = SETTLE_RATIO, field: str = None,
            log=print) -> float:
    """The measured rho for a device condition, from the phase-2a cell.

    `field` names the measurement to read ('marginal_vote_lag1_acf' by default,
    'all_vote_lag1_acf' for the on-chip variant). Both are recorded by every
    phase-2a cell, so the variant costs no extra measurement.
    """
    field = field or "marginal_vote_lag1_acf"
    key = rho_key(device_class, mode, r0, sigma_delta, chip_seed, settle_ratio) \
        + (field,)
    if key in _RHOS:
        return _RHOS[key]
    cid = rho_cid(device_class, mode, r0, sigma_delta,
                  key[4], settle_ratio)
    payload = fresh_result(cid, chash)
    if payload is None:
        payload = compute_rho(device_class, mode, r0, git_sha(), chash,
                              sigma_delta=sigma_delta, chip_seed=key[4],
                              settle_ratio=settle_ratio, log=log)
    _RHOS[key] = (float(payload["rho_used"])
                  if field == "marginal_vote_lag1_acf"
                  else clamp_rho(payload.get(field)))
    return _RHOS[key]


# ---------------------------------------------------------------------------
# metric bookkeeping
# ---------------------------------------------------------------------------

def _slice(decisions, labels, samples, ref, mask) -> dict:
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "stopping_error": None,
                "stopping_error_wilson95": None,
                "stopping_error_over_alpha": None, "task_accuracy": None,
                "mean_samples": None}
    wrong = int((decisions[mask] != ref[mask]).sum())
    lo, hi = wilson_interval(wrong, n)
    return {
        "n": n,
        "stopping_error": wrong / n,
        "stopping_error_wilson95": [lo, hi],
        "stopping_error_over_alpha": (wrong / n) / ALPHA,
        "task_accuracy": float((decisions[mask] == labels[mask]).mean()),
        "mean_samples": float(np.mean(samples[mask])),
    }


def strata_report(decisions, labels, samples, ref, keep, marginal) -> dict:
    """Stopping error stratified by block-A reference margin (PROJECT_STATE
    lesson 1: the per-image guarantee is diluted to invisibility by easy
    images, so the marginal stratum is the quantity the paper reports).
    Block-B reference ties are excluded from every stratum."""
    return {
        "all": _slice(decisions, labels, samples, ref, keep),
        "marginal": _slice(decisions, labels, samples, ref, keep & marginal),
        "easy": _slice(decisions, labels, samples, ref, keep & ~marginal),
    }


def termination_report(decisions, labels, samples, ref, keep, fired,
                       marginal) -> dict:
    """Fired vs truncated (lesson 2). Only the fired channel is alpha-bounded;
    truncation error is budget error and must never be blended into it. The
    marginal-stratum fired/truncated split is the pair the headline figure
    actually compares against alpha."""
    return {
        "fired": _slice(decisions, labels, samples, ref, keep & fired),
        "truncated": _slice(decisions, labels, samples, ref, keep & ~fired),
        "fired_marginal": _slice(decisions, labels, samples, ref,
                                 keep & fired & marginal),
        "truncated_marginal": _slice(decisions, labels, samples, ref,
                                     keep & ~fired & marginal),
    }


def cost_block(samples, r0, dims=DIMS, mode: str = "settle",
               settle_ratio: float = SETTLE_RATIO) -> dict:
    """Frozen cost proxy, charged against the network that was actually run.

    The throttled LATENCY equivalent is only defined against an operating
    point, so it is reported for finite r0 only (at r0=inf there is no clock to
    throttle). The settle wait is charged only in settle mode, where it is
    physically paid once per input.
    """
    if np.isinf(r0):
        return cost_report(samples, dims)
    return cost_report(samples, dims, r0_operating=float(r0),
                       r0_throttled=R0_THROTTLED,
                       settle_ratio=(float(settle_ratio) if mode == "settle"
                                     else None))


# ---------------------------------------------------------------------------
# phase 2 — one adaptive cell
# ---------------------------------------------------------------------------

def evaluate_cell(model, xs, ys, source, spec: dict, ref, tie, marginal,
                  r0, t_max: int, lanes: int, warm=None, warmup_per_lane=0,
                  collect_votes: bool = True, mode: str = "settle",
                  settle_ratio: float = SETTLE_RATIO,
                  tie_seed_base: int = TIE_SEED,
                  per_image: bool = True) -> dict:
    """Run one (rule, source, model, data) cell and reduce it to metrics.

    Pure w.r.t. the filesystem — the test harness calls this directly with a
    tiny synthetic model, and the phase-2 runner wraps it with loading,
    provenance stamping and the atomic write.

    `per_image=True` keeps the raw per-image decision / samples_used / fired /
    tie arrays in the result (audit r1 item 2). They are what makes every
    post-hoc restriction the audit asked for — q >= 0.75, margin cuts at
    0.8/0.9/0.95, cluster bootstraps — computable WITHOUT re-running the grid,
    and they let a reader recompute every aggregate in the cell from the raw
    numbers. Compactly encoded they add ~70 KB to a cell.
    """
    t0 = time.time()
    out = evaluate_lanes_adaptive(
        model, xs, ys, source, make_rule_factory(spec, tie_seed_base),
        t_max=t_max, lanes=lanes, warmup_x=warm,
        warmup_per_lane=warmup_per_lane, collect_votes=collect_votes,
        reference=ref, reference_ties=tie)
    keep = ~np.asarray(tie, dtype=bool)
    marginal = np.asarray(marginal, dtype=bool)
    d, s = out["decisions"], out["samples_used"]
    labels, fired = out["labels"], out["fired"]
    acf_m, n_m = ((float("nan"), 0) if not collect_votes
                  else vote_lag1_acf(out["votes"], marginal))
    acf_a, n_a = ((float("nan"), 0) if not collect_votes
                  else vote_lag1_acf(out["votes"]))
    res = {
        "n_images": int(xs.shape[0]),
        "reference_tie_count": int(out["reference_tie_count"]),
        "stopping_error": strata_report(d, labels, s, ref, keep, marginal),
        "by_termination": termination_report(d, labels, s, ref, keep, fired,
                                             marginal),
        "truncation_rate": float(out["truncation_rate"]),
        "plurality_tie_rate": float(out["plurality_tie_rate"]),
        "plurality_tie_rate_marginal": float(
            out["plurality_tie"][marginal].mean()) if marginal.any() else None,
        "task_accuracy": float(out["accuracy"]),
        "mean_samples": float(out["mean_samples"]),
        "mean_samples_marginal": (float(np.mean(s[keep & marginal]))
                                  if (keep & marginal).any() else None),
        "early_stop_frac": float(out["early_stop_frac"]),
        "vote_lag1_acf": {"marginal": acf_m, "marginal_n_images": n_m,
                          "all": acf_a, "all_n_images": n_a},
        "cost": cost_block(s, r0, model.dims, mode=mode,
                           settle_ratio=settle_ratio),
        "wall_seconds": time.time() - t0,
        "_out": out,
    }
    if collect_votes:
        res["vote_diagnostics_marginal"] = vote_diagnostics(
            out["votes"], marginal, max_lag=ACF_MAX_LAG)
    if per_image:
        res["per_image"] = {
            "decisions": [int(v) for v in d],
            "samples_used": [int(v) for v in s],
            "fired": [int(v) for v in fired],
            # ties are rare (order 1e-4 per image), so an index list is far
            # smaller than a 10k flag array and loses nothing
            "plurality_tie_index": [int(i) for i in
                                    np.flatnonzero(out["plurality_tie"])],
            "note": ("xs order, aligned with the phase-1 reference arrays of "
                     "the same (dataset, train_seed); labels/reference/margins "
                     "live in the phase-1 cell and are not duplicated here; "
                     "plurality_tie_index lists the images whose class came "
                     "out of a random tie-break"),
        }
    return res


def run_phase2_cell(cell: dict, sha: str, chash: str, log=print) -> dict:
    cfg = cell["config"]
    model = get_model(cfg["dataset"], cfg["train_seed"])
    xs, ys = split_images(cfg["dataset"], "test")
    ref, tie, marginal = get_reference(cfg["dataset"], cfg["train_seed"],
                                       "test", chash, log=log)
    r0 = cell["r0"]
    rho = (get_rho(cfg["device_class"], cfg["mode"], r0, chash,
                   sigma_delta=cfg["sigma_delta"], chip_seed=cfg["chip_seed"],
                   settle_ratio=cfg["settle_ratio"],
                   field=rho_field(cfg["rule"]), log=log)
           if cfg["rule"] in FIXEDRHO_RULES else None)
    spec = rule_spec(cfg["rule"], model.dims[-1], T_MAX, rho=rho,
                     t_min=cfg["t_min"])
    src = make_source(cfg["device_class"], cfg["mode"], r0,
                      chip_seed=cfg["chip_seed"], noise_seed=cfg["noise_seed"],
                      sigma_delta=cfg["sigma_delta"],
                      settle_ratio=cfg["settle_ratio"])
    warm, wpl = warmup_pool(cfg["dataset"], cfg["mode"], LANES, r0)
    res = evaluate_cell(model, xs, ys, src, spec, ref, tie, marginal, r0,
                        t_max=T_MAX, lanes=LANES, warm=warm,
                        warmup_per_lane=wpl, mode=cfg["mode"],
                        settle_ratio=cfg["settle_ratio"],
                        tie_seed_base=TIE_SEED + cfg["noise_seed"])
    res.pop("_out")
    payload = {**stamp(cell["cid"], 2, sha, chash),
               "config": {**cfg, "r0": r0_tag(r0), "rule_spec": spec,
                          "rho_used": rho,
                          "rho_field": (rho_field(cfg["rule"])
                                        if cfg["rule"] in FIXEDRHO_RULES
                                        else None),
                          "split": "test", "t_max": T_MAX,
                          "t_ref": T_REF, "lanes": LANES,
                          "margin_cut": MARGIN_CUT, "alpha": ALPHA,
                          "tie_seed_base": TIE_SEED + cfg["noise_seed"],
                          "warmup_per_lane": wpl, "dims": list(DIMS),
                          "reference_convention": "split-sample"},
               **res}
    write_atomic(cell_path(cell["cid"]), payload, compact=True)
    return payload


# ---------------------------------------------------------------------------
# phase 3 — fixed-N ladder (prefix decoding of one full-budget stream)
# ---------------------------------------------------------------------------

def ladder_from_votes(votes, labels, ref, keep, marginal, dims, r0,
                      ns=LADDER, mode: str = "settle",
                      settle_ratio: float = SETTLE_RATIO,
                      tie_rng=None) -> list:
    """Fixed-N metrics for every N in the ladder, read off the prefixes of a
    single t_max-pass vote stream.

    Legitimate because of the protocol already fixed in infer_adaptive: every
    lane is QUERIED for the full budget whatever the rule does, and only the
    reads are charged. The device trajectory is therefore identical for every
    N, and the first N votes of the recorded stream are exactly the votes an
    N-pass readout on that same chip would have seen. It also makes the F3
    frontier free instead of nine times the compute. The consequence, stated in
    Methods: the rungs are offline replays of ONE stream, so they are perfectly
    correlated with each other and with the adaptive cells of the same cell id.

    Ties in the argmax are broken at random (audit r1 item 1) with `tie_rng`;
    at small N most images are tied, so lowest-id tie-breaking here would bias
    the frontier the adaptive rules are compared against.
    """
    votes = np.asarray(votes)
    rng = np.random.default_rng(TIE_SEED) if tie_rng is None else tie_rng
    n, t = votes.shape
    rows = np.arange(n)
    counts = np.zeros((n, tuple(dims)[-1]), dtype=np.int32)
    want = {int(v) for v in ns if v <= t}
    rungs = []
    for step in range(t):
        counts[rows, votes[:, step]] += 1
        if (step + 1) in want:
            top = counts.max(1, keepdims=True)
            tied = counts == top
            n_tied = tied.sum(1)
            # random argmax: rank the tied entries by a fresh uniform draw
            noise = rng.random(counts.shape) * tied
            dec = noise.argmax(1).astype(np.int64)
            samples = np.full(n, step + 1, dtype=np.int64)
            rungs.append({
                "n_passes": step + 1,
                "stopping_error": strata_report(dec, labels, samples, ref,
                                                keep, marginal),
                "task_accuracy": float((dec == labels).mean()),
                "top_counts_mean": float(counts.max(1).mean()),
                "plurality_tie_rate": float((n_tied > 1).mean()),
                "cost": cost_block(samples, r0, dims, mode=mode,
                                   settle_ratio=settle_ratio),
            })
    return rungs


def run_phase3_cell(cell: dict, sha: str, chash: str, log=print) -> dict:
    cfg = cell["config"]
    model = get_model(cfg["dataset"], cfg["train_seed"])
    xs, ys = split_images(cfg["dataset"], "test")
    ref, tie, marginal = get_reference(cfg["dataset"], cfg["train_seed"],
                                       "test", chash, log=log)
    r0 = cell["r0"]
    spec = rule_spec("fixed", model.dims[-1], T_MAX)
    src = make_source(cfg["device_class"], cfg["mode"], r0,
                      chip_seed=cfg["chip_seed"], noise_seed=cfg["noise_seed"],
                      sigma_delta=cfg["sigma_delta"],
                      settle_ratio=cfg["settle_ratio"])
    warm, wpl = warmup_pool(cfg["dataset"], cfg["mode"], LANES, r0)
    res = evaluate_cell(model, xs, ys, src, spec, ref, tie, marginal, r0,
                        t_max=T_MAX, lanes=LANES, warm=warm,
                        warmup_per_lane=wpl, mode=cfg["mode"],
                        settle_ratio=cfg["settle_ratio"],
                        tie_seed_base=TIE_SEED + cfg["noise_seed"],
                        per_image=False)
    out = res.pop("_out")
    keep = ~np.asarray(tie, dtype=bool)
    rungs = ladder_from_votes(
        out["votes"], out["labels"], ref, keep, marginal, model.dims, r0,
        mode=cfg["mode"], settle_ratio=cfg["settle_ratio"],
        tie_rng=np.random.default_rng(TIE_SEED + cfg["noise_seed"]))
    payload = {**stamp(cell["cid"], 3, sha, chash),
               "config": {**cfg, "r0": r0_tag(r0), "split": "test",
                          "ladder": list(LADDER), "t_max": T_MAX,
                          "t_ref": T_REF, "lanes": LANES,
                          "margin_cut": MARGIN_CUT, "alpha": ALPHA,
                          "warmup_per_lane": wpl, "dims": list(DIMS),
                          "reference_convention": "split-sample"},
               "vote_lag1_acf": res["vote_lag1_acf"],
               "vote_diagnostics_marginal": res.get(
                   "vote_diagnostics_marginal"),
               "reference_tie_count": res["reference_tie_count"],
               "n_images": res["n_images"],
               "ladder": rungs,
               "wall_seconds": res["wall_seconds"]}
    write_atomic(cell_path(cell["cid"]), payload, compact=True)
    return payload


# ---------------------------------------------------------------------------
# phase 4 — calibration panel (RQ3, demoted)
# ---------------------------------------------------------------------------

def run_phase4_cell(cell: dict, sha: str, chash: str, log=print) -> dict:
    cfg = cell["config"]
    model = get_model(cfg["dataset"], cfg["train_seed"])
    xs, ys = split_images(cfg["dataset"], "test")
    ref, tie, marginal = get_reference(cfg["dataset"], cfg["train_seed"],
                                       "test", chash, log=log)
    r0 = cell["r0"]
    spec = rule_spec("fixed", model.dims[-1], T_MAX)
    src = make_source(cfg["device_class"], cfg["mode"], r0,
                      chip_seed=cfg["chip_seed"], noise_seed=cfg["noise_seed"],
                      sigma_delta=cfg["sigma_delta"],
                      settle_ratio=cfg["settle_ratio"])
    warm, wpl = warmup_pool(cfg["dataset"], cfg["mode"], LANES, r0)
    t0 = time.time()
    out = evaluate_lanes_adaptive(
        model, xs, ys, src, make_rule_factory(spec), t_max=T_MAX, lanes=LANES,
        warmup_x=warm, warmup_per_lane=wpl, collect_votes=True,
        reference=ref, reference_ties=tie)
    votes, labels = out["votes"], out["labels"]
    n, k = votes.shape[0], model.dims[-1]
    rows = np.arange(n)
    counts = np.zeros((n, k), dtype=np.int32)
    rng = np.random.default_rng(TIE_SEED + cfg["noise_seed"])
    want = {int(v) for v in CALIB_NS}
    panels = []
    for step in range(T_MAX):
        counts[rows, votes[:, step]] += 1
        if (step + 1) in want:
            tied = counts == counts.max(1, keepdims=True)
            dec = (rng.random(counts.shape) * tied).argmax(1).astype(np.int64)
            top = counts.max(1).astype(np.int64)
            panels.append({
                "n_passes": step + 1,
                "top_counts": [int(v) for v in top],
                "decision": [int(v) for v in dec],
                "correct": [int(v) for v in (dec == labels)],
                "agrees_reference": [int(v) for v in (dec == ref)],
                "plurality_tie_rate": float((tied.sum(1) > 1).mean()),
            })
    payload = {
        **stamp(cell["cid"], 4, sha, chash),
        "config": {**cfg, "r0": r0_tag(r0), "split": "test",
                   "fixed_ns": list(CALIB_NS), "t_max": T_MAX, "t_ref": T_REF,
                   "lanes": LANES, "margin_cut": MARGIN_CUT,
                   "warmup_per_lane": wpl, "dims": list(DIMS),
                   "reference_convention": "split-sample"},
        "note": ("confidence = top_counts / n_passes (vote share of the "
                 "decided class); raw pairs only, ECE/Brier/NLL computed at "
                 "writing time with 15 equal-mass bins; no temperature "
                 "scaling applied here"),
        "n_images": n,
        "labels": [int(v) for v in labels],
        "reference": [int(v) for v in ref],
        "marginal": [int(v) for v in marginal],
        "reference_tie": [int(v) for v in tie],
        "panels": panels,
        "wall_seconds": time.time() - t0,
    }
    write_atomic(cell_path(cell["cid"]), payload, compact=True)
    return payload


# ---------------------------------------------------------------------------
# cell enumeration (pure — no io, so --dry-run and the tests are cheap)
# ---------------------------------------------------------------------------

def is_headline_r0(r0) -> bool:
    return r0_tag(r0) in {r0_tag(x) for x in HEADLINE_R0}


def noise_seeds_for(dataset: str, r0) -> tuple:
    headline = is_headline_r0(r0)
    if dataset == "fashion":
        return NOISE_SEEDS[:3] if headline else NOISE_SEEDS[:2]
    return NOISE_SEEDS[:5] if headline else NOISE_SEEDS[:2]


def train_seeds_for(dataset: str, arm) -> tuple:
    if dataset == "fashion":
        return (0,)
    return TRAIN_SEEDS if tuple(arm) == PRIMARY_ARM else (0,)


def p2_cell(dataset: str, train_seed: int, device_class: str, mode: str,
            rule: str, noise_seed: int, r0, sigma_delta: float = 0.0,
            chip_seed=None, t_min: int = T_MIN,
            settle_ratio: float = SETTLE_RATIO) -> dict:
    """One phase-2 cell descriptor. chip_seed defaults to train_seed."""
    cs = train_seed if chip_seed is None else int(chip_seed)
    cid = (f"p2_{dataset}_ts{train_seed}_{device_class}-{mode}"
           f"_r{r0_tag(r0)}_{rule}_ns{noise_seed}"
           + variant_tag(sigma_delta, cs, t_min, settle_ratio))
    return {"cid": cid, "phase": 2, "r0": r0,
            "config": {"dataset": dataset, "train_seed": train_seed,
                       "chip_seed": cs, "device_class": device_class,
                       "mode": mode, "rule": rule, "noise_seed": noise_seed,
                       "sigma_delta": float(sigma_delta),
                       "centering": CENTERING, "t_min": int(t_min),
                       "settle_ratio": float(settle_ratio)}}


def phase0_tasks() -> list:
    return [{"cid": f"p0_ckpt_{ds}_seed{s}", "phase": 0, "dataset": ds,
             "train_seed": s}
            for ds in DATASETS for s in TRAIN_SEEDS]


def phase1_tasks() -> list:
    tasks = [{"cid": ref_cid(ds, s, "test"), "phase": 1, "dataset": ds,
              "train_seed": s, "split": "test"}
             for ds in DATASETS for s in TRAIN_SEEDS]
    tasks.append({"cid": ref_cid("mnist", 0, "val2000"), "phase": 1,
                  "dataset": "mnist", "train_seed": 0, "split": "val2000"})
    return tasks


def phase2a_tasks() -> list:
    """One rho per device condition used by a fixedrho cell."""
    seen, tasks = set(), []
    for cell in phase2_cells():
        cfg = cell["config"]
        if cfg["rule"] not in FIXEDRHO_RULES:
            continue
        key = rho_key(cfg["device_class"], cfg["mode"], cell["r0"],
                      cfg["sigma_delta"], cfg["chip_seed"],
                      cfg["settle_ratio"])
        if key in seen:
            continue
        seen.add(key)
        tasks.append({"cid": rho_cid(cfg["device_class"], cfg["mode"],
                                     cell["r0"], cfg["sigma_delta"], key[4],
                                     cfg["settle_ratio"]),
                      "phase": "2a", "device_class": cfg["device_class"],
                      "mode": cfg["mode"], "r0": cell["r0"],
                      "sigma_delta": cfg["sigma_delta"], "chip_seed": key[4],
                      "settle_ratio": cfg["settle_ratio"]})
    return tasks


def phase2_cells() -> list:
    cells, seen = [], set()

    def add(cell):
        if cell["cid"] in seen:
            return
        seen.add(cell["cid"])
        cells.append(cell)

    # (1) main grid: every rule on every arm (coverage note (a))
    for dataset in DATASETS:
        for arm in ARMS:
            for train_seed in train_seeds_for(dataset, arm):
                for rule in RULES:
                    for r0 in R0S:
                        dev, mode = arm_key(arm[0], arm[1], r0)
                        for ns in noise_seeds_for(dataset, r0):
                            add(p2_cell(dataset, train_seed, dev, mode, rule,
                                        ns, r0))

    # (2) arrhcap: the attempt-frequency-ceiling arm, worst corner (note (f))
    for rule in CAP_RULES:
        for r0 in CAP_R0S:
            for ns in NOISE_SEEDS[:2]:
                add(p2_cell("mnist", 0, CAP_ARM[0], CAP_ARM[1], rule, ns, r0))

    # (3) sigma_delta column: device variation, 3 chip seeds (note (g))
    for sd in SIGMA_DELTAS:
        for cs in SIGMA_CHIP_SEEDS:
            for rule in SIGMA_RULES:
                for r0 in SIGMA_R0S:
                    add(p2_cell("mnist", 0, PRIMARY_ARM[0], PRIMARY_ARM[1],
                                rule, NOISE_SEEDS[0], r0, sigma_delta=sd,
                                chip_seed=cs))

    # (4) t_min sensitivity, headline r0 incl. the ideal calibration point
    for tm in T_MIN_SENS:
        for rule in SENS_RULES:
            for r0 in HEADLINE_R0:
                dev, mode = arm_key(PRIMARY_ARM[0], PRIMARY_ARM[1], r0)
                for ns in NOISE_SEEDS[:2]:
                    add(p2_cell("mnist", 0, dev, mode, rule, ns, r0,
                                t_min=tm))

    # (5) settle_ratio sensitivity (finite r0 only: the ideal source has no
    #     settle interval, so its cells are shared with the default rows)
    for sr in SETTLE_RATIO_SENS:
        for rule in SENS_RULES:
            for r0 in SENS_R0S_FINITE:
                for ns in NOISE_SEEDS[:2]:
                    add(p2_cell("mnist", 0, PRIMARY_ARM[0], PRIMARY_ARM[1],
                                rule, ns, r0, settle_ratio=sr))

    # (6) on-chip rho variant (coverage note (i)): the same Markov rule fed the
    #     ACF over ALL fluctuating inputs instead of the block-A marginal mask.
    #     Primary arm only — it is a check on the deployment claim, not a new
    #     axis — and all three train seeds, because the claim it supports is a
    #     compliance count over replicate cells.
    for ts in TRAIN_SEEDS:
        for r0 in ONCHIP_R0S:
            for ns in ONCHIP_NOISE_SEEDS:
                add(p2_cell("mnist", ts, PRIMARY_ARM[0], PRIMARY_ARM[1],
                            ONCHIP_RULE, ns, r0))
    return cells


def phase3_cells() -> list:
    cells, seen = [], set()
    for dataset in DATASETS:
        for arm in ARMS:
            for r0 in R0S:
                dev, mode = arm_key(arm[0], arm[1], r0)
                headline = is_headline_r0(r0)
                for ns in (NOISE_SEEDS[:3] if headline else NOISE_SEEDS[:1]):
                    cid = (f"p3_{dataset}_ts0_{dev}-{mode}_r{r0_tag(r0)}"
                           f"_ladder_ns{ns}")
                    if cid in seen:
                        continue
                    seen.add(cid)
                    cells.append({
                        "cid": cid, "phase": 3, "r0": r0,
                        "config": {"dataset": dataset, "train_seed": 0,
                                   "chip_seed": 0, "device_class": dev,
                                   "mode": mode, "noise_seed": ns,
                                   "sigma_delta": 0.0, "centering": CENTERING,
                                   "t_min": T_MIN,
                                   "settle_ratio": SETTLE_RATIO}})
    # the matched-budget frontier is read off the IDEAL ladder at all three
    # train seeds, because F1(b) is reported per train seed (audit r1 item 11)
    for ts in TRAIN_SEEDS[1:]:
        for ns in NOISE_SEEDS[:1]:
            cells.append({
                "cid": f"p3_mnist_ts{ts}_ideal-ideal_rinf_ladder_ns{ns}",
                "phase": 3, "r0": np.inf,
                "config": {"dataset": "mnist", "train_seed": ts,
                           "chip_seed": ts, "device_class": "ideal",
                           "mode": "ideal", "noise_seed": ns,
                           "sigma_delta": 0.0, "centering": CENTERING,
                           "t_min": T_MIN, "settle_ratio": SETTLE_RATIO}})
        for r0 in (1.0, 0.5):
            for ns in NOISE_SEEDS[:1]:
                cells.append({
                    "cid": (f"p3_mnist_ts{ts}_{PRIMARY_ARM[0]}-"
                            f"{PRIMARY_ARM[1]}_r{r0_tag(r0)}_ladder_ns{ns}"),
                    "phase": 3, "r0": r0,
                    "config": {"dataset": "mnist", "train_seed": ts,
                               "chip_seed": ts,
                               "device_class": PRIMARY_ARM[0],
                               "mode": PRIMARY_ARM[1], "noise_seed": ns,
                               "sigma_delta": 0.0, "centering": CENTERING,
                               "t_min": T_MIN,
                               "settle_ratio": SETTLE_RATIO}})
    return cells


def phase4_cells() -> list:
    cells = []
    dev, mode = PRIMARY_ARM
    for r0 in (np.inf, 0.5):
        dk, mk = arm_key(dev, mode, r0)
        for ns in NOISE_SEEDS[:3]:
            cells.append({
                "cid": f"p4_mnist_ts0_{dk}-{mk}_r{r0_tag(r0)}_calib_ns{ns}",
                "phase": 4, "r0": r0,
                "config": {"dataset": "mnist", "train_seed": 0, "chip_seed": 0,
                           "device_class": dk, "mode": mk, "noise_seed": ns,
                           "sigma_delta": 0.0, "centering": CENTERING,
                           "t_min": T_MIN, "settle_ratio": SETTLE_RATIO}})
    return cells


def tasks_for_phase(phase: str) -> list:
    return {"0": phase0_tasks(), "1": phase1_tasks(),
            "2": phase2a_tasks() + phase2_cells(),
            "3": phase3_cells(), "4": phase4_cells()}[phase]


def all_tasks() -> list:
    return [t for p in ("0", "1", "2", "3", "4") for t in tasks_for_phase(p)]


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def cell_headline(payload: dict) -> str:
    if payload.get("phase") == 2:
        m = payload["stopping_error"]["marginal"]
        f = payload["by_termination"]["fired"]
        marg = ("n/a" if m["n"] == 0 else
                f"{m['stopping_error']:.4f} "
                f"({m['stopping_error_over_alpha']:.2f}x alpha)")
        return (f"err(marg)={marg} trunc={payload['truncation_rate']:.3f} "
                f"mean_n={payload['mean_samples']:.1f} fired_n={f['n']} "
                f"ties={payload['plurality_tie_rate']:.4f}")
    if payload.get("phase") == 3:
        last = payload["ladder"][-1]
        return (f"acc@{last['n_passes']}={last['task_accuracy']:.4f} "
                f"rungs={len(payload['ladder'])}")
    if payload.get("phase") == 4:
        return f"panels={len(payload['panels'])}"
    return ""


def run_task(task: dict, sha: str, chash: str, log=print):
    phase = str(task["phase"])
    if phase == "0":
        ensure_checkpoint(task["dataset"], task["train_seed"], log=log)
        return None
    if phase == "1":
        return compute_reference(task["dataset"], task["train_seed"],
                                 task["split"], sha, chash, log=log)
    if phase == "2a":
        return compute_rho(task["device_class"], task["mode"], task["r0"],
                           sha, chash, sigma_delta=task["sigma_delta"],
                           chip_seed=task["chip_seed"],
                           settle_ratio=task["settle_ratio"], log=log)
    return {"2": run_phase2_cell, "3": run_phase3_cell,
            "4": run_phase4_cell}[phase](task, sha, chash, log=log)


def task_done(task: dict, chash: str) -> bool:
    if str(task["phase"]) == "0":
        return ckpt_path(task["dataset"], task["train_seed"]).exists()
    return fresh_result(task["cid"], chash) is not None


# ---------------------------------------------------------------------------
# --verify: recompute one cell into scratch and diff (never overwrite)
# ---------------------------------------------------------------------------

VOLATILE_KEYS = {"wall_seconds", "git_dirty", "git_sha", "code_hash"}


def diff_payload(a, b, path: str = "", rel: float = 1e-9) -> list:
    """Recursive diff of two cell payloads, ignoring the volatile stamps.

    Numbers must agree to `rel` (they should be bit-identical: same seeds, same
    draw order), lists must match elementwise, and a key present on one side
    only is a difference. Returns a list of human-readable paths.
    """
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in VOLATILE_KEYS:
                continue
            if k not in a or k not in b:
                side = "stored" if k not in a else "recomputed"
                out.append(f"{path}.{k}: missing in {side}")
                continue
            out += diff_payload(a[k], b[k], f"{path}.{k}", rel)
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff_payload(x, y, f"{path}[{i}]", rel)
            if len(out) > 40:
                out.append(f"{path}: ... more differences suppressed")
                return out
        return out
    if isinstance(a, bool) or isinstance(b, bool):
        return [] if a == b else [f"{path}: {a} vs {b}"]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b:
            return []
        if (np.isnan(a) if isinstance(a, float) else False) and \
           (np.isnan(b) if isinstance(b, float) else False):
            return []
        scale = max(abs(a), abs(b), 1e-300)
        return ([] if abs(a - b) / scale <= rel
                else [f"{path}: {a} vs {b}"])
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def verify_cell(cid: str, chash: str, log=print) -> int:
    """Recompute `cid` into a scratch dir and diff against the stored payload.

    The stored cell is never touched. Dependencies (the phase-1 reference, the
    phase-2a rho) are loaded from the REAL grid first so verification does not
    re-run a 1024-pass reference; only the named cell is recomputed.
    """
    global GRID
    path = cell_path(cid)
    if not path.exists():
        log(f"verify {cid}: NOT ON DISK")
        return 2
    stored = json.loads(path.read_text())
    tasks = {t["cid"]: t for t in all_tasks()}
    if cid not in tasks:
        log(f"verify {cid}: not enumerated by the current grid definition")
        return 2
    task = tasks[cid]
    if str(task["phase"]) in ("2", "3", "4"):     # prime dependency caches
        cfg = task["config"]
        get_reference(cfg["dataset"], cfg["train_seed"], "test", chash,
                      log=log)
        if cfg.get("rule") in FIXEDRHO_RULES:
            get_rho(cfg["device_class"], cfg["mode"], task["r0"], chash,
                    sigma_delta=cfg["sigma_delta"],
                    chip_seed=cfg["chip_seed"],
                    settle_ratio=cfg["settle_ratio"],
                    field=rho_field(cfg["rule"]), log=log)
    real = GRID
    with tempfile.TemporaryDirectory(prefix="apbit-verify-") as td:
        GRID = Path(td)
        try:
            fresh = run_task(task, git_sha(), chash, log=log)
        finally:
            GRID = real
    if fresh is None:
        log(f"verify {cid}: phase 0 task, nothing to diff")
        return 0
    diffs = diff_payload(stored, fresh)
    if not diffs:
        log(f"verify {cid}: IDENTICAL "
            f"(ignoring {sorted(VOLATILE_KEYS)})")
        return 0
    log(f"verify {cid}: {len(diffs)} DIFFERENCE(S)")
    for d in diffs[:40]:
        log(f"  {d}")
    return 1


def main(argv=None):
    global GRID
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", nargs="+", default=["all"],
                    choices=["0", "1", "2", "3", "4", "all"],
                    help="phases to run, in the order given")
    ap.add_argument("--only", default=None,
                    help="substring filter on the cell id")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the cells that would run, then exit")
    ap.add_argument("--verify", default=None, metavar="CELL_ID",
                    help="recompute one cell into scratch and diff it against "
                         "the stored payload; never writes to the grid")
    ap.add_argument("--grid-dir", default=None,
                    help="override the output directory "
                         "(default results/grid2)")
    args = ap.parse_args(argv)

    if args.grid_dir:
        GRID = Path(args.grid_dir)
    sha, chash = git_sha(), code_hash()
    dirty = git_dirty()
    GRID.mkdir(parents=True, exist_ok=True)
    print(f"grid driver: code_hash {chash[:12]} @ git {sha[:8]}"
          f"{' (DIRTY TREE)' if dirty else ''} -> {GRID}", flush=True)

    if args.verify:
        return verify_cell(args.verify, chash,
                           log=lambda m: print(m, flush=True))

    phases = ["0", "1", "2", "3", "4"] if "all" in args.phase else args.phase
    plan = []
    for phase in phases:
        for task in tasks_for_phase(phase):
            if args.only and args.only not in task["cid"]:
                continue
            plan.append(task)

    counts = {}
    for task in plan:
        counts[str(task["phase"])] = counts.get(str(task["phase"]), 0) + 1
    print("planned cells: " + ", ".join(f"phase {k}: {v}"
                                        for k, v in sorted(counts.items()))
          + f"  (total {len(plan)})", flush=True)

    if args.dry_run:
        for task in plan:
            mark = "done" if task_done(task, chash) else "TODO"
            print(f"  [{mark}] {task['cid']}", flush=True)
        return 0

    t_start = time.time()
    ran = skipped = 0
    for i, task in enumerate(plan, 1):
        if task_done(task, chash):
            skipped += 1
            continue
        t0 = time.time()
        print(f"[{i}/{len(plan)}] {task['cid']}", flush=True)
        payload = run_task(task, sha, chash, log=lambda m: print(m, flush=True))
        ran += 1
        head = cell_headline(payload) if payload else ""
        print(f"    done {time.time() - t0:.1f}s  {head}", flush=True)
    print(f"grid: {ran} run, {skipped} reused, "
          f"{time.time() - t_start:.0f}s total", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
