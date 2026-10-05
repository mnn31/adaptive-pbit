# adaptive-pbit

**Correlation-aware sequential stopping for p-bit neural network inference.**

Sequential stopping rules for stochastic-inference networks — Adaptive-
Consistency-style vote-count rules, Wald SPRTs, e-processes — assume the passes
they aggregate are i.i.d. A p-bit device sampled faster than its own
correlation time does not deliver i.i.d. passes. This project measures how far
such a rule's realized error moves away from its nominal level as the sampling
ratio r0 degrades, and shows that the fix belongs in the EVIDENCE, not the
clock: replacing the i.i.d. likelihood with the joint likelihood of a two-state
Markov vote model (Wald's upper threshold unchanged) keeps the stopping error
within the configured level down to r0 = 1 without slowing the sampler; at
faster sampling it only partly reduces the excess.

Two error quantities are kept strictly apart everywhere:

* **stopping error** — P(adaptive decision != the network's own asymptotic
  decision). This is what a rule's nominal level speaks about.
* **task error** — disagreement with the true label. Not bounded by any rule.

Substrate and checkpoints are ported from the author's earlier fixed-budget
study in `fixed_budget/`: a
784-256-128-10 stochastic binary MLP trained with a straight-through estimator,
stochastic at inference, driven by a telegraph-noise sMTJ model.

**Status: complete.** `results/grid2/` holds all 1,055 per-cell result records
used by the manuscript; every figure is regenerated from them by
`experiments/make_figures.py`.

## Layout

```
src/apbit/
  smtj.py             telegraph-noise sMTJ source (ported) + rate ceiling
  model.py            SBNN + straight-through estimator (ported)
  data.py, train.py   MNIST / Fashion-MNIST loading and STE training (ported)
  infer.py            fixed-budget lane-stream evaluation (ported)
  stopping.py         FixedN, DirichletStop, WaldSPRT, MarkovSPRT, EProcessStop
  infer_adaptive.py   per-lane adaptive protocol + split-sample reference
  cost.py             frozen cost proxy (bits, MACs, throttling LATENCY)
  diagnostics.py      vote ACF(k), inflation factor, entropy rate, Wilson
tests/                physics/model validation + rule theory + driver plumbing
experiments/
  run_grid.py         the phased, resumable, content-hash-stamped grid driver
  make_figures.py     every paper figure, from the cells, in one run
  precheck*.py        the week-1 go/no-go runs (superseded; kept for the record)
results/grid2/        per-cell result records used by the paper
fixed_budget/         the fixed-budget study: code, tests, result records, checkpoints
```

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"          # or: pip install -r requirements-frozen.txt
python -m pytest tests/          # ~1 min, no network, no dataset needed

python experiments/run_grid.py --dry-run          # what would run, and its state
python experiments/run_grid.py                    # ~8 h, one CPU core, sequential
python experiments/make_figures.py                # every figure + manifest + captions
```

The grid is deliberately single-process: the randomness sources are stateful
objects whose draw order IS the physics simulated, so no worker pool may
interleave them. Every cell is written atomically the moment it finishes, and
re-running skips any cell already on disk at the current code hash, so the run
is safely resumable (`nohup ... &` is the intended usage; ~8 h on one core of an
M-series laptop, dominated by phase 2's 906 cells at ~25 s each).

Freshness is by CONTENT HASH, not git SHA: `code_hash` = sha256 over
`src/apbit/*.py` + `experiments/run_grid.py`. A manuscript-only commit does not
invalidate cells; an uncommitted edit to the simulator does. Cells with no
`code_hash` come from an earlier protocol and are never reused.

```bash
python experiments/run_grid.py --verify CELL_ID   # recompute into scratch + diff
```

`--verify` never writes to the grid: it recomputes one named cell in a
temporary directory and diffs it against what is stored, ignoring only
`wall_seconds`, `git_sha`, `git_dirty` and `code_hash`.

## Figures

`experiments/make_figures.py` reads `results/grid2/` and writes
`paper/figs/F1..F8*.pdf` (created on first run), plus `figures_manifest.json` (provenance, every cell
consumed, every cell missing) and `captions_draft.md` (the numbers, generated).
Manuscript captions may not state anything `captions_draft.md` does not. A
figure built while the grid is still filling is drawn, but flagged PARTIAL on
stderr and in the manifest; a panel with no data is drawn as an explicit "DATA
PENDING" placeholder, never as an empty axis.

## Statistical conventions (binding)

* Reference decision: majority over T_ref=1024 ideal-source passes of
  **block B**; the marginal stratum is defined by the margin of an
  **independent block A** (split-sample, so stratum selection and scoring
  target do not share a noise term). Block-B ties are excluded and counted.
* The marginal stratum (block-A margin < 0.9) is where the per-image guarantee
  is visible; aggregate numbers are diluted by easy images and are reported as
  such, never as the headline.
* Primary normalization is calibration-referenced: each rule against its OWN
  ideal-randomness error. The raw level is secondary, and the SPRT rules are
  scored against alpha/(1-beta) — the bound Wald's inequalities actually give —
  never alpha.
* Intervals are cluster-robust over train seeds (t(2) on three cluster means)
  wherever three train seeds exist; otherwise min-max, labelled as a range and
  not as a confidence interval. Per-cell Wilson intervals are never pooled.
* Fired and truncated decisions are reported as separate channels: only the
  fired channel is bounded, truncation error is budget error.
* Plurality ties are broken uniformly at random from a seeded per-lane stream,
  and the tie rate is reported with every error number.
* Cost is a proxy: random bits and MACs, both exact counts. Clock throttling is
  charged in LATENCY-equivalent passes only — bits and MACs do not scale with a
  slowdown — and the settle overhead is reported alongside. **No joules are
  claimed anywhere.**

## License

MIT (`LICENSE`).

## AI disclosure

Simulation code, experiment orchestration, and manuscript drafting were done
with substantial assistance from AI tools (Anthropic Claude), under the
author's direction and review; see the manuscript's AI-use statement.

## Provenance

Every result cell in `results/grid2/` records the content hash of the
exact simulator and driver code that produced it; `--verify` in
`experiments/run_grid.py` recomputes any cell into a scratch directory and
diffs. The full development history (including the pre-registration
timestamps referenced in the paper) is retained privately and is available
to editors and reviewers on request.
