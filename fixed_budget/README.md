# fixed_budget/

Fixed-budget study: stochastic binary neural network (SBNN) inference under
autocorrelated sMTJ randomness: what happens to classification accuracy when
p-bit devices are sampled faster than their correlation time, and what
recovers the loss. The adaptive-stopping study lives in the repository root.

## Layout
- `src/smtjnn/`: model (SBNN + STE), telegraph sMTJ source, protocols, training
- `tests/`: physics + protocol validation (pytest)
- `experiments/`: run_v2.py (all phases), analysis / figure scripts;
  sllg_*.py and fig_*.py for the device check and the letter's figures
- `micromagnetics/`: mumax3 run generator, queue runner and analysis
- `results/v2/`: one JSON per (config, seed), stamped with its git SHA
- `results/*.pt`: trained checkpoints
- `results/micromagnetics/`, `results/sllg_matched/`, `results/validation/`:
  mumax3, matched-macrospin and side-by-side records (`results/sllg/` is the
  earlier 3.5 kT macrospin, kept as a sensitivity check)

## Reproduce (run inside fixed_budget/)
    pip install -e ".[dev]"
    python -m pytest tests/
    python experiments/run_v2.py --phase 0   # train (val split; test untouched)
    python experiments/run_v2.py --phase 1   # ideal-randomness anchors
    python experiments/run_v2.py --phase 2   # sampling-ratio sweep
    python experiments/run_v2.py --phase 3   # device variability
    python experiments/run_v2.py --phase 4   # correlation-aware training
    python experiments/run_v2.py --phase 5   # lane-count invariance
    python experiments/run_v2.py --phase 2 --dataset fashion

MNIST and Fashion-MNIST are downloaded by torchvision into `data/` on first
use. CPU-only; a few hours on a laptop. Git SHAs in the `results/v2/`
records refer to the development history before this public repository.

## Device check of the telegraph model (letter revision)
Three steps, each documented below or in its own README, each reading the
records of the step before:

1. **Micromagnetics** (`micromagnetics/`, mumax3 on a GPU): a 20 nm
   perpendicular disk at 300 K under field bias, with an easy-plane control.
   Run list, queue runner and analysis are in `micromagnetics/README.md`;
   the reduced records are in `results/micromagnetics/`. Production step
   25 fs; the 50 and 12.5 fs runs are the time-step check.
2. **Matched macrospin** (`src/smtjnn/sllg.py`, `experiments/sllg_*.py`,
   any CUDA GPU): one command,
   `bash experiments/sllg_matched.sh results/sllg_matched cuda:0`, fits the
   barrier to the mumax3 tau_c(0), characterizes the device, trains six
   networks on macrospin activations and runs the classifier with macrospin
   neurons next to the telegraph model. Records in `results/sllg_matched/`.
3. **Figures and numbers of the letter** (CPU, seconds):

       python experiments/fig_validation.py build     # results/validation/validation.json
       python experiments/fig_validation.py draw figs/gupta1.pdf
       python experiments/fig_letter.py figs          # gupta2.pdf, gupta3.pdf

   Fig. 1 puts the mumax3 disk and the matched macrospin on common axes
   against the telegraph model; Figs. 2 and 3 redraw the telegraph records of
   `results/v2/` with the macrospin columns of
   `results/sllg_matched/classifier_table.json` as open symbols. Steps 1 and
   2 need not be rerun for step 3: all records are in the repository.

Nothing in these steps depends on a particular machine. The mumax3 binary
and its kernel cache are set at the top of `micromagnetics/run_queue.py`
(or through `MUMAX3_DIR` and `MUMAX3_CACHE`);
the macrospin scripts take the device with `--gpu`.
`validation.json` must be rebuilt whenever either record set changes, and
the macrospin must be rematched (`sllg_matched.sh` into an empty directory)
whenever the mumax3 summary changes.

## Macrospin (sLLG) check of the telegraph model
`src/smtjnn/sllg.py` replaces the two-state telegraph neuron with a
stochastic Landau-Lifshitz-Gilbert macrospin: uniaxial perpendicular
anisotropy, Brown thermal field, stochastic Heun (Stratonovich) with |m|
renormalized each step, run as a Triton kernel on CUDA (a plain PyTorch
step is the reference and the CPU fallback). The neuron is read as
sign(mz). `MacrospinSource` has the same interface as `TelegraphSource`,
so `infer.evaluate_lanes` and `train.train` run it unchanged.

Device. The macrospin has a barrier, a moment and alpha = 0.1 (dt = 1 ps).
The matched device (`results/sllg_matched/`) takes the field-to-I slope
measured on the mumax3 disk and fits one parameter, the barrier: the value
at which the macrospin's zero-bias 1/e read-autocorrelation time equals
the mumax3 tau_c(0). The fit uses Brown's Fokker-Planck equation for mz
(`sllg.fp_tau_1e`, `sllg.match_barrier`), which agrees with the sLLG runs
to 0.2%. The fitted barrier (3.91 kT, `match.json`) is higher than the
3.56 kT read off the histogram of the disk-averaged mz, so that histogram
understates the barrier of the equivalent single spin. An earlier device
(barrier 3.5 kT, slope 0.056 per mT, `results/sllg/`) is kept as a
sensitivity check. Rerun
`--part match` whenever the mumax3 summary changes.

Definitions: tau_c(0) is the 1/e time of the zero-bias read
autocorrelation (as for the mumax3 disk) and r0 = read interval /
tau_c(0); a network input I is applied as the field Bz = I / slope at
every I, with no clipping, so large inputs saturate the device and remove
its minority well. Dwell times and escape rates use the mumax3 analysis's
own estimator (flips out of a state over the time spent in it, every run
counted). Under bias the read autocorrelation is not one exponential, so
each bias record also gives the tail fit with its amplitude and the
integrated time.

    python -m pytest tests/test_sllg.py tests/test_sllg_match.py
    bash experiments/sllg_matched.sh results/sllg_matched cuda:0

`sllg_matched.sh` runs every step in order into an empty directory: the
match (2 min, CPU), the device parts (`dt bias step summary`), the six
training runs, the telegraph-model columns on CPU, the macrospin
classifier runs, the time-step check and the table. The networks are
trained before any run that tests them; a run whose network has no
checkpoint stops with an error, and the last step (`--part table
--strict`) fails if any cell is missing. Every part skips records that
exist, so a set built on an older mumax3 summary must be moved aside
before the script is run again.

The 3.5 kT set: the same commands with `--out results/sllg` and
`--delta 3.5 --slope 0.056` (the defaults) and `--dev-dir results/sllg`,
`--groups legacy`.

Classifier sources: `sllg` (macrospin neurons), `telegraph` (the paper's
model, same 500 lanes), `tcap` (telegraph model whose rate growth
2f cosh I is capped at the macrospin's relaxation rate after a strong
field reversal, `TelegraphSource(rate_cap=...)`; a diagnostic for the
streaming gap), `tlaw_reversal` and `tlaw_zero` (telegraph model whose rate
at each input is the macrospin's measured step-response rate, tau_c(0) over
the 1/e time after a field step to that input from the opposite input or
from zero field, `TelegraphSource(rate_law=...)`; tabulated at the five
step fields, nominal I = 1, 2, 4, 6, 8, each placed at the network input
that the classifier's field-to-I map gives that field, and held above the
last one), `equilibrium` (independent reads from the macrospin's
exact P(+)) and `ideal`. Networks: the ideal-trained seed-0 network, the
telegraph-trained correlation-aware networks, and networks trained the
same way on macrospin activations (`experiments/sllg_train.py`, in
`$M/nets/`, with their training logs).

Records: `device_*.json` and `device.json` (tau_c(0), the field-to-I map,
time scales against the cosh law), `match.json` (the barrier fit),
`clf/` (one record per classifier run; telegraph runs of the shared
networks are in `results/sllg/clf/`), `classifier_table.{json,csv}` (all
sources side by side, the rate caps and rate laws, the share of the
telegraph-to-macrospin gap each diagnostic closes, and the 90% crossings
with their interpolation ranges; a mean is over every noise seed on
record, and the `n` columns give the count). `results/sllg/clf_tailfit/`
holds earlier 3.5 kT records whose r0 used the tail-fit time instead of the 1/e time.

`code_sha256` in a record is the hash of the script versions that wrote it.
`experiments/sllg_classifier.py` was edited after the classifier runs (a
guard against a zero gap in `gap_closed`, used only by `--part table`, and
docstring wording), so its present hash differs from the one in the `clf/`
records; no record depends on the edit. `micromagnetics/analyze.py` and
`experiments/sllg_device.py` are unchanged since the records were written.
