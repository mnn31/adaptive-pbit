# micromagnetics/

Micromagnetic check of the telegraph device model used in the letter. mumax3
simulates a low-barrier perpendicular free layer at 300 K, and the thresholded
read sign(m_z) is compared with the two-state model: stationary law
P(+) = (1 + tanh I)/2, correlation time tau_c(I) = tau_c(0)/cosh I, and the
propagator p + (1[s=+] - p) exp(-t/tau_c). An easy-plane disk (Ku = 0) is run
as a control.

## Devices
- **P (perpendicular):** disk 20 nm x 1 nm, 2 nm x 2 nm x 1 nm cells (80 cells),
  Ms = 1.0 MA/m, A = 15 pJ/m, Ku = 6.0e5 J/m^3 along z, alpha = 0.1, T = 300 K.
  Ku is a mesh-level input calibrated at temperature; the effective barrier is
  measured from the m_z histogram. Bias: perpendicular field B_z (and, as a
  check, a Slonczewski current through a +z fixed layer).
- **E (easy-plane):** the same disk with Ku = 0, read along x, biased with B_x.

Each mumax3 process simulates an 8x8 array of identical disks (64 nm pitch,
one region each) and tables every disk's average magnetization every 10 ps
(5 ps for device E). Heun solver with a fixed step. **The production step is
25 fs**: every number under bias comes from 25 fs runs, including the
spin-torque points and the easy-plane control in a field. The study was first
run at 50 fs. At +18 and +36 mT the 50 fs correlation times came out shorter
than at 25 fs, so the 50 fs runs, with the 12.5 fs runs, are kept
only as the time-step check (see "Time step" below for how strong that
evidence is). The array-pitch control is a zero-field 50 fs run; the
alpha = 0.03 check is at 25 fs. The first 20 ns of every run is discarded. That is only
about 1.4 tau_c(0), which is enough because each disk starts up or down with
the probability expected at its field, not from a common state; `p_up_blocks`
in the records gives P(+) in the first 20 ns after the discarded start, the
next 40 ns and the rest, and shows no drift.

## Files
- `mx3gen.py`: writes the mumax3 script of a disk array.
- `make_runs.py`: the full run list (bias points, seeds, checks, step response,
  easy-plane, spin-torque probe); `--stt J1 J2` writes the spin-torque runs;
  `--sets eq step --dt 25e-15 --seed-offset 10000` writes the 25 fs production
  rerun of the equilibrium and step sets (same plan, seeds shifted by 10000);
  `--extra` writes the runs added afterwards at 25 fs (more seeds at -9, -27,
  -36, 54 and 72 mT, and the easy-plane step check); `--controls` writes the
  25 fs repeats of the spin-torque runs and of the easy plane in a field, two
  more seeds of the alpha = 0.03 check and five more of the 12.5 fs zero-field
  point. `--stt-step J` writes the current-step check (equilibrium runs at
  current density J and a square wave between -J and +J, 25 fs);
  `--mesh-cal KU ...` and `--mesh KU` write the mesh check with 1 nm cells
  (a short zero-field run per anisotropy constant, then the check at one).
  Weak-bias steps (0 and +-18 mT) are run twice: with 50 and 60 ns half
  periods, and with 80 ns, which is longer than a settle of 4 tau_c(0).
- `run_queue.py`: runs a job directory through CUDA MPS, several processes per
  GPU, and packs each finished table into a float16 `.npz`. Runners on several
  GPUs can share one directory.
- `mmstats.py`: the statistics (dwell times and escape rates, with
  `escape_rates` as the reference definition of the flip-count rate, Kaplan-Meier
  survival, autocorrelation, stale reads, histogram barrier, step fits, the
  high-barrier macrospin rate formula).
- `analyze.py`: reduces a run directory to the records in
  `results/micromagnetics/`.
- `analyze_checks.py`: reduces the current-step and mesh checks against those
  records, which it reads and does not change.

## Reproduce (GPU with mumax3 3.12 and CUDA MPS)
    python make_runs.py RUNS                       # 125 jobs, 50 fs study and checks
    python run_queue.py RUNS --gpu 0 --slots 8
    python make_runs.py RUNS_STT --stt 1.4e10 2.4e10   # chosen from the probe in RUNS
    python run_queue.py RUNS_STT --gpu 0 --slots 4
    python make_runs.py RUNS_25 --sets eq step --dt 25e-15 --seed-offset 10000   # 70 jobs
    python run_queue.py RUNS_25 --gpu 0 --slots 14
    python make_runs.py RUNS_25X --extra           # 16 jobs
    python run_queue.py RUNS_25X --gpu 0 --slots 8
    python make_runs.py RUNS_25C --controls        # 25 jobs
    python run_queue.py RUNS_25C --gpu 0 --slots 13
    python analyze.py RUNS RUNS_STT RUNS_25 RUNS_25X RUNS_25C ../results/micromagnetics
    python make_runs.py RUNS_STEPJ --stt-step 4.4e10   # 6 jobs; I near 3.9, as the 72 mT step
    python run_queue.py RUNS_STEPJ --gpu 0 --slots 6
    python make_runs.py RUNS_MCAL --mesh-cal 6.0e5 6.1e5 6.2e5 6.3e5 6.4e5   # 5 jobs
    python run_queue.py RUNS_MCAL --gpu 0 --slots 5
    python make_runs.py RUNS_MESH --mesh 6.2e5         # 6 jobs; Ku chosen from the scan
    python run_queue.py RUNS_MESH --gpu 0 --slots 6
    python analyze_checks.py RUNS_STEPJ RUNS_MCAL RUNS_MESH ../results/micromagnetics

`analyze.py` sorts the runs by set and time step from their parameter files,
so the order of the directories does not matter. The paths of the mumax3
binary and kernel cache are set at the top of `run_queue.py` (or with the
environment variables `MUMAX3_DIR` and `MUMAX3_CACHE`). The 50 fs study
is about 35 000 ns of 64-disk simulation in 50 fs-equivalent steps (about 14
GPU-hours on one RTX 4090 with MPS); the 25 fs rerun and the added runs are
about 46 000 more, run on two GPUs. Packed trajectories are not in the
repository; the reduced records are.

## Records (`results/micromagnetics/`)
- `summary.json`: the headline numbers of all records below.
- `device_p_equilibrium.json`: per field point (seed mean and standard error):
  P(+), I, histogram peaks and well depths, dwell rates and exponential tests
  (hysteretic threshold +-0.4 and plain sign; Kaplan-Meier deviation D against
  the same statistic for a simulated telegraph process with the same rates,
  window and sampling), correlation times (1/e time, dwell rates, tail fit),
  nearest-neighbour correlation; field-to-I fit; tau_c(I) against the cosh law,
  a constant and the high-barrier macrospin formula (Brown), which is outside
  its range of validity at this barrier and is only a reference; stale reads
  and read autocorrelation at r0 = 0.02 to 4 against the telegraph propagator.
- `device_p_curves.npz`: autocorrelation (log lag grid) and m_z histogram per point.
- `checks.json`: time step, damping and array-pitch checks. `time_step` puts
  the 50, 25 and 12.5 fs runs side by side at every field (12.5 fs at 0, 18 and
  36 mT only), with their differences in units of the error;
  `time_step_pooled` is the same comparison with the seeds of +B and -B
  pooled; `dt50` and
  `dt12.5` hold the full 50 and 12.5 fs points, `field_slope_by_dt` the
  field-to-I slope at each step, and `rerun_vs_earlier_25fs` compares the
  production rerun with the earlier 25 fs check runs that are pooled with it.
- `step_response.json`, `step_curves.npz`: ensemble read mean after field steps
  (25 fs; `steps_dt50` and the `dt50_` curves are the same steps at 50 fs).
  `tau_step_over_cosh` is the step time over the cosh-law time for each
  estimate of I in `I_target_options`.
- `easy_plane.json`: device E statistics per in-plane field (25 fs);
  `easy_plane_dt50.json`: the same at 50 fs.
- `spin_torque.json`: current-biased points. `stt` is the 25 fs set, compared
  with the 25 fs zero-bias point; `stt_dt50` and `sttprobe_dt50` are the 50 fs
  runs (the probe only served to choose the currents).
- `run_manifest.json`: every job with its parameters and wall time.
- `spin_torque_step.json`, `spin_torque_step_curves.npz`: read mean after a
  reversal of the spin-torque current between -4.4e10 and +4.4e10 A/m^2, no
  field (25 fs), with the equilibrium runs at that current. Fields as in
  `step_response.json`; `tau_step_over_cosh` is the step time over
  tau_c(0) / cosh I with tau_c(0) from the field-free production runs.
- `mesh_check.json`: the runs with 1 nm cells (see "Mesh" below), per
  anisotropy constant in `by_ku`, beside the 2 nm production values in
  `reference_2nm`.

## Reading the records
- **Which tau_c in the stale-read comparison.** tau_c(0) is defined as the 1/e
  time of the read autocorrelation C. At zero bias the stale-read probability
  is (1 + C)/2, so the propagator built on the 1/e time (`telegraph_efold`)
  equals the measurement at r0 = 1 by construction and is pulled towards it
  nearby; that row is marked `by_construction` and is not evidence for the
  model. The same holds for any comparison of two devices at r0 = 1 when each
  is scaled by its own 1/e time. `telegraph_dwell` uses 1/(k_up + k_down) from counted flips instead
  and `telegraph_tail` the tail fit; `all_se` carries the error of each tau.
  The flip-count time is shorter than the 1/e time (some flips are reversed
  at once and lose little correlation), so the device is less stale than
  `telegraph_dwell` at short lags and more stale from about r0 = 0.5.
- **Dwell rates.** An escape rate is the number of flips out of a state over
  the time spent in it, with every run counted: the run under way at the
  window start ends in an observed flip, and only the last run of a disk is
  censored (`mmstats.escape_rates`). The mean of whole dwells (first and last
  runs dropped) is not used for rates: a window a few dwells long keeps mostly
  the short ones, so it is biased short (tested in
  `tests/test_micromagnetics_stats.py`). `flips` and `n_flips` are these flip counts. The shape statistics
  (`cv`, `D`, `short_excess`) use whole dwells only, those that began inside
  the window (`n_dwells` of them completed), through a Kaplan-Meier curve.
  `short_excess` is the fraction of dwells shorter than 5% of the mean dwell
  over the exponential value; with few dwells it is noisy, and
  `short_excess_null_sd` gives its per-seed spread for a telegraph process.
  `cv_null_mean` is the telegraph value of `cv` in the same window, below 1
  because the window cuts long dwells.
- **Cosh-law ratios.** `tau_*_over_cosh` is tau(I) cosh(I) / tau(0) with both
  times from the same estimator; 1 means the cosh law holds.
- **Step response.** The relaxation time and the remaining memory
  (s(t) - s_inf)/(s0 - s_inf) depend on the asymptote s_inf, most at weak bias
  where the step is small. Each is given for a fitted asymptote (`free`), the
  equilibrium-run value (`eq_asymptote`) and the mean of the last fifth of the
  segment (`tail_asymptote`), with errors from resampling the disk-steps.
  `tail_start_over_pred_tau` says how relaxed the tail window should be. For
  strongly driven steps the response is not a single exponential and the free
  fit overshoots abs(s_inf) = 1; use the fixed asymptotes there.
- **Large bias.** `few_flips` marks points with fewer than 30 flips. There the
  read is single-state and its 1/e time measures in-well motion, not switching.
  Where a seed never visits the minority state (108 mT) the logit is unbounded
  and I is not measured. `I_model` is then the field-fit line
  (`I_model_source`), which is an extrapolation from abs(I) <= 2.1 to about 6
  and is not reliable: the map bends below the line at large field.
  `field_to_I.extrapolated_I` gives I there under the line, an odd cubic
  (a B + c B^3) over the fit range, an odd cubic over every measured point,
  and the largest measured abs(I) as a lower bound, each with its error. A step into such a field carries all four in
  `I_target_options`, and `tau_step_over_cosh` gives the slowdown against the
  cosh law for each. Quote that slowdown as the range over the estimates, not
  as one number; only the lower bound is free of extrapolation.
- **Step fits at strong drive.** At +-72 and +-108 mT the three-parameter fit
  gives a relaxation time 11 to 39% longer than the fixed-asymptote fits
  (17 and 11% at +-72 mT, 32 and 38% at +-108 mT against the equilibrium
  asymptote; `tau_step_free_over_fixed` in each step record, and
  `ranges.step_tau_free_over_fixed_absB_ge_72` in `summary.json`), with an
  asymptote beyond abs(s) = 1. The quoted errors are per asymptote and do
  not include this difference: the slowdown against the cosh law at +72 mT
  is 3.33 +- 0.38 with the equilibrium asymptote and 3.89 +- 0.44 with the
  fitted one. `tau_step_over_cosh` lists both. At +-108 mT no seed visits
  the minority state, so the equilibrium asymptote is exactly +-1 with no
  error and the step errors there contain no asymptote uncertainty; the
  fitted asymptote of 1.03 shows that a single exponential does not describe
  that response.
- **Time step.** At zero bias 50 and 25 fs agree; the 12.5 fs correlation
  time (8 seeds) is 9% above the 25 fs one, 1.8 standard errors. Under bias
  the evidence for a time-step effect is the +18 and +36 mT points, where the
  50 fs correlation times are shorter than the 25 fs ones (`checks.json`,
  `time_step`); the mirrored fields -18 and -36 mT, with fewer seeds, show no
  difference, and the 50 fs errors rest on two to eight seeds per field
  (four at 0 and +36 mT, eight at +18 mT). `time_step_pooled`
  pools +B with -B and is the fairer comparison: there the 25 fs 1/e time is
  2.5 and 3.9 standard errors above the 50 fs one at 18 and 36 mT (1.8 and
  2.4 in the cosh-law ratio), and the 1/e time shows no resolved gap at the
  other fields. The dwell-rate time has pooled gaps of 2.9 and 2.8 standard
  errors at 27 and 36 mT (1.8 and 1.6 in the cosh-law ratio).
  The histogram barrier also moves with the step: 3.534, 3.557 and 3.616 kT
  at 50, 25 and 12.5 fs, the last 1.9 standard errors above the 25 fs value.
  Neither the absolute tau_c(0) nor the barrier is shown to be converged to
  better than about 10% and 0.1 kT.
  25 and 12.5 fs agree within their errors under bias. The 25 fs field-to-I
  slope is 2% below the 50 fs one. 25 fs is the production step because it is
  the finer of the two steps with full statistics; the 50 fs runs are not
  shown to be wrong. Compare with the cosh law through `tau_*_over_cosh`,
  where tau_c(0) and I come from the same time step; a 12.5 fs ratio therefore
  carries the error of the 12.5 fs tau_c(0).
- **Mesh.** Ku is a mesh-level input. Thermal motion of the cells lowers the
  barrier (the zero-temperature value for this disk is Ku V minus the
  demagnetizing energy difference, 46.4 - 39.1 = 7.3 kT, against 3.56 kT
  measured at 300 K with 2 nm cells), and a finer mesh has more thermal modes,
  so the same Ku gives a lower barrier with 1 nm cells. `mesh_check.json`
  therefore holds a scan of Ku at 1 nm (`meshcal_*` runs, 28 ns each) and a
  longer set at the Ku whose barrier is closest to the 2 nm one (`mesh_*`,
  `meshstep_*`). The 1 nm runs use a 6.25 fs step, because the stiffest
  exchange mode scales as 1 / cell^2, and a 48 nm pitch instead of 64 nm to
  limit the grid; `neighbour_corr` in each record shows what that costs.
  Each Ku is measured in units of its own tau_c(0). I at 72 mT is not
  measured on the fine mesh, so the step comparison uses tau_step / tau_c(0)
  and, for the cosh-law ratio, I = 4 x I(18 mT) on both meshes; the
  asymptote of the fixed-asymptote fit is then tanh of that I, not an
  equilibrium run. The scan runs are short: where a run ends before the read
  decorrelates, its correlation time is missing. Result: with 1 nm cells
  Ku = 6.0e5 J/m^3 gives a barrier of 2.74 kT and Ku = 6.2e5 gives
  3.67 +- 0.03 kT with tau_c(0) = 13.7 +- 1.3 ns; at 6.2e5 the read-change
  factor at r0 = 0.02 is 1.99 (2 nm: 2.10) and the step time at +-72 mT is
  0.157 +- 0.018 and 0.148 +- 0.017 tau_c(0) (2 nm: 0.137 +- 0.007 and
  0.142 +- 0.011).
- **Current steps.** At 4.4e10 A/m^2 the measured input is I = 3.69 +- 0.10
  (20 flips in four seeds). After a reversal of the current the read mean
  relaxes in 0.126 and 0.133 tau_c(0) (equilibrium asymptote), 2.5 to 2.7
  times the cosh-law time, or 3.0 times with a fitted asymptote; the field
  step at I = 3.89 gives 3.3 to 3.9.
- **Dwell test.** The null is 200 telegraph replicas per seed, reduced
  exactly as the data; `seedmean_D_p_value` is the fraction of replicas whose
  seed-mean D is at least the measured one.
- **Lint.** Run `ruff check micromagnetics` from `fixed_budget/` (inside this
  directory ruff sorts the sibling-module imports differently).
