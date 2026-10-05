# Author: Manan Gupta <mnn@yogins.com>
"""Telegraph-noise sMTJ neuron model.

Ported unchanged from the smtj-sbnn project (paper 1).

Each neuron is a two-state fluctuator s ∈ {−1,+1} with input-biased
Néel–Arrhenius escape rates

    k(−→+) = f · e^{+I},   k(+→−) = f · e^{−I},   f = f0 · e^{−δΔ}

giving stationary law P(+1) = (1+tanh I)/2 (the p-bit activation) and
correlation time τ_corr(I) = 1/(2 f cosh I). Exact two-state propagator
over an interval Δt with I held constant during the interval:

    P(s' = +1 | s) = p_eq + (1[s=+1] − p_eq) · exp(−Δt/τ_corr)

Control knob: r0 = Δt · 2 f0 = Δt / τ_corr(I=0) for a nominal device.
NOTE the 2τ rule of Daniels et al. is stated in MEAN DWELL time; at I=0
τ_dwell = 2 τ_corr, so rule-compliant sampling is r0 ≥ 4 (not 2).

Variant `bias_independent_tau=True` keeps the same stationary law but a
fixed total rate (k₊₋ + k₋₊ = 2f): the easy-plane-like regime where
fluctuation rate is not Arrhenius-limited. Then Δt/τ_corr = r0·e^{−δΔ}
with no cosh(I) factor.

ATTEMPT-FREQUENCY RATE CEILING (`rate_ceiling_r0`, audit r1 item 4).
The Arrhenius form is a large-barrier asymptotic: an escape rate cannot
exceed the attempt rate 1/τ0 (τ0 ≈ 0.1–1 ns for an sMTJ), so the honest
total rate is

    k_tot(I) = min(2 f cosh I, 1/τ0),      f = f0 e^{−δΔ}.

Divide by the nominal zero-bias total rate 2 f0 and define the
dimensionless ceiling

    C ≡ 1/(2 f0 τ0) = τ_corr(I=0, nominal) / τ0        (`rate_ceiling_r0`)

so that k_tot/(2 f0) = min(e^{−δΔ} cosh I, C) and, with
ratio ∈ {r0, settle_ratio} standing for Δt·2f0,

    Δt/τ_capped = ratio · min(e^{−δΔ} · cosh I, C)
                = min(ratio · rf · cosh I,  ratio · C).

The cap binds for |I| ≥ arccosh(C/rf); above that the device stops
self-decorrelating with bias, so the per-interval ACF has a FLOOR
exp(−ratio·C) that no amount of input drive can beat. That is why this is
the worst corner for this project: the bias-driven speed-up is exactly the
mechanism that made Arrhenius devices the weak-effect class (PROJECT_STATE
lesson 3), and the ceiling switches it off. The `arrhcap` arm uses
C = τ_corr(0)/τ0 = 2 — total rate saturating at twice the nominal
zero-bias rate, |I| effectively capped at arccosh 2 = 1.317, device ACF
floor exp(−0.5·2) = 0.368 at r0 = 0.5 (the physics-audit worst corner,
whose measured marginal-stratum VOTE ACF is ≈ 0.30). `None` (default)
disables the ceiling and reproduces the paper-1 model bit-for-bit.

Device variation: δΔ ~ N(μ, σ_Δ²) per (layer, neuron), fixed per chip.
centering='median' → μ=0 (median τ preserved; E[τ] grows as e^{σ²/2});
centering='mean-tau' → μ=−σ_Δ²/2 (E[τ] preserved; isolates dispersion
from mean slowdown).

Reset ("settle") mode physics: an idle device equilibrates to p=1/2, NOT
to the upcoming input's stationary law. Applying input I then relaxes the
state toward p_eq(I) with time constant τ_corr(I). We simulate this
explicitly: the first query of each input applies the propagator over the
settle interval t_settle = settle_ratio · τ_corr(0-nominal) starting from
the device's prior state (idle equilibrium for a fresh chip, or the
previous input's final state). settle_ratio=∞ recovers ideal per-input
equilibration; throughput accounting must charge t_settle per input.
"""

import numpy as np
import torch


class TelegraphSource:
    """Stateful sMTJ randomness source.

    Modes:
      streaming=True: state persists across inputs; every query interval is Δt.
      streaming=False (settle mode): on each new input (signaled by
        new_input()), the first query's interval is t_settle instead of Δt;
        state persists through the settle (physical relaxation), so what the
        settle erases depends on settle_ratio, honestly.
    """

    def __init__(self, r0: float, sigma_delta: float = 0.0, chip_seed: int = 0,
                 noise_seed: int = 0, streaming: bool = False,
                 settle_ratio: float = 4.0, centering: str = "median",
                 bias_independent_tau: bool = False,
                 rate_ceiling_r0: float = None):
        if not (r0 > 0):
            raise ValueError("r0 must be positive (np.inf = ideal)")
        if centering not in ("median", "mean-tau"):
            raise ValueError(centering)
        if rate_ceiling_r0 is not None and not (rate_ceiling_r0 > 0):
            raise ValueError("rate_ceiling_r0 must be positive (None = off)")
        self.rate_ceiling_r0 = (None if rate_ceiling_r0 is None
                                else float(rate_ceiling_r0))
        self.r0 = float(r0)
        self.sigma_delta = float(sigma_delta)
        self.chip_seed = chip_seed
        self.streaming = streaming
        self.settle_ratio = float(settle_ratio)
        self.centering = centering
        self.bias_independent_tau = bias_independent_tau
        self.rng = np.random.default_rng(noise_seed)
        self.state: dict[int, np.ndarray] = {}
        self.rate_factor: dict[int, np.ndarray] = {}
        self._fresh: dict[int, bool] = {}

    # -- protocol hooks -----------------------------------------------------
    def new_input(self):
        """Called once per input presentation (before its T passes)."""
        for k in self._fresh:
            self._fresh[k] = True

    def reset_batch(self):  # backwards-compatible alias
        self.new_input()

    # -- internals ----------------------------------------------------------
    def _rate_factor(self, n: int, layer: int) -> np.ndarray:
        rf = self.rate_factor.get(layer)
        if rf is None:
            crng = np.random.default_rng(self.chip_seed + 104729 * layer)
            mu = -0.5 * self.sigma_delta ** 2 if self.centering == "mean-tau" \
                else 0.0
            delta = crng.normal(mu, self.sigma_delta, size=n)
            rf = np.exp(-delta)
            self.rate_factor[layer] = rf
        if rf.shape[0] != n:
            raise ValueError(f"layer {layer}: device count changed "
                             f"{rf.shape[0]} -> {n}")
        return rf

    def _dt_over_tau(self, i_pre: np.ndarray, ratio: float, layer: int,
                     n: int) -> np.ndarray:
        """Δt/τ for this interval; see the module docstring for the ceiling.

        Uncapped:  ratio · rf · cosh I   (Arrhenius) or ratio · rf (easy-plane).
        Capped:    ratio · min(rf · cosh I, C) with C = rate_ceiling_r0, the
        attempt-frequency ceiling 1/τ0 expressed in units of the nominal
        zero-bias total rate 2 f0.
        """
        rf = self._rate_factor(n, layer)[None, :]
        if self.bias_independent_tau:
            eff = np.broadcast_to(rf, i_pre.shape)
        else:
            eff = rf * np.cosh(i_pre)
        if self.rate_ceiling_r0 is not None:
            eff = np.minimum(eff, self.rate_ceiling_r0)
        return ratio * eff

    # -- sampling -----------------------------------------------------------
    def sample(self, i: torch.Tensor, layer: int) -> torch.Tensor:
        """i: pre-activations [B, N] (float). Returns ±1 samples."""
        i_np = i.detach().cpu().numpy().astype(np.float64)
        b, n = i_np.shape
        p_np = 0.5 * (1.0 + np.tanh(i_np))
        s = self.state.get(layer)
        if s is not None and s.shape != (b, n):
            raise ValueError(f"layer {layer}: batch shape changed "
                             f"{s.shape} -> {(b, n)}; streams must use a "
                             f"constant batch size")
        if s is None:
            # brand-new chip: idle equilibrium (I = 0), then treated below
            # as a fresh input with a settle (or first stream step)
            s = np.where(self.rng.random((b, n)) < 0.5, 1.0, -1.0)
            self._fresh[layer] = True
        if np.isinf(self.r0) and (self.streaming or
                                  np.isinf(self.settle_ratio)):
            p_next = p_np
        else:
            first = self._fresh.get(layer, True)
            if self.streaming or not first:
                ratio = self.r0
            else:
                ratio = self.settle_ratio  # settle interval before 1st read
            if np.isinf(ratio):
                p_next = p_np
            else:
                lam = np.exp(-self._dt_over_tau(i_np, ratio, layer, n))
                p_next = p_np + ((s > 0).astype(np.float64) - p_np) * lam
        s = np.where(self.rng.random((b, n)) < p_next, 1.0, -1.0)
        self.state[layer] = s
        self._fresh[layer] = False
        return torch.from_numpy(s).to(dtype=i.dtype, device=i.device)


class IdealSource:
    """i.i.d. uniform comparison against p — the software baseline."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def new_input(self):
        pass

    def reset_batch(self):
        pass

    def sample(self, i: torch.Tensor, layer: int) -> torch.Tensor:
        i_np = i.detach().cpu().numpy().astype(np.float64)
        p = 0.5 * (1.0 + np.tanh(i_np))
        u = self.rng.random(i_np.shape)
        return torch.from_numpy(np.where(u < p, 1.0, -1.0)).to(
            dtype=i.dtype, device=i.device)
