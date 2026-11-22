# Author: Manan Gupta <mnn@yogins.com>
"""Stochastic Landau-Lifshitz-Gilbert macrospin neurons.

Each neuron is one single-domain magnet with unit moment direction m and
reduced energy

    E / kT = -delta * mz^2 - moment * Bz * mz

(delta > 0: perpendicular easy axis with barrier delta kT; delta < 0: easy
plane). `moment` is Ms V / kT in 1/T, so moment * Bz is the Zeeman energy
in units of kT. The dynamics are the Landau-Lifshitz form of the Gilbert
equation with Brown's thermal field,

    dm/dt = -gamma / (1 + alpha^2) [m x B + alpha m x (m x B)],
    B = (2 delta / moment) mz z + Bz z + B_th,
    <B_th,i(t) B_th,j(t')> = 2 alpha / (gamma moment) delta_ij delta(t - t'),

integrated with the stochastic Heun scheme (Stratonovich; the same thermal
field in predictor and corrector) and |m| renormalized after every step,
as in Garcia-Palacios and Lazaro, PRB 58, 14937 (1998), and as mumax3 does.

The neuron is read by thresholding: s = sign(mz). Its input I from the
network is applied as a perpendicular field Bz = I * tesla_per_unit_i and
held constant between reads, exactly like the telegraph model's propagator.

On CUDA the integration runs in a Triton kernel that keeps each spin in
registers for a whole read interval and draws its thermal field from
Philox counters (seed, step, spin), so runs are reproducible. Elsewhere it
falls back to the plain PyTorch step `heun_step`, the reference version
the kernel is tested against.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs
    triton = None

GAMMA = 1.7595e11  # gyromagnetic ratio, rad/(s T); mumax3's GammaLL


@dataclass(frozen=True)
class Macrospin:
    """Device parameters: barrier delta (kT), moment Ms V / kT (1/T), alpha."""

    delta: float
    moment: float
    alpha: float = 0.1

    # Anisotropy field 2 delta kT / (Ms V), in T. Negative for an easy-plane magnet.
    @property
    def b_anis(self) -> float:
        return 2.0 * self.delta / self.moment

    # Standard deviation of each thermal-field component over one step dt (s), in T.
    def noise_sigma(self, dt: float) -> float:
        return math.sqrt(2.0 * self.alpha / (GAMMA * self.moment * dt))


# ---------------------------------------------------------------- equilibrium
_GL_X, _GL_W = np.polynomial.legendre.leggauss(400)
_GL_X = 0.5 * (_GL_X + 1.0)  # nodes on [0, 1]
_GL_W = 0.5 * _GL_W


# Exact Boltzmann P(mz > 0) at field b_z (T), by Gauss-Legendre quadrature over mz.
# Works on a scalar or an array of fields; a scalar gives a float back.
def equilibrium_p_plus(dev: Macrospin, b_z) -> np.ndarray:
    b = np.asarray(b_z, dtype=np.float64).reshape(-1)
    out = np.empty_like(b)
    for a in range(0, b.size, 4096):
        h = dev.moment * b[a:a + 4096, None]
        up = dev.delta * _GL_X ** 2 + h * _GL_X   # log density at +mz
        dn = dev.delta * _GL_X ** 2 - h * _GL_X   # log density at -mz
        top = np.maximum(up.max(1), dn.max(1))[:, None]
        zp = (np.exp(up - top) * _GL_W).sum(1)
        zm = (np.exp(dn - top) * _GL_W).sum(1)
        out[a:a + 4096] = zp / (zp + zm)
    return out.reshape(np.shape(b_z)) if np.ndim(b_z) else float(out[0])


# Exact Boltzmann <mz>, <mz^2>, <|mz|> and <sign mz> at one field b_z (T).
def equilibrium_moments(dev: Macrospin, b_z: float) -> dict:
    x = np.concatenate([-_GL_X[::-1], _GL_X])
    w = np.concatenate([_GL_W[::-1], _GL_W])
    logp = dev.delta * x ** 2 + dev.moment * b_z * x
    p = w * np.exp(logp - logp.max())
    p /= p.sum()
    return {"mz": float((p * x).sum()), "mz2": float((p * x * x).sum()),
            "abs_mz": float((p * np.abs(x)).sum()),
            "sign": float((p * np.sign(x)).sum())}


# Least-squares slope (per T) of I = atanh<sign mz> against Bz, fitted through the
# origin over |I| <= i_max. This is the field-to-input map of the device.
def logit_slope(dev: Macrospin, i_max: float = 2.0) -> float:
    # widen the field range until it reaches I = i_max, then fit on a fine grid
    b_hi = 1.0 / dev.moment
    while np.arctanh(2 * equilibrium_p_plus(dev, b_hi) - 1) < i_max:
        b_hi *= 1.5
    b = np.linspace(0.0, b_hi, 2001)[1:]
    i = np.arctanh(2 * equilibrium_p_plus(dev, b) - 1)
    keep = i <= i_max
    b, i = b[keep], i[keep]
    return float((b * i).sum() / (b * b).sum())


# Macrospin with barrier delta (kT) whose logit slope over |I| <= 2 equals `slope`
# (per T). Bisects on the moment in log space.
def matched_device(delta: float, slope: float, alpha: float = 0.1) -> Macrospin:
    lo, hi = 1.0, 1e4
    for _ in range(100):  # the slope grows monotonically with the moment
        mid = math.sqrt(lo * hi)
        if logit_slope(Macrospin(delta, mid, alpha)) < slope:
            lo = mid
        else:
            hi = mid
    return Macrospin(delta, math.sqrt(lo * hi), alpha)


# ------------------------------------------------- Brown's Fokker-Planck in mz
# Brown's free-diffusion time (1 + alpha^2) Ms V / (2 alpha gamma kT), in s.
def brown_time(dev: Macrospin) -> float:
    return (1.0 + dev.alpha ** 2) * dev.moment / (2.0 * dev.alpha * GAMMA)


# Exact stationary autocorrelation of sign(mz) (or of mz) at times t (s), from
# Brown's Fokker-Planck equation in mz. Used to fit the barrier without running spins.
def fp_read_acf(dev: Macrospin, t, b_z: float = 0.0, observable="sign",
                cells: int = 2000) -> np.ndarray:
    """Stationary autocorrelation of sign(mz) (or mz) at times t, exactly.

    With an axial field the energy depends on mz alone and mz obeys Brown's
    equation 2 tau_N dW/dt = d/dx[(1 - x^2)(dW/dx + W dU/dx)], U = E / kT.
    It is discretized as a birth-death chain on `cells` cells of mz whose
    rates obey detailed balance with the cell Boltzmann weights, so the
    symmetrized generator is tridiagonal and its eigenmodes give C(t).
    """
    from scipy.linalg import eigh_tridiagonal

    edges = np.linspace(-1.0, 1.0, cells + 1)
    x = 0.5 * (edges[1:] + edges[:-1])
    h = edges[1] - edges[0]
    u = -(dev.delta * x ** 2 + dev.moment * b_z * x)
    d = (1.0 - edges[1:-1] ** 2) / (2.0 * brown_time(dev) * h * h)
    du = u[1:] - u[:-1]
    up, down = d * np.exp(-du / 2), d * np.exp(du / 2)
    diag = -(np.r_[up, 0.0] + np.r_[0.0, down])
    lam, vec = eigh_tridiagonal(diag, d)   # sqrt(up * down) = d
    pi = np.exp(-(u - u.min()))
    pi /= pi.sum()
    f = np.sign(x) if observable == "sign" else x
    df = f - (pi * f).sum()
    w = (vec.T @ (np.sqrt(pi) * df)) ** 2
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    c = (w[None, :] * np.exp(np.outer(t, np.minimum(lam, 0.0)))).sum(1)
    return c / w.sum()


# 1/e time (s) of the read autocorrelation from Brown's equation at field b_z (T).
def fp_tau_1e(dev: Macrospin, b_z: float = 0.0) -> float:
    from scipy.optimize import brentq

    hi = brown_time(dev)
    while fp_read_acf(dev, hi, b_z)[0] > np.exp(-1):
        hi *= 2.0
    return brentq(lambda s: fp_read_acf(dev, s, b_z)[0] - np.exp(-1),
                  0.0, hi, xtol=1e-15, rtol=1e-10)


# Barrier (kT) at which the slope-matched macrospin has zero-bias 1/e read time
# tau_target (s). Returns the matched Macrospin.
def match_barrier(tau_target: float, slope: float, alpha: float = 0.1,
                  lo: float = 2.0, hi: float = 7.0) -> Macrospin:
    from scipy.optimize import brentq

    # log ratio of the model 1/e time to the target; zero at the matched barrier
    def gap(delta):
        return math.log(fp_tau_1e(matched_device(delta, slope, alpha))
                        / tau_target)

    delta = brentq(gap, lo, hi, xtol=1e-5)
    return matched_device(delta, slope, alpha)


# n unit vectors drawn from the Boltzmann law at field b_z (T), shape [3, n]. mz comes
# from inverting the tabulated CDF, the azimuth is uniform.
def sample_equilibrium(dev: Macrospin, b_z: float, n: int,
                       rng: np.random.Generator) -> np.ndarray:
    grid = np.linspace(-1.0, 1.0, 200_001)
    logp = dev.delta * grid ** 2 + dev.moment * b_z * grid
    p = np.exp(logp - logp.max())
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]))])
    # inverse-CDF draw of mz on a fine grid (trapezoid CDF)
    mz = np.interp(rng.random(n) * cdf[-1], cdf, grid)
    phi = rng.random(n) * 2.0 * np.pi
    rho = np.sqrt(np.clip(1.0 - mz ** 2, 0.0, None))
    return np.stack([rho * np.cos(phi), rho * np.sin(phi), mz])


# ---------------------------------------------------------- reference stepper
# m x B + alpha m x (m x B) for [3, n] tensors, the bracket of the LL equation
def _torque(m: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    mxb = torch.cross(m, b, dim=0)
    return mxb + alpha * torch.cross(m, mxb, dim=0)


# One stochastic Heun step of all spins. m and noise are [3, n] (noise is unit
# normal), b_z is [n] in T, dt in s. Returns the renormalized m.
def heun_step(m: torch.Tensor, b_z: torch.Tensor, dev: Macrospin, dt: float,
              noise: torch.Tensor) -> torch.Tensor:
    c = -GAMMA * dt / (1.0 + dev.alpha ** 2)
    b_th = dev.noise_sigma(dt) * noise

    # thermal field plus the applied and anisotropy fields, which point along z
    def field(mm):
        b = b_th.clone()
        b[2] += b_z + dev.b_anis * mm[2]
        return b

    # predictor, then corrector with the same thermal field (Stratonovich)
    k1 = _torque(m, field(m), dev.alpha)
    mp = m + c * k1
    k2 = _torque(mp, field(mp), dev.alpha)
    m = m + 0.5 * c * (k1 + k2)
    return m / m.norm(dim=0, keepdim=True)


# --------------------------------------------------------------- GPU kernel
if triton is not None:
    # scalar version of _torque for use inside the kernel
    @triton.jit
    def _torque_tl(mx, my, mz, bx, by, bz, alpha):
        # m x B + alpha m x (m x B), componentwise
        ax = my * bz - mz * by
        ay = mz * bx - mx * bz
        az = mx * by - my * bx
        tx = ax + alpha * (my * az - mz * ay)
        ty = ay + alpha * (mz * ax - mx * az)
        tz = az + alpha * (mx * ay - my * ax)
        return tx, ty, tz

    # Heun steps for one block of spins, kept in registers for all n_steps. Each
    # (step, spin) pair gets its own Philox counter, so a run is reproducible and
    # does not depend on how the steps are split into calls.
    @triton.jit(do_not_specialize=["n_steps", "step0", "seed"])
    def _heun_kernel(m_ptr, bz_ptr, n, n_steps, step0, seed, b_anis, sigma,
                     c, alpha, BLOCK: tl.constexpr):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = idx < n
        mx = tl.load(m_ptr + idx, mask=mask, other=0.0)
        my = tl.load(m_ptr + n + idx, mask=mask, other=0.0)
        mz = tl.load(m_ptr + 2 * n + idx, mask=mask, other=1.0)
        bz = tl.load(bz_ptr + idx, mask=mask, other=0.0)
        s0 = step0.to(tl.int64)
        for k in range(n_steps):
            # one Philox counter per (step, spin): independent, reproducible
            off = (s0 + k) * n + idx
            g1, g2, g3, _ = tl.randn4x(seed, off)
            hx = sigma * g1
            hy = sigma * g2
            hz = bz + sigma * g3
            t1x, t1y, t1z = _torque_tl(mx, my, mz, hx, hy, hz + b_anis * mz,
                                       alpha)
            px = mx + c * t1x
            py = my + c * t1y
            pz = mz + c * t1z
            t2x, t2y, t2z = _torque_tl(px, py, pz, hx, hy, hz + b_anis * pz,
                                       alpha)
            mx = mx + 0.5 * c * (t1x + t2x)
            my = my + 0.5 * c * (t1y + t2y)
            mz = mz + 0.5 * c * (t1z + t2z)
            inv = 1.0 / tl.sqrt(mx * mx + my * my + mz * mz)
            mx = mx * inv
            my = my * inv
            mz = mz * inv
        tl.store(m_ptr + idx, mx, mask=mask)
        tl.store(m_ptr + n + idx, my, mask=mask)
        tl.store(m_ptr + 2 * n + idx, mz, mask=mask)


class MacrospinArray:
    """n independent macrospins with their own thermal noise stream."""

    # n spins all starting at +z. step counts every step taken so far and feeds the
    # kernel's noise counter; the kernel is used only on CUDA in float32.
    def __init__(self, dev: Macrospin, n: int, dt: float, seed: int = 0,
                 device="cpu", dtype=torch.float32):
        self.dev = dev
        self.n = n
        self.dt = float(dt)
        self.seed = int(seed) % (2 ** 31 - 1)
        self.device = torch.device(device)
        self.step = 0  # noise counter: total steps taken so far
        self.m = torch.zeros(3, n, dtype=dtype, device=self.device)
        self.m[2] = 1.0
        self._cpu_gen = torch.Generator().manual_seed(self.seed)
        self.use_kernel = (self.device.type == "cuda" and triton is not None
                           and dtype == torch.float32)

    # Draw every spin from the Boltzmann law at one field b_z (T).
    def set_equilibrium(self, b_z: float = 0.0, rng=None):
        rng = rng or np.random.default_rng(self.seed + 7)
        m = sample_equilibrium(self.dev, b_z, self.n, rng)
        self.m = torch.as_tensor(m, dtype=self.m.dtype, device=self.device)
        return self

    # Advance all spins n_steps under a held field b_z (T, scalar or [n]). thermal=False
    # turns the noise off, which the tests use for deterministic relaxation.
    def run(self, b_z, n_steps: int, thermal: bool = True):
        if torch.is_tensor(b_z):
            b = b_z.to(self.device, self.m.dtype).expand(self.n).contiguous()
        else:
            b = torch.full((self.n,), float(b_z), dtype=self.m.dtype,
                           device=self.device)
        if n_steps <= 0:
            return self
        if self.use_kernel:
            # all n_steps in one launch; the noise counter starts at self.step
            d = self.dev
            sigma = d.noise_sigma(self.dt) if thermal else 0.0
            c = -GAMMA * self.dt / (1.0 + d.alpha ** 2)
            self.m = self.m.contiguous()
            grid = (triton.cdiv(self.n, 256),)
            with torch.cuda.device(self.device):
                _heun_kernel[grid](self.m, b, self.n, int(n_steps),
                                   self.step, self.seed, d.b_anis, sigma, c,
                                   d.alpha, BLOCK=256)
        else:
            # reference path: one PyTorch Heun step at a time, CPU generator
            for _ in range(n_steps):
                noise = torch.randn(3, self.n, generator=self._cpu_gen,
                                    dtype=self.m.dtype).to(self.device)
                if not thermal:
                    noise.zero_()
                self.m = heun_step(self.m, b, self.dev, self.dt, noise)
        self.step += int(n_steps)
        return self

    # Thresholded read: +1 where mz >= 0, else -1.
    def read(self) -> torch.Tensor:
        return torch.where(self.m[2] >= 0, 1.0, -1.0)


# ------------------------------------------------------- classifier sources
class MacrospinSource:
    """Drop-in replacement for TelegraphSource with sLLG macrospin neurons.

    Same protocol hooks and semantics: in settle mode the first read of each
    input comes after settle_ratio * tau_c0 under that input, later reads
    every r0 * tau_c0; in streaming mode every read interval is r0 * tau_c0
    and the state carries over from the previous input. tau_c0 is the
    macrospin's own measured zero-bias correlation time of sign(mz). A
    brand-new chip starts in zero-field (idle) equilibrium. Inputs are
    mapped linearly to field, Bz = I * tesla_per_unit_i, at every I: no
    clipping, so beyond the linear range the device's own saturation (and
    loss of its minority well) is what the network sees.
    """

    # r0 and settle_ratio are in units of tau_c0 (s); dt is the integration step (s).
    # One MacrospinArray is made per layer on first use.
    def __init__(self, r0: float, dev: Macrospin, tau_c0: float,
                 tesla_per_unit_i: float, dt: float = 1e-12,
                 streaming: bool = False, settle_ratio: float = 4.0,
                 noise_seed: int = 0, device="cuda"):
        if not (r0 > 0 and np.isfinite(r0)):
            raise ValueError("r0 must be positive and finite")
        self.r0 = float(r0)
        self.dev = dev
        self.tau_c0 = float(tau_c0)
        self.tesla_per_unit_i = float(tesla_per_unit_i)
        self.dt = float(dt)
        self.streaming = streaming
        self.settle_ratio = float(settle_ratio)
        self.noise_seed = noise_seed
        self.device = device
        self.read_steps = self._steps(self.r0)
        self.settle_steps = self._steps(self.settle_ratio)
        self.arrays: dict[int, MacrospinArray] = {}
        self._fresh: dict[int, bool] = {}
        self._shape: dict[int, tuple] = {}

    # number of integration steps in an interval of ratio * tau_c0
    def _steps(self, ratio: float) -> int:
        return max(1, round(ratio * self.tau_c0 / self.dt))

    # Called once per input presentation (before its T passes): the next read of
    # every layer comes after a settle.
    def new_input(self):
        for k in self._fresh:
            self._fresh[k] = True

    # old name for new_input, kept for the shared inference loop
    def reset_batch(self):
        self.new_input()

    # The spins for one layer, created on first use in zero-field equilibrium with a
    # seed derived from (noise_seed, layer). The batch shape may not change afterwards.
    def _array(self, layer: int, shape: tuple) -> MacrospinArray:
        arr = self.arrays.get(layer)
        if arr is None:
            ss = np.random.SeedSequence([self.noise_seed, layer])
            seed = int(ss.generate_state(1)[0] % (2 ** 31 - 1))
            arr = MacrospinArray(self.dev, shape[0] * shape[1], self.dt,
                                 seed=seed, device=self.device)
            arr.set_equilibrium(0.0, np.random.default_rng(ss))
            self.arrays[layer] = arr
            self._shape[layer] = shape
            self._fresh[layer] = True
        elif self._shape[layer] != shape:
            raise ValueError(f"layer {layer}: batch shape changed "
                             f"{self._shape[layer]} -> {shape}")
        return arr

    # i: pre-activations [B, N]. Runs the spins for one settle or one read interval
    # under Bz = I * tesla_per_unit_i and returns the +-1 reads, shape [B, N].
    def sample(self, i: torch.Tensor, layer: int) -> torch.Tensor:
        arr = self._array(layer, tuple(i.shape))
        # first read of a new input in settle mode waits settle_ratio * tau_c0
        settle = not self.streaming and self._fresh.get(layer, True)
        n_steps = self.settle_steps if settle else self.read_steps
        b_z = i.detach().reshape(-1).to(self.device, torch.float32)
        arr.run(b_z * self.tesla_per_unit_i, n_steps)
        self._fresh[layer] = False
        s = arr.read().reshape(i.shape)
        return s.to(dtype=i.dtype, device=i.device)


class EquilibriumSource:
    """Independent reads from the macrospin's exact Boltzmann P(+) at Bz(I).

    The infinitely-slow-read limit: the device's static activation shape
    with no time correlation at all.
    """

    # tabulate the device P(+) once on a fine grid of network inputs
    def __init__(self, dev: Macrospin, tesla_per_unit_i: float,
                 seed: int = 0):
        self.dev = dev
        self.tesla_per_unit_i = float(tesla_per_unit_i)
        self.rng = np.random.default_rng(seed)
        # P(+) on a fine input grid; beyond +-60 it is 0 or 1 to double precision
        self._i = np.linspace(-60.0, 60.0, 240_001)
        self._p = equilibrium_p_plus(dev, self._i * self.tesla_per_unit_i)

    # no state, nothing to reset
    def new_input(self):
        pass

    # no state, nothing to reset
    def reset_batch(self):
        pass

    # Device P(+) at network input I, interpolated from the table.
    def p_plus(self, i_np: np.ndarray) -> np.ndarray:
        return np.interp(i_np, self._i, self._p)

    # independent +-1 reads with the device's equilibrium P(+) at each input
    def sample(self, i: torch.Tensor, layer: int) -> torch.Tensor:
        i_np = i.detach().cpu().numpy().astype(np.float64)
        u = self.rng.random(i_np.shape)
        s = np.where(u < self.p_plus(i_np), 1.0, -1.0)
        return torch.from_numpy(s).to(dtype=i.dtype, device=i.device)
