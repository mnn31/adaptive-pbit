# Author: Manan Gupta <mnn@yogins.com>
"""Physics checks for the sLLG macrospin neuron (smtjnn.sllg)."""

import numpy as np
import pytest
import torch

from smtjnn import sllg
from smtjnn.sllg import (
    GAMMA,
    EquilibriumSource,
    Macrospin,
    MacrospinArray,
    MacrospinSource,
    matched_device,
)

DEV = Macrospin(delta=3.5, moment=72.28, alpha=0.1)
FAST = Macrospin(delta=3.5, moment=72.28, alpha=1.0)  # same statics, quicker
HAS_KERNEL = torch.cuda.is_available() and sllg.triton is not None


# n spins tilted by theta from +z in the x-z plane
def _tilted(theta, n=1):
    m = torch.zeros(3, n, dtype=torch.float64)
    m[0], m[2] = np.sin(theta), np.cos(theta)
    return m


# ---------- zero temperature ----------

# at zero temperature the precession frequency and small-angle damping match the LLG
# values
def test_precession_frequency_and_damping():
    dev = Macrospin(3.5, 72.28, alpha=0.01)
    theta, b_z, dt = 0.05, 0.05, 1e-13
    arr = MacrospinArray(dev, 1, dt, dtype=torch.float64)
    arr.m = _tilted(theta)
    steps = 7000
    mx = np.empty(steps)
    mz = np.empty(steps)
    for k in range(steps):
        arr.run(b_z, 1, thermal=False)
        mx[k], mz[k] = arr.m[0, 0].item(), arr.m[2, 0].item()
    # precession about z in the field b_anis mz + Bz
    w = GAMMA / (1 + dev.alpha ** 2) * (dev.b_anis * np.cos(theta) + b_z)
    up = np.nonzero((mx[:-1] < 0) & (mx[1:] >= 0))[0]
    frac = -mx[up] / (mx[up + 1] - mx[up])
    t_up = (up + 1 + frac) * dt
    period = np.diff(t_up).mean()
    assert abs(2 * np.pi / period / w - 1) < 2e-3
    # small-angle damping: theta decays as exp(-alpha w0 t)
    w0 = GAMMA / (1 + dev.alpha ** 2) * (dev.b_anis + b_z)
    t_end = steps * dt
    th_end = np.arccos(mz[-1])
    assert abs(th_end / (theta * np.exp(-dev.alpha * w0 * t_end)) - 1) < 0.01


# the Triton kernel follows the float64 reference stepper without noise
@pytest.mark.skipif(not HAS_KERNEL, reason="needs CUDA and Triton")
def test_kernel_matches_reference_at_zero_temperature():
    rng = np.random.default_rng(0)
    m0 = sllg.sample_equilibrium(DEV, 0.0, 512, rng)
    b = torch.linspace(-0.3, 0.3, 512)
    gpu = MacrospinArray(DEV, 512, 1e-12, device="cuda")
    gpu.m = torch.tensor(m0, dtype=torch.float32, device="cuda")
    ref = MacrospinArray(DEV, 512, 1e-12, dtype=torch.float64)
    ref.m = torch.tensor(m0)
    gpu.run(b.cuda(), 2000, thermal=False)
    ref.run(b.double(), 2000, thermal=False)
    assert (gpu.m.cpu().double() - ref.m).abs().max() < 1e-3


# ---------- equilibrium ----------

# matched_device hits the requested logit slope, and P(+) = 1/2 at zero field
def test_matched_device_has_requested_slope():
    dev = matched_device(3.5, 56.0)
    assert abs(sllg.logit_slope(dev) / 56.0 - 1) < 1e-6
    assert abs(sllg.equilibrium_p_plus(dev, 0.0) - 0.5) < 1e-12


# sampled directions have the exact Boltzmann <mz^2> and <sign mz> and unit length
def test_equilibrium_sampler_matches_boltzmann():
    rng = np.random.default_rng(1)
    for b in [0.0, 0.02]:
        m = sllg.sample_equilibrium(DEV, b, 400_000, rng)
        ex = sllg.equilibrium_moments(DEV, b)
        assert abs((m[2] ** 2).mean() - ex["mz2"]) < 3e-3
        assert abs(np.sign(m[2]).mean() - ex["sign"]) < 6e-3
        assert np.allclose((m ** 2).sum(0), 1.0)


# the CPU dynamics relax from +z to the Boltzmann P(+) and <mz^2>
@pytest.mark.parametrize("b_z", [0.0, 1.0 / 56.0])
def test_dynamics_relax_to_boltzmann(b_z):
    # start every spin at +z, let the thermal dynamics equilibrate, then
    # compare P(+) and <mz^2> with the exact Boltzmann values
    n, dt = 2048, 1e-12
    arr = MacrospinArray(FAST, n, dt, seed=3, dtype=torch.float64)
    arr.run(b_z, 8000)
    p, mz2 = [], []
    for _ in range(40):
        arr.run(b_z, 100)
        p.append((arr.m[2] > 0).double().mean().item())
        mz2.append((arr.m[2] ** 2).mean().item())
    ex = sllg.equilibrium_moments(FAST, b_z)
    assert abs(np.mean(p) - 0.5 * (1 + ex["sign"])) < 0.015
    assert abs(np.mean(mz2) - ex["mz2"]) < 0.01


# the GPU kernel keeps spins drawn from equilibrium in equilibrium
@pytest.mark.skipif(not HAS_KERNEL, reason="needs CUDA and Triton")
@pytest.mark.parametrize("b_z", [0.0, 1.0 / 56.0, 3.0 / 56.0])
def test_kernel_holds_boltzmann(b_z):
    # the production kernel at the production damping and step: spins drawn
    # from exact equilibrium must stay there, with and without a field
    n = 65536
    arr = MacrospinArray(DEV, n, 1e-12, seed=7, device="cuda")
    arr.set_equilibrium(b_z, np.random.default_rng(7))
    b = torch.full((n,), b_z, device="cuda")
    p, mz2 = [], []
    for _ in range(40):
        arr.run(b, 2000)
        p.append((arr.m[2] > 0).double().mean().item())
        mz2.append((arr.m[2] ** 2).double().mean().item())
    ex = sllg.equilibrium_moments(DEV, b_z)
    # the tolerance scales with the minority population, so the check stays
    # sharp at 3/56 T where P(-) is only 0.3%
    p_minus = 0.5 * (1 - ex["sign"])
    assert abs(np.mean(p) - (1 - p_minus)) < min(0.004, 0.15 * p_minus)
    assert abs(np.mean(mz2) - ex["mz2"]) < 0.002


# the read correlation across 1 ns agrees at dt = 0.5 and 2 ps
def test_read_correlation_converges_in_dt():
    # correlation of sign(mz) across 1 ns from exact equilibrium, two steps
    lag = 1e-9
    corr = {}
    for dt in [0.5e-12, 2e-12]:
        arr = MacrospinArray(FAST, 4096, dt, seed=5, dtype=torch.float64)
        arr.set_equilibrium(0.0, np.random.default_rng(5))
        c = []
        for _ in range(6):
            s0 = arr.read()
            arr.run(0.0, round(lag / dt))
            c.append((s0 * arr.read()).mean().item())
        corr[dt] = np.mean(c)
    assert 0.3 < corr[2e-12] < 0.9
    assert abs(corr[0.5e-12] - corr[2e-12]) < 0.03


# ---------- classifier sources ----------

# MacrospinSource gives +-1 reads, settles to P(+) at I = 1, and keeps state between
# fast reads
def test_source_reads_and_settle():
    src = MacrospinSource(r0=0.05, dev=FAST, tau_c0=2.5e-9,
                          tesla_per_unit_i=1 / 56.0, dt=2e-12,
                          streaming=False, settle_ratio=4.0, noise_seed=0,
                          device="cpu")
    i = torch.full((1, 3000), 1.0)
    src.new_input()
    s = src.sample(i, layer=0)
    assert s.shape == i.shape and set(s.unique().tolist()) <= {-1.0, 1.0}
    # four tau_c of settle from idle equilibrium: close to P(+) at I = 1
    p_eq = sllg.equilibrium_p_plus(FAST, 1 / 56.0)
    assert abs((s > 0).double().mean().item() - p_eq) < 0.03
    # a fast read right after keeps most of the state
    s2 = src.sample(i, layer=0)
    assert (s * s2).mean().item() > 0.6
    with pytest.raises(ValueError):
        src.sample(torch.zeros(2, 3), layer=0)


# EquilibriumSource has the device's P(+), symmetric about I = 0 and saturating at 1
def test_equilibrium_source_shape():
    src = EquilibriumSource(DEV, 1 / 56.0, seed=0)
    p = src.p_plus(np.array([-1.0, 0.0, 1.0, 100.0]))
    assert abs(p[1] - 0.5) < 1e-9 and abs(p[0] + p[2] - 1) < 1e-9
    assert abs(p[2] - sllg.equilibrium_p_plus(DEV, 1 / 56.0)) < 1e-6
    assert p[3] == pytest.approx(1.0)
    s = src.sample(torch.zeros(4, 5000), layer=0)
    assert abs(s.mean().item()) < 0.03
