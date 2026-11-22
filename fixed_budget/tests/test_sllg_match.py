# Author: Manan Gupta <mnn@yogins.com>
"""Barrier matching (Brown's Fokker-Planck) and the telegraph rate variants."""

import numpy as np
import torch

from smtjnn.sllg import (
    Macrospin,
    brown_time,
    fp_read_acf,
    fp_tau_1e,
    match_barrier,
    matched_device,
)
from smtjnn.smtj import TelegraphSource


# with no anisotropy, <mz(0) mz(t)> is exp(-t / tau_N), Brown's free-diffusion result
def test_fp_isotropic_mz_is_brown_exponential():
    dev = Macrospin(0.0, 70.0, 0.1)
    t = np.array([0.3, 1.0, 2.5]) * brown_time(dev)
    c = fp_read_acf(dev, t, observable="mz")
    assert np.allclose(c, np.exp(-t / brown_time(dev)), rtol=1e-3)


# the 1/e read time barely moves when the mz grid is refined from 2000 to 4000 cells
def test_fp_converges_in_cells():
    dev = Macrospin(3.9, 71.0, 0.1)
    a = fp_tau_1e(dev)
    t = np.linspace(0, 3 * a, 3001)
    c = fp_read_acf(dev, t, cells=4000)
    b = np.interp(np.exp(-1), c[::-1], t[::-1])
    assert abs(a / b - 1) < 1e-3


# a higher barrier gives a longer read correlation time
def test_fp_tau_grows_with_barrier():
    taus = [fp_tau_1e(matched_device(d, 56.7)) for d in (3.0, 3.5, 4.0)]
    assert taus[0] < taus[1] < taus[2]


# match_barrier recovers the barrier that produced a given 1/e time
def test_match_barrier_round_trip():
    target = fp_tau_1e(matched_device(3.7, 56.7))
    dev = match_barrier(target, 56.7)
    assert abs(dev.delta - 3.7) < 1e-3
    assert abs(fp_tau_1e(dev) / target - 1) < 1e-4


# a cap of 1 gives the bias-independent rate model, read for read
def test_rate_cap_one_equals_bias_independent_rate():
    i = torch.tensor(np.random.default_rng(0).normal(0, 3, (64, 32)),
                     dtype=torch.float32)
    a = TelegraphSource(0.3, streaming=True, noise_seed=5, rate_cap=1.0)
    b = TelegraphSource(0.3, streaming=True, noise_seed=5,
                        bias_independent_tau=True)
    for _ in range(5):
        assert torch.equal(a.sample(i, 0), b.sample(i, 0))


# a cap far above cosh(3) changes nothing
def test_large_rate_cap_equals_cosh_law():
    i = torch.tensor(np.random.default_rng(1).normal(0, 1, (64, 32)),
                     dtype=torch.float32).clamp(-3, 3)
    a = TelegraphSource(0.3, streaming=True, noise_seed=5, rate_cap=1e3)
    b = TelegraphSource(0.3, streaming=True, noise_seed=5)
    for _ in range(5):
        assert torch.equal(a.sample(i, 0), b.sample(i, 0))


# a rate law tabulated from cosh itself reproduces the cosh law, read for read
def test_rate_law_on_cosh_table_equals_cosh_law():
    i = torch.tensor(np.random.default_rng(2).normal(0, 2, (64, 32)),
                     dtype=torch.float32).clamp(-5, 5)
    grid = np.linspace(0, 5, 20001)
    a = TelegraphSource(0.3, streaming=True, noise_seed=5,
                        rate_law=(grid, np.cosh(grid)))
    b = TelegraphSource(0.3, streaming=True, noise_seed=5)
    same = [torch.equal(a.sample(i, 0), b.sample(i, 0)) for _ in range(5)]
    assert all(same)


# the tabulated law interpolates in log g and is held flat past the last point
def test_rate_law_is_held_beyond_the_grid():
    src = TelegraphSource(0.3, streaming=True, rate_law=([0, 2], [1, 3]))
    x = src._dt_over_tau(np.array([[0.0, 1.0, -2.0, 9.0]]), 0.3, 0, 4)
    assert np.allclose(x, 0.3 * np.array([[1, np.sqrt(3), 3, 3]]))
