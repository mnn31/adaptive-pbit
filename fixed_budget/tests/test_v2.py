# Author: Manan Gupta <mnn@yogins.com>
import numpy as np
import pytest
import torch

from smtjnn.model import SBNN, stochastic_sign
from smtjnn.smtj import IdealSource, TelegraphSource


def I_of(p):
    return float(np.arctanh(2 * p - 1))


def _chain(src, i_scalar, n_dev, n_steps, streaming_new_input=False):
    i = torch.full((1, n_dev), i_scalar, dtype=torch.float64)
    out = np.empty((n_steps, n_dev))
    for t in range(n_steps):
        out[t] = src.sample(i, layer=0).numpy()[0]
    return out


# ---------- stationary law ----------

def test_stationary_marginal_all_variants():
    p_target = 0.73
    for kw in [dict(r0=np.inf), dict(r0=1.0), dict(r0=0.2),
               dict(r0=0.2, bias_independent_tau=True)]:
        src = TelegraphSource(streaming=True, noise_seed=1, **kw)
        traj = _chain(src, I_of(p_target), 400, 4000)
        # discard burn-in
        frac = (traj[200:] > 0).mean()
        assert abs(frac - p_target) < 0.01, (kw, frac)


# ---------- autocorrelation ----------

def _acf(traj, k):
    x = traj - traj.mean()
    return (x[:-k] * x[k:]).mean() / x.var()


def test_acf_at_zero_bias():
    r0 = 0.5
    src = TelegraphSource(r0=r0, streaming=True, noise_seed=2)
    traj = _chain(src, 0.0, 2000, 3000)
    for k in [1, 2, 4]:
        assert abs(_acf(traj[100:], k) - np.exp(-k * r0)) < 0.02


def test_acf_at_nonzero_bias_has_cosh_speedup():
    r0, i = 0.4, 0.8
    src = TelegraphSource(r0=r0, streaming=True, noise_seed=3)
    traj = _chain(src, i, 3000, 3000)
    expect = np.exp(-r0 * np.cosh(i))
    assert abs(_acf(traj[100:], 1) - expect) < 0.02


def test_bias_independent_variant_acf_ignores_input():
    r0, i = 0.4, 0.8
    src = TelegraphSource(r0=r0, streaming=True, noise_seed=4,
                          bias_independent_tau=True)
    traj = _chain(src, i, 3000, 3000)
    assert abs(_acf(traj[100:], 1) - np.exp(-r0)) < 0.02


# ---------- settle physics ----------

def test_first_read_after_settle_matches_theory():
    # fresh chip: idle equilibrium (p=1/2), settle rho under input I,
    # marginal of first read: p + (1/2 - p) * exp(-rho * cosh I)
    rho, i = 2.0, 0.9
    p = 0.5 * (1 + np.tanh(i))
    n = 200_000
    src = TelegraphSource(r0=0.02, streaming=False, settle_ratio=rho,
                          noise_seed=5)
    s = src.sample(torch.full((1, n), i, dtype=torch.float64), layer=0)
    frac = (s.numpy() > 0).mean()
    expect = p + (0.5 - p) * np.exp(-rho * np.cosh(i))
    assert abs(frac - expect) < 0.005, (frac, expect)


def test_infinite_settle_recovers_equilibrium_first_read():
    i = 0.9
    p = 0.5 * (1 + np.tanh(i))
    src = TelegraphSource(r0=0.02, streaming=False, settle_ratio=np.inf,
                          noise_seed=6)
    s = src.sample(torch.full((1, 200_000), i, dtype=torch.float64), layer=0)
    assert abs((s.numpy() > 0).mean() - p) < 0.005


def test_settle_applies_only_to_first_read_of_each_input():
    # after new_input, first interval is settle_ratio, subsequent are r0
    src = TelegraphSource(r0=0.01, streaming=False, settle_ratio=np.inf,
                          noise_seed=7)
    i = torch.zeros((1, 50_000), dtype=torch.float64)
    a = src.sample(i, layer=0).numpy()
    b = src.sample(i, layer=0).numpy()  # r0=0.01: nearly frozen
    assert (a == b).mean() > 0.98
    src.new_input()
    c = src.sample(i, layer=0).numpy()  # fresh equilibrium again
    assert 0.4 < (b == c).mean() < 0.6


# ---------- variation model ----------

def test_mean_tau_centering_preserves_mean_tau():
    sd = 1.2
    a = TelegraphSource(r0=0.5, sigma_delta=sd, centering="median",
                        chip_seed=1)
    b = TelegraphSource(r0=0.5, sigma_delta=sd, centering="mean-tau",
                        chip_seed=1)
    n = 200_000
    a._rate_factor(n, 0), b._rate_factor(n, 0)
    tau_a = (1.0 / a.rate_factor[0]).mean()   # tau factor = e^{+delta}
    tau_b = (1.0 / b.rate_factor[0]).mean()
    assert tau_a > 1.5                         # median centering inflates E[tau]
    assert abs(tau_b - 1.0) < 0.02             # mean-tau centering fixes it


def test_chip_seed_fixed_noise_seed_free():
    a = TelegraphSource(r0=0.5, sigma_delta=1.0, chip_seed=7, noise_seed=1)
    b = TelegraphSource(r0=0.5, sigma_delta=1.0, chip_seed=7, noise_seed=9)
    i = torch.zeros((2, 64), dtype=torch.float64)
    a.sample(i, 0), b.sample(i, 0)
    assert np.allclose(a.rate_factor[0], b.rate_factor[0])


# ---------- protocol safety ----------

def test_batch_shape_change_raises():
    src = TelegraphSource(r0=0.5, streaming=True, noise_seed=8)
    src.sample(torch.zeros((10, 4), dtype=torch.float64), 0)
    with pytest.raises(ValueError, match="batch shape"):
        src.sample(torch.zeros((7, 4), dtype=torch.float64), 0)


def test_streaming_state_survives_new_input():
    src = TelegraphSource(r0=0.01, streaming=True, noise_seed=9)
    i = torch.zeros((1, 50_000), dtype=torch.float64)
    a = src.sample(i, 0).numpy()
    src.new_input()
    b = src.sample(i, 0).numpy()
    assert (a == b).mean() > 0.98  # streaming: no reset between inputs


# ---------- model / STE ----------

def test_ste_gradient_is_sech2():
    i = torch.tensor([0.3, -1.2], requires_grad=True)
    u = torch.tensor([0.0, 0.99])
    stochastic_sign(i, u).sum().backward()
    assert np.allclose(i.grad.numpy(),
                       1 - np.tanh([0.3, -1.2]) ** 2, atol=1e-6)


def test_model_gradients_flow_through_physical_source():
    model = SBNN((10, 8, 4))
    model.train()
    out = model(torch.randn(3, 10), source=IdealSource(seed=0))
    out.sum().backward()
    assert all(l.weight.grad is not None and l.weight.grad.abs().sum() > 0
               for l in model.layers)


def test_ideal_source_marginal():
    src = IdealSource(seed=3)
    i = torch.full((1000, 100), I_of(0.25))
    s = src.sample(i, 0)
    assert abs((s == 1).float().mean().item() - 0.25) < 0.01
