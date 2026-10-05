# Author: Manan Gupta <mnn@yogins.com>
"""Stochastic binary MLP (p-bit style: stochastic at inference).

Each hidden unit fires s = +1 with probability p = (1 + tanh(I))/2 at every
query. Training uses a straight-through estimator (gradient of the mean
activation tanh(I)); the Bernoulli draw is delegated to a pluggable
`source.sample(i, layer)` receiving the raw pre-activations, so physical
randomness models (telegraph sMTJ) can compute their own dynamics from I.
"""

import torch
from torch import nn


class _StochasticSignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i, u):
        ctx.save_for_backward(i)
        p = 0.5 * (1.0 + torch.tanh(i))
        return torch.where(u < p, 1.0, -1.0)

    @staticmethod
    def backward(ctx, grad_out):
        (i,) = ctx.saved_tensors
        return grad_out * (1.0 - torch.tanh(i) ** 2), None


def stochastic_sign(i: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return _StochasticSignSTE.apply(i, u)


class SBNN(nn.Module):
    """MLP with stochastic bipolar hidden activations, real-valued readout."""

    def __init__(self, dims=(784, 256, 128, 10)):
        super().__init__()
        self.dims = tuple(dims)
        self.layers = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])
        )

    def forward(self, x: torch.Tensor, source=None) -> torch.Tensor:
        """One stochastic pass. `source=None`: ideal torch randomness
        (training default). Otherwise `source.sample(i, layer)` supplies
        the draws; in training mode a straight-through surrogate keeps
        gradients flowing through tanh(I)."""
        h = x
        for li, layer in enumerate(self.layers[:-1]):
            i = layer(h)
            if source is None:
                u = torch.rand_like(i)
                h = stochastic_sign(i, u)
            else:
                s = source.sample(i, layer=li)
                if self.training:
                    m = torch.tanh(i)
                    h = s.detach() + m - m.detach()
                else:
                    h = s
        return self.layers[-1](h)
