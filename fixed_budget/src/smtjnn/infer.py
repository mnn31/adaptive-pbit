# Author: Manan Gupta <mnn@yogins.com>
"""Inference protocols.

Lane-stream evaluation: the test set is split into B independent lanes,
each a true sequential stream of L = N/B images processed one at a time.
Device state persists along a lane exactly as it would on B parallel
chips. For streaming sources, `warmup` unscored images (drawn from the
training set) precede each lane so that scored images are in steady
state — this removes the warm-start artifact where early images see a
freshly equilibrated device. Batch shape is constant (B) at every step,
so no mid-stream state re-initialization can occur.

For settle-mode (streaming=False) sources, `source.new_input()` is called
before each image, triggering the physical settle interval.
"""

import torch


@torch.no_grad()
def evaluate_lanes(model, xs, ys, source, T: int = 16, lanes: int = 100,
                   warmup_x=None, warmup_per_lane: int = 0):
    """xs: [N, D] test images, ys: [N] labels. Returns accuracy over all N.

    Images are laid out lane-major: lane b processes xs[b], xs[B+b],
    xs[2B+b], ... sequentially. warmup_x: [M, D] pool of unscored images
    (training set); warmup_per_lane of them are streamed through each lane
    first.
    """
    model.eval()
    n = xs.shape[0]
    if n % lanes != 0:
        raise ValueError(f"{n} images not divisible into {lanes} lanes")
    steps = n // lanes

    if warmup_per_lane > 0:
        if warmup_x is None or warmup_x.shape[0] < warmup_per_lane * lanes:
            raise ValueError("insufficient warmup pool")
        for t in range(warmup_per_lane):
            batch = warmup_x[t * lanes:(t + 1) * lanes]
            source.new_input()
            for _ in range(T):
                model(batch, source=source)

    correct = 0
    for t in range(steps):
        batch = xs[t * lanes:(t + 1) * lanes]
        labels = ys[t * lanes:(t + 1) * lanes]
        source.new_input()
        logits = torch.zeros(lanes, model.dims[-1])
        for _ in range(T):
            logits += model(batch, source=source)
        correct += (logits.argmax(1) == labels).sum().item()
    return correct / n


@torch.no_grad()
def evaluate_ideal(model, xs, ys, source, T: int = 16, batch: int = 1000):
    """Plain batched evaluation for stateless (ideal) sources."""
    model.eval()
    correct = 0
    for a in range(0, xs.shape[0], batch):
        x, y = xs[a:a + batch], ys[a:a + batch]
        logits = torch.zeros(x.shape[0], model.dims[-1])
        for _ in range(T):
            logits += model(x, source=source)
        correct += (logits.argmax(1) == y).sum().item()
    return correct / xs.shape[0]
