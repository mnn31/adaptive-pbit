# Author: Manan Gupta <mnn@yogins.com>
"""Training: validation-split monitoring, test set untouched.

`source`: optional randomness source used in the forward pass
(correlation-aware training). Device state runs continuously across
minibatches; `new_input()` is signaled per minibatch.
"""

import argparse
import time
from pathlib import Path

import torch
from torch import nn

from .data import splits, train_loader
from .model import SBNN

RESULTS = Path(__file__).resolve().parents[2] / "results"


def train(dataset="mnist", dims=(784, 256, 128, 10), epochs=30, lr=1e-3,
          seed=0, device="cpu", ckpt_name=None, source=None, log=print):
    torch.manual_seed(seed)
    dl = train_loader(dataset)
    _, _, vx, vy, _, _ = splits(dataset)
    vx, vy = vx.to(device), vy.to(device)
    model = SBNN(dims).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        t0 = time.time()
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            if source is not None:
                source.new_input()
            opt.zero_grad()
            loss = loss_fn(model(x, source=source), y)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            correct = (model(vx).argmax(1) == vy).sum().item()
        log(f"epoch {ep + 1}/{epochs} loss {loss.item():.4f} "
            f"val@T=1 {correct / len(vy):.4f} ({time.time() - t0:.1f}s)")

    RESULTS.mkdir(parents=True, exist_ok=True)
    name = ckpt_name or f"sbnn_{dataset}_seed{seed}.pt"
    path = RESULTS / name
    torch.save({"dims": dims, "state_dict": model.state_dict(),
                "dataset": dataset, "seed": seed, "epochs": epochs,
                "noise_aware": source is not None}, path)
    log(f"saved {path}")
    return model, path


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = SBNN(ckpt["dims"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    train(dataset=args.dataset, epochs=args.epochs, seed=args.seed,
          device=args.device)
