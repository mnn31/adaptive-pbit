# Author: Manan Gupta <mnn@yogins.com>
"""Dataset loading with a train/val/test split.

The last 5k images of the official training set form a validation split
used for all model-selection decisions; the 10k test set is touched only
for final reported numbers.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
VAL_SIZE = 5000

_LOADERS = {
    "mnist": datasets.MNIST,
    "fashion": datasets.FashionMNIST,
}


def _flat(name, train):
    ds = _LOADERS[name](DATA_DIR, train=train, download=True,
                        transform=transforms.ToTensor())
    xs = torch.stack([x for x, _ in ds]).reshape(len(ds), -1)
    ys = torch.tensor([y for _, y in ds])
    return xs, ys


def splits(name: str = "mnist"):
    """Returns (train_x, train_y, val_x, val_y, test_x, test_y)."""
    xs, ys = _flat(name, True)
    tx, ty = _flat(name, False)
    return (xs[:-VAL_SIZE], ys[:-VAL_SIZE],
            xs[-VAL_SIZE:], ys[-VAL_SIZE:], tx, ty)


def train_loader(name: str = "mnist", batch_size: int = 256):
    trx, try_, *_ = splits(name)
    return DataLoader(TensorDataset(trx, try_), batch_size=batch_size,
                      shuffle=True, drop_last=True)
