"""MNIST loading for the Lipschitz-diagnostics experiment.

Pixel values are kept in [0, 1] (ToTensor() only, no ImageNet-style
Normalize) so that raw pixel differences remain directly interpretable --
the covariance/Mahalanobis work in distance.py operates on these same
values, and a mean/std normalization would silently rescale distances in
ways that would need to be undone there.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

torch.set_default_dtype(torch.float64)

DATA_ROOT = Path(__file__).resolve().parent / "data"


@dataclass
class MNISTData:
    x_flat: torch.Tensor    # (N, 784) float64 in [0, 1]
    x_image: torch.Tensor   # (N, 1, 28, 28) float64 in [0, 1]
    y: torch.Tensor         # (N,) int64 in {0, ..., 9}

    def __len__(self):
        return self.x_flat.shape[0]


def load_mnist(root=DATA_ROOT, train=True):
    """Load MNIST via torchvision, values in [0, 1], both flattened (784,)
    and image (1, 28, 28) views of the same underlying pixels."""
    transform = transforms.ToTensor()  # scales uint8 [0,255] -> float32 [0,1], no further normalization
    dataset = datasets.MNIST(root=str(root), train=train, download=True, transform=transform)

    # Materialize the whole (small, 60k/10k image) dataset as tensors up front --
    # the Lipschitz estimators operate on plain tensors of points, not on a
    # DataLoader, so there's no benefit to lazy loading here.
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    x_image, y = next(iter(loader))
    x_image = x_image.to(torch.get_default_dtype())
    y = y.to(torch.int64)
    x_flat = x_image.reshape(x_image.shape[0], -1)

    return MNISTData(x_flat=x_flat, x_image=x_image, y=y)


def get_dev_subset(data, n, seed):
    """A small, fixed, seeded subset of `data` for fast iteration during
    development. Same seed always returns the same subset."""
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(data), generator=generator)[:n]
    return MNISTData(x_flat=data.x_flat[idx], x_image=data.x_image[idx], y=data.y[idx])


def make_loader(x, y, batch_size=128, shuffle=True, seed=None):
    """Wrap (x, y) tensors in a DataLoader for train_classifier."""
    dataset = TensorDataset(x, y)
    if shuffle and seed is not None:
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
