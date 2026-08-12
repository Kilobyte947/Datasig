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


def stratified_subset_idx(y, n_points, seed, exclude_idx=None):
    """Index tensor for a subset of `n_points` drawn from `y` (integer
    class labels), split as evenly as possible across classes -- unlike
    get_dev_subset's plain uniform random draw, which can (and at small n
    often does) under-represent some classes by chance. Used where a query
    set needs to actually cover every digit 0-9, not just whatever a
    single random draw happens to include.

    If `exclude_idx` is given, those indices are excluded before sampling
    -- used to keep this subset disjoint from an existing query set drawn
    from the same pool (e.g. run_mnist_experiment's `query_idx`).

    `n_points` must divide evenly across the number of classes present in
    `y` (10, for MNIST); the returned subset has exactly n_points points.
    """
    generator = torch.Generator().manual_seed(seed)
    classes = torch.unique(y)
    n_per_class = n_points // len(classes)
    assert n_per_class * len(classes) == n_points, (
        f"n_points={n_points} does not divide evenly across {len(classes)} classes")

    exclude_mask = torch.zeros(y.shape[0], dtype=torch.bool)
    if exclude_idx is not None:
        exclude_mask[exclude_idx] = True

    chosen = []
    for c in classes.tolist():
        class_idx = ((y == c) & ~exclude_mask).nonzero(as_tuple=True)[0]
        assert class_idx.shape[0] >= n_per_class, (
            f"class {c} has only {class_idx.shape[0]} points available, need {n_per_class}")
        perm = class_idx[torch.randperm(class_idx.shape[0], generator=generator)]
        chosen.append(perm[:n_per_class])

    idx = torch.cat(chosen)
    return idx[torch.randperm(idx.shape[0], generator=generator)]  # shuffle across classes


def make_loader(x, y, batch_size=128, shuffle=True, seed=None):
    """Wrap (x, y) tensors in a DataLoader for train_classifier."""
    dataset = TensorDataset(x, y)
    if shuffle and seed is not None:
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
