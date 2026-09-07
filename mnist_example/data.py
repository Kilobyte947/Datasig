"""This file loads MNIST data, and hands back three views of the same pixels: 
1. flattened 784-dimensional vectors (what the estimators actually operate on)
2. the original 28x28 image shape (needed for the CNN, which expects a 2D image structure to run its convolution filters over)
3. integer digit labels

Pixel values are kept in [0, 1] (ToTensor() only, no ImageNet-style Normalize) so that raw pixel differences remain directly interpretable.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

torch.set_default_dtype(torch.float64) # needed everywhere to avoid silent float32->float64 upcasting and the resulting performance hit.

DATA_ROOT = Path(__file__).resolve().parent / "data"


@dataclass
class MNISTData:
    x_flat: torch.Tensor    # (N, 784) float64 in [0, 1]
    x_image: torch.Tensor   # (N, 1, 28, 28) float64 in [0, 1]
    y: torch.Tensor         # (N,) int64 in {0, ..., 9}

    def __len__(self):
        return self.x_flat.shape[0]


def load_mnist(root=DATA_ROOT, train=True):
    """Load the MNIST dataset and return it as a MNISTData dataclass."""
    transform = transforms.ToTensor()  # scales uint8 [0,255] -> float32 [0,1], no further normalization
    dataset = datasets.MNIST(root=str(root), train=train, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    x_image, y = next(iter(loader))
    x_image = x_image.to(torch.get_default_dtype())
    y = y.to(torch.int64)
    x_flat = x_image.reshape(x_image.shape[0], -1)

    return MNISTData(x_flat=x_flat, x_image=x_image, y=y)


def get_dev_subset(data, n, seed):
    """A small, fixed, seeded subset of `data` for fast iteration during development. 
    Same seed always returns the same subset."""
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(data), generator=generator)[:n]
    return MNISTData(x_flat=data.x_flat[idx], x_image=data.x_image[idx], y=data.y[idx])


def stratified_subset_idx(y, n_points, seed, exclude_idx=None):
    """Return a stratified subset of indices from y, with n_points total, evenly distributed across classes. 
    Important as MNIST is imbalanced and random sampling would have added its own noise on top, especially for small n_points.
    
    If `exclude_idx` is provided, those indices are excluded from the sampling - 
    useful for creating a validation set (`query set`) that is disjoint from the training set. 
    Used in sub-method comparison and ratio-distribution set.
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
    """Make a DataLoader from x and y tensors, with optional shuffling and seeding."""
    dataset = TensorDataset(x, y)
    if shuffle and seed is not None:
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
