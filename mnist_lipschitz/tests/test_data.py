import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mnist_lipschitz.data import load_mnist, get_dev_subset


def test_load_mnist_train_shapes_and_ranges():
    data = load_mnist(train=True)
    N = len(data)
    assert N == 60000
    assert data.x_flat.shape == (N, 784)
    assert data.x_image.shape == (N, 1, 28, 28)
    assert data.y.shape == (N,)
    assert data.x_flat.dtype == torch.float64
    assert data.x_image.dtype == torch.float64
    assert data.y.dtype == torch.int64


def test_load_mnist_test_split_shapes():
    data = load_mnist(train=False)
    assert len(data) == 10000
    assert data.x_flat.shape == (10000, 784)


def test_pixel_value_range():
    data = load_mnist(train=True)
    assert data.x_flat.min().item() >= 0.0
    assert data.x_flat.max().item() <= 1.0
    # MNIST digits are mostly-black backgrounds with white strokes -- both
    # ends of the range should actually be present, not just technically allowed.
    assert data.x_flat.max().item() > 0.9
    assert data.x_flat.min().item() == 0.0


def test_label_range():
    data = load_mnist(train=True)
    assert data.y.min().item() >= 0
    assert data.y.max().item() <= 9
    assert set(data.y.unique().tolist()) == set(range(10))


def test_flat_and_image_views_agree():
    data = load_mnist(train=True)
    assert torch.allclose(data.x_image.reshape(len(data), -1), data.x_flat)


def test_dev_subset_shape_and_seed_reproducibility():
    data = load_mnist(train=True)
    sub1 = get_dev_subset(data, n=2000, seed=0)
    sub2 = get_dev_subset(data, n=2000, seed=0)
    sub3 = get_dev_subset(data, n=2000, seed=1)

    assert len(sub1) == 2000
    assert sub1.x_flat.shape == (2000, 784)
    assert torch.equal(sub1.x_flat, sub2.x_flat)
    assert torch.equal(sub1.y, sub2.y)
    assert not torch.equal(sub1.x_flat, sub3.x_flat)
