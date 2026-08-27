from pathlib import Path

import torch

from signature_distance.data_pool import load_eval_pool

# Own local cache (self-contained within signature_distance, no dependency on
# any other experiment's data directory) so tests don't need network access
# after the first run.
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def test_shapes_and_dtypes():
    n_per_class = 5
    images, labels = load_eval_pool(n_per_class=n_per_class, seed=0, root=str(DATA_ROOT))
    assert images.shape == (10 * n_per_class, 28, 28)
    assert labels.shape == (10 * n_per_class,)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64


def test_exact_n_per_class_and_sorted_labels():
    n_per_class = 7
    images, labels = load_eval_pool(n_per_class=n_per_class, seed=0, root=str(DATA_ROOT))
    expected_labels = torch.arange(10).repeat_interleave(n_per_class)
    assert torch.equal(labels, expected_labels)
    for c in range(10):
        assert (labels == c).sum().item() == n_per_class


def test_pixel_values_in_unit_interval():
    images, _ = load_eval_pool(n_per_class=5, seed=0, root=str(DATA_ROOT))
    assert images.min().item() >= 0.0
    assert images.max().item() <= 1.0


def test_determinism_same_seed():
    images1, labels1 = load_eval_pool(n_per_class=5, seed=0, root=str(DATA_ROOT))
    images2, labels2 = load_eval_pool(n_per_class=5, seed=0, root=str(DATA_ROOT))
    assert torch.equal(images1, images2)
    assert torch.equal(labels1, labels2)


def test_determinism_different_seed_differs():
    images1, _ = load_eval_pool(n_per_class=5, seed=0, root=str(DATA_ROOT))
    images2, _ = load_eval_pool(n_per_class=5, seed=1, root=str(DATA_ROOT))
    assert not torch.equal(images1, images2)
