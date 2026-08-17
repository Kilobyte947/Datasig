import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mnist_lipschitz.augmentation import random_affine_augment


def test_zero_params_is_identity_up_to_interpolation():
    torch.manual_seed(0)
    x = torch.rand(4, 1, 28, 28)
    generator = torch.Generator().manual_seed(0)
    out = random_affine_augment(x, degrees=0.0, translate=0.0, generator=generator)
    assert out.shape == x.shape
    assert torch.allclose(out, x, atol=1e-6)


def test_shape_and_dtype_preserved():
    torch.manual_seed(0)
    x = torch.rand(6, 1, 28, 28)
    generator = torch.Generator().manual_seed(1)
    out = random_affine_augment(x, degrees=10.0, translate=0.1, generator=generator)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_values_stay_within_original_range():
    """Bilinear interpolation over the sampled grid is a convex combination
    of nearby original pixel values (or zero, for the zero-padded
    out-of-bounds case), so output values can't exceed the original min/max."""
    torch.manual_seed(0)
    x = torch.rand(8, 1, 28, 28)
    generator = torch.Generator().manual_seed(2)
    out = random_affine_augment(x, degrees=15.0, translate=0.15, generator=generator)
    assert out.min().item() >= -1e-9
    assert out.max().item() <= x.max().item() + 1e-9


def test_seeded_reproducibility():
    torch.manual_seed(0)
    x = torch.rand(5, 1, 28, 28)
    out1 = random_affine_augment(x, degrees=10.0, translate=0.1, generator=torch.Generator().manual_seed(42))
    out2 = random_affine_augment(x, degrees=10.0, translate=0.1, generator=torch.Generator().manual_seed(42))
    assert torch.equal(out1, out2)


def test_nonzero_params_actually_change_the_image():
    torch.manual_seed(0)
    x = torch.rand(4, 1, 28, 28)
    generator = torch.Generator().manual_seed(3)
    out = random_affine_augment(x, degrees=15.0, translate=0.15, generator=generator)
    assert not torch.allclose(out, x)
