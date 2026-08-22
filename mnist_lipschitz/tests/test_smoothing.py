import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mnist_lipschitz.smoothing import gaussian_blur_embedding, smoothed_cross_terms_embedding
from mnist_lipschitz.embeddings import local_patch_cross_terms


def test_sigma_zero_is_exact_identity():
    torch.manual_seed(0)
    x = torch.randn(10, 784)
    assert torch.equal(gaussian_blur_embedding(x, sigma=0), x)
    assert torch.equal(gaussian_blur_embedding(x, sigma=-1.0), x)


def test_shape_preserved_for_batched_and_single_input():
    torch.manual_seed(1)
    for sigma in (0, 0.5, 1.0, 3.0):
        x_batch = torch.rand(7, 784)
        assert gaussian_blur_embedding(x_batch, sigma).shape == (7, 784)

        x_single = torch.rand(784)
        assert gaussian_blur_embedding(x_single, sigma).shape == (784,)


def test_blur_conserves_total_intensity_for_an_interior_point_source():
    """A single 'on' pixel away from the border, blurred: mass should spread to its neighbors
    (they become nonzero) while total intensity is conserved (each 1D pass uses a normalized
    kernel, and none of that single point's mass reaches the zero-padded border at these sigmas),
    and the center pixel's own value should drop -- a hand-checkable sanity check on the blur
    itself, not just its shape."""
    x = torch.zeros(28, 28)
    x[14, 14] = 1.0
    x_flat = x.reshape(-1)

    blurred = gaussian_blur_embedding(x_flat, sigma=1.0).reshape(28, 28)

    assert blurred[14, 14].item() < 1.0
    assert blurred[13, 14].item() > 0.0
    assert blurred[15, 14].item() > 0.0
    assert blurred[14, 13].item() > 0.0
    assert blurred[14, 15].item() > 0.0
    assert torch.isclose(blurred.sum(), torch.tensor(1.0), atol=1e-6)


def test_larger_sigma_spreads_more_than_smaller_sigma():
    x = torch.zeros(28, 28)
    x[14, 14] = 1.0
    x_flat = x.reshape(-1)

    blurred_small = gaussian_blur_embedding(x_flat, sigma=0.5).reshape(28, 28)
    blurred_large = gaussian_blur_embedding(x_flat, sigma=2.0).reshape(28, 28)

    # peak value should fall as sigma grows (energy spread over a wider area)
    assert blurred_large[14, 14].item() < blurred_small[14, 14].item()
    # a pixel several steps away should pick up more intensity under the larger blur
    assert blurred_large[10, 14].item() > blurred_small[10, 14].item()


def test_border_padding_is_zero_not_reflected():
    """A bright pixel in the corner: the out-of-image neighbors it blurs into must be treated as
    background (0), matching local_patch_cross_terms's own zero-padding convention -- so the
    surviving intensity (sum over the real 28x28 grid) should be strictly less than 1, not
    conserved, since some of the kernel's mass falls outside the image and is discarded."""
    x = torch.zeros(28, 28)
    x[0, 0] = 1.0
    x_flat = x.reshape(-1)

    blurred = gaussian_blur_embedding(x_flat, sigma=1.0).reshape(28, 28)
    assert blurred.sum().item() < 1.0
    assert blurred[0, 0].item() > 0.0


def test_composition_at_sigma_zero_matches_plain_local_patch_cross_terms():
    torch.manual_seed(2)
    x_flat = torch.rand(5, 784)
    x_image = x_flat.reshape(5, 28, 28)

    composed = smoothed_cross_terms_embedding(x_flat, sigma=0)
    direct = local_patch_cross_terms(x_image)

    assert torch.equal(composed, direct)


def test_composition_output_shape():
    torch.manual_seed(3)
    for sigma in (0, 0.5, 2.0):
        x_flat = torch.rand(6, 784)
        embedded = smoothed_cross_terms_embedding(x_flat, sigma)
        assert embedded.shape == (6, 3920)


def test_composition_changes_with_sigma():
    """Sanity check that blurring actually feeds through into the cross-term output -- sigma>0
    should give a different result than sigma=0 on a non-trivial (non-uniform) image."""
    torch.manual_seed(4)
    x_flat = torch.rand(3, 784)

    unblurred = smoothed_cross_terms_embedding(x_flat, sigma=0)
    blurred = smoothed_cross_terms_embedding(x_flat, sigma=1.5)

    assert not torch.allclose(unblurred, blurred)


def test_composition_single_example_matches_batch_leading_dim_flexibility():
    """embed_fn callers (e.g. gradient_norm_estimate's jacrev/vmap path, see smoothing.py's
    docstring) call this on a single (784,) example, not just a batch -- must work the same way
    local_patch_cross_terms itself does."""
    torch.manual_seed(5)
    x_flat = torch.rand(4, 784)

    batched = smoothed_cross_terms_embedding(x_flat, sigma=1.0)
    for i in range(4):
        single = smoothed_cross_terms_embedding(x_flat[i], sigma=1.0)
        assert torch.allclose(single, batched[i], atol=1e-10)
