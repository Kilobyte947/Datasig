import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mnist_lipschitz.smoothing import (
    gaussian_blur_embedding, smoothed_cross_terms_embedding, _gaussian_kernel_1d, RADIUS_MULTIPLIER,
)
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


def test_default_radius_multiplier_matches_constant():
    torch.manual_seed(6)
    x = torch.rand(5, 784)
    default = gaussian_blur_embedding(x, sigma=1.0)
    explicit = gaussian_blur_embedding(x, sigma=1.0, radius_multiplier=RADIUS_MULTIPLIER)
    assert torch.equal(default, explicit)


def test_larger_radius_multiplier_gives_wider_kernel():
    sigma = 1.0
    sizes = [_gaussian_kernel_1d(sigma, radius_multiplier=m).shape[0] for m in (2, 3, 4, 5, 6)]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


def test_larger_radius_multiplier_converges_toward_the_continuous_gaussian():
    """Each kernel is renormalized to sum to 1 regardless of radius_multiplier, so the
    distinguishing effect isn't total mass -- it's how much of the *true* (infinite-support)
    Gaussian's tail gets included before that renormalization. A larger radius_multiplier includes
    more of the tail, so it should approximate the continuous Gaussian more closely: the kernel
    should change less between radius_multiplier=5 and 6 than between 2 and 3, since 5/6 have
    already captured nearly all the real mass (>99.99% within 5 sigma) while 2/3 are still cutting
    off a non-negligible amount."""
    sigma = 1.0
    k2 = _gaussian_kernel_1d(sigma, radius_multiplier=2)
    k3 = _gaussian_kernel_1d(sigma, radius_multiplier=3)
    k5 = _gaussian_kernel_1d(sigma, radius_multiplier=5)
    k6 = _gaussian_kernel_1d(sigma, radius_multiplier=6)

    # compare only the overlapping center region (smaller kernels are a strict sub-length)
    def center_diff(k_small, k_large):
        pad = (k_large.shape[0] - k_small.shape[0]) // 2
        return (k_large[pad:pad + k_small.shape[0]] - k_small).abs().sum().item()

    diff_2_3 = center_diff(k2, k3)
    diff_5_6 = center_diff(k5, k6)
    assert diff_5_6 < diff_2_3


def test_radius_multiplier_composes_through_smoothed_cross_terms_embedding():
    """radius_multiplier=1 (a visibly narrower kernel than any default this project is likely to
    pick) is deliberately used as the "different" contrast value here, not a value close to
    RADIUS_MULTIPLIER -- large multipliers converge toward the same continuous-Gaussian
    approximation (see test_larger_radius_multiplier_converges_toward_the_continuous_gaussian and
    the radius_multiplier sweep's own finding that 4/5/6 give near-identical downstream numbers),
    so a contrast value near the current default could coincidentally fall within torch.allclose's
    tolerance regardless of which specific default this project settles on."""
    torch.manual_seed(7)
    x_flat = torch.rand(3, 784)
    default = smoothed_cross_terms_embedding(x_flat, sigma=1.0)
    explicit = smoothed_cross_terms_embedding(x_flat, sigma=1.0, radius_multiplier=RADIUS_MULTIPLIER)
    different = smoothed_cross_terms_embedding(x_flat, sigma=1.0, radius_multiplier=1)
    assert torch.equal(default, explicit)
    assert not torch.allclose(default, different)
