"""Spatial smoothing as a pre-processing step for `embeddings.py::local_patch_cross_terms`.

`local_patch_cross_terms` failed epsilon selection categorically under Mahalanobis distance
(cv 0.91-1.45 against a 0.05 stability bound at every regularization level tried -- see
`README.md`'s "Epsilon selection fails categorically for this embedding" section). The working
hypothesis there was that MNIST's mostly-black background makes cross-term features near-zero
everywhere except the stroke, so the resampled covariance is dominated by which few pixels happen
to be "on" in a given subsample, rather than a stable population-level structure.

This module tests a direct fix for that: blur the raw image first (spreading stroke intensity
into more, non-exactly-zero neighboring pixels) *before* computing cross-terms, so the covariance
has more non-degenerate signal to work with. The obvious risk is the opposite failure mode --
enough blur to make different digits visually indistinguishable would destroy the very
locality/margin signal this project is trying to measure, so smoothing strength (`sigma`) needs to
be swept and evaluated, not picked once. See `run_experiment.py::run_smoothing_sweep` and
`notebook_smoothing.ipynb` for that sweep.
"""

import torch
import torch.nn.functional as F

from mnist_lipschitz.embeddings import local_patch_cross_terms

torch.set_default_dtype(torch.float64)


def _gaussian_kernel_1d(sigma):
    """Normalized 1D Gaussian kernel, radius `max(1, round(3*sigma))` (99.7% of the mass for a
    true Gaussian), so kernel size scales with `sigma` rather than being fixed -- a fixed small
    kernel would silently truncate a large-sigma blur, and a fixed large one wastes computation at
    small sigma. Built explicitly (not via `torchvision.transforms.functional.gaussian_blur`) so
    `sigma=0` can be handled as an exact identity case by the caller rather than needing a
    kernel-size-1 special case here, matching this project's convention of implementing its own
    numerical primitives directly rather than a library helper with different edge-case behavior
    (`augmentation.py`'s `affine_grid`/`grid_sample` is the same style).
    """
    radius = max(1, int(round(3 * sigma)))
    positions = torch.arange(-radius, radius + 1, dtype=torch.get_default_dtype())
    kernel = torch.exp(-(positions ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def gaussian_blur_embedding(x, sigma):
    """Applies an isotropic 2D Gaussian blur (standard deviation `sigma`, in pixels) to each 28x28
    MNIST image, via a separable convolution (horizontal 1D pass, then vertical) with zero
    padding -- zero padding matches `local_patch_cross_terms`'s own border convention (out-of-image
    neighbors are 0, i.e. background), appropriate here since MNIST's actual image border is
    already background in every example.

    `x`: `(..., 784)` flat pixel vectors -- any number of leading batch dims (including zero),
    matching `local_patch_cross_terms`'s flexibility so this also composes with
    `gradient_norm_estimate`'s per-point `jacrev`/`vmap` embed_fn path. Returns the same shape as
    `x` -- this is a fixed-784-dimension transform of the image, not a dimensionality-changing
    embedding (unlike `elementwise_embedding`/`local_patch_cross_terms`, which do change dimension)
    -- so it composes cleanly as a pre-processing step ahead of either.

    `sigma <= 0` is an exact identity (returns `x` unchanged, not merely a near-identity from a
    tiny kernel) -- this is what lets a `sigma` sweep include an unblurred baseline directly
    comparable to the existing raw `local_patch_cross_terms` result.
    """
    if sigma <= 0:
        return x

    orig_shape = x.shape
    x_img = x.reshape(-1, 1, 28, 28)

    kernel = _gaussian_kernel_1d(sigma)
    radius = kernel.shape[0] // 2
    kernel_h = kernel.view(1, 1, 1, -1)
    kernel_v = kernel.view(1, 1, -1, 1)

    blurred = F.conv2d(x_img, kernel_h, padding=(0, radius))
    blurred = F.conv2d(blurred, kernel_v, padding=(radius, 0))

    return blurred.reshape(orig_shape)


def smoothed_cross_terms_embedding(x_flat, sigma):
    """Composition helper: blur first (`gaussian_blur_embedding`), then compute spatial cross-term
    features on the blurred image (`embeddings.py::local_patch_cross_terms`) -- deliberately kept
    as a separate function rather than a `local_patch_cross_terms` parameter, so the un-composed
    embedding (still used elsewhere, e.g. the existing epsilon-selection negative result) is
    untouched by this module.

    `x_flat`: `(..., 784)` flat pixel vectors, same leading-dims flexibility as
    `gaussian_blur_embedding`/`local_patch_cross_terms`. Returns `(..., 3920)`, matching
    `local_patch_cross_terms`'s own output dimension (blurring doesn't change dimensionality).

    `sigma=0` reduces to plain `local_patch_cross_terms` on the unblurred image exactly (via
    `gaussian_blur_embedding`'s identity case), which is what lets a `sigma` sweep starting at 0
    reproduce the existing, already-documented `local_patch_cross_terms` epsilon-instability result
    as its own baseline row, not a separate one-off comparison.
    """
    blurred_flat = gaussian_blur_embedding(x_flat, sigma)
    blurred_image = blurred_flat.reshape(*blurred_flat.shape[:-1], 28, 28)
    return local_patch_cross_terms(blurred_image)
