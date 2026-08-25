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


RADIUS_MULTIPLIER = 5
"""Default `radius = round(RADIUS_MULTIPLIER * sigma)` used by `_gaussian_kernel_1d`/
`gaussian_blur_embedding`. Not a library default -- this module never calls a library blur function
at all (see `_gaussian_kernel_1d`'s docstring). `3` was the value used throughout this project's
existing smoothing-strength sweep (`commit 18d3549`) without having itself been swept; a direct
sweep over `radius_multiplier` in `{2, 3, 4, 5, 6}` at the already-established best `sigma=1`
(`results/radius_multiplier_sweep_results.json` / `notebook_radius_multiplier_sweep.ipynb`) found
`5` gives the best epsilon-selection stability (cv=0.0110, vs. `3`'s 0.0233) while `4`/`5`/`6`
converge to essentially identical purity and ratio-distribution numbers -- so `5` is picked for its
stability margin, not because `4`/`6` perform any differently downstream.

**Important, larger finding from that sweep, not just about this constant**: every
`radius_multiplier` tested passed the `cv<=0.05` stability bound, sharply contradicting `sigma=1`'s
own previously-recorded result (`radius_multiplier=3`, cv=0.0754, failing) at nominally the exact
same configuration. Purity matched exactly (0.8146) between the two runs, confirming the model and
embedding were reproducing identically -- the discrepancy traced to a real numerical bug in
`estimators.py::gradient_norm_estimate`'s embed_fn-pullback path, fixed (`torch.linalg.solve` ->
`torch.linalg.pinv`) between the two runs for an unrelated reason (`distance.py::truncated_precision`
needing a pseudoinverse-tolerant solve). Directly confirmed: `Q(x)`'s condition number for this
embedding reaches ~1e18-1e21 (far beyond float64's ~1e15-1e16 useful precision), and the old
`torch.linalg.solve`-based dual norm produced numerically meaningless values (single-subsample mean
L_hat ~11 million, with per-point values differing from the pinv-based ones by up to a factor of
~9e13) on exactly this matrix. This means the *entire* smoothing-sweep and
`local_patch_cross_terms` "categorical epsilon-selection failure" narrative documented elsewhere in
this project was measured under that same buggy numerics and needs independent re-verification
under the fix -- see `README.md`'s smoothing sub-experiment for the corrected re-run, not this
constant's own sweep, which is unaffected in its own right (its numbers were already computed with
the fixed code)."""


def _gaussian_kernel_1d(sigma, radius_multiplier=RADIUS_MULTIPLIER):
    """Normalized 1D Gaussian kernel, radius `max(1, round(radius_multiplier*sigma))`, so kernel
    size scales with `sigma` rather than being fixed -- a fixed small kernel would silently
    truncate a large-sigma blur, and a fixed large one wastes computation at small sigma. Built
    explicitly (not via `torchvision.transforms.functional.gaussian_blur`, nor any other library
    blur function) so `sigma=0` can be handled as an exact identity case by the caller rather than
    needing a kernel-size-1 special case here, matching this project's convention of implementing
    its own numerical primitives directly rather than a library helper with different edge-case
    behavior (`augmentation.py`'s `affine_grid`/`grid_sample` is the same style) -- there is no
    library "default" `radius_multiplier` inherited from anywhere; `RADIUS_MULTIPLIER=3` is this
    project's own empirically-swept choice (see that constant's docstring).
    """
    radius = max(1, int(round(radius_multiplier * sigma)))
    positions = torch.arange(-radius, radius + 1, dtype=torch.get_default_dtype())
    kernel = torch.exp(-(positions ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def gaussian_blur_embedding(x, sigma, radius_multiplier=RADIUS_MULTIPLIER):
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
    comparable to the existing raw `local_patch_cross_terms` result. `radius_multiplier` controls
    how many multiples of `sigma` the kernel extends to before being cut off (see
    `RADIUS_MULTIPLIER`'s docstring for why the default is `3`, not left unset for existing
    callers).
    """
    if sigma <= 0:
        return x

    orig_shape = x.shape
    x_img = x.reshape(-1, 1, 28, 28)

    kernel = _gaussian_kernel_1d(sigma, radius_multiplier=radius_multiplier)
    radius = kernel.shape[0] // 2
    kernel_h = kernel.view(1, 1, 1, -1)
    kernel_v = kernel.view(1, 1, -1, 1)

    blurred = F.conv2d(x_img, kernel_h, padding=(0, radius))
    blurred = F.conv2d(blurred, kernel_v, padding=(radius, 0))

    return blurred.reshape(orig_shape)


def smoothed_cross_terms_embedding(x_flat, sigma, radius_multiplier=RADIUS_MULTIPLIER):
    """Composition helper: blur first (`gaussian_blur_embedding`), then compute spatial cross-term
    features on the blurred image (`embeddings.py::local_patch_cross_terms`) -- deliberately kept
    as a separate function rather than a `local_patch_cross_terms` parameter, so the un-composed
    embedding (still used elsewhere, e.g. the existing epsilon-selection negative result) is
    untouched by this module.

    `x_flat`: `(..., 784)` flat pixel vectors, same leading-dims flexibility as
    `gaussian_blur_embedding`/`local_patch_cross_terms`. Returns `(..., 3920)`, matching
    `local_patch_cross_terms`'s own output dimension (blurring doesn't change dimensionality).
    `radius_multiplier` is passed straight through to `gaussian_blur_embedding` -- see
    `RADIUS_MULTIPLIER`'s docstring for why its default is `3`, not left unset here.

    `sigma=0` reduces to plain `local_patch_cross_terms` on the unblurred image exactly (via
    `gaussian_blur_embedding`'s identity case), which is what lets a `sigma` sweep starting at 0
    reproduce the existing, already-documented `local_patch_cross_terms` epsilon-instability result
    as its own baseline row, not a separate one-off comparison.
    """
    blurred_flat = gaussian_blur_embedding(x_flat, sigma, radius_multiplier=radius_multiplier)
    blurred_image = blurred_flat.reshape(*blurred_flat.shape[:-1], 28, 28)
    return local_patch_cross_terms(blurred_image)
