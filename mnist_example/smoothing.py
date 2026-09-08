"""Spatial smoothing as a pre-processing step for local_patch_cross_terms, to test whether
blurring the image before computing cross-term features fixes that embedding's Mahalanobis
epsilon-selection instability."""

import torch
import torch.nn.functional as F
from mnist_example.embeddings import local_patch_cross_terms
torch.set_default_dtype(torch.float64)


RADIUS_MULTIPLIER = 5
"""Default radius = round(RADIUS_MULTIPLIER * sigma) for the Gaussian kernel. Chosen by sweeping
{2, 3, 4, 5, 6} at sigma=1: gives the best epsilon-selection stability (cv=0.0110) while 4/5/6
converge to the same downstream numbers.
"""

def _gaussian_kernel_1d(sigma, radius_multiplier=RADIUS_MULTIPLIER):
    """Normalised 1D Gaussian kernel, with radius scaling with sigma rather than fixed."""
    radius = max(1, int(round(radius_multiplier * sigma)))
    positions = torch.arange(-radius, radius + 1, dtype=torch.get_default_dtype())
    kernel = torch.exp(-(positions ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def gaussian_blur_embedding(x, sigma, radius_multiplier=RADIUS_MULTIPLIER):
    """Applies an isotropic Gaussian blur to each 28x28 MNIST image, via a separable convolution
    with zero padding. x is (..., 784); returns the same shape. sigma <= 0 returns x unchanged."""
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
    """Blurs the image, then computes local_patch_cross_terms on the result. x_flat is (..., 784);
    returns (..., 3920). sigma=0 reduces exactly to local_patch_cross_terms on the unblurred image."""
    blurred_flat = gaussian_blur_embedding(x_flat, sigma, radius_multiplier=radius_multiplier)
    blurred_image = blurred_flat.reshape(*blurred_flat.shape[:-1], 28, 28)
    return local_patch_cross_terms(blurred_image)
