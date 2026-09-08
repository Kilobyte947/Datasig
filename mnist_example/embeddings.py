"""Feature-space embeddings for MNIST pixel vectors."""

import torch

def elementwise_embedding(x_flat, degree):
    """Maps x to (x, x^2, ..., x^degree), applied independently per pixel and concatenated along the feature axis. 
    x_flat is (N, 784). Returns (N, 784*degree). degree=1 returns x_flat unchanged."""
    return torch.cat([x_flat ** k for k in range(1, degree + 1)], dim=-1)


def local_patch_cross_terms(x_image):
    """Maps each pixel to its raw value plus its product with each of its four "forward" spatial
    neighbours (right, down-left, down, down-right), zero-padded at the border. x_image is (..., H, W) 
    with any number of leading batch dimensions. Returns (..., H*W*5): the raw pixels followed by the
    four cross-term blocks, each in the same raster order."""
    H, W = x_image.shape[-2:]
    padded = torch.nn.functional.pad(x_image, (1, 1, 1, 1))  # zero border, (..., H+2, W+2)

    offsets = [(0, 1), (1, -1), (1, 0), (1, 1)]  # right, down-left, down, down-right
    cross_terms = []
    for di, dj in offsets:
        neighbor = padded[..., 1 + di:1 + di + H, 1 + dj:1 + dj + W]
        cross_terms.append((x_image * neighbor).reshape(*x_image.shape[:-2], -1))

    raw = x_image.reshape(*x_image.shape[:-2], -1)
    return torch.cat([raw] + cross_terms, dim=-1)
