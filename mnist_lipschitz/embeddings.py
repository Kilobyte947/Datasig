"""Feature-space embeddings for MNIST pixel vectors, generalizing toy_lipschitz/embeddings.py's polynomial_embedding
convention (elementwise powers, no cross-terms) to the 784-pixel setting.
"""

import torch


def elementwise_embedding(x_flat, degree):
    """Map x -> (x, x**2, ..., x**degree), applied independently per pixel (elementwise power), then concatenated
    along the feature axis.

    x_flat: (N, 784) raw pixel vectors. Returns (N, 784*degree): [x, x**2, ..., x**degree] concatenated in that
    block order.

    Matches toy_lipschitz/embeddings.py::polynomial_embedding's convention: this is 784 independent
    degree-`degree` polynomials (one per pixel) stacked together, not a single degree-`degree` polynomial in all
    784 pixels jointly (which would have combinatorially many cross-terms, e.g. x_i * x_j for i != j, and isn't
    what's built here).

    degree=1 returns x_flat unchanged (concatenating a single term is the identity) -- this is what lets the
    embedded Mahalanobis pipeline be checked directly against the existing raw-pixel one (see
    tests/test_embeddings.py).
    """
    return torch.cat([x_flat ** k for k in range(1, degree + 1)], dim=-1)


def local_patch_cross_terms(x_image):
    """Maps each pixel of an image to its raw value plus one cross-term product with each of its
    "forward" immediate spatial neighbors in a 3x3 window: right `(i,j+1)`, down-left `(i+1,j-1)`,
    down `(i+1,j)`, down-right `(i+1,j+1)`. These four directions cover every one of the (up to) 8
    neighbors in a 3x3 window exactly once per unordered adjacent pixel pair -- e.g. a pixel's
    "left" neighbor is some other pixel's "right" neighbor, so including both directions would just
    duplicate the same product as two columns. This matches the "cross-term" terminology already
    used in this file's `elementwise_embedding` docstring (`x_i * x_j` for `i != j`, each pair
    counted once), just restricted to spatially-adjacent pairs instead of all `784 choose 2` pairs.

    Pixels outside the image border (e.g. the "right" neighbor of a rightmost-column pixel) are
    treated as 0 via zero-padding, so every pixel contributes the same fixed 4 cross-term slots
    regardless of position -- a border pixel's out-of-bounds slots are then just 0, not omitted, so
    the output dimension stays fixed regardless of image size.

    x_image: `(..., H, W)` -- any number of leading batch dims (including zero, so this composes
    both with a single `(H, W)` image and a batch `(N, H, W)`). Works directly on
    `MNISTData.x_image` after squeezing its channel dim (`x_image.squeeze(1)`); for use as this
    package's `embed_fn(x_flat)` convention elsewhere (`distance.py`, `estimators.py`,
    `run_experiment.py`), wrap it: `lambda x: local_patch_cross_terms(x.reshape(*x.shape[:-1], 28, 28))`.

    Returns `(..., H*W*5)`: the `H*W` raw pixels (raster order) followed by 4 blocks of `H*W`
    cross-terms each (right, down-left, down, down-right), each block in the same raster order as
    the raw pixels.
    """
    H, W = x_image.shape[-2:]
    padded = torch.nn.functional.pad(x_image, (1, 1, 1, 1))  # zero border, (..., H+2, W+2)

    offsets = [(0, 1), (1, -1), (1, 0), (1, 1)]  # right, down-left, down, down-right
    cross_terms = []
    for di, dj in offsets:
        neighbor = padded[..., 1 + di:1 + di + H, 1 + dj:1 + dj + W]
        cross_terms.append((x_image * neighbor).reshape(*x_image.shape[:-2], -1))

    raw = x_image.reshape(*x_image.shape[:-2], -1)
    return torch.cat([raw] + cross_terms, dim=-1)
