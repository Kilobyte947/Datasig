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
