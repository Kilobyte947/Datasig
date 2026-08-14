"""Instead of assuming raw Euclidean distance in x is the right notion of "close," embed x into a richer feature space (polynomial powers,
optionally with f(x) appended) and derive a Mahalanobis-style distance from the empirical covariance of the embedded points.

Put together:x → embed → compute covariance of embedded points → invert → use as the weighting matrix for distance
"""

import torch

torch.set_default_dtype(torch.float64)


def _as_tensor(v):
    return torch.as_tensor(v, dtype=torch.get_default_dtype())


def polynomial_embedding(x, degree):
    """Map x -> (x, x^2, ..., x^degree), elementwise. Shape (N, degree).
    A linear function of this embedding is a`degree` polynomial in x, 
    so a distance measured in the embedded space can reflect curvature
    that raw Euclidean distance in x cannot. Supports d=1 inputs only.
    """
    x = _as_tensor(x).reshape(-1)
    return torch.stack([x ** k for k in range(1, degree + 1)], dim=-1)


def augmented_embedding(x, degree, f_vals):
    """polynomial_embedding(x, degree) with an additional feature column
    appended.

    f_vals: (N,) precomputed scalar values to append as an extra feature (e.g. a function's output at each x). 
    Intended for cases where the appended values come from a function independent of the one whose Lipschitz behaviour is being measured; 
    using a function's own output to build the metric that measures its own sensitivity is
    self-referential and degrades the resulting estimate (see new_distance_measure.md).
    """
    poly = polynomial_embedding(x, degree)
    f_vals = _as_tensor(f_vals).reshape(-1, 1)
    return torch.cat([poly, f_vals], dim=-1)


def empirical_covariance(z, eps=1e-8):
    """Empirical covariance matrix of embedded points z, shape (N, k).
    A small ridge term eps*I is added before returning, so the result is always invertible even when z's features are collinear or z is low-rank.
    """
    z = _as_tensor(z)
    z_centered = z - z.mean(dim=0, keepdim=True)
    N = z.shape[0]
    cov = (z_centered.T @ z_centered) / (N - 1)
    return cov + eps * torch.eye(cov.shape[0])


def precision_from_covariance(cov):
    """Inverse of a covariance matrix: Sigma^-1.
    Used as the quadratic-form matrix for Mahalanobis distance, d(a,b)^2 = (a-b)^T Sigma^-1 (a-b), 
    which reweights each direction in proportion to how little the data varies along it.

    Covariance tells you "how much do things vary in this direction".
    Its inverse tells you "how much should a fixed amount of movement in this direction count," 
    which is smaller in directions where there's naturally lots of spread, and larger in directions where there's naturally very little spread.
    """
    return torch.linalg.inv(cov)
