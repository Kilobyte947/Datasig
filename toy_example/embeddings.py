"""Embeds x into a richer feature space (polynomial powers, optionally with f(x) appended) and
derives a Mahalanobis-style distance from the empirical covariance of the embedded points."""

import torch
torch.set_default_dtype(torch.float64)

def _as_tensor(v):
    return torch.as_tensor(v, dtype=torch.get_default_dtype())


def polynomial_embedding(x, degree):
    """Maps x to (x, x^2, ..., x^degree), elementwise. Supports 1D input only. Returns (N, degree)."""
    x = _as_tensor(x).reshape(-1)
    return torch.stack([x ** k for k in range(1, degree + 1)], dim=-1)


def augmented_embedding(x, degree, f_vals):
    """polynomial_embedding(x, degree) with an extra feature column appended. f_vals should come from
    a function independent of the one whose Lipschitz behaviour is being measured — using a function's
    own output to build the metric that measures its own sensitivity is self-referential and degrades
    the resulting estimate."""
    poly = polynomial_embedding(x, degree)
    f_vals = _as_tensor(f_vals).reshape(-1, 1)
    return torch.cat([poly, f_vals], dim=-1)


def empirical_covariance(z, eps=1e-8):
    """Empirical covariance of embedded points z, (N, k), with a ridge term added so the result is 
    always invertible."""
    z = _as_tensor(z)
    N = z.shape[0]
    if N <= 1:
        raise ValueError(f"empirical_covariance needs at least 2 points, got N={N}")
    z_centered = z - z.mean(dim=0, keepdim=True)
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


def _mahalanobis_dist(diff, precision):
    """Mahalanobis distance sqrt(diff^T @ precision @ diff), reweighting each direction by precision
    instead of treating all directions equally. Reduces to Euclidean distance if precision is the
    identity matrix."""
    quad = torch.einsum("...i,ij,...j->...", diff, precision, diff)
    return quad.clamp_min(0.0).sqrt()
