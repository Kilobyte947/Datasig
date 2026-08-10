"""Feature embeddings and covariance-derived metrics (Terry's feedback,
meeting 2): instead of assuming raw Euclidean distance in x is the right
notion of "close," embed x into a richer feature space (polynomial powers,
optionally with f(x) appended) and derive a Mahalanobis-style distance from
the empirical covariance of the embedded points.
"""

import torch

torch.set_default_dtype(torch.float64)


def _as_tensor(v):
    return torch.as_tensor(v, dtype=torch.get_default_dtype())


def polynomial_embedding(x, degree):
    """Map x (N,) or (N,1) -> (x, x^2, ..., x^degree), shape (N, degree).
    d=1 only. A linear function of this embedding is a degree-`degree`
    polynomial in x -- Terry's point 1/2: a clump near one point in x
    looks different from the same-size clump elsewhere once you're
    measuring in this embedded space rather than raw x.
    """
    x = _as_tensor(x).reshape(-1)
    return torch.stack([x ** k for k in range(1, degree + 1)], dim=-1)


def augmented_embedding(x, degree, f_vals):
    """polynomial_embedding(x, degree) with f(x) appended as an extra
    feature column -- Terry's point 6: "embedding should include not only
    x, x^2, ... but also f(x), which we already have."
    f_vals: (N,) precomputed f(x), e.g. the trained model's output.
    """
    poly = polynomial_embedding(x, degree)
    f_vals = _as_tensor(f_vals).reshape(-1, 1)
    return torch.cat([poly, f_vals], dim=-1)


def empirical_covariance(z, eps=1e-8):
    """Covariance matrix of embedded points z (N, k), with a small ridge
    eps*I added so it's always invertible."""
    z = _as_tensor(z)
    z_centered = z - z.mean(dim=0, keepdim=True)
    N = z.shape[0]
    cov = (z_centered.T @ z_centered) / (N - 1)
    return cov + eps * torch.eye(cov.shape[0])


def precision_from_covariance(cov):
    """Sigma^{-1} -- the quadratic-form matrix used for Mahalanobis
    distance: d(a,b)^2 = (a-b)^T Sigma^{-1} (a-b). This is the "L2 norm
    in the dual space, pulled back" Terry described in points 3-4."""
    return torch.linalg.inv(cov)
