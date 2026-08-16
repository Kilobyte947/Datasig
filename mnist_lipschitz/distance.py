"""This file implements the two distance functions used by estimators.py -- plain Euclidean distance, and ridge-regularized Mahalanobis
distance over MNIST pixel space. The Mahalanobis precision matrix is built from the SVD of the (centered) pixel matrix directly
(svd_ridge_precision) rather than by first forming and inverting a (784, 784) covariance matrix: forming X^T @ X squares the condition
number of the data before any regularization is even applied, and MNIST's pixel matrix is already ill-conditioned (many border pixels
are 0 in every image) -- working from X's own singular values instead avoids that amplification. covariance_eigenvalues and
sweep_epsilon are built the same way, so nothing in this file ever needs to form the (784, 784) covariance explicitly.
"""

import torch

torch.set_default_dtype(torch.float64)

def euclidean_distance_fn(x, y):
    """||x-y||_2. Plain Euclidean distance between x and y, row-wise if x/y are batches of the same length (broadcasts otherwise). """
    return (x - y).norm(p=2, dim=-1)


def svd_ridge_precision(x_flat, epsilon):
    """Ridge-regularized Mahalanobis precision matrix: (Sigma + epsilon*I)^-1, where Sigma is the empirical covariance of centered
    `x_flat` -- built directly from the SVD of `x_flat` itself, without ever forming Sigma as a (784, 784) matrix.

    x_flat: (N, 784) raw (uncentered) pixel vectors.

    Math: for the centered X (N, 784), the economy/reduced SVD X = U @ diag(S) @ V^T gives
    X^T @ X = V @ diag(S^2) @ V^T (since U^T @ U = I), so Sigma = X^T @ X / (N-1) = V @ diag(S^2/(N-1)) @ V^T,
    and therefore (Sigma + epsilon*I)^-1 = V @ diag(1 / (S^2/(N-1) + epsilon)) @ V^T.

    This is the numerically preferred route: forming X^T @ X squares X's condition number (cond(X^T @ X) = cond(X)^2),
    amplifying floating-point error before the ridge term even gets added -- the classic reason to prefer an SVD over
    the normal-equations (X^T @ X) approach when X is ill-conditioned, which MNIST's pixel matrix is (many always-zero
    border pixels; see covariance_eigenvalues).

    The construction above requires the reduced SVD's V to be a full 784x784 orthonormal basis, which is exactly what
    `torch.linalg.svd(..., full_matrices=False)` returns whenever N >= 784 (true everywhere `x_flat` is used in this
    codebase -- always the full-size MNIST training set against 784 pixels). If N < 784, V would only span an
    N-dimensional subspace, and the missing 784-N directions would get no epsilon regularization at all.
    """
    x_centered = x_flat - x_flat.mean(dim=0, keepdim=True)
    N = x_flat.shape[0]
    _, S, Vh = torch.linalg.svd(x_centered, full_matrices=False)
    V = Vh.T
    return V @ torch.diag(1.0 / (S**2 / (N - 1) + epsilon)) @ V.T


def mahalanobis_distance(x, y, precision):
    """Mahalanobis distance between x and y, given a precision matrix P = Sigma^-1: sqrt((x-y)^T P (x-y))
    x, y: (N, 784) or (784,) or (1, 784). precision: (784, 784). Returns (N,) or scalar.
    """
    diff = x - y
    quad = torch.einsum("...i,ij,...j->...", diff, precision, diff)
    return quad.clamp_min(0.0).sqrt()


def make_mahalanobis_distance_fn(precision):
    """Returns a distance_fn(x, y) closure over a fixed precision matrix, for direct use as estimators.py's `distance_fn` argument."""
    return lambda x, y: mahalanobis_distance(x, y, precision)


def covariance_eigenvalues(x_flat):
    """Eigenvalues of the (symmetric, PSD) empirical covariance of centered `x_flat`, sorted descending -- computed from `x_flat`'s
    own singular values (S^2/(N-1), see svd_ridge_precision) rather than a formed covariance matrix.

    x_flat: (N, 784) raw (uncentered) pixel vectors.

    Expected to include several ~0 values in practice (MNIST's constant-zero border pixels), which is why
    svd_ridge_precision needs epsilon - this makes that rank-deficiency directly visible rather than only
    inferred from a large condition number.
    """
    x_centered = x_flat - x_flat.mean(dim=0, keepdim=True)
    N = x_flat.shape[0]
    _, S, _ = torch.linalg.svd(x_centered, full_matrices=False)
    return S**2 / (N - 1)  # S is already sorted descending, so this is too


def sweep_epsilon(x_flat, epsilon_values):
    """Condition number of Sigma + epsilon*I for each candidate epsilon, where Sigma is the empirical covariance of centered
    `x_flat` -- computed from covariance_eigenvalues's singular-value-derived eigenvalues rather than a formed Sigma matrix.
    Sigma + epsilon*I is symmetric PSD, so its condition number is just the ratio of its largest to smallest eigenvalue
    (each of Sigma's own eigenvalues shifted by the same epsilon).

    x_flat: (N, 784) raw (uncentered) pixel vectors.
    """
    eigenvalues = covariance_eigenvalues(x_flat)  # sorted descending
    return [((eigenvalues[0] + eps) / (eigenvalues[-1] + eps)).item() for eps in epsilon_values]
