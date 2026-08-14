"""This file implements the two distance functions used by estimators.py -- plain Euclidean distance, and ridge-regularized Mahalanobis
distance over MNIST pixel space -- plus sweep_epsilon, the pure linear-algebra half of epsilon selection (condition number vs. epsilon).
MNIST's raw pixel covariance is rank-deficient in practice (many border pixels are 0 in every image), so it can't be inverted directly.
Adding `epsilon * I` before inverting (ridge regularization) fixes this; see Checkpoint 5 for how epsilon is selected.
"""

import torch

torch.set_default_dtype(torch.float64)

def euclidean_distance_fn(x, y):
    """||x-y||_2. Plain Euclidean distance between x and y, row-wise if x/y are batches of the same length (broadcasts otherwise). """
    return (x - y).norm(p=2, dim=-1)


def pixel_covariance(x_flat):
    """Empirical covariance of centered, flattened pixel vectors.
    x_flat: (N, 784). Returns (784, 784) Sigma."""
    x_centered = x_flat - x_flat.mean(dim=0, keepdim=True)
    N = x_flat.shape[0]
    return (x_centered.T @ x_centered) / (N - 1)

def ridge_precision(Sigma, epsilon):
    """Ridge-regularized precision matrix: (Sigma + epsilon * I)^-1. 
    Done to make Sigma invertible, since MNIST's pixel covariance is rank-deficient in practice (many border pixels are 0 in every image, 
    so their variance and covariance with everything else -- is exactly 0)."""
    d = Sigma.shape[0]
    return torch.linalg.inv(Sigma + epsilon * torch.eye(d))

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


def covariance_eigenvalues(Sigma):
    """Eigenvalues of the (symmetric, PSD) pixel covariance Sigma, sorted
    descending. Expected to include several ~0 values in practice (MNIST's
    constant-zero border pixels), which is why ridge_precision needs
    epsilon - this makes that rank-deficiency directly visible rather
    than only inferred from a large condition number."""
    eigenvalues = torch.linalg.eigvalsh(Sigma)
    return eigenvalues.flip(0)


def sweep_epsilon(Sigma, epsilon_values):
    """Condition number of Sigma + epsilon*I for each candidate epsilon."""
    d = Sigma.shape[0]
    identity = torch.eye(d)
    return [torch.linalg.cond(Sigma + eps * identity).item() for eps in epsilon_values]
