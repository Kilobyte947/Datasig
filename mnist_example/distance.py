"""Distance functions used by estimators.py: Euclidean, ridge-regularised Mahalanobis, and
truncated-eigenvalue Mahalanobis, over pixel or embedded feature space.
"""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

torch.set_default_dtype(torch.float64)

def euclidean_distance_fn(x, y):
    """||x-y||_2. Plain Euclidean distance between x and y, 
    row-wise if x/y are batches of the same length (broadcasts otherwise). """
    return (x - y).norm(p=2, dim=-1)


def covariance_eigenbasis(x_flat):
    """Eigenvectors and eigenvalues of the empirical covariance of centered x_flat, sorted descending, 
    computed from x_flat's own SVD. x_flat is (N, D) with N >= D. Returns (V, eigenvalues)."""
    x_centered = x_flat - x_flat.mean(dim=0, keepdim=True)
    N = x_flat.shape[0]
    _, S, Vh = torch.linalg.svd(x_centered, full_matrices=False)
    V = Vh.T
    eigenvalues = S**2 / (N - 1)
    return V, eigenvalues


def svd_ridge_precision(x_flat, epsilon):
    """Ridge-regularized Mahalanobis precision matrix: (Sigma + epsilon*I)^-1, where Sigma is the empirical covariance of centered `x_flat`

    x_flat: (N, D) raw (uncentered) feature vectors.

    Math: Sigma = V @ diag(eigenvalues) @ V^T (see `covariance_eigenbasis`), so
    (Sigma + epsilon*I)^-1 = V @ diag(1 / (eigenvalues + epsilon)) @ V^T.
    """
    V, eigenvalues = covariance_eigenbasis(x_flat)
    return V @ torch.diag(1.0 / (eigenvalues + epsilon)) @ V.T


def truncated_precision(x_flat, k):
    """Truncated-eigenvalue Mahalanobis precision matrix: keeps only the top k eigenvectors of the 
    covariance and discards the rest, rather than regularising near-singular directions with a ridge term. 
    Returns a rank-k, positive-semidefinite (D, D) matrix."""
    V, eigenvalues = covariance_eigenbasis(x_flat)
    V_k, eigenvalues_k = V[:, :k], eigenvalues[:k]
    return V_k @ torch.diag(1.0 / eigenvalues_k) @ V_k.T


def mahalanobis_distance(x, y, precision):
    """Mahalanobis distance between x and y, given a precision matrix P = Sigma^-1: sqrt((x-y)^T P (x-y))
    x, y: (N, 784) or (784,) or (1, 784). precision: (784, 784). Returns (N,) or scalar.
    """
    diff = x - y
    quad = torch.einsum("...i,ij,...j->...", diff, precision, diff)
    return quad.clamp_min(0.0).sqrt()


def make_mahalanobis_distance_fn(precision, embed_fn=None):
    """Wraps a fixed precision matrix as a distance_fn(x, y) closure. If embed_fn is given, x and y 
    are mapped through it before the distance is computed; precision must then be sized for the embedded space."""
    if embed_fn is None:
        return lambda x, y: mahalanobis_distance(x, y, precision)
    return lambda x, y: mahalanobis_distance(embed_fn(x), embed_fn(y), precision)


def make_truncated_mahalanobis_distance_fn(x_flat, k, embed_fn=None):
    """Fits a truncated-eigenvalue precision matrix on x_flat (or embed_fn(x_flat)) and wraps it as a distance_fn."""
    x_for_cov = embed_fn(x_flat) if embed_fn is not None else x_flat
    precision = truncated_precision(x_for_cov, k)
    return make_mahalanobis_distance_fn(precision, embed_fn=embed_fn)


def covariance_eigenvalues(x_flat):
    """Eigenvalues of the empirical covariance of centered x_flat, sorted descending."""
    return covariance_eigenbasis(x_flat)[1]


def sweep_epsilon(x_flat, epsilon_values):
    """Condition number of Sigma + epsilon*I for each candidate epsilon."""
    eigenvalues = covariance_eigenvalues(x_flat)  # sorted descending
    return [((eigenvalues[0] + eps) / (eigenvalues[-1] + eps)).item() for eps in epsilon_values]


def sweep_k_condition_numbers(x_flat, k_values):
    """Condition number of the retained top-k eigenvalues, for each candidate k."""
    eigenvalues = covariance_eigenvalues(x_flat)  # sorted descending
    return [(eigenvalues[0] / eigenvalues[k - 1]).item() for k in k_values]


def knn_label_purity(embedded, labels, k=5):
    """Mean fraction of each point's k nearest neighbours, in the given embedding, that share its true label. 
    A well-clustered embedding scores well above the chance baseline (0.10 for MNIST's 10 classes)."""
    embedded_np = embedded.detach().cpu().numpy() if hasattr(embedded, "detach") else np.asarray(embedded)
    labels_np = labels.detach().cpu().numpy() if hasattr(labels, "detach") else np.asarray(labels)

    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(embedded_np)
    _, neighbor_idx = nn.kneighbors(embedded_np)
    neighbor_idx = neighbor_idx[:, 1:]  # drop each point's own (zero-distance) self-match

    matches = (labels_np[neighbor_idx] == labels_np[:, None]).mean()
    return float(matches)


def class_separation_ratio(x_subset, y_subset, distance_fn):
    """Mean between-class distance over mean within-class distance under distance_fn, using true labels
    — a property of the metric itself, independent of any model. Returns a dict with within_mean, between_mean, ratio, and pair counts."""
    N = x_subset.shape[0]
    ii, jj = torch.triu_indices(N, N, offset=1)
    dists = distance_fn(x_subset[ii], x_subset[jj])
    same_class = y_subset[ii] == y_subset[jj]

    within_mean = dists[same_class].mean().item()
    between_mean = dists[~same_class].mean().item()
    return {
        "within_mean": within_mean, "between_mean": between_mean,
        "ratio": between_mean / within_mean,
        "n_within": int(same_class.sum().item()), "n_between": int((~same_class).sum().item()),
    }
