"""This file implements the distance functions used by estimators.py -- plain Euclidean distance, ridge-regularized Mahalanobis
distance, and truncated-eigenvalue Mahalanobis distance, all over MNIST pixel (or embedded feature) space. Every Mahalanobis-family
precision matrix here is built from the SVD of the (centered) feature matrix directly (covariance_eigenbasis) rather than by first
forming and inverting a (D, D) covariance matrix: forming X^T @ X squares the condition number of the data before any regularization
is even applied, and MNIST's pixel matrix (and every embedded feature space built from it elsewhere in this project) is already
ill-conditioned (many border pixels are 0 in every image) -- working from X's own singular values instead avoids that amplification.
Nothing in this file ever needs to form the (D, D) covariance explicitly.
"""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

torch.set_default_dtype(torch.float64)

def euclidean_distance_fn(x, y):
    """||x-y||_2. Plain Euclidean distance between x and y, row-wise if x/y are batches of the same length (broadcasts otherwise). """
    return (x - y).norm(p=2, dim=-1)


def covariance_eigenbasis(x_flat):
    """Full eigendecomposition (V, eigenvalues) of the empirical covariance of centered `x_flat`, sorted descending by
    eigenvalue -- computed once from `x_flat`'s own SVD and shared by every Mahalanobis-family construction in this file
    (`svd_ridge_precision`'s ridge regularization, `truncated_precision`'s top-k truncation, `covariance_eigenvalues`),
    so the SVD itself is never duplicated across those call sites.

    x_flat: (N, D) raw (uncentered) feature vectors, N >= D (needed for the reduced SVD's V to be a full D-dimensional
    orthonormal basis -- true everywhere this is called in this codebase, always a full training-set-sized pool against
    at most a few thousand feature dimensions). Returns `V` ((D, D), columns are eigenvectors) and `eigenvalues` ((D,),
    sorted descending, `S**2/(N-1)` where `S` is `x_flat`'s own singular values -- see `svd_ridge_precision`'s docstring
    for why this route is numerically preferred over forming `X^T @ X` directly).
    """
    x_centered = x_flat - x_flat.mean(dim=0, keepdim=True)
    N = x_flat.shape[0]
    _, S, Vh = torch.linalg.svd(x_centered, full_matrices=False)
    V = Vh.T
    eigenvalues = S**2 / (N - 1)
    return V, eigenvalues


def svd_ridge_precision(x_flat, epsilon):
    """Ridge-regularized Mahalanobis precision matrix: (Sigma + epsilon*I)^-1, where Sigma is the empirical covariance of centered
    `x_flat` -- built directly from `covariance_eigenbasis`, without ever forming Sigma as a (D, D) matrix.

    x_flat: (N, D) raw (uncentered) feature vectors.

    Math: Sigma = V @ diag(eigenvalues) @ V^T (see `covariance_eigenbasis`), so
    (Sigma + epsilon*I)^-1 = V @ diag(1 / (eigenvalues + epsilon)) @ V^T.

    This is the numerically preferred route: forming X^T @ X squares X's condition number (cond(X^T @ X) = cond(X)^2),
    amplifying floating-point error before the ridge term even gets added -- the classic reason to prefer an SVD over
    the normal-equations (X^T @ X) approach when X is ill-conditioned, which MNIST's pixel matrix is (many always-zero
    border pixels; see covariance_eigenvalues). See `covariance_eigenbasis`'s docstring for the `N >= D` requirement
    this construction relies on.
    """
    V, eigenvalues = covariance_eigenbasis(x_flat)
    return V @ torch.diag(1.0 / (eigenvalues + epsilon)) @ V.T


def truncated_precision(x_flat, k):
    """Truncated-eigenvalue Mahalanobis precision matrix: keeps only the top-`k` eigenvectors/eigenvalues of the empirical
    covariance of centered `x_flat` (by descending eigenvalue) and discards the rest entirely, rather than
    `svd_ridge_precision`'s ridge regularization (which keeps all `D` directions and adds `epsilon` to stabilize the
    near-singular ones). `P = V_k @ diag(1/eigenvalues_k) @ V_k^T` -- a rank-`k`, positive-semidefinite matrix (the
    discarded `D-k` directions contribute exactly 0 to any Mahalanobis distance computed with this `P`, not a
    regularized-but-nonzero amount).

    **Motivation**: `svd_ridge_precision`'s ridge regularization stabilizes near-singular directions by adding `epsilon`,
    but on feature spaces with many near-zero-variance directions (`embeddings.py::local_patch_cross_terms`'s spatial
    cross-terms, `smoothing.py`'s smoothed variant) this project has repeatedly found that *regularizing* those
    directions -- rather than removing them -- is itself the source of resampling instability (`distance_measures.md`'s "Epsilon
    selection fails categorically" finding, cv 0.91-1.45 against a 0.05 bound at every epsilon tried): a near-zero
    eigenvalue makes `1/(eigenvalue+epsilon)` extremely sensitive to exactly which points land in a given resample.
    Discarding those directions outright removes that sensitivity by construction, at the cost of a metric that's blind
    to variation along the discarded directions -- see `run_experiment.py::k_stability_check` for whether this actually
    fixes the instability in practice, not just in principle.

    `x_flat`: (N, D) raw (uncentered) feature vectors, same `N >= D` requirement as `covariance_eigenbasis`. `k`: number
    of top-variance eigenvectors to keep, `1 <= k <= D`.

    Returned `P` is a (D, D) dense matrix, suitable for direct use anywhere this project already threads a `precision`
    matrix through -- `make_mahalanobis_distance_fn`, `gradient_norm_estimate`, `k_stability_check` -- rather than a new,
    separate distance-computation code path; truncated Mahalanobis distance is mathematically just Mahalanobis distance
    with this specific rank-`k` precision matrix, reusing `mahalanobis_distance`'s existing formula and tests.
    """
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
    """Returns a distance_fn(x, y) closure over a fixed precision matrix, for direct use as estimators.py's `distance_fn` argument.

    If `embed_fn` is given, both x and y are mapped through it before the Mahalanobis distance is computed -- lets
    the metric be defined over an embedded feature space (e.g. embeddings.py::elementwise_embedding) instead of
    raw pixel space, matching toy_example's embed_fn convention for pairwise_lipschitz/local_perturbation_lipschitz.
    `precision` must then be sized for the embedded space, not raw x (e.g. `svd_ridge_precision(embed_fn(x_flat), epsilon)`,
    or `truncated_precision(embed_fn(x_flat), k)`).

    Leaving `embed_fn` unset (the default) leaves existing behavior exactly unchanged.
    """
    if embed_fn is None:
        return lambda x, y: mahalanobis_distance(x, y, precision)
    return lambda x, y: mahalanobis_distance(embed_fn(x), embed_fn(y), precision)


def make_truncated_mahalanobis_distance_fn(x_flat, k, embed_fn=None):
    """One-shot convenience: fits `truncated_precision` on `embed_fn(x_flat)` (or raw `x_flat` if `embed_fn` is `None`)
    and wraps it via `make_mahalanobis_distance_fn` -- since truncated Mahalanobis distance is just Mahalanobis distance
    with a rank-`k` precision matrix, this composes the existing pieces rather than a new distance formula.

    `x_flat`: (N, D) raw (uncentered) feature vectors to fit the covariance on (typically the full training set, matching
    every other precision matrix fit in this project). `k`: number of top-variance eigenvectors to keep (see
    `truncated_precision`). `embed_fn`: same convention as `make_mahalanobis_distance_fn` -- leaving it unset uses raw
    pixel space.
    """
    x_for_cov = embed_fn(x_flat) if embed_fn is not None else x_flat
    precision = truncated_precision(x_for_cov, k)
    return make_mahalanobis_distance_fn(precision, embed_fn=embed_fn)


def covariance_eigenvalues(x_flat):
    """Eigenvalues of the (symmetric, PSD) empirical covariance of centered `x_flat`, sorted descending -- thin wrapper
    around `covariance_eigenbasis` for callers that only need the eigenvalues, not the eigenvectors too.

    x_flat: (N, D) raw (uncentered) feature vectors.

    Expected to include several ~0 values in practice on raw MNIST pixels (constant-zero border pixels), which is why
    svd_ridge_precision needs epsilon - this makes that rank-deficiency directly visible rather than only
    inferred from a large condition number.
    """
    return covariance_eigenbasis(x_flat)[1]


def sweep_epsilon(x_flat, epsilon_values):
    """Condition number of Sigma + epsilon*I for each candidate epsilon, where Sigma is the empirical covariance of centered
    `x_flat` -- computed from covariance_eigenvalues's singular-value-derived eigenvalues rather than a formed Sigma matrix.
    Sigma + epsilon*I is symmetric PSD, so its condition number is just the ratio of its largest to smallest eigenvalue
    (each of Sigma's own eigenvalues shifted by the same epsilon).

    x_flat: (N, 784) raw (uncentered) pixel vectors.
    """
    eigenvalues = covariance_eigenvalues(x_flat)  # sorted descending
    return [((eigenvalues[0] + eps) / (eigenvalues[-1] + eps)).item() for eps in epsilon_values]


def sweep_k_condition_numbers(x_flat, k_values):
    """Condition number of the retained top-`k` eigenvalues for each candidate `k`, where the eigenvalues come from the
    empirical covariance of centered `x_flat` -- the `truncated_precision` analogue of `sweep_epsilon`. Since truncation
    discards the bottom `D-k` eigenvalues rather than shifting all of them by `epsilon`, the condition number here is
    just the ratio of the largest retained eigenvalue to the smallest retained one (`eigenvalues[0] / eigenvalues[k-1]`),
    not a function of any regularization strength.

    x_flat: (N, D) raw (uncentered) feature vectors.
    """
    eigenvalues = covariance_eigenvalues(x_flat)  # sorted descending
    return [(eigenvalues[0] / eigenvalues[k - 1]).item() for k in k_values]


def knn_label_purity(embedded, labels, k=5):
    """For each point, the fraction of its `k` nearest neighbors (Euclidean, in the given embedded
    space, excluding itself) that share its true label, averaged over every point -- a well-
    clustered-by-digit embedding scores well above the 10-class chance baseline (0.10); a
    scattered one sits close to it. Used as a quantitative embedding-quality check anywhere a
    metric or embedding needs one (`run_experiment.py::run_smoothing_sweep`,
    `umap_embedding.py`'s own validation) -- kept here rather than in `umap_embedding.py` so
    callers that have nothing to do with UMAP don't pull in that dependency.

    `embedded`: (N, d) array/tensor of embedded coordinates. `labels`: (N,) integer true labels,
    same order.
    """
    embedded_np = embedded.detach().cpu().numpy() if hasattr(embedded, "detach") else np.asarray(embedded)
    labels_np = labels.detach().cpu().numpy() if hasattr(labels, "detach") else np.asarray(labels)

    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(embedded_np)
    _, neighbor_idx = nn.kneighbors(embedded_np)
    neighbor_idx = neighbor_idx[:, 1:]  # drop each point's own (zero-distance) self-match

    matches = (labels_np[neighbor_idx] == labels_np[:, None]).mean()
    return float(matches)


def class_separation_ratio(x_subset, y_subset, distance_fn):
    """Mean between-class / mean within-class distance under `distance_fn`, using true labels --
    a metric-quality check independent of any model's margin behavior: does this distance metric
    intrinsically place different digits farther apart than same-digit pairs, or not? Higher is
    better separation. Uses all `N*(N-1)/2` pairs (`torch.triu_indices`, matching
    `estimators.py::pairwise_lipschitz_all`'s all-pairs convention), not just near-neighbors.

    This answers a different question than `run_experiment.py::run_ratio_distribution_analysis`'s
    near/all ratio: that ratio is about whether a *model's* margin sensitivity is elevated on
    near-neighbor pairs (a statement about the model, using the metric only to pick pairs); this
    is a statement about the metric itself, using only ground-truth class labels, with no model or
    margin involved at all.

    `x_subset`: (N, 784) raw pixel vectors -- `distance_fn` is called directly on raw x/y and is
    expected to handle any embedding internally (matching `run_ratio_distribution_analysis`'s
    `distance_fn` convention, e.g. `make_mahalanobis_distance_fn(precision, embed_fn=embed_fn)`),
    not pre-embedded input. `y_subset`: (N,) true integer labels, same order.
    """
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
