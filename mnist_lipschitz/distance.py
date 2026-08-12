"""Ridge-regularized Mahalanobis distance over MNIST pixel space.

MNIST's raw pixel covariance is rank-deficient in practice (many border
pixels are 0 in every image, so their variance -- and covariance with
everything else -- is exactly 0), so it cannot be inverted directly. Adding
`epsilon * I` before inverting (ridge regularization) is the standard fix;
Checkpoint 5's epsilon sweep picks how much.

Representation choice: a dense (784, 784) precision matrix, computed via
`torch.linalg.inv(Sigma + epsilon*I)`, rather than a Cholesky factor reused
via `torch.cholesky_solve`. A dense inverse at this size is cheap (~10ms,
measured -- negligible next to model training/evaluation), and it lets this
module hand back exactly the same kind of object (`precision`, a plain
matrix) that `estimators.py`'s `pairwise_lipschitz`/`local_perturbation_lipschitz`
(quadratic-form distance) and `gradient_norm_estimate` (dual-norm via
`precision`) both already expect -- one representation, reused everywhere,
rather than maintaining a solve-based path alongside a dense-matrix path
for what is fundamentally the same quadratic form.
"""

import torch

torch.set_default_dtype(torch.float64)


def pixel_covariance(x_flat):
    """Empirical covariance of centered, flattened pixel vectors.
    x_flat: (N, 784). Returns (784, 784) Sigma."""
    x_centered = x_flat - x_flat.mean(dim=0, keepdim=True)
    N = x_flat.shape[0]
    return (x_centered.T @ x_centered) / (N - 1)


def ridge_precision(Sigma, epsilon):
    """Dense precision matrix P = (Sigma + epsilon*I)^{-1}."""
    d = Sigma.shape[0]
    return torch.linalg.inv(Sigma + epsilon * torch.eye(d))


def mahalanobis_distance(x, y, precision):
    """sqrt((x-y)^T P (x-y)), row-wise if x/y are batches of the same
    length (broadcasts otherwise). Matches the `distance_fn(x, y)`
    signature expected by estimators.py."""
    diff = x - y
    quad = torch.einsum("...i,ij,...j->...", diff, precision, diff)
    return quad.clamp_min(0.0).sqrt()


def make_mahalanobis_distance_fn(precision):
    """Returns a distance_fn(x, y) closure over a fixed precision matrix,
    for direct use as estimators.py's `distance_fn` argument."""
    return lambda x, y: mahalanobis_distance(x, y, precision)
