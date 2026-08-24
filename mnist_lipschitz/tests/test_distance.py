import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mnist_lipschitz.data import load_mnist, get_dev_subset
from mnist_lipschitz.distance import (
    svd_ridge_precision, mahalanobis_distance, covariance_eigenvalues, covariance_eigenbasis,
    truncated_precision, make_mahalanobis_distance_fn, make_truncated_mahalanobis_distance_fn,
    sweep_k_condition_numbers,
)
from mnist_lipschitz.embeddings import elementwise_embedding


def test_mahalanobis_converges_to_scaled_euclidean_as_epsilon_grows():
    """As epsilon -> large, Sigma + epsilon*I ~= epsilon*I, so precision ~=
    (1/epsilon)*I and mahalanobis_distance(x,y) ~= ||x-y||_2 / sqrt(epsilon).
    This is the direct analogue of Experiment 1's degree-1 closed-form
    check, and the main defense against a sign/inversion bug here, since
    there's no ground-truth L* at MNIST scale to catch one otherwise."""
    torch.manual_seed(0)
    d = 20
    x_flat = torch.randn(500, d)  # N >= d, so svd_ridge_precision's construction is exact (see its docstring)

    epsilon = 1e6
    precision = svd_ridge_precision(x_flat, epsilon)

    x = torch.randn(15, d)
    y = torch.randn(15, d)
    maha = mahalanobis_distance(x, y, precision)
    euclidean = (x - y).norm(p=2, dim=-1)

    expected_ratio = 1.0 / (epsilon ** 0.5)
    actual_ratio = maha / euclidean
    assert torch.allclose(actual_ratio, torch.full_like(actual_ratio, expected_ratio), rtol=1e-3)


def test_svd_ridge_precision_well_conditioned_despite_singular_raw_covariance():
    """Real MNIST pixel covariance is expected to be (near-)singular --
    border pixels are 0 in every image, so their variance and covariance
    with everything else is exactly 0. Verify this directly via the
    eigenvalues derived from x_flat's own SVD, then verify the
    ridge-regularized precision matrix is well-conditioned and correctly
    inverts Sigma + epsilon*I."""
    train = load_mnist(train=True)
    dev = get_dev_subset(train, n=2000, seed=0)

    eigenvalues = covariance_eigenvalues(dev.x_flat)
    assert eigenvalues.shape == (784,)
    n_near_zero = (eigenvalues < 1e-8).sum().item()
    assert n_near_zero > 0, "expected raw pixel covariance to be rank-deficient (constant border pixels)"
    assert eigenvalues[-1].item() < eigenvalues[0].item() * 1e-6, \
        "expected severe ill-conditioning in the raw (unregularized) covariance"

    epsilon = 1.0
    ridge_cond = ((eigenvalues[0] + epsilon) / (eigenvalues[-1] + epsilon)).item()
    assert torch.isfinite(torch.tensor(ridge_cond))
    assert ridge_cond < 1e8, f"expected ridge-regularized covariance to be well-conditioned, got cond={ridge_cond:.3e}"

    precision = svd_ridge_precision(dev.x_flat, epsilon)
    assert torch.isfinite(precision).all()

    # (Sigma + eps*I) @ precision should be close to the identity. Sigma is formed here only for this independent
    # check, not inside svd_ridge_precision itself -- verifies the SVD-based precision is the correct inverse
    # without relying on svd_ridge_precision's own internal math to also be the test's oracle.
    x_centered = dev.x_flat - dev.x_flat.mean(dim=0, keepdim=True)
    N = dev.x_flat.shape[0]
    Sigma = (x_centered.T @ x_centered) / (N - 1)
    identity_check = (Sigma + epsilon * torch.eye(784)) @ precision
    assert torch.allclose(identity_check, torch.eye(784), atol=1e-6)


def test_covariance_eigenvalues_sorted_descending_and_nonnegative():
    torch.manual_seed(2)
    d = 30
    x_flat = torch.randn(200, d)  # N >= d

    eigenvalues = covariance_eigenvalues(x_flat)

    assert eigenvalues.shape == (d,)
    # descending order
    assert (eigenvalues[:-1] >= eigenvalues[1:] - 1e-9).all()
    # PSD -> eigenvalues should be (numerically) non-negative
    assert eigenvalues.min().item() > -1e-8
    # matches eigvalsh of the explicitly formed covariance, just computed differently
    x_centered = x_flat - x_flat.mean(dim=0, keepdim=True)
    N = x_flat.shape[0]
    Sigma = (x_centered.T @ x_centered) / (N - 1)
    expected = torch.linalg.eigvalsh(Sigma).flip(0)
    assert torch.allclose(eigenvalues, expected, atol=1e-6)


def test_covariance_eigenvalues_on_real_mnist_confirms_rank_deficiency():
    """Direct visibility into the same singularity
    test_svd_ridge_precision_well_conditioned_despite_singular_raw_covariance
    already checks: the smallest eigenvalue of the real pixel covariance
    should be ~0 (border pixels are constant zero across every image)."""
    train = load_mnist(train=True)
    dev = get_dev_subset(train, n=2000, seed=0)

    eigenvalues = covariance_eigenvalues(dev.x_flat)
    assert eigenvalues.shape == (784,)
    assert eigenvalues[0].item() > eigenvalues[-1].item()
    assert eigenvalues[-1].item() < 1e-6, "expected the smallest eigenvalue to be ~0 (rank-deficient covariance)"


def test_mahalanobis_distance_symmetric_and_nonnegative():
    torch.manual_seed(1)
    d = 10
    A = torch.randn(d, d)
    precision = A @ A.T + d * torch.eye(d)
    x = torch.randn(8, d)
    y = torch.randn(8, d)

    d_xy = mahalanobis_distance(x, y, precision)
    d_yx = mahalanobis_distance(y, x, precision)
    assert torch.allclose(d_xy, d_yx, atol=1e-9)
    assert (d_xy >= 0).all()
    assert torch.allclose(mahalanobis_distance(x, x, precision), torch.zeros(8), atol=1e-9)


def test_covariance_eigenbasis_matches_covariance_eigenvalues_and_reconstructs_covariance():
    """covariance_eigenvalues is now a thin wrapper around covariance_eigenbasis -- verify they
    still agree exactly, and that V/eigenvalues together reconstruct the explicitly-formed
    covariance (an independent oracle, not covariance_eigenbasis's own internal math)."""
    torch.manual_seed(3)
    d = 15
    x_flat = torch.randn(300, d)

    V, eigenvalues = covariance_eigenbasis(x_flat)
    assert V.shape == (d, d)
    assert eigenvalues.shape == (d,)
    assert torch.allclose(eigenvalues, covariance_eigenvalues(x_flat))

    x_centered = x_flat - x_flat.mean(dim=0, keepdim=True)
    N = x_flat.shape[0]
    Sigma = (x_centered.T @ x_centered) / (N - 1)
    reconstructed = V @ torch.diag(eigenvalues) @ V.T
    assert torch.allclose(reconstructed, Sigma, atol=1e-6)


def test_truncated_precision_exactly_discards_the_lowest_variance_direction():
    """Hand-checkable example, by construction rather than random data: 3 independent axes with
    exactly-diagonal empirical covariance (paired +/- samples per axis, so cross-axis sample
    covariance is exactly 0, not just small) and a large variance gap between axis 1 and axis 2 --
    a > b >> c. truncated_precision(x_flat, k=2) must discard axis 2 (the smallest-variance
    direction) *entirely*, unlike ridge regularization, which would still give it a small but
    nonzero weight. Checked directly: two points differing ONLY along axis 2 must have Mahalanobis
    distance exactly 0 under this truncated precision."""
    a, b, c = 10.0, 5.0, 0.01
    x_flat = torch.tensor([
        [a, 0.0, 0.0], [-a, 0.0, 0.0],
        [0.0, b, 0.0], [0.0, -b, 0.0],
        [0.0, 0.0, c], [0.0, 0.0, -c],
    ])
    N = x_flat.shape[0]
    # exactly-diagonal covariance by construction -- cross terms average to 0 exactly
    x_centered = x_flat - x_flat.mean(dim=0, keepdim=True)
    Sigma = (x_centered.T @ x_centered) / (N - 1)
    assert torch.allclose(Sigma, torch.diag(torch.diagonal(Sigma)), atol=1e-12)

    precision = truncated_precision(x_flat, k=2)

    # differ only along the discarded axis 2 -- must be exactly 0 distance, not just small
    x = torch.tensor([0.0, 0.0, 5.0])
    y = torch.tensor([0.0, 0.0, -5.0])
    assert mahalanobis_distance(x, y, precision).item() == 0.0

    # differ along a retained axis (axis 0) -- must be nonzero, and match the analytic eigenvalue
    x2 = torch.tensor([1.0, 0.0, 0.0])
    y2 = torch.tensor([0.0, 0.0, 0.0])
    expected_eigenvalue_0 = (2 * a**2) / (N - 1)
    expected_dist = (1.0 / expected_eigenvalue_0) ** 0.5
    assert torch.isclose(mahalanobis_distance(x2, y2, precision), torch.tensor(expected_dist), rtol=1e-6)


def test_truncated_precision_has_rank_k():
    torch.manual_seed(4)
    d = 20
    x_flat = torch.randn(300, d)
    for k in (1, 5, 20):
        precision = truncated_precision(x_flat, k)
        assert precision.shape == (d, d)
        rank = torch.linalg.matrix_rank(precision, atol=1e-8)
        assert rank.item() == k


def test_truncated_precision_at_full_rank_matches_exact_covariance_inverse():
    """At k=d (no truncation), truncated_precision should be the exact inverse of the empirical
    covariance -- checked against torch.linalg.inv on the explicitly-formed covariance, an
    independent oracle. Uses well-separated per-axis variances (not raw pixels, which are
    genuinely singular) so the covariance is actually invertible at full rank."""
    torch.manual_seed(5)
    d = 10
    N = 500
    scales = torch.linspace(1.0, 5.0, d)
    x_flat = torch.randn(N, d) * scales

    precision = truncated_precision(x_flat, k=d)

    x_centered = x_flat - x_flat.mean(dim=0, keepdim=True)
    Sigma = (x_centered.T @ x_centered) / (N - 1)
    expected_precision = torch.linalg.inv(Sigma)
    assert torch.allclose(precision, expected_precision, atol=1e-6)


def test_make_truncated_mahalanobis_distance_fn_matches_manual_composition():
    torch.manual_seed(6)
    d = 12
    x_flat = torch.randn(200, d)
    k = 5

    distance_fn = make_truncated_mahalanobis_distance_fn(x_flat, k)
    precision = truncated_precision(x_flat, k)
    expected_fn = make_mahalanobis_distance_fn(precision)

    x = torch.randn(10, d)
    y = torch.randn(10, d)
    assert torch.allclose(distance_fn(x, y), expected_fn(x, y))


def test_make_truncated_mahalanobis_distance_fn_composes_with_embed_fn_identity_case():
    """degree=1 of elementwise_embedding is the identity (see embeddings.py), so a truncated
    Mahalanobis distance_fn built with embed_fn=elementwise_embedding(., 1) must exactly reproduce
    the raw-pixel-space (embed_fn=None) version -- same checkpoint pattern
    test_embedded_mahalanobis_matches_raw_pixel_at_degree_1 already established for
    svd_ridge_precision."""
    torch.manual_seed(7)
    d = 8
    x_flat = torch.randn(150, d)
    k = 4

    embed_fn = lambda x: elementwise_embedding(x, degree=1)
    raw_fn = make_truncated_mahalanobis_distance_fn(x_flat, k)
    embedded_fn = make_truncated_mahalanobis_distance_fn(x_flat, k, embed_fn=embed_fn)

    x = torch.randn(9, d)
    y = torch.randn(9, d)
    assert torch.allclose(raw_fn(x, y), embedded_fn(x, y))


def test_sweep_k_condition_numbers_matches_manual_ratio():
    torch.manual_seed(8)
    d = 25
    x_flat = torch.randn(400, d)
    k_values = [3, 10, 25]

    eigenvalues = covariance_eigenvalues(x_flat)
    expected = [(eigenvalues[0] / eigenvalues[k - 1]).item() for k in k_values]

    assert sweep_k_condition_numbers(x_flat, k_values) == expected
    # condition number should be non-decreasing as k grows (retaining more, including
    # smaller-eigenvalue directions, can only widen or match the ratio)
    conds = sweep_k_condition_numbers(x_flat, k_values)
    assert conds[0] <= conds[1] <= conds[2]
