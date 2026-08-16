import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mnist_lipschitz.data import load_mnist, get_dev_subset
from mnist_lipschitz.distance import svd_ridge_precision, mahalanobis_distance, covariance_eigenvalues


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
