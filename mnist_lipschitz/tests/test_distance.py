import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mnist_lipschitz.data import load_mnist, get_dev_subset
from mnist_lipschitz.distance import pixel_covariance, ridge_precision, mahalanobis_distance


def test_mahalanobis_converges_to_scaled_euclidean_as_epsilon_grows():
    """As epsilon -> large, Sigma + epsilon*I ~= epsilon*I, so precision ~=
    (1/epsilon)*I and mahalanobis_distance(x,y) ~= ||x-y||_2 / sqrt(epsilon).
    This is the direct analogue of Experiment 1's degree-1 closed-form
    check, and the main defense against a sign/inversion bug here, since
    there's no ground-truth L* at MNIST scale to catch one otherwise."""
    torch.manual_seed(0)
    d = 20
    Sigma = torch.randn(d, d)
    Sigma = Sigma @ Sigma.T  # PSD, not necessarily well-conditioned

    epsilon = 1e6
    precision = ridge_precision(Sigma, epsilon)

    x = torch.randn(15, d)
    y = torch.randn(15, d)
    maha = mahalanobis_distance(x, y, precision)
    euclidean = (x - y).norm(p=2, dim=-1)

    expected_ratio = 1.0 / (epsilon ** 0.5)
    actual_ratio = maha / euclidean
    assert torch.allclose(actual_ratio, torch.full_like(actual_ratio, expected_ratio), rtol=1e-3)


def test_ridge_precision_well_conditioned_despite_singular_raw_covariance():
    """Real MNIST pixel covariance is expected to be (near-)singular --
    border pixels are 0 in every image, so their variance and covariance
    with everything else is exactly 0. Verify this directly, then verify
    the ridge-regularized version is well-conditioned."""
    train = load_mnist(train=True)
    dev = get_dev_subset(train, n=2000, seed=0)
    Sigma = pixel_covariance(dev.x_flat)

    assert Sigma.shape == (784, 784)

    raw_rank = torch.linalg.matrix_rank(Sigma).item()
    assert raw_rank < 784, "expected raw pixel covariance to be rank-deficient (constant border pixels)"

    raw_cond = torch.linalg.cond(Sigma).item()
    assert raw_cond > 1e10, f"expected raw covariance to be severely ill-conditioned, got cond={raw_cond:.3e}"

    epsilon = 1.0
    Sigma_ridge = Sigma + epsilon * torch.eye(784)
    ridge_cond = torch.linalg.cond(Sigma_ridge).item()
    assert torch.isfinite(torch.tensor(ridge_cond))
    assert ridge_cond < 1e8, f"expected ridge-regularized covariance to be well-conditioned, got cond={ridge_cond:.3e}"

    precision = ridge_precision(Sigma, epsilon)
    assert torch.isfinite(precision).all()
    # (Sigma + eps*I) @ precision should be close to the identity.
    identity_check = Sigma_ridge @ precision
    assert torch.allclose(identity_check, torch.eye(784), atol=1e-6)


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
