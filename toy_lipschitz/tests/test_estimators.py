import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from toy_lipschitz.toy_functions import tier_a_f, tier_a_true_L
from toy_lipschitz.estimators import (
    pairwise_lipschitz,
    local_perturbation_lipschitz,
    gradient_norm_estimate,
    gradient_norm_estimate_grid,
    local_perturbation_lipschitz_grid,
)


W = torch.tensor([3.0, -4.0])
B = 0.0  # so x0=(0,0) lies exactly on the hyperplane w^T x + b = 0
A = 2.0
TRUE_L = tier_a_true_L(W, A, norm="l2")  # A * ||w||_2 = 10.0


def test_pairwise_lipschitz_approaches_true_L_near_hyperplane():
    x0 = torch.zeros(2)
    generator = torch.Generator().manual_seed(0)
    N = 2000
    deltas = (torch.rand(N, 2, generator=generator) * 2 - 1) * 0.3
    x = x0 + deltas
    y = tier_a_f(x, W, B, A)

    L_hat, i, j = pairwise_lipschitz(x, y, norm="l2")

    rel_err = abs(L_hat - TRUE_L) / TRUE_L
    assert rel_err < 0.10
    assert L_hat <= TRUE_L * 1.01  # sup of |Δf|/|Δx| cannot exceed the true Lipschitz constant (up to fp slack)


def test_pairwise_lipschitz_improves_with_more_samples():
    x0 = torch.zeros(2)

    def rel_err(N, seed):
        g = torch.Generator().manual_seed(seed)
        deltas = (torch.rand(N, 2, generator=g) * 2 - 1) * 0.3
        x = x0 + deltas
        y = tier_a_f(x, W, B, A)
        L_hat, _, _ = pairwise_lipschitz(x, y, norm="l2")
        return abs(L_hat - TRUE_L) / TRUE_L

    assert rel_err(2000, seed=1) <= rel_err(50, seed=1) + 1e-9


def test_pairwise_lipschitz_max_pairs_subsampling_runs_and_is_reasonable():
    x0 = torch.zeros(2)
    generator = torch.Generator().manual_seed(2)
    N = 500
    deltas = (torch.rand(N, 2, generator=generator) * 2 - 1) * 0.3
    x = x0 + deltas
    y = tier_a_f(x, W, B, A)

    L_hat, i, j = pairwise_lipschitz(x, y, norm="l2", max_pairs=2000, seed=3)
    assert L_hat <= TRUE_L * 1.01
    assert i != j


def test_gradient_norm_estimate_matches_true_L_on_hyperplane():
    f = lambda x: tier_a_f(x, W, B, A)
    x0 = torch.zeros(2)
    L_hat = gradient_norm_estimate(f, x0, norm="l2")
    assert abs(L_hat - TRUE_L) < 1e-9


def test_gradient_norm_estimate_grid_matches_pointwise():
    f = lambda x: tier_a_f(x, W, B, A)
    X = torch.tensor([[0.0, 0.0], [1.0, -1.0], [-2.0, 0.5]])
    grid_vals = gradient_norm_estimate_grid(f, X, norm="l2")
    for k in range(X.shape[0]):
        pointwise = gradient_norm_estimate(f, X[k], norm="l2")
        assert abs(grid_vals[k].item() - pointwise) < 1e-9


def test_local_perturbation_lipschitz_bounded_by_true_L_near_hyperplane():
    f = lambda x: tier_a_f(x, W, B, A)
    x0 = torch.zeros(2)
    L_hat, x_prime = local_perturbation_lipschitz(f, x0, radius=0.3, n_samples=500, norm="l2", seed=4)
    assert L_hat <= TRUE_L * 1.01
    assert L_hat > 0.5 * TRUE_L  # should be reasonably close since x0 is the true argmax


def test_local_perturbation_lipschitz_grid_matches_length():
    f = lambda x: tier_a_f(x, W, B, A)
    X = torch.tensor([[0.0, 0.0], [1.0, -1.0], [-2.0, 0.5]])
    vals = local_perturbation_lipschitz_grid(f, X, radius=0.3, n_samples=50, norm="l2", seed=5)
    assert vals.shape == (3,)
    assert (vals >= 0).all()
