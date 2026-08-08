import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from toy_lipschitz.toy_functions import tier_a_f, tier_a_grad, tier_a_true_L


def _random_case(d, seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(d, generator=g)
    b = torch.randn(1, generator=g).item()
    A = torch.randn(1, generator=g).item() * 3
    x = torch.randn(d, generator=g)
    return x, w, b, A


def test_tier_a_grad_matches_autograd_d1():
    for seed in range(20):
        x, w, b, A = _random_case(1, seed)
        x = x.clone().requires_grad_(True)
        y = tier_a_f(x, w, b, A)
        (autograd_grad,) = torch.autograd.grad(y, x)
        analytic_grad = tier_a_grad(x.detach(), w, b, A)
        assert torch.allclose(autograd_grad, analytic_grad, atol=1e-9, rtol=1e-9)


def test_tier_a_grad_matches_autograd_d2():
    for seed in range(20):
        x, w, b, A = _random_case(2, seed + 1000)
        x = x.clone().requires_grad_(True)
        y = tier_a_f(x, w, b, A)
        (autograd_grad,) = torch.autograd.grad(y, x)
        analytic_grad = tier_a_grad(x.detach(), w, b, A)
        assert torch.allclose(autograd_grad, analytic_grad, atol=1e-9, rtol=1e-9)


def test_tier_a_grad_batched():
    x, w, b, A = _random_case(2, 42)
    x_batch = torch.randn(16, 2)
    x_batch.requires_grad_(True)
    y = tier_a_f(x_batch, w, b, A)
    autograd_grad = torch.autograd.grad(y.sum(), x_batch)[0]
    analytic_grad = tier_a_grad(x_batch.detach(), w, b, A)
    assert torch.allclose(autograd_grad, analytic_grad, atol=1e-9, rtol=1e-9)


def test_tier_a_true_L_matches_max_grad_norm_on_hyperplane():
    w = torch.tensor([3.0, -4.0])
    b = 0.0
    A = 2.0
    x0 = torch.zeros(2)  # on the hyperplane w^T x + b = 0, where |grad| is maximal
    g = tier_a_grad(x0, w, b, A)
    L_star = tier_a_true_L(w, A, norm="l2")
    assert abs(float(g.norm(p=2)) - L_star) < 1e-9
