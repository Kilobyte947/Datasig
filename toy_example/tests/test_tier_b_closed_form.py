import torch

from toy_example.toy_functions import tier_b_f, tier_b_grad


def _random_components(d, n_ridges, seed):
    g = torch.Generator().manual_seed(seed)
    components = []
    for _ in range(n_ridges):
        w = torch.randn(d, generator=g)
        b = torch.randn(1, generator=g).item()
        A = torch.randn(1, generator=g).item() * 3
        components.append({"w": w, "b": b, "A": A})
    return components


def test_tier_b_grad_matches_autograd_d1():
    components = _random_components(1, 3, seed=0)
    for seed in range(20):
        g = torch.Generator().manual_seed(seed + 500)
        x = torch.randn(1, generator=g).requires_grad_(True)
        y = tier_b_f(x, components)
        (autograd_grad,) = torch.autograd.grad(y, x)
        analytic_grad = tier_b_grad(x.detach(), components)
        assert torch.allclose(autograd_grad, analytic_grad, atol=1e-9, rtol=1e-9)


def test_tier_b_grad_matches_autograd_d2():
    components = _random_components(2, 3, seed=1)
    for seed in range(20):
        g = torch.Generator().manual_seed(seed + 2000)
        x = torch.randn(2, generator=g).requires_grad_(True)
        y = tier_b_f(x, components)
        (autograd_grad,) = torch.autograd.grad(y, x)
        analytic_grad = tier_b_grad(x.detach(), components)
        assert torch.allclose(autograd_grad, analytic_grad, atol=1e-9, rtol=1e-9)


def test_tier_b_grad_batched():
    components = _random_components(2, 3, seed=42)
    x_batch = torch.randn(16, 2)
    x_batch.requires_grad_(True)
    y = tier_b_f(x_batch, components)
    autograd_grad = torch.autograd.grad(y.sum(), x_batch)[0]
    analytic_grad = tier_b_grad(x_batch.detach(), components)
    assert torch.allclose(autograd_grad, analytic_grad, atol=1e-9, rtol=1e-9)


def test_tier_b_f_is_sum_of_components():
    # tier_b_f's own definition (sum of tier_a_f terms) checked against a
    # manual sum, independent of tier_b_grad's correctness above.
    components = _random_components(2, 4, seed=7)
    x = torch.randn(10, 2)
    y = tier_b_f(x, components)

    from toy_example.toy_functions import tier_a_f
    manual = sum(tier_a_f(x, c["w"], c["b"], c["A"]) for c in components)
    assert torch.allclose(y, manual, atol=1e-12)
