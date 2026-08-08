"""Ground-truth functions f* for the toy Lipschitz experiment.

Tier A: a single ridge f*(x) = A * tanh(w^T x + b), with a closed-form
Lipschitz constant. Tier B: a sum of such ridges, whose true Lipschitz
constant has no closed form and is estimated via dense grid search plus
gradient-ascent refinement.
"""

import torch

torch.set_default_dtype(torch.float64)


def _as_tensor(v):
    return torch.as_tensor(v, dtype=torch.get_default_dtype())


def tier_a_f(x, w, b, A):
    """f*(x) = A * tanh(w^T x + b). x: (..., d), w: (d,)."""
    x = _as_tensor(x)
    w = _as_tensor(w)
    z = (x * w).sum(dim=-1) + b
    return A * torch.tanh(z)


def tier_a_grad(x, w, b, A):
    """Analytic gradient: A * w * (1 - tanh(w^T x + b)^2). Shape (..., d)."""
    x = _as_tensor(x)
    w = _as_tensor(w)
    z = (x * w).sum(dim=-1) + b
    scale = A * (1 - torch.tanh(z) ** 2)
    return scale.unsqueeze(-1) * w


def tier_a_true_L(w, A, norm="l2"):
    """Return A * ||w||_2 (or ||w||_1 for norm='l1').

    Note: the norm='l1' case is A * ||w||_1 as a simplified (non-dual)
    convention, not the true L1-distance dual norm A * ||w||_inf. All
    correctness checkpoints in this project use norm='l2', where the
    dual-norm identity holds exactly (Cauchy-Schwarz).
    """
    w = _as_tensor(w)
    if norm == "l2":
        return float(A * w.norm(p=2))
    elif norm == "l1":
        return float(A * w.norm(p=1))
    raise ValueError(f"unknown norm: {norm}")


def tier_b_f(x, components):
    """f*(x) = sum_k A_k * tanh(w_k^T x + b_k).

    components: list of dicts {"w":..., "b":..., "A":...}
    """
    total = 0.0
    for c in components:
        total = total + tier_a_f(x, c["w"], c["b"], c["A"])
    return total


def tier_b_grad(x, components):
    """Analytic gradient via linearity of the sum. Shape (..., d)."""
    total = None
    for c in components:
        g = tier_a_grad(x, c["w"], c["b"], c["A"])
        total = g if total is None else total + g
    return total


def _grad_norm(g, norm):
    if norm == "l2":
        return g.norm(p=2, dim=-1)
    elif norm == "l1":
        # dual norm of l1 distance is l_inf on the gradient side
        return g.norm(p=float("inf"), dim=-1)
    raise ValueError(f"unknown norm: {norm}")


def _dense_grid(domain, d, grid_points):
    lo, hi = domain
    if d == 1:
        xs = torch.linspace(lo, hi, grid_points).unsqueeze(-1)
    elif d == 2:
        n_side = max(2, int(round(grid_points ** 0.5)))
        g1 = torch.linspace(lo, hi, n_side)
        g2 = torch.linspace(lo, hi, n_side)
        gg1, gg2 = torch.meshgrid(g1, g2, indexing="ij")
        xs = torch.stack([gg1.reshape(-1), gg2.reshape(-1)], dim=-1)
    else:
        raise NotImplementedError("tier_b_true_L only supports d in {1, 2}")
    return xs


def tier_b_true_L(components, domain, norm="l2", grid_points=200000, n_restarts=50, seed=None):
    """Numerically-exact-enough estimate of the true global Lipschitz constant.

    1. Evaluate ||grad f*(x)|| on a dense grid over `domain`, take the max
       as a first estimate.
    2. Refine with gradient ascent on ||grad f*(x)|| from `n_restarts`
       random starts (torch.optim.LBFGS), clamped to stay inside `domain`.
    3. Return (L_star, x_star): the max over grid search and all restarts,
       plus the argmax location.
    """
    d = _as_tensor(components[0]["w"]).numel()
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    lo, hi = domain

    xs = _dense_grid(domain, d, grid_points)
    grid_norms = _grad_norm(tier_b_grad(xs, components), norm)
    grid_max, grid_idx = grid_norms.max(dim=0)
    best_L = grid_max.item()
    best_x = xs[grid_idx].clone()

    for _ in range(n_restarts):
        if generator is not None:
            x0 = torch.rand(d, generator=generator) * (hi - lo) + lo
        else:
            x0 = torch.rand(d) * (hi - lo) + lo
        x0 = x0.clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS([x0], lr=0.5, max_iter=20, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            g = tier_b_grad(x0, components)
            loss = -_grad_norm(g.unsqueeze(0), norm).squeeze(0)
            loss.backward()
            return loss

        for _ in range(5):
            optimizer.step(closure)
            with torch.no_grad():
                x0.clamp_(lo, hi)

        with torch.no_grad():
            final_norm = _grad_norm(tier_b_grad(x0, components).unsqueeze(0), norm).squeeze(0).item()
        if final_norm > best_L:
            best_L = final_norm
            best_x = x0.detach().clone()

    return best_L, best_x
