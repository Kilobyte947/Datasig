"""Adversarial attack primitives: FGSM, PGD, and a magnitude-matched random
control perturbation - the only three ways an image gets perturbed anywhere
in this package. All three are fresh, standard implementations (Goodfellow
et al. 2015 for FGSM, Madry et al. 2018 for PGD), independent of any model
architecture - they take a trained model and operate on its cross-entropy
loss/gradient, nothing method (A/B/C)-specific.
"""

import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)


def fgsm_attack(model, x, y, epsilon: float) -> torch.Tensor:
    """Single-step FGSM: clip(x + epsilon * sign(grad_x CE(model(x), y)), 0, 1).
    Standard formula (Goodfellow et al. 2015): cross-entropy loss, L_inf
    single step, clip to the valid [0,1] pixel range. `epsilon=0` returns
    `x` unchanged."""
    x = x.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    (grad,) = torch.autograd.grad(loss, x)
    x_adv = x.detach() + epsilon * grad.sign()
    return x_adv.clamp(0.0, 1.0)


def pgd_attack(model, x, y, epsilon: float, num_steps: int = 10, alpha: float = None,
               random_start: bool = True, generator=None) -> torch.Tensor:
    """Standard L_inf PGD (Madry et al. 2018): `num_steps` steps of
    gradient ascent on cross-entropy loss with step size `alpha` (defaults
    to the common `2.5 * epsilon / num_steps` heuristic), each step
    projected back onto the L_inf epsilon-ball around `x` and clamped to
    [0, 1]. Random start within the epsilon-ball by default
    (`random_start=True`) - standard practice, avoids every attack starting
    from the same point on the loss surface. `epsilon=0` returns `x`
    unchanged (the epsilon-ball projection pins every step back to `x`
    itself, and the random-start delta range collapses to a point)."""
    if alpha is None:
        alpha = 2.5 * epsilon / num_steps if num_steps > 0 else 0.0

    x = x.detach()
    if random_start and epsilon > 0:
        delta = torch.empty_like(x).uniform_(-epsilon, epsilon, generator=generator)
        x_adv = (x + delta).clamp(0.0, 1.0)
    else:
        x_adv = x.clone()

    for _ in range(num_steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        (grad,) = torch.autograd.grad(loss, x_adv)
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
        x_adv = x_adv.clamp(0.0, 1.0)

    return x_adv.detach()


def random_noise_perturbation(x, l2_budget: torch.Tensor, generator=None) -> torch.Tensor:
    """Control perturbation: random, non-gradient-directed noise with the
    SAME per-example L2 norm as a given attack's perturbation (`l2_budget`,
    shape (N,)), clipped to [0, 1]. Used to check whether a distance
    measure separates genuinely *adversarial* shifts from equally-large
    but undirected ones, not just any shift of similar magnitude."""
    n = x.shape[0]
    flat_shape = x.reshape(n, -1).shape
    noise = torch.randn(flat_shape, generator=generator)
    noise = noise / noise.norm(dim=1, keepdim=True) * l2_budget.unsqueeze(1)
    x_control = x.reshape(n, -1) + noise
    return x_control.reshape(x.shape).clamp(0.0, 1.0)
