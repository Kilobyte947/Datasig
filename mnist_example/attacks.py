"""FGSM (Goodfellow et al. 2015) and PGD (Madry et al. 2018) adversarial example generation.

model must accept flat (N, 784) pixel input and return (N, num_classes) logits — wrap a trained
SmallCNN in models.FlattenedInputWrapper first.
"""

import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)


def _generator(seed):
    """A seeded torch.Generator, or None if seed is None."""
    return torch.Generator().manual_seed(seed) if seed is not None else None


def fgsm_attack(model, x, y, epsilon, loss_fn=None):
    """Single-step FGSM: x_adv = clip(x + epsilon * sign(grad_x loss(model(x), y)), 0, 1).
    loss_fn defaults to cross-entropy. Returns x_adv, clipped to [0, 1]."""
    loss_fn = loss_fn or F.cross_entropy
    x = x.detach().clone().requires_grad_(True)
    loss = loss_fn(model(x), y)
    (grad,) = torch.autograd.grad(loss, x)
    x_adv = x.detach() + epsilon * grad.sign()
    return x_adv.clamp(0.0, 1.0)


def pgd_attack(model, x, y, epsilon, alpha, num_steps, num_restarts=1, loss_fn=None, seed=None):
    """Multi-step projected gradient ascent on loss_fn, within the L-infinity epsilon ball around x
    and the valid [0, 1] pixel range. The first restart starts from x itself; further restarts start
    from a random point in the epsilon ball. Returns the per-example best x_adv across restarts, by
    cross-entropy loss."""
    loss_fn = loss_fn or F.cross_entropy
    generator = _generator(seed)
    x = x.detach()

    best_x_adv = None
    best_loss = None
    for restart in range(num_restarts):
        if restart == 0:
            x_adv = x.clone()
        else:
            if generator is not None:
                noise = (torch.rand(x.shape, generator=generator) * 2 - 1) * epsilon
            else:
                noise = (torch.rand_like(x) * 2 - 1) * epsilon
            x_adv = (x + noise).clamp(0.0, 1.0)

        for _ in range(num_steps):
            x_adv = x_adv.detach().requires_grad_(True)
            loss = loss_fn(model(x_adv), y)
            (grad,) = torch.autograd.grad(loss, x_adv)
            x_adv = x_adv.detach() + alpha * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)  # L_inf ball projection
            x_adv = x_adv.clamp(0.0, 1.0)

        with torch.no_grad():
            per_example_loss = F.cross_entropy(model(x_adv), y, reduction="none")

        if best_x_adv is None:
            best_x_adv, best_loss = x_adv, per_example_loss
        else:
            improved = per_example_loss > best_loss
            best_x_adv = torch.where(improved.unsqueeze(-1), x_adv, best_x_adv)
            best_loss = torch.where(improved, per_example_loss, best_loss)

    return best_x_adv
