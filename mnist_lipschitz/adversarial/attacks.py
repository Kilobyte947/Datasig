"""FGSM (Goodfellow et al. 2015) and PGD (Madry et al. 2018) adversarial example generation.

`model` here is expected to accept flat (N, 784) pixel input and return (N, num_classes) raw
logits -- the same flat-input convention every estimator in mnist_lipschitz/estimators.py uses,
so a trained SmallCNN must be wrapped in `models.FlattenedInputWrapper` before being passed in
here, exactly as layer_decomposition.py does for its own L_full computation.
"""

import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)


def _generator(seed):
    return torch.Generator().manual_seed(seed) if seed is not None else None


def fgsm_attack(model, x, y, epsilon, loss_fn=None):
    """Single-step FGSM: x_adv = clip(x + epsilon * sign(grad_x loss(model(x), y)), 0, 1).

    `loss_fn(logits, y) -> scalar`, defaulting to `torch.nn.functional.cross_entropy` (mean
    reduction) -- the loss FGSM/PGD classically maximize, which is a *different* objective from
    the Euclidean logit-space ratio (`achieved_ratio` in run_experiment.py) this sub-experiment
    ultimately measures; see that module's docstring for why the distinction matters.

    `epsilon=0` returns `x` unchanged (up to the `[0, 1]` clamp, which is a no-op for already
    valid pixel input): `sign(grad) * 0 == 0` regardless of the gradient's value.

    Returns x_adv, clipped to the valid pixel range `[0, 1]` (MNIST pixels live in `[0, 1]`, see
    data.py -- there is no separate epsilon-ball projection here since a single step's maximum
    excursion is already exactly `epsilon` in every coordinate).
    """
    loss_fn = loss_fn or F.cross_entropy
    x = x.detach().clone().requires_grad_(True)
    loss = loss_fn(model(x), y)
    (grad,) = torch.autograd.grad(loss, x)
    x_adv = x.detach() + epsilon * grad.sign()
    return x_adv.clamp(0.0, 1.0)


def pgd_attack(model, x, y, epsilon, alpha, num_steps, num_restarts=1, loss_fn=None, seed=None):
    """Multi-step projected gradient ascent on `loss_fn` (see `fgsm_attack`'s docstring for why
    this is a different objective from the ratio measured afterward), within the L_inf epsilon
    ball around `x` and the valid `[0, 1]` pixel range.

    Restart 0 always starts from the clean `x` itself (no randomization). This is deliberate, not
    an oversight: it guarantees that `pgd_attack(..., num_steps=1, alpha=epsilon, num_restarts=1)`
    performs the exact same single gradient step as `fgsm_attack` -- FGSM is a special case of
    PGD's search, not a separate code path -- which is checked directly in
    `tests/test_attacks.py`, not just asserted. Every restart beyond the first (`num_restarts >
    1`) starts from a point drawn uniformly at random from the epsilon ball (clipped to `[0, 1]`),
    per Madry et al. 2018's recommended practice for escaping local optima a single deterministic
    start might land in.

    Across restarts, the returned `x_adv` is selected **per example** (not per batch): whichever
    restart's final iterate achieves the higher loss for that specific point. The comparison
    always uses plain cross-entropy with `reduction="none"` -- independent of whatever `loss_fn`
    drove the per-step gradient ascent -- since an arbitrary custom `loss_fn` isn't guaranteed to
    support `reduction="none"`, and only a per-example scalar is needed to compare restarts, not
    the attack's actual optimization objective.

    At every step, the gradient-ascent update is clamped first to the L_inf epsilon ball around
    `x` (`[x - epsilon, x + epsilon]`), then to the valid pixel range `[0, 1]` -- both are
    per-coordinate interval constraints, so this successive clamping is exactly equivalent to
    projecting onto their intersection in one step, not an approximation.
    """
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
