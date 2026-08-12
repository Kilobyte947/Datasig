"""The three Lipschitz sub-methods, generalized from toy_lipschitz/estimators.py
to operate on a classifier's margin_fn over high-dimensional (784-d) MNIST
inputs, under a pluggable distance metric (plain Euclidean or ridge-regularized
Mahalanobis, see distance.py).

Kept as three clearly separate functions throughout, exactly as in
Experiment 1 -- local-perturbation and gradient-norm are related but
distinct quantities and must never be conflated into a single "local
estimate" without labeling which one it is.
"""

import torch

torch.set_default_dtype(torch.float64)


def _generator(seed):
    return torch.Generator().manual_seed(seed) if seed is not None else None


def euclidean_distance_fn(x, y):
    """Plain L2 distance, row-wise if x/y are batches of the same length.
    The default `distance_fn` for all three estimators below."""
    return (x - y).norm(p=2, dim=-1)


def pairwise_lipschitz(model, x_batch, y_batch, margin_fn, distance_fn=euclidean_distance_fn,
                        max_pairs=None, seed=None):
    """L_hat = max_{i != j} |margin(x_i) - margin(x_j)| / distance_fn(x_i, x_j).

    O(N^2) pairs for N points. At MNIST scale we keep N modest (a few
    hundred points, passed in by the caller) rather than defaulting to
    random pair subsampling -- with N ~ 200-300 the full pair set is only
    ~20-45k pairs, cheap to score directly and exhaustive rather than a
    random sample of it. `max_pairs` is still supported (matching
    toy_lipschitz's interface) as a safety valve if a caller passes a
    larger N than intended.

    Returns (L_hat, i_argmax, j_argmax).
    """
    with torch.no_grad():
        margins = margin_fn(model, x_batch, y_batch)

    N = x_batch.shape[0]
    total_pairs = N * (N - 1) // 2

    if max_pairs is None or total_pairs <= max_pairs:
        ii, jj = torch.triu_indices(N, N, offset=1)
    else:
        generator = _generator(seed)
        if generator is not None:
            ii = torch.randint(0, N, (max_pairs,), generator=generator)
            jj = torch.randint(0, N, (max_pairs,), generator=generator)
        else:
            ii = torch.randint(0, N, (max_pairs,))
            jj = torch.randint(0, N, (max_pairs,))
        keep = ii != jj
        ii, jj = ii[keep], jj[keep]

    with torch.no_grad():
        dist = distance_fn(x_batch[ii], x_batch[jj])
    dy = (margins[ii] - margins[jj]).abs()

    valid = dist > 1e-12
    ratio = torch.where(valid, dy / dist.clamp_min(1e-12), torch.zeros_like(dist))
    L_hat, idx = ratio.max(dim=0)
    return L_hat.item(), ii[idx].item(), jj[idx].item()


def local_perturbation_lipschitz(model, x_batch, y_batch, margin_fn, distance_fn=euclidean_distance_fn,
                                  radius=1.0, n_directions=20, seed=None):
    """Finite-difference local estimate, per point in x_batch: sample
    `n_directions` random unit vectors in raw 784-d pixel space, scale to
    length `radius`, perturb x -> x+delta (label held fixed at the
    original true class y, since this measures how fast the *true-class*
    margin degrades under a small push), and take
    max_direction |margin(x)-margin(x+delta)| / distance_fn(x, x+delta).

    Directions are always sampled in raw pixel space regardless of the
    distance metric in use -- only distance_fn (the denominator) changes
    between Euclidean and Mahalanobis; the sampling distribution does not
    (this is deliberate, see README's Design decisions section).

    Returns the full (N,) array of per-point local estimates, not just the
    overall max -- needed for the submethod-agreement plot and consistent
    with toy_lipschitz's `_grid` convention.
    """
    generator = _generator(seed)
    N, d = x_batch.shape

    with torch.no_grad():
        margin_x = margin_fn(model, x_batch, y_batch)

    best = torch.full((N,), -float("inf"))
    for _ in range(n_directions):
        if generator is not None:
            raw = torch.randn(N, d, generator=generator)
        else:
            raw = torch.randn(N, d)
        unit = raw / raw.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        delta = unit * radius
        x_prime = x_batch + delta

        with torch.no_grad():
            margin_xp = margin_fn(model, x_prime, y_batch)
            dist = distance_fn(x_batch, x_prime)

        ratio = (margin_xp - margin_x).abs() / dist.clamp_min(1e-12)
        best = torch.maximum(best, ratio)

    return best


def gradient_norm_estimate(model, x_batch, y_batch, margin_fn, precision=None):
    """Autograd-based LOCAL/infinitesimal estimate, per point: the dual norm
    of grad(margin_fn) w.r.t. x, under the metric in use.

    Plain Euclidean (precision=None): ||grad||_2.

    Mahalanobis with precision matrix P (P = Sigma^{-1}, the same P used
    directly as the quadratic form in distance_fn's Mahalanobis distance):
    the correct dual-norm expression is sqrt(grad^T P^{-1} grad) =
    sqrt(grad^T Sigma grad) -- note this uses Sigma = P^{-1}, NOT P itself;
    inverting P a second time here (rather than reusing Sigma from
    distance.py directly) is deliberate for interface consistency (every
    estimator here takes `precision`, matching pairwise_lipschitz's and
    local_perturbation_lipschitz's Mahalanobis argument), and is cheap: a
    784x784 dense inverse takes ~10ms, negligible next to model
    training/evaluation.

    Returns the full (N,) array of per-point gradient-norm estimates.
    """
    x = x_batch.clone().requires_grad_(True)
    margins = margin_fn(model, x, y_batch)
    (grad,) = torch.autograd.grad(margins.sum(), x)

    if precision is None:
        return grad.norm(p=2, dim=-1).detach()

    sigma = torch.linalg.inv(precision)  # P^{-1}, i.e. Sigma itself -- see docstring
    quad = torch.einsum("ni,ij,nj->n", grad, sigma, grad)
    return quad.clamp_min(0.0).sqrt().detach()
