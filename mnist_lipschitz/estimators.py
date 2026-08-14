"""The three Lipschitz sub-methods (pairwise, local-perturbation, gradient-norm), generalized from toy_lipschitz/estimators.py to operate
on a classifier's margin_fn over 784-d MNIST inputs, under a pluggable distance metric (Euclidean or Mahalanobis, see distance.py). 
Kept as three separate functions throughout.
"""

import torch

from mnist_lipschitz.distance import euclidean_distance_fn

torch.set_default_dtype(torch.float64)

def _generator(seed):
    return torch.Generator().manual_seed(seed) if seed is not None else None


def pairwise_lipschitz(model, x_batch, y_batch, margin_fn, distance_fn=euclidean_distance_fn,
                        max_pairs=None, seed=None):
    """L_hat = max_{i != j} |margin(x_i) - margin(x_j)| / distance_fn(x_i, x_j).

    All N*(N-1)/2 pairs by default; if max_pairs is set and exceeded,
    subsamples pairs randomly instead. At MNIST scale N is kept modest
    (a few hundred points), so the full pair set (~20-45k) is cheap
    enough to score exhaustively -- max_pairs is a safety valve, not the
    default path.

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


def ratio_and_components_for_pairs(model, x_batch, y_batch, margin_fn, distance_fn, ii, jj):
    """|margin_i - margin_j| / distance_fn(x_i, x_j) for the given pairs, 
    plus its two raw components (dist, margin_diff) - for inspecting why a pair scored high: 
    far apart in the metric, vs. genuinely divergent margins. 
    Returns (ratio, dist, margin_diff), each shape (len(ii),).
    """
    with torch.no_grad():
        margins = margin_fn(model, x_batch, y_batch)
        dist = distance_fn(x_batch[ii], x_batch[jj])
    margin_diff = (margins[ii] - margins[jj]).abs()
    valid = dist > 1e-12
    ratio = torch.where(valid, margin_diff / dist.clamp_min(1e-12), torch.zeros_like(dist))
    return ratio, dist, margin_diff


def ratio_for_pairs(model, x_batch, y_batch, margin_fn, distance_fn, ii, jj):
    """Same as ratio_and_components_for_pairs, ratio only. 
    Shared core of pairwise_lipschitz/pairwise_lipschitz_all - takes any ii/jj pairing (not just all-i<j), 
    so callers with a different pairing scheme (a subsample, or external nearest-neighbor pairs) reuse this instead of
    recomputing margins/distances themselves.
    """
    ratio, _, _ = ratio_and_components_for_pairs(model, x_batch, y_batch, margin_fn, distance_fn, ii, jj)
    return ratio


def pairwise_lipschitz_all(model, x_batch, y_batch, margin_fn, distance_fn=euclidean_distance_fn,
                            max_pairs=None, seed=None):
    """Same as pairwise_lipschitz, but returns the full (num_pairs,) ratio array and its ii/jj indices instead 
    of collapsing to the max - needed for the ratio distribution (histogram, or picking out specific high-ratio pairs), 
    not just its extreme value.

    Returns (ratio, ii, jj): ratio[k] is the ratio for (x_batch[ii[k]], x_batch[jj[k]]).
    """
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

    ratio = ratio_for_pairs(model, x_batch, y_batch, margin_fn, distance_fn, ii, jj)
    return ratio, ii, jj


def local_perturbation_lipschitz(model, x_batch, y_batch, margin_fn, distance_fn=euclidean_distance_fn,
                                  radius=1.0, n_directions=40, seed=None):
    """Finite-difference local estimate per point: sample n_directions random unit vectors in raw pixel space, scale to `radius`, 
    perturb x -> x+delta (true label held fixed), take max_direction |margin(x)-margin(x+delta)| / distance_fn(x, x+delta).

    Perturbation directions are always sampled in raw pixel space regardless of distance_fn - only the denominator changes 
    between Euclidean and Mahalanobis, not the sampling itself.

    Returns the full (N,) array of per-point estimates, not just the max.
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
    """Autograd exact/infinitesimal estimate per point: the dual norm of
    grad(margin_fn) wrt x.

    Euclidean (precision=None): ||grad||_2.
    Mahalanobis (precision=P=Sigma^-1): sqrt(grad^T Sigma grad) - note this uses Sigma = P^-1, not P itself; 
    inverted back here for interface consistency with pairwise_lipschitz/local_perturbation_lipschitz, which take `precision` directly. 

    Returns the full (N,) array of per-point estimates.
    """
    x = x_batch.clone().requires_grad_(True)
    margins = margin_fn(model, x, y_batch)
    (grad,) = torch.autograd.grad(margins.sum(), x)

    if precision is None:
        return grad.norm(p=2, dim=-1).detach()

    sigma = torch.linalg.inv(precision)  # P^{-1}, i.e. Sigma itself -- see docstring
    quad = torch.einsum("ni,ij,nj->n", grad, sigma, grad)
    return quad.clamp_min(0.0).sqrt().detach()
