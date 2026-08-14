"""The three Lipschitz estimators: 
1. pairwise (data or model), 
2. local perturbation (finite-difference)
3. gradient norm (infinitesimal/autograd)
These are kept as clearly separate functions throughout and are related but distinct quantities. 
"""

import torch

torch.set_default_dtype(torch.float64)

def _generator(seed):
    return torch.Generator().manual_seed(seed) if seed is not None else None

def _as_tensor(v):
    return torch.as_tensor(v, dtype=torch.get_default_dtype())

def _norm(v, norm, dim=-1):
    """Ordinary straight-line distance between two points, sqrt(sum((x_i - x_j)^2)). Used by pairwise_lipschitz and local_sample_density."""
    if norm == "l2":
        return v.norm(p=2, dim=dim)
    elif norm == "l1":
        return v.norm(p=1, dim=dim)
    raise ValueError(f"unknown norm: {norm}")


def _mahalanobis_dist(diff, precision):
    """Weighted distance in embedded feature space: reweights each direction by `precision` (inverse covariance) instead of treating all
    directions equally like plain Euclidean distance -- movement in directions the data naturally varies a lot counts for less than
    movement in directions it barely varies at all. Reduces to plain Euclidean distance if `precision` is the identity matrix.

    diff: (..., k) embedded-space differences (not raw x).
    precision: (k, k), typically an inverse covariance.
    Returns sqrt(diff^T @ precision @ diff), clamped at 0 first (guards against floating-point rounding pushing a near-zero value negative).
    """
    quad = torch.einsum("...i,ij,...j->...", diff, precision, diff)
    return quad.clamp_min(0.0).sqrt()


def pairwise_lipschitz(x, y, norm="l2", max_pairs=None, seed=None, embed_fn=None, precision=None):
    """Takes a pair of points and computes |y_i - y_j| / distance(x_i, x_j), for each pair, then reports the biggest ratio found:

    L_hat = max_{i != j} |y_i - y_j| / d(x_i, x_j) 

    x: (N, d) tensor/array, y: (N,) tensor/array (scalar outputs only). This is GLOBAL

    If max_pairs is set and the number of unordered pairs exceeds it, subsample pairs rather than computing all N*(N-1)/2.

    distance d(x_i, x_j) is computed one of two ways:
      - Default (embed_fn/precision not given): plain distance in raw x (Euclidean if norm="l2", or l1 if norm="l1").
      - If embed_fn and precision are both given: raw x is first transformed via phi = embed_fn(x) (e.g. into polynomial features),
        then d is the Mahalanobis distance between phi_i and phi_j -- sqrt((phi_i - phi_j)^T @ precision @ (phi_i - phi_j)) -- using
        `precision` to weight some directions more than others, instead of treating every direction equally the way plain distance does.
        `norm` is ignored in this case.

    Return (L_hat, i_argmax, j_argmax).
    """
    x = _as_tensor(x)
    y = _as_tensor(y).reshape(-1)
    N = x.shape[0]
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

    if embed_fn is not None and precision is not None:
        phi = embed_fn(x)
        dist = _mahalanobis_dist(phi[ii] - phi[jj], precision)
    else:
        diff = x[ii] - x[jj]
        dist = _norm(diff, norm)

    dy = (y[ii] - y[jj]).abs()

    valid = dist > 1e-12
    ratio = torch.where(valid, dy / dist.clamp_min(1e-12), torch.zeros_like(dist))
    L_hat, idx = ratio.max(dim=0)
    return L_hat.item(), ii[idx].item(), jj[idx].item()


def _sample_in_ball(center, radius, n_samples, generator, max_attempts=10000):
    d = center.shape[0]
    collected = []
    n_collected = 0
    attempts = 0
    batch = max(n_samples * 4, 256)
    while n_collected < n_samples and attempts < max_attempts:
        attempts += 1
        if generator is not None:
            u = (torch.rand(batch, d, generator=generator) * 2 - 1) * radius
        else:
            u = (torch.rand(batch, d) * 2 - 1) * radius
        dist = u.norm(p=2, dim=-1)
        mask = (dist <= radius) & (dist > 1e-12)
        keep = u[mask]
        if keep.shape[0] > 0:
            collected.append(keep)
            n_collected += keep.shape[0]
    if n_collected < n_samples:
        raise RuntimeError(f"could not sample {n_samples} points in ball after {max_attempts} batches")
    return torch.cat(collected, dim=0)[:n_samples]


def local_perturbation_lipschitz(f, x0, radius, n_samples, norm="l2", seed=None, embed_fn=None, precision=None):
    """Sample n_samples points x' within an L2 ball of `radius` around x0, evaluate f at x0 and each x', 
    return max |f(x')-f(x0)| / d(x0,x') and the maximizing x'. 
    f must accept a batch of points (N, d) and return a batch of scalars (N,).

    d(x0,x') is one of two things:
      - Default: plain distance in raw x (per `norm`).
      - If embed_fn AND precision are both given: Mahalanobis distance
        between embed_fn(x0) and embed_fn(x'), weighted by `precision`
        (same convention as pairwise_lipschitz). `norm` is ignored.
    """

    x0 = _as_tensor(x0).reshape(-1)
    generator = _generator(seed)
    deltas = _sample_in_ball(x0, radius, n_samples, generator)
    x_primes = x0.unsqueeze(0) + deltas

    with torch.no_grad():
        f_x0 = f(x0.unsqueeze(0)).reshape(-1)[0]
        f_xp = f(x_primes).reshape(-1)

    if embed_fn is not None and precision is not None:
        phi_diff = embed_fn(x_primes) - embed_fn(x0.unsqueeze(0))
        dist = _mahalanobis_dist(phi_diff, precision)
    else:
        dist = _norm(deltas, norm)

    ratio = (f_xp - f_x0).abs() / dist.clamp_min(1e-12)
    L_hat, idx = ratio.max(dim=0)
    return L_hat.item(), x_primes[idx]


def gradient_norm_estimate(f, x0, norm="l2"):
    """A separate, exact-derivative quantity which gives autograd-based local/infinitesimal estimate (true slope at at that location, no perturbation needed).
    Requires_grad on x0, forward pass,  backward, return ||grad||_2 (or the dual norm, l_inf, for norm='l1').
    A separate, exact-derivative quantity -- do not conflate with local_perturbation_lipschitz above, which is finite-difference (perturb and measure), not exact.
    """
    x0 = _as_tensor(x0).reshape(-1).clone().requires_grad_(True)
    y = f(x0.unsqueeze(0)).reshape(-1)[0]
    (grad,) = torch.autograd.grad(y, x0)
    if norm == "l2":
        return grad.norm(p=2).item()
    elif norm == "l1":
        return grad.norm(p=float("inf")).item()
    raise ValueError(f"unknown norm: {norm}")


def gradient_norm_estimate_grid(f, X, norm="l2"):
    """Vectorized gradient_norm_estimate over every row of X (N, d).

    Valid whenever f applies independently per batch row (true of tier_b_f and of standard feedforward models like TinyMLP): 
    summing the outputs before calling autograd.grad decouples into per-row gradients. 
    Returns the full (N,) array of per-point estimates, not just the max -- needed for the sweep/heatmap plots.
    """
    X = _as_tensor(X).clone().requires_grad_(True)
    y = f(X).reshape(-1)
    (grad,) = torch.autograd.grad(y.sum(), X)
    if norm == "l2":
        return grad.norm(p=2, dim=-1).detach()
    elif norm == "l1":
        return grad.norm(p=float("inf"), dim=-1).detach()
    raise ValueError(f"unknown norm: {norm}")


def local_perturbation_lipschitz_grid(f, X, radius, n_samples, norm="l2", seed=None, embed_fn=None, precision=None):
    """local_perturbation_lipschitz evaluated at every row of X (N, d).
    Returns the full (N,) array of L_hat values, not just the max.
    embed_fn/precision, if given, are passed through unchanged (seelocal_perturbation_lipschitz).
    """
    X = _as_tensor(X)
    results = torch.empty(X.shape[0])
    for i in range(X.shape[0]):
        point_seed = None if seed is None else seed + i
        L_hat, _ = local_perturbation_lipschitz(f, X[i], radius, n_samples, norm=norm, seed=point_seed,
                                                  embed_fn=embed_fn, precision=precision)
        results[i] = L_hat
    return results


def local_sample_density(x_query, x_train, radius, norm="l2"):
    """This is not a Lipschitz quantity. It is a convergence coverage check (a count).
    For each query point, count training points within `radius` (per `norm`). 
    A low local Lipschitz estimate can mean "genuinely smooth here" or  just "barely sampled here," and this is what tells the two apart.
    """
    x_query = _as_tensor(x_query)
    x_train = _as_tensor(x_train)
    diff = x_query.unsqueeze(1) - x_train.unsqueeze(0)  # (Q, N, d)
    dist = _norm(diff, norm, dim=-1)
    return (dist <= radius).sum(dim=-1)
