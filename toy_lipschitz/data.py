"""Sampling schemes for the toy Lipschitz experiment."""

import torch

torch.set_default_dtype(torch.float64)


def _generator(seed):
    return torch.Generator().manual_seed(seed) if seed is not None else None


def sample_uniform(N, domain, d, seed=None):
    """Uniform i.i.d. samples in the box `domain`. Returns (N, d)."""
    lo, hi = domain
    g = _generator(seed)
    if g is not None:
        u = torch.rand(N, d, generator=g)
    else:
        u = torch.rand(N, d)
    return u * (hi - lo) + lo


def _rejection_sample(n_needed, domain, d, gap_center, gap_radius, inside_gap, generator, max_attempts=10000):
    lo, hi = domain
    gap_center = torch.as_tensor(gap_center, dtype=torch.get_default_dtype()).reshape(d)
    collected = []
    n_collected = 0
    attempts = 0
    batch = max(n_needed * 4, 256)
    while n_collected < n_needed and attempts < max_attempts:
        attempts += 1
        if generator is not None:
            candidates = torch.rand(batch, d, generator=generator) * (hi - lo) + lo
        else:
            candidates = torch.rand(batch, d) * (hi - lo) + lo
        dist = (candidates - gap_center).norm(p=2, dim=-1)
        mask = dist < gap_radius if inside_gap else dist >= gap_radius
        keep = candidates[mask]
        if keep.shape[0] > 0:
            collected.append(keep)
            n_collected += keep.shape[0]
    if n_collected < n_needed:
        raise RuntimeError(
            f"rejection sampling failed to collect {n_needed} points "
            f"(inside_gap={inside_gap}) after {max_attempts} batches"
        )
    return torch.cat(collected, dim=0)[:n_needed]


def sample_with_gap(N, domain, d, gap_center, gap_radius, gap_fraction, seed=None):
    """Sample so that only `gap_fraction` of the density falls within
    `gap_radius` of `gap_center` (an L2 ball). Implemented via rejection
    sampling: most points are drawn uniformly outside the gap ball, a small
    number are drawn uniformly inside it (intersected with `domain`).
    """
    generator = _generator(seed)
    n_gap = int(round(N * gap_fraction))
    n_outside = N - n_gap

    parts = []
    if n_outside > 0:
        parts.append(_rejection_sample(n_outside, domain, d, gap_center, gap_radius, inside_gap=False, generator=generator))
    if n_gap > 0:
        parts.append(_rejection_sample(n_gap, domain, d, gap_center, gap_radius, inside_gap=True, generator=generator))
    x = torch.cat(parts, dim=0)

    if generator is not None:
        perm = torch.randperm(x.shape[0], generator=generator)
    else:
        perm = torch.randperm(x.shape[0])
    return x[perm]


def make_dataset(f_star, x):
    """Return (x, y) with y = f_star(x). f_star is always evaluated exactly,
    with no external noise added on top -- noise, if ever needed, belongs in
    the definition of f_star itself (a different, still-fixed function),
    not as a stochastic perturbation of a fixed function's output."""
    return x, f_star(x)
