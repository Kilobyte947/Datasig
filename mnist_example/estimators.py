"""The three Lipschitz sub-methods (pairwise, local-perturbation, gradient-norm), generalised to
a classifier's margin function over 784-dimensional MNIST inputs, under a pluggable distance metric.
"""

import torch
from mnist_example.distance import euclidean_distance_fn
torch.set_default_dtype(torch.float64)

def _generator(seed):
    """A seeded torch.Generator, or None if seed is None."""
    return torch.Generator().manual_seed(seed) if seed is not None else None

def _diff_norm(diff):
    """||diff|| per row: absolute value for a scalar-per-example difference, Euclidean norm for a 
    vector-per-example one."""
    if diff.dim() <= 1:
        return diff.abs()
    return diff.flatten(start_dim=1).norm(p=2, dim=-1)


def linear_layer_lipschitz(linear_layer):
    """Exact Lipschitz constant of an nn.Linear layer: the spectral norm of its weight matrix."""
    W = linear_layer.weight.detach()
    return torch.linalg.matrix_norm(W, ord=2).item()


def pairwise_lipschitz(model, x_batch, y_batch, output_fn, distance_fn=euclidean_distance_fn,
                        max_pairs=None, seed=None):
    """L_hat = max_{i != j} ||output(x_i) - output(x_j)|| / distance_fn(x_i, x_j), over all pairs in 
    x_batch (or a random subsample if there are more than max_pairs). 
    Returns (L_hat, i_argmax, j_argmax)."""
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

    ratio, _, _ = ratio_and_components_for_pairs(model, x_batch, y_batch, output_fn, distance_fn, ii, jj)
    L_hat, idx = ratio.max(dim=0)
    return L_hat.item(), ii[idx].item(), jj[idx].item()


def ratio_and_components_for_pairs(model, x_batch, y_batch, output_fn, distance_fn, ii, jj):
    """The ratio for a given set of pairs (ii, jj), along with its two components: dist (the
    denominator) and output_diff (the numerator). Returns (ratio, dist, output_diff)."""
    with torch.no_grad():
        outputs = output_fn(model, x_batch, y_batch)
        dist = distance_fn(x_batch[ii], x_batch[jj])
    output_diff = _diff_norm(outputs[ii] - outputs[jj])
    valid = dist > 1e-12
    ratio = torch.where(valid, output_diff / dist.clamp_min(1e-12), torch.zeros_like(dist))
    return ratio, dist, output_diff


def ratio_for_pairs(model, x_batch, y_batch, output_fn, distance_fn, ii, jj):
    """The ratio ||output_i - output_j|| / distance_fn(x_i, x_j) for a given set of pairs (ii, jj)."""
    ratio, _, _ = ratio_and_components_for_pairs(model, x_batch, y_batch, output_fn, distance_fn, ii, jj)
    return ratio


def pairwise_lipschitz_all(model, x_batch, y_batch, output_fn, distance_fn=euclidean_distance_fn,
                            max_pairs=None, seed=None):
    """Same computation as pairwise_lipschitz, but returns the full ratio array instead of just its maximum. 
    Returns (ratio, ii, jj)."""
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

    ratio = ratio_for_pairs(model, x_batch, y_batch, output_fn, distance_fn, ii, jj)
    return ratio, ii, jj


def local_perturbation_lipschitz(model, x_batch, y_batch, output_fn, distance_fn=euclidean_distance_fn,
                                  radius=1.0, n_directions=40, seed=None):
    """Finite-difference local estimate, per point: samples n_directions random unit vectors in pixel
    space, scaled to radius, and takes the maximum ratio of output change to distance_fn across them.
    Returns the full (N,) array of per-point estimates."""
    generator = _generator(seed)
    N, d = x_batch.shape

    with torch.no_grad():
        output_x = output_fn(model, x_batch, y_batch)

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
            output_xp = output_fn(model, x_prime, y_batch)
            dist = distance_fn(x_batch, x_prime)

        ratio = _diff_norm(output_xp - output_x) / dist.clamp_min(1e-12)
        best = torch.maximum(best, ratio)

    return best


def gradient_norm_estimate(model, x_batch, y_batch, output_fn, precision=None, embed_fn=None):
    """Gradient-norm local estimate, per point, under the given metric.
    
    For a scalar output_fn: the exact per-example gradient norm. Plain Euclidean if precision is None. 
    With a Mahalanobis precision matrix P, the correct dual norm is sqrt(grad^T Sigma grad) 
    where Sigma = P^-1 (computed via pinv, which also handles P's rank-deficient case correctly). 
    If embed_fn is given, the metric is pulled back through embed_fn's Jacobian J at each point, 
    Q(x) = J^T P J, and the dual norm becomes sqrt(grad^T Q(x)^-1 grad) — Q varies per point whenever
    embed_fn is nonlinear.
    
    For a vector-valued output_fn, the local Lipschitz constant is the spectral norm of its Jacobian, 
    computed per example. Mahalanobis is not supported in this case and raises NotImplementedError.
    
    Returns the full (N,) array of per-point estimates."""

    x = x_batch.clone().requires_grad_(True)
    outputs = output_fn(model, x, y_batch)

    if outputs.dim() <= 1:
        (grad,) = torch.autograd.grad(outputs.sum(), x)
        grad = grad.detach()

        if precision is None:
            return grad.norm(p=2, dim=-1)

        if embed_fn is None:
            sigma = torch.linalg.pinv(precision)  # pinv(P) -- Sigma for full-rank P, minimum-norm generalization for rank-deficient P; see docstring
            quad = torch.einsum("ni,ij,nj->n", grad, sigma, grad)
            return quad.clamp_min(0.0).sqrt()

        x_detached = x_batch.detach()
        jacobian_fn = torch.func.jacrev(embed_fn)
        J = torch.func.vmap(jacobian_fn)(x_detached)  # (N, D, d): D = embed_fn output dim, d = input dim
        Q = torch.einsum("nDi,DE,nEj->nij", J, precision, J)  # (N, d, d), the per-point pullback metric

        Q_pinv = torch.linalg.pinv(Q)  # pinv(Q(x)), same full-rank-equivalence/rank-deficient reasoning as above
        u = torch.einsum("nij,nj->ni", Q_pinv, grad)  # pinv(Q(x)) @ grad, per point
        quad = (grad * u).sum(dim=-1)
        return quad.clamp_min(0.0).sqrt()

    if precision is not None or embed_fn is not None:
        raise NotImplementedError(
            "gradient_norm_estimate: Mahalanobis (precision != None) is not implemented "
            "for vector-valued output_fn -- see docstring.")

    N = x_batch.shape[0]
    spectral_norms = torch.empty(N)
    for i in range(N):
        xi = x_batch[i].detach()
        yi = y_batch[i:i + 1]

        def _single_example_output(x_flat, yi=yi):
            return output_fn(model, x_flat.unsqueeze(0), yi).squeeze(0)

        J = torch.autograd.functional.jacobian(_single_example_output, xi)  # (d, input_dim)
        spectral_norms[i] = torch.linalg.matrix_norm(J, ord=2)

    return spectral_norms.detach()
