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


def _diff_norm(diff):
    """||diff|| per row, generalizing over a scalar-per-example difference
    (shape (M,), e.g. margin_fn's output) or a vector-per-example
    difference (shape (M, d), e.g. an extractor's flattened features).
    For the scalar case this is exactly |diff| -- the L2 norm of a 1-d
    difference is its absolute value -- so every estimator below behaves
    identically to before when handed a scalar-valued `output_fn` (a
    margin function); passing a vector-valued `output_fn` (e.g.
    `model.extractor`) is a strict generalization, not a behavior change
    for existing callers. See layer_decomposition.py.
    """
    if diff.dim() <= 1:
        return diff.abs()
    return diff.flatten(start_dim=1).norm(p=2, dim=-1)


def euclidean_distance_fn(x, y):
    """Plain L2 distance, row-wise if x/y are batches of the same length.
    The default `distance_fn` for all three estimators below."""
    return (x - y).norm(p=2, dim=-1)


def linear_layer_lipschitz(linear_layer):
    """Exact Lipschitz constant (Euclidean) of an `nn.Linear` layer: the
    spectral norm (largest singular value) of its weight matrix `W`. For
    `y = Wx + b`, `||y1-y2|| = ||W(x1-x2)|| <= ||W||_2 ||x1-x2||`, with
    equality attained along `W`'s top right-singular vector -- so this is a
    tight closed-form bound, not an approximation, unlike every other
    Lipschitz quantity in this module (all of which are empirical
    estimates). Used both directly (`layer_decomposition.py`'s `L_head_exact`)
    and as the closed-form target the pairwise/local/gradient estimators
    are calibrated against when applied to `model.head` (see
    `L_head_estimated`).
    """
    W = linear_layer.weight.detach()
    return torch.linalg.matrix_norm(W, ord=2).item()


def pairwise_lipschitz(model, x_batch, y_batch, output_fn, distance_fn=euclidean_distance_fn,
                        max_pairs=None, seed=None):
    """L_hat = max_{i != j} ||output(x_i) - output(x_j)|| / distance_fn(x_i, x_j).

    `output_fn(model, x, y) -> Tensor` may return a scalar per example
    (shape (N,), e.g. a margin function) or a vector per example (shape
    (N, d), e.g. a feature extractor) -- see `_diff_norm`. `y` need not be
    used by `output_fn` (a wrapper that ignores it, e.g. for
    `model.extractor`, is fine); it's threaded through unconditionally so
    every `output_fn` shares the same 3-argument call convention.

    O(N^2) pairs for N points. At MNIST scale we keep N modest (a few
    hundred points, passed in by the caller) rather than defaulting to
    random pair subsampling -- with N ~ 200-300 the full pair set is only
    ~20-45k pairs, cheap to score directly and exhaustive rather than a
    random sample of it. `max_pairs` is still supported (matching
    toy_lipschitz's interface) as a safety valve if a caller passes a
    larger N than intended.

    Returns (L_hat, i_argmax, j_argmax).
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

    ratio, _, _ = ratio_and_components_for_pairs(model, x_batch, y_batch, output_fn, distance_fn, ii, jj)
    L_hat, idx = ratio.max(dim=0)
    return L_hat.item(), ii[idx].item(), jj[idx].item()


def ratio_and_components_for_pairs(model, x_batch, y_batch, output_fn, distance_fn, ii, jj):
    """Like ratio_for_pairs below, but also returns the two raw components
    the ratio is built from: `dist` (distance_fn(x_i, x_j), the
    denominator) and `output_diff` (||output_i - output_j||, the
    numerator -- `_diff_norm` generalizes this over scalar or vector
    `output_fn` outputs). For callers that want to inspect *why* a
    specific pair has a high or low ratio -- e.g. two points that are
    simply far apart in the metric vs. two points whose outputs genuinely
    diverge a lot -- not just the combined number. Returns
    (ratio, dist, output_diff), each shape (len(ii),).
    """
    with torch.no_grad():
        outputs = output_fn(model, x_batch, y_batch)
        dist = distance_fn(x_batch[ii], x_batch[jj])
    output_diff = _diff_norm(outputs[ii] - outputs[jj])
    valid = dist > 1e-12
    ratio = torch.where(valid, output_diff / dist.clamp_min(1e-12), torch.zeros_like(dist))
    return ratio, dist, output_diff


def ratio_for_pairs(model, x_batch, y_batch, output_fn, distance_fn, ii, jj):
    """The shared core of pairwise_lipschitz/pairwise_lipschitz_all: given
    index tensors `ii`/`jj` into x_batch/y_batch -- any pairing scheme, not
    just all-i<j (e.g. a random subsample, or nearest-neighbor pairs from
    an external index) -- returns the (len(ii),) array of
    ||output_i - output_j|| / distance_fn(x_i, x_j) ratios for exactly
    those pairs. Factored out so callers that need a different pairing
    scheme can reuse the same output/distance/ratio computation without
    duplicating it (see pairwise_lipschitz_all below, and
    run_experiment.py's nearest-neighbor ratio check).
    """
    ratio, _, _ = ratio_and_components_for_pairs(model, x_batch, y_batch, output_fn, distance_fn, ii, jj)
    return ratio


def pairwise_lipschitz_all(model, x_batch, y_batch, output_fn, distance_fn=euclidean_distance_fn,
                            max_pairs=None, seed=None):
    """Same computation pairwise_lipschitz does internally (margins,
    pairwise distances, ratios over all i<j pairs, with the same
    max_pairs subsampling fallback), but returns the full (num_pairs,)
    ratio array and its index arrays instead of collapsing to the max --
    needed to look at the ratio *distribution* (a histogram, or picking
    out specific high-ratio pairs to inspect as images), not just its
    extreme value.

    Returns (ratio, ii, jj): ratio is (num_pairs,), ii/jj are the row/col
    index tensors into x_batch/y_batch such that ratio[k] is the ratio for
    the pair (x_batch[ii[k]], x_batch[jj[k]]).
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

    ratio = ratio_for_pairs(model, x_batch, y_batch, output_fn, distance_fn, ii, jj)
    return ratio, ii, jj


def local_perturbation_lipschitz(model, x_batch, y_batch, output_fn, distance_fn=euclidean_distance_fn,
                                  radius=1.0, n_directions=40, seed=None):
    """Finite-difference local estimate, per point in x_batch: sample
    `n_directions` random unit vectors in raw 784-d pixel space, scale to
    length `radius`, perturb x -> x+delta (label held fixed at the
    original true class y, since this measures how fast `output_fn`
    changes for the *same* label under a small push), and take
    max_direction ||output(x)-output(x+delta)|| / distance_fn(x, x+delta).

    `output_fn` may be scalar-per-example (a margin function) or
    vector-per-example (e.g. `model.extractor`) -- see `_diff_norm`, which
    generalizes the numerator over both without changing behavior for the
    existing scalar case.

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
    """Autograd-based LOCAL/infinitesimal estimate, per point: the dual norm
    of grad(output_fn) w.r.t. x, under the metric in use.

    Two paths, depending on whether `output_fn` is scalar-per-example
    (shape (N,), e.g. a margin function) or vector-per-example (shape
    (N, d) with d>1, e.g. `model.extractor`):

    - **Scalar**: a single backward pass gives the exact per-example
      gradient (N, input_dim) directly, since summing a per-example scalar
      over the batch doesn't mix examples. Plain Euclidean
      (precision=None): ||grad||_2. Mahalanobis with precision matrix P
      (P = Sigma^{-1}, the same P used directly as the quadratic form in
      distance_fn's Mahalanobis distance):

      - `embed_fn=None`: `P` is sized for raw x, and the dual-norm
        expression is sqrt(grad^T P^{-1} grad) = sqrt(grad^T Sigma grad)
        -- note this uses Sigma = P^{-1}, NOT P itself; inverting P a
        second time here (rather than reusing Sigma from distance.py
        directly) is deliberate for interface consistency (every
        estimator here takes `precision`, matching pairwise_lipschitz's
        and local_perturbation_lipschitz's Mahalanobis argument).
        Computed via `torch.linalg.pinv`, not `torch.linalg.inv`: for a
        full-rank `P` (every ridge-regularized `svd_ridge_precision`
        matrix -- `epsilon > 0` always shifts every eigenvalue away from
        0) the Moore-Penrose pseudoinverse is mathematically identical to
        the ordinary inverse, so this is a strict generalization, not a
        behavior change, for every pre-existing caller (checked directly
        in `tests/test_estimators.py`, not just asserted). It also
        correctly handles `distance.py::truncated_precision`'s
        deliberately rank-deficient `P` (only the top-k eigendirections
        retained, the rest discarded entirely rather than regularized):
        `pinv` there gives the *minimum-norm* Sigma, which treats the
        discarded directions' contribution to `Sigma` as exactly 0 rather
        than raising (`inv` would) or blowing up to infinity -- exactly
        the intended reading of "this metric cannot see gradient
        components outside the directions it retains," not "the
        Lipschitz constant is undefined/infinite whenever the gradient
        has any component off the retained subspace."
      - `embed_fn` given: `P` is sized for the *embedded* feature space
        (`embed_fn(x).shape[-1]`), e.g.
        `embeddings.py::elementwise_embedding`. The distance being
        differentiated is now `||embed_fn(x) - embed_fn(y)||_P`, so the
        correct dual norm is the *pullback* of that metric through
        `embed_fn`: for a perturbation `dx`, `embed_fn(x+dx) - embed_fn(x)
        ~= J(x) @ dx` (J = embed_fn's Jacobian at x), so the local metric
        on raw x is `Q(x) = J(x)^T @ P @ J(x)` -- generally **different
        per point** (unlike the fixed Sigma above) whenever embed_fn is
        nonlinear, since J(x) itself depends on x. The dual norm is then
        `sqrt(grad^T Q(x)^-1 grad)`, computed via a batched pseudoinverse
        (`torch.linalg.pinv`, same full-rank-equivalence/rank-deficient-
        handling reasoning as the `embed_fn=None` case above -- `Q(x)` is
        singular whenever `P` is rank-deficient with rank less than the
        raw input dimension, even if `P` itself is comparatively large,
        since `rank(J^T P J) <= rank(P)`) rather than forming `Q(x)^-1`
        explicitly via a linear solve. `J(x)` is computed via
        `torch.func.jacrev`/`vmap`, so this works for *any* embed_fn, not
        just a specific structural form (e.g. elementwise powers) --
        genuine automatic differentiation through embed_fn, not a
        hand-derived formula specific to one embedding.

        When `embed_fn` is linear (in particular the identity, degree=1
        of `elementwise_embedding`), `J(x)` is constant across x, `Q(x)`
        reduces to a single fixed matrix, and this reduces exactly to the
        `embed_fn=None` formula above with `P` replaced by that fixed
        `J^T @ P @ J` -- checked directly in
        `tests/test_estimators.py`, not just derived and trusted (this
        embedded metric is otherwise unverifiable at MNIST scale, with no
        `L*` to check against).
    - **Vector** (new): the local Lipschitz constant of a vector-valued
      map is the spectral norm (largest singular value) of its Jacobian,
      not a plain gradient norm -- there is no single "gradient" to take
      the norm of once the output has more than one component. Computed
      exactly per example via `torch.autograd.functional.jacobian` (one
      full Jacobian per point, looped -- not batched/vectorized; ~0.15s/point
      measured for the 784->1568 CNN extractor in this project, acceptable
      at the small query-set sizes used here, but don't hand this path a
      large N). **Mahalanobis (`precision is not None`) is not supported
      for vector-valued `output_fn`** -- the correct generalization (the
      spectral norm of `J @ Sigma^{1/2}`, requiring a matrix square root)
      is out of scope for this project's current use of vector outputs
      (layer_decomposition.py's pixel-space-only `L_extractor`) and is
      deliberately left unimplemented rather than shipped unverified;
      raises `NotImplementedError` if both apply.

    Returns the full (N,) array of per-point gradient-norm estimates.
    """
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
