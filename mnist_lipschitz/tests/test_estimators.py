import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch.nn as nn

from mnist_lipschitz.models import LogisticRegressionModel, margin_fn
from mnist_lipschitz.estimators import (
    euclidean_distance_fn,
    pairwise_lipschitz,
    local_perturbation_lipschitz,
    gradient_norm_estimate,
    ratio_and_components_for_pairs,
    ratio_for_pairs,
    linear_layer_lipschitz,
)

# --- Critical checkpoint: closed-form agreement on logistic regression ---
#
# With num_classes=2, margin_fn(model, x, y=0) = logit_0(x) - logit_1(x)
# (the "max over j != y_true" in margin_fn has only one choice of j left),
# which is EXACTLY (w_0 - w_1)^T x + (b_0 - b_1) -- a genuinely linear
# function of x, with an exact closed-form Euclidean Lipschitz constant of
# ||w_0 - w_1||_2, attained at every point (constant gradient). This is the
# one remaining closed-form check available at MNIST scale, and it must
# pass before the same estimator code is trusted on the MLP/CNN.
#
# Low input dimension (5, not MNIST's 784) is deliberate and confined to
# this test only: local_perturbation_lipschitz samples random directions in
# pixel space, and how well a batch of random directions covers the true
# gradient direction (needed for its ratio to approach the closed-form
# value) degrades sharply with dimension. d=5 lets a modest n_directions
# get within the stated 10% tolerance; this tests the *estimator
# implementation*, not MNIST-scale numbers (those come from Checkpoint 6).

TOLERANCE = 0.10


def _linear_margin_setup(d=5, seed=0):
    torch.manual_seed(seed)
    model = LogisticRegressionModel(input_dim=d, num_classes=2)
    w_diff = (model.linear.weight[0] - model.linear.weight[1]).detach()
    L_star = w_diff.norm(p=2).item()
    return model, w_diff, L_star


def test_pairwise_lipschitz_matches_closed_form_exactly_along_gradient_direction():
    d = 5
    model, w_diff, L_star = _linear_margin_setup(d=d)
    unit = w_diff / w_diff.norm(p=2)

    # Points constructed strictly along the gradient direction: for an
    # exactly linear function this makes the pairwise ratio equal L_star
    # exactly (not just approximately), by construction.
    ts = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    x_batch = ts.unsqueeze(-1) * unit.unsqueeze(0)
    y_batch = torch.zeros(x_batch.shape[0], dtype=torch.int64)

    L_hat, i, j = pairwise_lipschitz(model, x_batch, y_batch, margin_fn, euclidean_distance_fn)
    assert abs(L_hat - L_star) < 1e-9


def test_pairwise_lipschitz_on_random_points_within_tolerance():
    d = 5
    model, w_diff, L_star = _linear_margin_setup(d=d)
    torch.manual_seed(1)
    x_batch = torch.randn(200, d)
    y_batch = torch.zeros(200, dtype=torch.int64)

    L_hat, i, j = pairwise_lipschitz(model, x_batch, y_batch, margin_fn, euclidean_distance_fn)
    # Can only ever underestimate L_star (Cauchy-Schwarz), never overshoot.
    assert L_hat <= L_star * 1.01
    rel_err = abs(L_hat - L_star) / L_star
    assert rel_err < TOLERANCE, f"pairwise rel_err={rel_err:.4f}"


def test_gradient_norm_estimate_matches_closed_form_exactly():
    d = 5
    model, w_diff, L_star = _linear_margin_setup(d=d)
    torch.manual_seed(2)
    x_batch = torch.randn(10, d)  # gradient is constant everywhere -- location shouldn't matter
    y_batch = torch.zeros(10, dtype=torch.int64)

    grad_vals = gradient_norm_estimate(model, x_batch, y_batch, margin_fn)
    assert grad_vals.shape == (10,)
    assert torch.allclose(grad_vals, torch.full((10,), L_star), atol=1e-9)


def test_local_perturbation_lipschitz_within_tolerance():
    d = 5
    model, w_diff, L_star = _linear_margin_setup(d=d)
    torch.manual_seed(3)
    x_batch = torch.randn(5, d)
    y_batch = torch.zeros(5, dtype=torch.int64)

    vals = local_perturbation_lipschitz(model, x_batch, y_batch, margin_fn,
                                         euclidean_distance_fn, radius=1.0, n_directions=1000, seed=4)
    assert vals.shape == (5,)
    L_hat = vals.max().item()
    assert L_hat <= L_star * 1.01
    rel_err = abs(L_hat - L_star) / L_star
    assert rel_err < TOLERANCE, f"local-perturbation rel_err={rel_err:.4f}"


def test_all_three_submethods_agree_with_each_other_and_closed_form():
    d = 5
    model, w_diff, L_star = _linear_margin_setup(d=d)
    unit = w_diff / w_diff.norm(p=2)

    torch.manual_seed(5)
    ts = torch.linspace(-3, 3, 200)
    x_pairwise = ts.unsqueeze(-1) * unit.unsqueeze(0)
    y_pairwise = torch.zeros(200, dtype=torch.int64)
    L_pairwise, _, _ = pairwise_lipschitz(model, x_pairwise, y_pairwise, margin_fn, euclidean_distance_fn)

    x_local = torch.randn(5, d)
    y_local = torch.zeros(5, dtype=torch.int64)
    L_local = local_perturbation_lipschitz(model, x_local, y_local, margin_fn,
                                            euclidean_distance_fn, radius=1.0, n_directions=1000, seed=6).max().item()

    x_grad = torch.randn(5, d)
    y_grad = torch.zeros(5, dtype=torch.int64)
    L_grad = gradient_norm_estimate(model, x_grad, y_grad, margin_fn).max().item()

    for name, L_hat in [("pairwise", L_pairwise), ("local", L_local), ("grad", L_grad)]:
        rel_err = abs(L_hat - L_star) / L_star
        assert rel_err < TOLERANCE, f"{name} rel_err={rel_err:.4f} vs L_star={L_star:.4f} (L_hat={L_hat:.4f})"


# --- Mahalanobis (precision) path: the "P vs P^{-1}" convention ---
#
# gradient_norm_estimate's dual-norm formula for a Mahalanobis metric with
# precision matrix P is sqrt(grad^T P^{-1} grad) -- easy to get backwards
# (using P instead of P^{-1}) with nothing to catch it once real MNIST data
# is involved, since there's no L* there. This checks it directly against
# an independent closed-form identity: for a fixed gradient vector g and
# precision matrix P, the maximizer of |g^T delta| subject to the
# Mahalanobis-unit-ball constraint delta^T P delta = 1 is
# delta* = P^{-1} g / sqrt(g^T P^{-1} g), which by construction achieves
# |g^T delta*| = sqrt(g^T P^{-1} g) exactly -- so gradient_norm_estimate's
# output must equal that construction's value if (and only if) it inverts
# precision the right way round.

def test_gradient_norm_estimate_mahalanobis_matches_independent_closed_form():
    d = 5
    torch.manual_seed(7)
    model, w_diff, _ = _linear_margin_setup(d=d, seed=7)
    x_batch = torch.randn(3, d)
    y_batch = torch.zeros(3, dtype=torch.int64)

    A = torch.randn(d, d)
    precision = A @ A.T + d * torch.eye(d)  # SPD, well-conditioned

    L_hat = gradient_norm_estimate(model, x_batch, y_batch, margin_fn, precision=precision)

    sigma = torch.linalg.inv(precision)
    expected = (w_diff @ sigma @ w_diff).clamp_min(0.0).sqrt().item()

    assert torch.allclose(L_hat, torch.full((3,), expected), atol=1e-8)

    # And the maximizer delta* actually attains it under the Mahalanobis
    # distance_fn used elsewhere in this module, confirming the formula
    # lines up with pairwise/local's distance convention, not just an
    # independent derivation.
    delta_star = sigma @ w_diff
    delta_star = delta_star / (w_diff @ sigma @ w_diff).sqrt()
    x0 = x_batch[0]
    x1 = x0 + delta_star
    maha_dist = (delta_star @ precision @ delta_star).clamp_min(0.0).sqrt()
    margin_diff = (margin_fn(model, x1.unsqueeze(0), y_batch[:1])
                   - margin_fn(model, x0.unsqueeze(0), y_batch[:1])).abs().item()
    assert abs(margin_diff / maha_dist.item() - expected) < 1e-6


def test_ratio_and_components_for_pairs_matches_ratio_for_pairs_and_is_self_consistent():
    """ratio_and_components_for_pairs must (a) return the same ratio
    ratio_for_pairs does -- it's a refactor of the same shared logic, not
    a reimplementation -- and (b) its own returned components must satisfy
    ratio == margin_diff / dist exactly, since that's the definition."""
    torch.manual_seed(9)
    model = LogisticRegressionModel(input_dim=10, num_classes=4)
    x_batch = torch.randn(12, 10)
    y_batch = torch.randint(0, 4, (12,))
    ii = torch.tensor([0, 1, 2, 3, 4])
    jj = torch.tensor([5, 6, 7, 8, 9])

    ratio, dist, margin_diff = ratio_and_components_for_pairs(
        model, x_batch, y_batch, margin_fn, euclidean_distance_fn, ii, jj)
    expected_ratio = ratio_for_pairs(model, x_batch, y_batch, margin_fn, euclidean_distance_fn, ii, jj)

    assert torch.allclose(ratio, expected_ratio)
    assert torch.allclose(ratio, margin_diff / dist.clamp_min(1e-12), atol=1e-9)
    assert (dist >= 0).all()
    assert (margin_diff >= 0).all()


def test_linear_layer_lipschitz_on_diagonal_matrix_with_known_singular_values():
    """A diagonal weight matrix's singular values are exactly the absolute
    values of its diagonal entries -- the largest one is the spectral norm
    by construction, no numerical approximation needed to know the answer."""
    layer = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.diag(torch.tensor([3.0, -7.0, 1.0, 5.0])))

    L = linear_layer_lipschitz(layer)
    assert abs(L - 7.0) < 1e-9


def test_linear_layer_lipschitz_matches_svdvals_max():
    torch.manual_seed(3)
    layer = nn.Linear(20, 8)  # non-square, with bias -- bias must not affect the Lipschitz constant

    L = linear_layer_lipschitz(layer)
    expected = torch.linalg.svdvals(layer.weight.detach()).max().item()
    assert abs(L - expected) < 1e-9


def test_linear_layer_lipschitz_ignores_bias():
    torch.manual_seed(4)
    layer = nn.Linear(6, 6)
    L_with_bias = linear_layer_lipschitz(layer)
    with torch.no_grad():
        layer.bias.zero_()
    L_no_bias = linear_layer_lipschitz(layer)
    assert L_with_bias == L_no_bias


# --- Vector-valued output_fn generalization (layer_decomposition's L_extractor) ---
#
# gradient_norm_estimate's vector-output path computes a per-example
# Jacobian spectral norm, a genuinely different (and more expensive)
# computation than the scalar-output fast path. A linear map's Jacobian
# equals its weight matrix everywhere (constant, not just locally), so
# this is a clean closed-form check: the estimate must match
# linear_layer_lipschitz's exact spectral norm at every point, not just on
# average.

def _linear_vector_output_fn(model, x, y):
    return model(x)  # `model` is an nn.Linear layer itself here; y unused


def test_gradient_norm_estimate_vector_path_matches_linear_layer_lipschitz_exactly():
    torch.manual_seed(8)
    layer = nn.Linear(10, 4)
    x_batch = torch.randn(6, 10)
    y_batch = torch.zeros(6, dtype=torch.int64)  # unused by _linear_vector_output_fn

    spectral_norms = gradient_norm_estimate(layer, x_batch, y_batch, _linear_vector_output_fn)
    expected = linear_layer_lipschitz(layer)

    assert spectral_norms.shape == (6,)
    assert torch.allclose(spectral_norms, torch.full((6,), expected), atol=1e-8)


def test_gradient_norm_estimate_vector_path_rejects_mahalanobis():
    torch.manual_seed(8)
    layer = nn.Linear(10, 4)
    x_batch = torch.randn(3, 10)
    y_batch = torch.zeros(3, dtype=torch.int64)
    precision = torch.eye(10)

    try:
        gradient_norm_estimate(layer, x_batch, y_batch, _linear_vector_output_fn, precision=precision)
        assert False, "expected NotImplementedError for vector output_fn + precision"
    except NotImplementedError:
        pass


def test_pairwise_and_local_perturbation_generalize_to_vector_output_and_stay_bounded():
    """No closed-form target for pairwise/local on a linear map beyond the
    Cauchy-Schwarz upper bound (linear_layer_lipschitz) -- random pairs
    won't generally align with the top singular direction -- so this
    checks the generalized vector-output path runs, returns sane shapes,
    and never exceeds that bound (up to float tolerance)."""
    torch.manual_seed(9)
    layer = nn.Linear(10, 4)
    x_batch = torch.randn(30, 10)
    y_batch = torch.zeros(30, dtype=torch.int64)
    L_star = linear_layer_lipschitz(layer)

    L_pairwise, i, j = pairwise_lipschitz(layer, x_batch, y_batch, _linear_vector_output_fn)
    assert L_pairwise <= L_star * 1.01
    assert i != j

    local_vals = local_perturbation_lipschitz(layer, x_batch, y_batch, _linear_vector_output_fn,
                                               radius=1.0, n_directions=50, seed=1)
    assert local_vals.shape == (30,)
    assert (local_vals <= L_star * 1.01).all()
