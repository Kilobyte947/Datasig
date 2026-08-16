import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mnist_lipschitz.data import load_mnist, get_dev_subset, make_loader
from mnist_lipschitz.models import SmallCNN, train_classifier
from mnist_lipschitz.estimators import linear_layer_lipschitz
from mnist_lipschitz.layer_decomposition import (
    extractor_output_fn,
    head_output_fn,
    fit_feature_normalizer,
    _effective_head_lipschitz_exact,
    layer_decomposition_experiment,
    run_cnn_width_sweep,
)


def _small_trained_cnn(seed=0, n_train=2000, epochs=3):
    torch.manual_seed(seed)
    train = load_mnist(train=True)
    test = load_mnist(train=False)
    dev = get_dev_subset(train, n=n_train, seed=seed)
    train_loader = make_loader(dev.x_image, dev.y, batch_size=128, shuffle=True, seed=seed)
    test_loader = make_loader(test.x_image, test.y, batch_size=1000, shuffle=False)
    model, train_acc, test_acc = train_classifier(
        SmallCNN(), train_loader, test_loader, epochs=epochs, lr=1e-3, verbose=False)
    return model, train, test


def test_extractor_output_fn_matches_direct_call():
    model, train, test = _small_trained_cnn()
    x = test.x_flat[:5]
    y = test.y[:5]
    out = extractor_output_fn(model, x, y)
    expected = model.extractor(x.reshape(5, 1, 28, 28))
    assert torch.equal(out, expected)


def test_head_output_fn_matches_direct_call():
    model, train, test = _small_trained_cnn()
    z = torch.randn(5, model.head.in_features)
    y = torch.zeros(5, dtype=torch.int64)
    out = head_output_fn(model, z, y)
    expected = model.head(z)
    assert torch.equal(out, expected)


def test_fit_feature_normalizer_shapes_and_positive_std():
    model, train, test = _small_trained_cnn()
    x_train = train.x_flat[:500]
    mean, std = fit_feature_normalizer(model, x_train)
    feat_dim = model.head.in_features
    assert mean.shape == (feat_dim,)
    assert std.shape == (feat_dim,)
    assert (std > 0).all()


def test_fit_feature_normalizer_is_deterministic_given_same_data():
    model, train, test = _small_trained_cnn()
    x_train = train.x_flat[:500]
    mean1, std1 = fit_feature_normalizer(model, x_train)
    mean2, std2 = fit_feature_normalizer(model, x_train)
    assert torch.equal(mean1, mean2)
    assert torch.equal(std1, std2)


def test_fit_feature_normalizer_floor_is_relative_not_just_absolute():
    """Regression test for a real bug found during development: an
    absolute-only std floor (e.g. 1e-8) barely constrains dead ReLU/MaxPool
    units (common -- ~6% of features on one trained checkpoint), and
    dividing by ~1e-8 inflated L_extractor's finite-difference estimate
    into the tens of millions on an actual trained CNN, an artifact of the
    floor rather than a real property of the network. The floor must scale
    with the *other* features' typical variance (their median std), not
    just an absolute constant that's meaningless relative to the data."""
    model, train, test = _small_trained_cnn()
    x_train = train.x_flat[:500]

    mean, std = fit_feature_normalizer(model, x_train, relative_floor=1e-2, absolute_floor=1e-8)

    with torch.no_grad():
        from mnist_lipschitz.layer_decomposition import extractor_output_fn
        raw_features_std = extractor_output_fn(model, x_train, None).std(dim=0)

    expected_floor = max(1e-2 * raw_features_std.median().item(), 1e-8)
    assert expected_floor > 1e-6, "test setup: expected a floor well above the old absolute-only 1e-8 default"
    assert std.min().item() >= expected_floor - 1e-12
    assert std.min().item() < 1.0  # sanity: the floor shouldn't be absurdly large either


def test_effective_head_lipschitz_exact_matches_plain_when_std_none():
    model, train, test = _small_trained_cnn()
    assert _effective_head_lipschitz_exact(model.head, std=None) == linear_layer_lipschitz(model.head)


def test_effective_head_lipschitz_exact_accounts_for_std_scaling():
    """Manual check against the closed-form derivation in the docstring:
    standardized_head(z) = head(z*std+mean) has effective weight
    W @ diag(std) -- computed independently here (raw svdvals on a
    manually-scaled weight matrix) rather than re-deriving the same
    formula the function itself uses."""
    torch.manual_seed(0)
    layer = nn.Linear(6, 3)
    std = torch.tensor([1.0, 2.0, 0.5, 3.0, 1.5, 0.25])

    L = _effective_head_lipschitz_exact(layer, std=std)

    W_effective = layer.weight.detach() * std.unsqueeze(0)
    expected = torch.linalg.svdvals(W_effective).max().item()
    assert abs(L - expected) < 1e-9


# --- Checkpoint: gradient method is exact for the (exactly linear) head,
# regardless of query-set size; pairwise/grid are lower-bound estimators
# that need enough samples/directions to converge -- kept as separate
# checks since they have genuinely different convergence guarantees. ---

def test_gradient_method_L_head_estimated_matches_exact_to_machine_precision():
    """The head is exactly linear, so the gradient method's per-example
    Jacobian IS the head's weight matrix everywhere -- L_head_estimated
    must match L_head_exact to near machine precision here, unlike
    pairwise/grid below."""
    model, train, test = _small_trained_cnn()
    x_query = test.x_flat[:8]
    y_query = test.y[:8]
    x_train_norm = train.x_flat[:300]

    result = layer_decomposition_experiment(
        model, x_query, y_query, x_train_for_norm=x_train_norm,
        method="gradient", normalize_features=True, seed=0, verbose=False)

    rel_err = abs(result["L_head_estimated"] - result["L_head_exact"]) / result["L_head_exact"]
    assert rel_err < 1e-6, f"gradient-method L_head_estimated should match L_head_exact exactly, rel_err={rel_err}"


def test_pairwise_and_grid_L_head_estimated_converges_toward_exact_with_enough_samples():
    """Reports the gap (informational -- printed, not just asserted away)
    rather than requiring exact equality, since pairwise/grid convergence
    genuinely depends on sample size. Does assert L_head_estimated never
    exceeds L_head_exact (Cauchy-Schwarz holds exactly for the head, since
    it's linear) and that with a generous query-point count the gap stays
    reasonably small.

    Expect "grid" to converge far worse than "pairwise" here: its random
    directions are sampled in the head's *input* space, i.e. the
    extractor's 1568-d flattened feature space, not raw 784-d pixel
    space -- the same convergence-degrades-sharply-with-dimension effect
    toy_lipschitz's local_perturbation tests found (a handful of random
    directions in ~1500 dimensions essentially never lands close to the
    top singular direction), not an estimator bug. The tolerance below is
    set loose enough to accommodate that honestly rather than papering
    over it with a tighter (and misleadingly reassuring) bound.
    """
    model, train, test = _small_trained_cnn()
    gen = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(test), generator=gen)[:150]
    x_query = test.x_flat[idx]
    y_query = test.y[idx]
    x_train_norm = train.x_flat[:300]

    for method, kwargs in [("pairwise", {}), ("grid", {"n_directions": 60, "radius": 2.0})]:
        result = layer_decomposition_experiment(
            model, x_query, y_query, x_train_for_norm=x_train_norm,
            method=method, normalize_features=True, seed=0, verbose=False, **kwargs)
        rel_gap = (result["L_head_exact"] - result["L_head_estimated"]) / result["L_head_exact"]
        print(f"[{method}] L_head_exact={result['L_head_exact']:.4f}  "
              f"L_head_estimated={result['L_head_estimated']:.4f}  rel_gap={rel_gap:.4f}")

        assert result["L_head_estimated"] <= result["L_head_exact"] * 1.01, \
            f"{method}: L_head_estimated should never exceed L_head_exact (Cauchy-Schwarz, linear head)"
        assert rel_gap < 0.9, \
            f"{method}: L_head_estimated too far from L_head_exact given {len(x_query)} query points"


def test_all_estimated_looseness_ratio_is_self_consistent_and_never_exceeds_exact_version():
    """product_estimated/looseness_ratio_estimated (using L_head_estimated
    instead of the exact spectral norm) must be arithmetically consistent
    with the other returned fields, and -- since L_head_estimated is only
    ever a lower bound on L_head_exact (Cauchy-Schwarz) -- must never
    exceed the exact-head version. Unlike looseness_ratio, it is NOT
    expected to stay >= 1 (see layer_decomposition_experiment's
    docstring), so that's deliberately not asserted here."""
    model, train, test = _small_trained_cnn()
    gen = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(test), generator=gen)[:150]
    x_query = test.x_flat[idx]
    y_query = test.y[idx]
    x_train_norm = train.x_flat[:300]

    result = layer_decomposition_experiment(
        model, x_query, y_query, x_train_for_norm=x_train_norm,
        method="pairwise", normalize_features=True, seed=0, verbose=False)

    expected_product = result["L_extractor_estimated"] * result["L_head_estimated"]
    expected_ratio = expected_product / result["L_full_estimated"]
    assert abs(result["product_estimated"] - expected_product) < 1e-9
    assert abs(result["looseness_ratio_estimated"] - expected_ratio) < 1e-9

    assert result["product_estimated"] <= result["product"] + 1e-9
    assert result["looseness_ratio_estimated"] <= result["looseness_ratio"] + 1e-9


def test_looseness_ratio_at_least_one_on_trained_cnn():
    """The central checkpoint: the empirical submultiplicative bound
    (Szegedy et al. 2014) should hold in practice given a reasonable
    amount of sampling. A small numerical tolerance accounts for floating
    point only -- a genuine violation would indicate an estimator sampling
    issue (see layer_decomposition_experiment's own warning) and should
    fail this test rather than being silently tolerated."""
    model, train, test = _small_trained_cnn()
    gen = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(test), generator=gen)[:150]
    x_query = test.x_flat[idx]
    y_query = test.y[idx]
    x_train_norm = train.x_flat[:300]

    for method in ["pairwise", "grid"]:
        result = layer_decomposition_experiment(
            model, x_query, y_query, x_train_for_norm=x_train_norm,
            method=method, normalize_features=True, seed=0, verbose=False)
        assert result["looseness_ratio"] >= 1.0 - 1e-6, \
            f"{method}: looseness_ratio={result['looseness_ratio']:.4f} < 1"


def test_looseness_ratio_at_least_one_across_width_sweep():
    """Checkpoint 8's explicit requirement: looseness_ratio >= 1 across
    all capacity sweep entries. Small/fast sweep config (few widths, few
    epochs, modest query points) -- this is a correctness checkpoint, not
    a full production sweep (see notebook_layer_decomposition.ipynb for
    that)."""
    df = run_cnn_width_sweep(
        widths=(4, 8, 16), epochs=3, train_subset_size=2000,
        n_query_points=60, n_train_norm_points=300,
        method="pairwise", normalize_features=True, seed=0, verbose=False, save_path=None)

    assert len(df) == 3
    violations = df[df["looseness_ratio"] < 1.0 - 1e-6]
    assert len(violations) == 0, f"looseness_ratio < 1 at widths {violations['width'].tolist()}"


def test_layer_decomposition_experiment_rejects_unknown_method():
    model, train, test = _small_trained_cnn()
    x_query = test.x_flat[:5]
    y_query = test.y[:5]
    try:
        layer_decomposition_experiment(model, x_query, y_query, method="not_a_real_method", normalize_features=False)
        assert False, "expected ValueError for an unknown method"
    except ValueError:
        pass
