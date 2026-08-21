import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mnist_lipschitz.data import load_mnist, get_dev_subset, make_loader
from mnist_lipschitz.models import StrongCNN, FlattenedInputWrapper, train_classifier
from mnist_lipschitz.estimators import euclidean_distance_fn, pairwise_lipschitz
from mnist_lipschitz.layer_decomposition import _effective_head_lipschitz_exact
from mnist_lipschitz.adversarial.strong_cnn_experiment import (
    strong_cnn_extractor_fn,
    strong_cnn_head_module,
    full_logits_fn,
    fit_strong_cnn_feature_normalizer,
    compute_strong_cnn_bounds,
    strong_cnn_head_layer_bound_check,
    strong_cnn_bound_comparison,
)

TOLERANCE = 1e-9


def _small_trained_strong_cnn(seed=0, n_train=800, epochs=2):
    torch.manual_seed(seed)
    train = load_mnist(train=True)
    test = load_mnist(train=False)
    dev = get_dev_subset(train, n=n_train, seed=seed)
    train_loader = make_loader(dev.x_image, dev.y, batch_size=128, shuffle=True, seed=seed)
    test_loader = make_loader(test.x_image, test.y, batch_size=1000, shuffle=False)
    model, train_acc, test_acc = train_classifier(
        StrongCNN(dropout_conv=0.25, dropout_fc=0.5), train_loader, test_loader,
        epochs=epochs, lr=1e-3, verbose=False)
    model.eval()
    return model, train, test


# --- composition identity ---

def test_composition_identity_matches_full_forward():
    """strong_cnn_head_module(model)(strong_cnn_extractor_fn(model, x, y)) must equal model(x)
    exactly -- mirrors test_layer_decomposition.py's own extractor/head-composition checks for
    SmallCNN, since this identity is what makes the externally-constructed split valid at all."""
    model, train, test = _small_trained_strong_cnn()
    x = test.x_flat[:5]
    y = test.y[:5]

    with torch.no_grad():
        features = strong_cnn_extractor_fn(model, x, y)
        head = strong_cnn_head_module(model)
        reconstructed = head(features)
        expected = model(x.reshape(5, 1, 28, 28))

    assert torch.equal(reconstructed, expected)


# --- eval-mode discipline ---

def test_batchnorm_train_mode_is_batch_dependent():
    """Demonstrates the UNDERLYING architectural fact _require_eval_mode exists to guard
    against: in train() mode, BatchNorm1d uses per-batch statistics, so the same row's output
    depends on which other rows share its batch -- calls model.features/model.classifier
    directly (bypassing strong_cnn_extractor_fn's guard) to isolate this from the guard itself."""
    model, train, test = _small_trained_strong_cnn()
    model.train()
    x_target = test.x_flat[0:1].reshape(1, 1, 28, 28)
    x_batch_a = torch.cat([x_target, test.x_flat[1:9].reshape(8, 1, 28, 28)], dim=0)
    x_batch_b = torch.cat([x_target, test.x_flat[9:17].reshape(8, 1, 28, 28)], dim=0)

    with torch.no_grad():
        out_a = model.classifier[:4](model.features(x_batch_a))[0]
        out_b = model.classifier[:4](model.features(x_batch_b))[0]

    assert not torch.equal(out_a, out_b), \
        "test setup: expected train-mode BatchNorm to be batch-composition-dependent"


def test_eval_mode_forward_passes_are_deterministic():
    model, train, test = _small_trained_strong_cnn()
    x = test.x_flat[:5]

    with torch.no_grad():
        out_1 = strong_cnn_extractor_fn(model, x, None)
        out_2 = strong_cnn_extractor_fn(model, x, None)

    assert torch.equal(out_1, out_2)


def test_strong_cnn_extractor_fn_requires_eval_mode():
    model, train, test = _small_trained_strong_cnn()
    model.train()
    try:
        strong_cnn_extractor_fn(model, test.x_flat[:5], None)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_compute_strong_cnn_bounds_requires_eval_mode():
    model, train, test = _small_trained_strong_cnn()
    model.train()
    try:
        compute_strong_cnn_bounds(model, test.x_flat[:20], test.y[:20], euclidean_distance_fn,
                                   train.x_flat[:50], seed=0, verbose=False)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_strong_cnn_head_layer_bound_check_requires_eval_mode():
    model, train, test = _small_trained_strong_cnn()
    x = test.x_flat[0]
    example = {"x": x, "x_adv": (x + 0.1).clamp(0.0, 1.0)}
    model.train()
    try:
        strong_cnn_head_layer_bound_check(model, example)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_strong_cnn_bound_comparison_requires_eval_mode():
    model, train, test = _small_trained_strong_cnn()
    model.train()
    try:
        strong_cnn_bound_comparison(model, test.x_flat[:20], test.y[:20], test.x_flat[:50], test.y[:50],
                                     train.x_flat[:50], euclidean_distance_fn,
                                     epsilons=(0.1,), pgd_num_steps=2, pgd_num_restarts=1,
                                     n_points=10, seed=0, verbose=False)
        assert False, "expected ValueError"
    except ValueError:
        pass


# --- self-consistency parity (weaker checkpoint than the SmallCNN precedent -- see module
# docstring: no independently-existing layer_decomposition_experiment-equivalent exists for
# StrongCNN to check against, so this instead checks compute_strong_cnn_bounds reduces to a
# from-scratch manual pairwise_lipschitz computation using the same closures) ---

def test_compute_strong_cnn_bounds_matches_manual_pairwise_computation():
    model, train, test = _small_trained_strong_cnn()
    x_query, y_query = test.x_flat[:40], test.y[:40]
    x_train_for_norm = train.x_flat[:200]

    result = compute_strong_cnn_bounds(
        model, x_query, y_query, euclidean_distance_fn, x_train_for_norm, seed=0, verbose=False)

    mean, std = fit_strong_cnn_feature_normalizer(model, x_train_for_norm)
    normalized_extractor_fn = lambda m, x, y: (strong_cnn_extractor_fn(m, x, y) - mean) / std
    head = strong_cnn_head_module(model)
    expected_L_head_exact = _effective_head_lipschitz_exact(head, std)
    expected_L_extractor, _, _ = pairwise_lipschitz(
        model, x_query, y_query, normalized_extractor_fn, distance_fn=euclidean_distance_fn, seed=0)
    wrapped = FlattenedInputWrapper(model)
    expected_L_full, _, _ = pairwise_lipschitz(
        wrapped, x_query, y_query, full_logits_fn, distance_fn=euclidean_distance_fn, seed=0)

    assert abs(result["L_head_exact"] - expected_L_head_exact) < TOLERANCE
    assert abs(result["L_extractor_estimated"] - expected_L_extractor) < TOLERANCE
    assert abs(result["L_full_estimated"] - expected_L_full) < TOLERANCE
    assert abs(result["product"] - expected_L_extractor * expected_L_head_exact) < TOLERANCE


def test_compute_strong_cnn_bounds_mahalanobis_differs_from_euclidean():
    from mnist_lipschitz.distance import svd_ridge_precision, mahalanobis_distance
    model, train, test = _small_trained_strong_cnn()
    x_query, y_query = test.x_flat[:40], test.y[:40]
    x_train_for_norm = train.x_flat[:200]
    precision = svd_ridge_precision(train.x_flat[:500], 0.01)
    maha_fn = lambda x, y: mahalanobis_distance(x, y, precision)

    euclidean_result = compute_strong_cnn_bounds(
        model, x_query, y_query, euclidean_distance_fn, x_train_for_norm, seed=0, verbose=False)
    maha_result = compute_strong_cnn_bounds(
        model, x_query, y_query, maha_fn, x_train_for_norm, seed=0, verbose=False)

    assert abs(euclidean_result["L_head_exact"] - maha_result["L_head_exact"]) < TOLERANCE
    assert euclidean_result["L_extractor_estimated"] != maha_result["L_extractor_estimated"]
    assert euclidean_result["L_full_estimated"] != maha_result["L_full_estimated"]


# --- strong_cnn_head_layer_bound_check ---

def test_strong_cnn_head_layer_bound_check_never_exceeds_its_own_bound():
    model, train, test = _small_trained_strong_cnn()
    torch.manual_seed(1)
    for i in range(5):
        x = test.x_flat[i]
        x_adv = (x + 0.15 * torch.randn_like(x)).clamp(0.0, 1.0)
        example = {"x": x, "x_adv": x_adv}
        result = strong_cnn_head_layer_bound_check(model, example)
        assert result["head_bound_tightness"] <= 1.0 + 1e-6, result


# --- fit_strong_cnn_feature_normalizer ---

def test_fit_strong_cnn_feature_normalizer_floor_is_relative_not_just_absolute():
    model, train, test = _small_trained_strong_cnn()
    x_train = train.x_flat[:200]

    mean, std = fit_strong_cnn_feature_normalizer(model, x_train, relative_floor=1e-2, absolute_floor=1e-8)

    with torch.no_grad():
        raw_std = strong_cnn_extractor_fn(model, x_train, None).std(dim=0)
    expected_floor = max(1e-2 * raw_std.median().item(), 1e-8)
    assert std.min().item() >= expected_floor - 1e-12
    assert mean.shape == (256,)
    assert std.shape == (256,)


# --- strong_cnn_bound_comparison end-to-end smoke test ---

def test_strong_cnn_bound_comparison_end_to_end_on_small_trained_cnn():
    model, train, test = _small_trained_strong_cnn()
    gen = torch.Generator().manual_seed(0)
    query_idx = torch.randperm(len(test), generator=gen)[:30]
    x_query, y_query = test.x_flat[query_idx], test.y[query_idx]
    mask = torch.ones(len(test), dtype=torch.bool)
    mask[query_idx] = False
    pool_idx = mask.nonzero(as_tuple=True)[0][:100]
    x_pool, y_pool = test.x_flat[pool_idx], test.y[pool_idx]
    x_train_for_norm = train.x_flat[:150]

    summary_df, sweep_results = strong_cnn_bound_comparison(
        model, x_query, y_query, x_pool, y_pool, x_train_for_norm, euclidean_distance_fn,
        epsilons=(0.1, 0.2), pgd_num_steps=3, pgd_num_restarts=1, n_points=20, seed=0, verbose=False)

    assert len(summary_df) == 4  # 2 epsilons x 2 methods (FGSM, PGD)
    assert (summary_df["L_full_estimated"] > 0).all()
    assert (summary_df["product_bound"] > 0).all()
    assert "x_eval" in sweep_results
    assert "per_case" in sweep_results
