import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mnist_lipschitz.data import load_mnist, get_dev_subset, make_loader
from mnist_lipschitz.models import SmallCNN, FlattenedInputWrapper, train_classifier
from mnist_lipschitz.estimators import euclidean_distance_fn
from mnist_lipschitz.distance import svd_ridge_precision, mahalanobis_distance
from mnist_lipschitz.layer_decomposition import layer_decomposition_experiment
from mnist_lipschitz.adversarial.run_experiment import (
    achieved_ratio,
    run_epsilon_sweep,
    run_bound_comparison,
    run_bound_comparison_with_distance_fn,
    run_cnn_adversarial_width_sweep,
    run_cnn_adversarial_width_sweep_with_distance_fn,
    compute_bounds_with_distance_fn,
    build_pixel_mahalanobis_distance_fn,
    most_and_least_sensitive_examples,
    head_layer_bound_check,
)


def _small_trained_cnn(seed=0, n_train=2000, epochs=3):
    torch.manual_seed(seed)
    train = load_mnist(train=True)
    test = load_mnist(train=False)
    dev = get_dev_subset(train, n=n_train, seed=seed)
    train_loader = make_loader(dev.x_image, dev.y, batch_size=128, shuffle=True, seed=seed)
    test_loader = make_loader(test.x_image, test.y, batch_size=1000, shuffle=False)
    model, train_acc, test_acc = train_classifier(
        SmallCNN(conv_channels=(8, 16)), train_loader, test_loader, epochs=epochs, lr=1e-3, verbose=False)
    return model, train, test


def _small_maha_distance_fn(train, epsilon=0.01, n=2000):
    precision = svd_ridge_precision(train.x_flat[:n], epsilon)
    return lambda x, y: mahalanobis_distance(x, y, precision)


# --- build_pixel_mahalanobis_distance_fn ---

def test_build_pixel_mahalanobis_distance_fn_basic_properties():
    train = load_mnist(train=True)
    distance_fn = build_pixel_mahalanobis_distance_fn(train.x_flat[:2000], epsilon=0.01)
    x = train.x_flat[:5]
    y = train.x_flat[5:10]

    assert torch.equal(distance_fn(x, x), torch.zeros(5))
    d_xy = distance_fn(x, y)
    d_yx = distance_fn(y, x)
    assert (d_xy > 0).all()
    assert torch.allclose(d_xy, d_yx, atol=1e-9)


# --- achieved_ratio with a Mahalanobis distance_fn ---

def test_achieved_ratio_with_mahalanobis_distance_fn_matches_manual_computation():
    model, train, test = _small_trained_cnn()
    wrapped = FlattenedInputWrapper(model)
    maha_fn = _small_maha_distance_fn(train)

    x = test.x_flat[:16]
    x_adv = (x + 0.05 * torch.randn_like(x)).clamp(0.0, 1.0)

    with torch.no_grad():
        f_x = wrapped(x)
        f_adv = wrapped(x_adv)
    expected_numerator = (f_x - f_adv).norm(p=2, dim=-1)
    expected_denominator = maha_fn(x, x_adv)
    expected = expected_numerator / expected_denominator.clamp_min(1e-12)

    actual = achieved_ratio(wrapped, x, x_adv, distance_fn=maha_fn)
    assert torch.allclose(actual, expected, atol=1e-9)


def test_achieved_ratio_default_distance_fn_unchanged():
    """Backward-compatibility checkpoint: achieved_ratio's default (no distance_fn passed) must
    still be plain Euclidean, exactly as before this parameter was added."""
    model, train, test = _small_trained_cnn()
    wrapped = FlattenedInputWrapper(model)
    x = test.x_flat[:16]
    x_adv = (x + 0.05 * torch.randn_like(x)).clamp(0.0, 1.0)

    default = achieved_ratio(wrapped, x, x_adv)
    explicit_euclidean = achieved_ratio(wrapped, x, x_adv, distance_fn=euclidean_distance_fn)
    assert torch.equal(default, explicit_euclidean)


# --- compute_bounds_with_distance_fn ---

def test_compute_bounds_with_distance_fn_matches_layer_decomposition_experiment_for_euclidean():
    """The central correctness checkpoint for the whole Mahalanobis extension: with
    distance_fn=euclidean_distance_fn, compute_bounds_with_distance_fn must reduce to EXACTLY
    layer_decomposition_experiment(method="pairwise", normalize_features=True)'s own numbers --
    not approximately, since both paths do the identical computation (same pairwise_lipschitz
    calls, same feature-standardization helpers), just reached via two different call sites."""
    model, train, test = _small_trained_cnn()
    x_query, y_query = test.x_flat[:60], test.y[:60]
    x_train_for_norm = train.x_flat[:300]

    generalized = compute_bounds_with_distance_fn(
        model, x_query, y_query, euclidean_distance_fn, x_train_for_norm, seed=0, verbose=False)
    original = layer_decomposition_experiment(
        model, x_query, y_query, x_train_for_norm=x_train_for_norm,
        method="pairwise", normalize_features=True, seed=0, verbose=False)

    assert abs(generalized["L_head_exact"] - original["L_head_exact"]) < 1e-9
    assert abs(generalized["L_extractor_estimated"] - original["L_extractor_estimated"]) < 1e-9
    assert abs(generalized["L_full_estimated"] - original["L_full_estimated"]) < 1e-9
    assert abs(generalized["product"] - original["product"]) < 1e-9
    assert abs(generalized["looseness_ratio"] - original["looseness_ratio"]) < 1e-9


def test_compute_bounds_with_distance_fn_mahalanobis_differs_from_euclidean():
    """Sanity check that switching distance_fn actually changes the bounds (not a no-op) --
    L_head_exact must stay IDENTICAL (see compute_bounds_with_distance_fn's key-insight docstring:
    it never depends on distance_fn), while L_extractor_estimated/L_full_estimated/product change."""
    model, train, test = _small_trained_cnn()
    x_query, y_query = test.x_flat[:60], test.y[:60]
    x_train_for_norm = train.x_flat[:300]
    maha_fn = _small_maha_distance_fn(train)

    euclidean_result = compute_bounds_with_distance_fn(
        model, x_query, y_query, euclidean_distance_fn, x_train_for_norm, seed=0, verbose=False)
    maha_result = compute_bounds_with_distance_fn(
        model, x_query, y_query, maha_fn, x_train_for_norm, seed=0, verbose=False)

    assert abs(euclidean_result["L_head_exact"] - maha_result["L_head_exact"]) < 1e-9
    assert euclidean_result["L_extractor_estimated"] != maha_result["L_extractor_estimated"]
    assert euclidean_result["L_full_estimated"] != maha_result["L_full_estimated"]


# --- run_bound_comparison_with_distance_fn ---

def test_run_bound_comparison_with_distance_fn_euclidean_matches_run_bound_comparison():
    """End-to-end parity: run_bound_comparison_with_distance_fn(..., euclidean_distance_fn) must
    produce the identical summary table (same bounds, same R_adv values -- same attacks, since
    attacks don't depend on distance_fn) as run_bound_comparison itself, given the same inputs."""
    model, train, test = _small_trained_cnn()
    gen = torch.Generator().manual_seed(0)
    query_idx = torch.randperm(len(test), generator=gen)[:40]
    x_query, y_query = test.x_flat[query_idx], test.y[query_idx]
    mask = torch.ones(len(test), dtype=torch.bool)
    mask[query_idx] = False
    pool_idx = mask.nonzero(as_tuple=True)[0][:300]
    x_pool, y_pool = test.x_flat[pool_idx], test.y[pool_idx]
    x_train_for_norm = train.x_flat[:200]

    df_original, sweep_original = run_bound_comparison(
        model, x_query, y_query, x_pool, y_pool, x_train_for_norm,
        epsilons=(0.1, 0.2), pgd_num_steps=5, pgd_num_restarts=2, n_points=50,
        seed=0, verbose=False)
    df_generalized, sweep_generalized = run_bound_comparison_with_distance_fn(
        model, x_query, y_query, x_pool, y_pool, x_train_for_norm, euclidean_distance_fn,
        epsilons=(0.1, 0.2), pgd_num_steps=5, pgd_num_restarts=2, n_points=50,
        seed=0, verbose=False)

    assert list(df_original.columns) == list(df_generalized.columns)
    for col in ("mean_R_adv", "max_R_adv", "L_full_estimated", "product_bound"):
        assert (df_original[col].values == df_generalized[col].values).all(), col
    for key in sweep_original["per_case"]:
        assert torch.equal(sweep_original["per_case"][key]["x_adv"],
                            sweep_generalized["per_case"][key]["x_adv"])
        assert torch.equal(sweep_original["per_case"][key]["R_adv"],
                            sweep_generalized["per_case"][key]["R_adv"])


# --- most_and_least_sensitive_examples + head_layer_bound_check under Mahalanobis ---

def test_most_and_least_sensitive_examples_pixel_distance_uses_mahalanobis_distance_fn():
    model, train, test = _small_trained_cnn()
    wrapped = FlattenedInputWrapper(model)
    maha_fn = _small_maha_distance_fn(train)
    x_pool, y_pool = test.x_flat[:300], test.y[:300]

    sweep = run_epsilon_sweep(wrapped, x_pool, y_pool, epsilons=(0.1, 0.2), pgd_num_steps=5,
                               pgd_num_restarts=2, n_points=50, distance_fn=maha_fn, seed=0, verbose=False)

    for example in most_and_least_sensitive_examples(wrapped, sweep, distance_fn=maha_fn):
        expected = maha_fn(example["x"], example["x_adv"]).item()
        assert abs(example["pixel_distance"] - expected) < 1e-9
        # R_adv must be self-consistent with the Mahalanobis pixel_distance, not a leftover
        # Euclidean value from some other code path.
        head_result = head_layer_bound_check(model, example)
        expected_R_adv = head_result["actual_logit_distance"] / example["pixel_distance"]
        assert abs(example["R_adv"] - expected_R_adv) < 1e-6


# --- run_cnn_adversarial_width_sweep_with_distance_fn ---

def test_run_cnn_adversarial_width_sweep_with_distance_fn_is_deterministic_vs_original():
    """Checkpoint for the determinism claim in run_cnn_adversarial_width_sweep_with_distance_fn's
    docstring: retraining with the same seed/config via the distance_fn-parametrized width sweep
    must reproduce IDENTICAL train_acc/test_acc (and therefore identical weights) to
    run_cnn_adversarial_width_sweep's own per-width models."""
    combined_original, _, _ = run_cnn_adversarial_width_sweep(
        widths=(4, 8), epochs=2, train_subset_size=800, n_query_points=20, n_train_norm_points=80,
        n_pool_points=150, n_attack_points=30, epsilons=(0.15,), pgd_num_steps=3, pgd_num_restarts=1,
        seed=0, verbose=False, save_path=None)
    combined_via_fn, _, _ = run_cnn_adversarial_width_sweep_with_distance_fn(
        euclidean_distance_fn, widths=(4, 8), epochs=2, train_subset_size=800, n_query_points=20,
        n_train_norm_points=80, n_pool_points=150, n_attack_points=30, epsilons=(0.15,),
        pgd_num_steps=3, pgd_num_restarts=1, seed=0, verbose=False, save_path=None)

    assert (combined_original["train_acc"].values == combined_via_fn["train_acc"].values).all()
    assert (combined_original["test_acc"].values == combined_via_fn["test_acc"].values).all()
    # Since the models are identical and distance_fn is identical (Euclidean both times), the
    # measured bounds/R_adv statistics must match too, not just accuracy.
    assert (combined_original["L_full_estimated"].values
            == combined_via_fn["L_full_estimated"].values).all()
    assert (combined_original["max_R_adv_pgd"].values
            == combined_via_fn["max_R_adv_pgd"].values).all()


def test_run_cnn_adversarial_width_sweep_with_distance_fn_mahalanobis_runs_end_to_end():
    train = load_mnist(train=True)
    maha_fn = build_pixel_mahalanobis_distance_fn(train.x_flat[:2000], epsilon=0.01)

    combined_df, per_width_summary_dfs, per_width_extremes = run_cnn_adversarial_width_sweep_with_distance_fn(
        maha_fn, widths=(4, 8), epochs=2, train_subset_size=800, n_query_points=20,
        n_train_norm_points=80, n_pool_points=150, n_attack_points=30, epsilons=(0.15,),
        pgd_num_steps=3, pgd_num_restarts=1, seed=0, verbose=False, save_path=None)

    assert len(combined_df) == 2
    assert set(per_width_summary_dfs.keys()) == {4, 8}
    for width, (most_sensitive, least_sensitive) in per_width_extremes.items():
        assert "feature_distance" in most_sensitive
        assert "pixel_distance" in most_sensitive
        assert most_sensitive["R_adv"] >= least_sensitive["R_adv"]
