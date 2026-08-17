import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mnist_lipschitz.data import load_mnist, get_dev_subset, make_loader
from mnist_lipschitz.models import LogisticRegressionModel, train_classifier
from mnist_lipschitz.distance import svd_ridge_precision, make_mahalanobis_distance_fn
from mnist_lipschitz.run_experiment import (
    sweep_epsilon, epsilon_stability_check, select_epsilon,
    run_ratio_distribution_analysis, run_embedding_degree_sweep,
)

EPSILON_VALUES = [1e-6, 1e-1, 10.0]  # near-singular, moderate, well-regularized


def _dev_model_and_data(seed=0):
    torch.manual_seed(seed)
    train = load_mnist(train=True)
    test = load_mnist(train=False)
    dev = get_dev_subset(train, n=2000, seed=seed)
    train_loader = make_loader(dev.x_flat, dev.y, batch_size=128, shuffle=True, seed=seed)
    test_loader = make_loader(test.x_flat, test.y, batch_size=1000, shuffle=False)
    model, _, _ = train_classifier(LogisticRegressionModel(), train_loader, test_loader,
                                    epochs=5, lr=1e-3, verbose=False)
    return model, dev


def test_condition_number_monotonically_non_increasing_with_epsilon():
    # Deterministic given fixed Sigma: cond(Sigma+eps*I) is provably
    # non-increasing in eps for eps > 0 (eigenvalues (lambda+eps) all get
    # pulled toward 1:1 ratios as eps grows), so this should never be flaky.
    torch.manual_seed(0)
    _, dev = _dev_model_and_data(seed=0)

    epsilon_values = [1e-6, 1e-3, 1e-1, 1.0, 10.0, 100.0]
    conds = sweep_epsilon(dev.x_flat, epsilon_values)

    for a, b in zip(conds, conds[1:]):
        assert b <= a + 1e-6, f"condition number increased: {conds}"


def test_instability_does_not_increase_from_near_singular_to_well_regularized():
    """If more regularization made estimates LESS reproducible across
    independent resamples, that would indicate a broken implementation
    (e.g. a sign error making the "ridge" term destabilizing instead of
    stabilizing). Checked at 3 well-separated epsilon values (not a dense
    sweep) specifically because the raw coefficient-of-variation numbers
    have real sampling noise at n_subsamples=10 -- a dense sweep over many
    close epsilon values is not reliably monotonic point-to-point on real
    MNIST subsamples (confirmed empirically during development), but the
    overall near-singular -> well-regularized trend is consistently
    non-increasing across repeated runs with different seeds.
    """
    model, dev = _dev_model_and_data(seed=0)
    results = epsilon_stability_check(model, dev, EPSILON_VALUES, n_subsamples=10,
                                       subsample_frac=0.8, n_points=100, seed=0, verbose=False)
    cv_values = [results[eps]["cv"] for eps in EPSILON_VALUES]

    assert all(torch.isfinite(torch.tensor(cv_values)))
    # Well-regularized epsilon must be clearly more stable than near-singular epsilon.
    assert cv_values[-1] < cv_values[0], f"instability increased end-to-end: {cv_values}"
    # The moderate epsilon shouldn't be meaningfully worse than the near-singular one either
    # (small tolerance for sampling noise in the near-flat low-epsilon region).
    assert cv_values[1] <= cv_values[0] * 1.2, f"instability increased at moderate epsilon: {cv_values}"


def test_select_epsilon_picks_smallest_epsilon_meeting_both_bounds():
    epsilon_values = [1e-6, 1e-1, 10.0]
    cond_numbers = [1e7, 50.0, 1.5]
    cv_values = [0.05, 0.02, 0.01]

    chosen = select_epsilon(epsilon_values, cond_numbers, cv_values, max_cond=1e4, max_cv=0.03, verbose=False)
    assert chosen == 1e-1  # smallest epsilon meeting both cond<=1e4 and cv<=0.03


def test_select_epsilon_falls_back_when_nothing_qualifies():
    epsilon_values = [1e-6, 1e-1, 10.0]
    cond_numbers = [1e7, 1e6, 1e5]  # none meet max_cond
    cv_values = [0.05, 0.02, 0.01]

    chosen = select_epsilon(epsilon_values, cond_numbers, cv_values, max_cond=1e2, max_cv=0.5, verbose=False)
    assert chosen == 10.0  # falls back to lowest cv


def test_run_embedding_degree_sweep_degree_1_matches_raw_pixel_baseline():
    """elementwise_embedding(., degree=1) is the identity (see tests/test_embeddings.py), so
    run_embedding_degree_sweep's degree=1 result -- selected epsilon, condition number, and
    ratio-distribution summary stats -- must exactly match what the pre-existing raw-pixel
    (embed_fn=None) pipeline produces on the identical pool/model/seed. Pinning `epsilon_pool`
    and `train`/`test` to the exact same objects in both paths (rather than re-drawing a "same
    size" pool, which would land on different actual points via a second independent
    torch.randperm call) is what makes this an exact-match checkpoint rather than merely
    close -- the checkpoint that the new embed_fn-aware plumbing doesn't silently change
    degree=1 behavior, before trusting it at degree > 1 where there's no raw-pixel baseline left
    to compare against."""
    model, dev = _dev_model_and_data(seed=0)
    test_data = get_dev_subset(load_mnist(train=False), n=300, seed=0)
    epsilon_pool = get_dev_subset(dev, n=len(dev), seed=0)

    epsilon_values = [1e-6, 1e-1, 10.0]
    shared_kwargs = dict(epsilon_values=epsilon_values, n_subsamples=5, subsample_frac=0.8,
                          stability_n_points=50, max_cond=1e4, max_cv=0.05,
                          n_ratio_points=100, k_neighbors=5, seed=0, verbose=False)

    # Raw-pixel baseline, computed directly (embed_fn=None throughout).
    cond_numbers = sweep_epsilon(epsilon_pool.x_flat, epsilon_values)
    stability_results = epsilon_stability_check(
        model, epsilon_pool, epsilon_values, n_subsamples=5, subsample_frac=0.8,
        n_points=50, seed=0, verbose=False)
    cv_values = [stability_results[eps]["cv"] for eps in epsilon_values]
    selected_epsilon = select_epsilon(epsilon_values, cond_numbers, cv_values,
                                       max_cond=1e4, max_cv=0.05, verbose=False)
    precision = svd_ridge_precision(dev.x_flat, selected_epsilon)
    distance_fn = make_mahalanobis_distance_fn(precision)
    baseline_ratio = run_ratio_distribution_analysis(
        model, "logistic_regression", "baseline_raw_pixel", test_data.x_flat, test_data.y,
        distance_fn, n_points=100, k_neighbors=5, seed=0, verbose=False)

    sweep_result = run_embedding_degree_sweep(
        degrees=(1,), lr_model=model, train=dev, test=test_data, epsilon_pool=epsilon_pool,
        **shared_kwargs)

    degree1 = sweep_result["degree_results"][1]
    assert degree1["selected_epsilon"] == selected_epsilon
    assert abs(degree1["cond_number_at_selected_epsilon"]
               - cond_numbers[epsilon_values.index(selected_epsilon)]) < 1e-6

    for key in ("all_pairs_mean", "near_neighbor_mean", "all_pairs_max", "near_neighbor_max"):
        assert abs(degree1["ratio_summary"][key] - baseline_ratio["summary"][key]) < 1e-6, (
            f"{key}: {degree1['ratio_summary'][key]} vs {baseline_ratio['summary'][key]}")
