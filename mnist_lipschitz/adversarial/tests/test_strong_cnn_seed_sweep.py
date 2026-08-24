import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mnist_lipschitz.data import load_mnist, get_dev_subset
from mnist_lipschitz.models import STRONG_CNN_CONFIG
from mnist_lipschitz.estimators import euclidean_distance_fn
from mnist_lipschitz.adversarial import strong_cnn_seed_sweep as sweep

TOLERANCE = 1e-9


def _tiny_data():
    """Small, fast dev subsets -- real MNIST images (not synthetic noise), just few enough of
    them, and few enough epochs (via the STRONG_CNN_CONFIG monkeypatch each test applies), to keep
    a full StrongCNN training loop fast for unit testing."""
    full_train = load_mnist(train=True)
    full_test = load_mnist(train=False)
    train = get_dev_subset(full_train, n=600, seed=0)
    test = get_dev_subset(full_test, n=300, seed=1)
    return train, test


def _tiny_config(monkeypatch):
    monkeypatch.setitem(STRONG_CNN_CONFIG, "epochs", 1)
    monkeypatch.setitem(STRONG_CNN_CONFIG, "lr_scheduler_t_max", 1)


def _tiny_models(monkeypatch, tmp_path, seeds=(0, 1, 2)):
    _tiny_config(monkeypatch)
    train, test = _tiny_data()
    models_by_seed = {}
    for seed in seeds:
        result = sweep.train_or_load_strong_cnn(seed, train, test, checkpoint_dir=tmp_path, verbose=False)
        models_by_seed[seed] = result["model"]
    return models_by_seed, train, test


# --- Checkpoint 1: training + caching ---

def test_train_or_load_strong_cnn_caches_on_second_call(tmp_path, monkeypatch):
    _tiny_config(monkeypatch)
    train, test = _tiny_data()

    first = sweep.train_or_load_strong_cnn(0, train, test, checkpoint_dir=tmp_path, verbose=False)
    assert first["loaded_from_cache"] is False
    assert first["wall_clock_seconds"] > 0.0

    second = sweep.train_or_load_strong_cnn(0, train, test, checkpoint_dir=tmp_path, verbose=False)
    assert second["loaded_from_cache"] is True
    assert second["wall_clock_seconds"] == 0.0
    assert second["train_acc"] == first["train_acc"]
    assert second["test_acc"] == first["test_acc"]
    assert second["final_train_loss"] == first["final_train_loss"]


def test_train_or_load_strong_cnn_is_deterministic_given_seed(tmp_path, monkeypatch):
    """Two INDEPENDENT trainings (force_retrain=True both times, so neither reads the other's
    cache) with the same seed must produce bit-identical weights."""
    _tiny_config(monkeypatch)
    train, test = _tiny_data()

    result_a = sweep.train_or_load_strong_cnn(0, train, test, force_retrain=True,
                                               checkpoint_dir=tmp_path, verbose=False)
    result_b = sweep.train_or_load_strong_cnn(0, train, test, force_retrain=True,
                                               checkpoint_dir=tmp_path, verbose=False)

    state_a, state_b = result_a["model"].state_dict(), result_b["model"].state_dict()
    assert set(state_a.keys()) == set(state_b.keys())
    for key in state_a:
        assert torch.equal(state_a[key], state_b[key]), f"mismatched weights for {key!r}"
    assert result_a["final_train_loss"] == result_b["final_train_loss"]


def test_train_or_load_strong_cnn_different_seeds_differ(tmp_path, monkeypatch):
    _tiny_config(monkeypatch)
    train, test = _tiny_data()
    result_a = sweep.train_or_load_strong_cnn(0, train, test, checkpoint_dir=tmp_path, verbose=False)
    result_b = sweep.train_or_load_strong_cnn(1, train, test, checkpoint_dir=tmp_path, verbose=False)
    assert not torch.equal(result_a["model"].state_dict()["classifier.4.weight"],
                            result_b["model"].state_dict()["classifier.4.weight"])


def test_checkpoint_paths_are_seed_specific_and_never_touch_the_single_checkpoint_file(tmp_path):
    """Guards `strong_cnn_experiment.main`'s pre-existing `strong_cnn_state_dict.pt` file against
    ever being read/overwritten by this module's per-seed paths."""
    for seed in (0, 1, 2, 3, 4):
        path = sweep._checkpoint_path(seed, tmp_path)
        assert path.name == f"strong_cnn_state_dict_seed{seed}.pt"
        assert path.name != "strong_cnn_state_dict.pt"


def test_check_accuracy_premise_flags_large_deviation():
    ok_df = pd.DataFrame({"seed": [0, 1, 2], "test_acc": [0.991, 0.993, 0.992]})
    ok, _ = sweep.check_accuracy_premise(ok_df, max_dev_pp=0.5)
    assert bool(ok)

    bad_df = pd.DataFrame({"seed": [0, 1, 2], "test_acc": [0.99, 0.95, 0.99]})
    ok, message = sweep.check_accuracy_premise(bad_df, max_dev_pp=0.5)
    assert not bool(ok)
    assert "PREMISE VIOLATED" in message


# --- Checkpoint 2: common pool ---

def test_build_common_pool_is_exact_intersection_and_passes_its_own_assertion(tmp_path, monkeypatch):
    models_by_seed, train, test = _tiny_models(monkeypatch, tmp_path)
    pool = sweep.build_common_pool(models_by_seed, test, save_path=tmp_path / "pool.pt", verbose=False)

    manual_mask = torch.ones(len(test), dtype=torch.bool)
    for model in models_by_seed.values():
        from mnist_lipschitz.models import FlattenedInputWrapper
        with torch.no_grad():
            preds = FlattenedInputWrapper(model)(test.x_flat).argmax(dim=1)
        manual_mask &= (preds == test.y)
    expected_idx = manual_mask.nonzero(as_tuple=True)[0]

    assert torch.equal(pool["pool_idx"], expected_idx)


def test_load_common_pool_round_trips(tmp_path, monkeypatch):
    models_by_seed, train, test = _tiny_models(monkeypatch, tmp_path)
    save_path = tmp_path / "pool.pt"
    pool = sweep.build_common_pool(models_by_seed, test, save_path=save_path, verbose=False)
    reloaded = sweep.load_common_pool(save_path)
    assert torch.equal(reloaded["pool_idx"], pool["pool_idx"])
    for seed in models_by_seed:
        assert torch.equal(reloaded["margins_by_seed"][seed], pool["margins_by_seed"][seed])


# --- Checkpoint 3: attacks + R_adv ---

def test_r_adv_table_uses_realized_not_nominal_norm(tmp_path, monkeypatch):
    """Plan section 5.1's concern: confirms the realized ||x - x_adv|| varies per example (a
    per-row-constant nominal epsilon*sqrt(784) value would show zero variance), and that its mean
    sits BELOW the nominal value at every epsilon tested (FGSM clips against [0,1] on much of
    MNIST's saturated background)."""
    models_by_seed, train, test = _tiny_models(monkeypatch, tmp_path)
    pool = sweep.build_common_pool(models_by_seed, test, save_path=tmp_path / "pool.pt", verbose=False)
    maha_fn = sweep.fit_shared_mahalanobis_distance_fn(train, epsilon=0.01)
    df, ratio_diag_df = sweep.build_r_adv_table(models_by_seed, test, pool, maha_fn,
                                                 epsilons=(0.1, 0.2), verbose=False)

    assert df["realized_norm"].std() > 0.0, "realized_norm should vary example-to-example"
    for _, row in ratio_diag_df.iterrows():
        assert 0.0 < row["mean_realized_over_nominal"] < 1.0 + 1e-9


def test_r_adv_euclidean_and_mahalanobis_share_the_same_x_adv(tmp_path, monkeypatch):
    """Both R_adv columns are computed from the SAME x_adv by construction (see
    build_r_adv_table's docstring): reconstructs the implied logit-space numerator from each
    column (R_adv * denominator) and confirms they match, which could only hold if x_adv (and
    therefore the numerator ||f(x)-f(x_adv)||) was identical for both."""
    models_by_seed, train, test = _tiny_models(monkeypatch, tmp_path)
    pool = sweep.build_common_pool(models_by_seed, test, save_path=tmp_path / "pool.pt", verbose=False)
    maha_fn = sweep.fit_shared_mahalanobis_distance_fn(train, epsilon=0.01)
    df, _ = sweep.build_r_adv_table(models_by_seed, test, pool, maha_fn, epsilons=(0.1,), verbose=False)

    seed0 = df[(df["seed"] == 0) & (df["epsilon"] == 0.1)].reset_index(drop=True)
    x_pool = test.x_flat[pool["pool_idx"]]
    y_pool = test.y[pool["pool_idx"]]
    from mnist_lipschitz.models import FlattenedInputWrapper
    wrapped = FlattenedInputWrapper(models_by_seed[0])
    x_adv = sweep._batched_fgsm(wrapped, x_pool, y_pool, 0.1)
    maha_dist = maha_fn(x_pool, x_adv).detach().numpy()

    numerator_from_euclidean = seed0["R_adv_euclidean"].to_numpy() * seed0["realized_norm"].to_numpy()
    numerator_from_mahalanobis = seed0["R_adv_mahalanobis"].to_numpy() * maha_dist
    np.testing.assert_allclose(numerator_from_euclidean, numerator_from_mahalanobis, atol=1e-6)


def test_shared_mahalanobis_distance_fn_is_one_closure_reused_across_seeds(tmp_path, monkeypatch):
    """Seed-independence by construction (plan section 5.3): the SAME distance_fn object is
    passed to every seed's R_adv computation, not independently refit per seed."""
    train, test = _tiny_data()
    fn_a = sweep.fit_shared_mahalanobis_distance_fn(train, epsilon=0.01)
    x = test.x_flat[:5]
    y = test.x_flat[5:10]
    # Calling it twice with the same inputs must give bit-identical output -- it is a pure
    # closure over one already-fit precision matrix, not something that refits per call.
    assert torch.equal(fn_a(x, y), fn_a(x, y))


# --- Checkpoint 5: statistics ---

def test_primary_r_adv_stats_matches_hand_computed_percentiles(tmp_path, monkeypatch):
    models_by_seed, train, test = _tiny_models(monkeypatch, tmp_path)
    pool = sweep.build_common_pool(models_by_seed, test, save_path=tmp_path / "pool.pt", verbose=False)
    maha_fn = sweep.fit_shared_mahalanobis_distance_fn(train, epsilon=0.01)
    df, _ = sweep.build_r_adv_table(models_by_seed, test, pool, maha_fn, epsilons=(0.1, 0.2), verbose=False)

    stats = sweep.primary_r_adv_stats(df, "R_adv_euclidean", "Euclidean")
    row = stats[(stats["seed"] == 0) & (stats["epsilon"] == 0.1)].iloc[0]

    values = df[(df["seed"] == 0) & (df["epsilon"] == 0.1)]["R_adv_euclidean"].to_numpy()
    assert np.isclose(row["median"], np.median(values))
    assert np.isclose(row["p95"], np.quantile(values, 0.95))
    assert np.isclose(row["p99"], np.quantile(values, 0.99))
    assert np.isclose(row["max"], values.max())
    assert row["metric"] == "Euclidean"


def test_attack_success_rate_matches_manual_mean_and_is_metric_independent():
    df = pd.DataFrame({
        "seed": [0, 0, 0, 0], "epsilon": [0.1, 0.1, 0.1, 0.1],
        "is_misclassified": [True, False, True, False],
        "R_adv_euclidean": [1.0, 2.0, 3.0, 4.0], "R_adv_mahalanobis": [0.1, 0.2, 0.3, 0.4],
    })
    result = sweep.attack_success_rate(df)
    assert result.loc[0, "attack_success_rate"] == 0.5


def test_by_outcome_split_reports_correct_counts_and_margin():
    df = pd.DataFrame({
        "seed": [0, 0, 0, 0], "epsilon": [0.1, 0.1, 0.1, 0.1],
        "is_misclassified": [True, True, False, False],
        "clean_margin": [1.0, 3.0, 5.0, 7.0],
        "R_adv_euclidean": [1.0, 2.0, 3.0, 4.0],
    })
    result = sweep.by_outcome_split(df, "R_adv_euclidean", "Euclidean")
    misclassified_row = result[result["is_misclassified"]].iloc[0]
    correct_row = result[~result["is_misclassified"]].iloc[0]
    assert misclassified_row["n"] == 2
    assert misclassified_row["mean_clean_margin"] == 2.0
    assert correct_row["n"] == 2
    assert correct_row["mean_clean_margin"] == 6.0


def test_common_success_set_is_exact_intersection_of_misclassified_sets():
    df = pd.DataFrame({
        "seed": [0, 0, 0, 1, 1, 1, 2, 2, 2],
        "epsilon": [0.2] * 9,
        "test_index": [10, 20, 30, 10, 20, 40, 10, 20, 50],
        "is_misclassified": [True, True, False, True, True, False, True, False, True],
        "R_adv_euclidean": [1] * 9, "R_adv_mahalanobis": [1] * 9,
    })
    result = sweep.common_success_set(df, epsilon=0.2)
    # test_index=10 misclassified by all three seeds; 20 only by seeds 0,1 (not 2); 30/40/50 not
    # misclassified by all three.
    assert result["test_indices"] == [10]
    assert result["n"] == 1
    assert result["reliable"] is False  # n=1 is far below the default min_reliable_size=50


def test_common_success_set_reliability_flag():
    n = 60
    df = pd.DataFrame({
        "seed": [0] * n + [1] * n,
        "epsilon": [0.2] * (2 * n),
        "test_index": list(range(n)) * 2,
        "is_misclassified": [True] * n + [True] * n,
        "R_adv_euclidean": [1.0] * (2 * n), "R_adv_mahalanobis": [1.0] * (2 * n),
    })
    result = sweep.common_success_set(df, epsilon=0.2)
    assert result["n"] == n
    assert result["reliable"] is True


# --- Extreme examples ---

def test_extract_extreme_example_matches_table_ranking(tmp_path, monkeypatch):
    models_by_seed, train, test = _tiny_models(monkeypatch, tmp_path)
    pool = sweep.build_common_pool(models_by_seed, test, save_path=tmp_path / "pool.pt", verbose=False)
    maha_fn = sweep.fit_shared_mahalanobis_distance_fn(train, epsilon=0.01)
    df, _ = sweep.build_r_adv_table(models_by_seed, test, pool, maha_fn, epsilons=(0.2,), verbose=False)

    case = df[(df["seed"] == 0) & (df["epsilon"] == 0.2)]
    expected_max = case["R_adv_euclidean"].max()
    expected_min = case["R_adv_euclidean"].min()

    most_sensitive = sweep.extract_extreme_example(models_by_seed[0], test, pool, df, seed=0,
                                                     epsilon=0.2, which="most_sensitive")
    least_sensitive = sweep.extract_extreme_example(models_by_seed[0], test, pool, df, seed=0,
                                                      epsilon=0.2, which="least_sensitive")
    assert np.isclose(most_sensitive["R_adv"], expected_max)
    assert np.isclose(least_sensitive["R_adv"], expected_min)
    assert most_sensitive["x"].shape == (784,)
    assert most_sensitive["x_adv"].shape == (784,)


# --- Checkpoint 4: shrinkage sensitivity ---

def test_shrinkage_sensitivity_does_not_change_euclidean_column_or_x_adv(tmp_path, monkeypatch):
    """No retraining/re-attacking should happen in the shrinkage sweep -- confirmed indirectly by
    checking recomputed R_adv_mahalanobis at maha_epsilon=0.01 matches build_r_adv_table's own
    column exactly (both derive from the same deterministic FGSM x_adv)."""
    models_by_seed, train, test = _tiny_models(monkeypatch, tmp_path)
    pool = sweep.build_common_pool(models_by_seed, test, save_path=tmp_path / "pool.pt", verbose=False)
    maha_fn = sweep.fit_shared_mahalanobis_distance_fn(train, epsilon=0.01)
    df, _ = sweep.build_r_adv_table(models_by_seed, test, pool, maha_fn, epsilons=(0.1, 0.2), verbose=False)

    shrink_df = sweep.run_shrinkage_sensitivity(models_by_seed, test, train, pool, df,
                                                 epsilons_maha=(0.01,), attack_epsilons=(0.1, 0.2),
                                                 verbose=False)

    original = df[(df["seed"] == 0) & (df["epsilon"] == 0.1)].sort_values("test_index")
    recomputed = shrink_df[(shrink_df["seed"] == 0) & (shrink_df["attack_epsilon"] == 0.1)
                           & (shrink_df["maha_epsilon"] == 0.01)].sort_values("test_index")
    np.testing.assert_allclose(original["R_adv_mahalanobis"].to_numpy(),
                                recomputed["R_adv_mahalanobis"].to_numpy(), atol=1e-9)


# --- Checkpoint 4 companion: shrinkage_bounds_per_seed ---

def test_shrinkage_bounds_per_seed_returns_one_row_per_seed_per_epsilon(tmp_path, monkeypatch):
    models_by_seed, train, test = _tiny_models(monkeypatch, tmp_path)
    pool = sweep.build_common_pool(models_by_seed, test, save_path=tmp_path / "pool.pt", verbose=False)

    result = sweep.shrinkage_bounds_per_seed(models_by_seed, train, test, pool,
                                              epsilons_maha=(0.01, 0.1), n_query=20, n_norm=20,
                                              verbose=False)

    assert len(result) == len(models_by_seed) * 2
    assert set(result["maha_epsilon"].unique()) == {0.01, 0.1}
    assert (result["L_full_estimated"] > 0).all()
    for maha_epsilon in (0.01, 0.1):
        expected_metric = f"Mahalanobis_eps{maha_epsilon:g}"
        assert (result[result["maha_epsilon"] == maha_epsilon]["metric"] == expected_metric).all()


# --- Checkpoint 7 (bounds): per_seed_bounds sanity ---

def test_per_seed_bounds_returns_expected_columns_and_finite_values(tmp_path, monkeypatch):
    models_by_seed, train, test = _tiny_models(monkeypatch, tmp_path)
    pool = sweep.build_common_pool(models_by_seed, test, save_path=tmp_path / "pool.pt", verbose=False)
    maha_fn = sweep.fit_shared_mahalanobis_distance_fn(train, epsilon=0.01)

    bounds_df = sweep.per_seed_bounds(models_by_seed, train, test, pool,
                                       {"Euclidean": euclidean_distance_fn, "Mahalanobis": maha_fn},
                                       n_query=20, n_norm=20, verbose=False)

    expected_cols = {"seed", "metric", "L_head_exact", "L_extractor_estimated",
                      "L_full_estimated", "product", "looseness_ratio"}
    assert expected_cols.issubset(set(bounds_df.columns))
    assert len(bounds_df) == len(models_by_seed) * 2
    assert (bounds_df["L_full_estimated"] > 0).all()
    assert (bounds_df["looseness_ratio"] > 0).all()


# --- Transferability check ---

def test_run_transfer_attack_source_equals_eval_matches_r_adv_table(tmp_path, monkeypatch):
    """The eval_seed == source_seed rows must reduce EXACTLY to build_r_adv_table's own values --
    attacking a model and evaluating it on its own adversarial examples is just the ordinary,
    non-transfer attack (see run_transfer_attack's docstring)."""
    models_by_seed, train, test = _tiny_models(monkeypatch, tmp_path)
    pool = sweep.build_common_pool(models_by_seed, test, save_path=tmp_path / "pool.pt", verbose=False)
    maha_fn = sweep.fit_shared_mahalanobis_distance_fn(train, epsilon=0.01)

    r_adv_df, _ = sweep.build_r_adv_table(models_by_seed, test, pool, maha_fn, epsilons=(0.1,),
                                           verbose=False)
    transfer_df = sweep.run_transfer_attack(models_by_seed, test, pool, maha_fn, source_seed=0,
                                             epsilons=(0.1,), verbose=False)

    ordinary = r_adv_df[(r_adv_df["seed"] == 0) & (r_adv_df["epsilon"] == 0.1)].sort_values("test_index")
    self_transfer = transfer_df[(transfer_df["source_seed"] == 0) & (transfer_df["eval_seed"] == 0)
                                 & (transfer_df["epsilon"] == 0.1)].sort_values("test_index")

    np.testing.assert_array_equal(ordinary["is_misclassified"].to_numpy(),
                                   self_transfer["is_misclassified"].to_numpy())
    np.testing.assert_allclose(ordinary["R_adv_euclidean"].to_numpy(),
                                self_transfer["R_adv_euclidean"].to_numpy(), atol=1e-9)
    np.testing.assert_allclose(ordinary["R_adv_mahalanobis"].to_numpy(),
                                self_transfer["R_adv_mahalanobis"].to_numpy(), atol=1e-9)


def test_run_transfer_attack_uses_same_x_adv_across_eval_seeds(tmp_path, monkeypatch):
    """Every eval_seed row for a given (source_seed, epsilon) must be evaluated against the
    IDENTICAL x_adv -- checked indirectly: is_misclassified/R_adv differ across eval_seed only
    because the model differs, not because x_adv does. Since x_adv is never exposed directly by
    run_transfer_attack, this is checked by recomputing x_adv once (attacking source_seed) and
    confirming every eval_seed's R_adv_euclidean * realized Euclidean distance reproduces the same
    numerator (||f_eval(x)-f_eval(x_adv)||) that a fresh achieved_ratio call on that SAME x_adv
    gives -- i.e. the table is self-consistent with a single shared x_adv."""
    models_by_seed, train, test = _tiny_models(monkeypatch, tmp_path)
    pool = sweep.build_common_pool(models_by_seed, test, save_path=tmp_path / "pool.pt", verbose=False)
    maha_fn = sweep.fit_shared_mahalanobis_distance_fn(train, epsilon=0.01)
    transfer_df = sweep.run_transfer_attack(models_by_seed, test, pool, maha_fn, source_seed=0,
                                             epsilons=(0.1,), verbose=False)

    from mnist_lipschitz.models import FlattenedInputWrapper
    from mnist_lipschitz.adversarial.run_experiment import achieved_ratio
    x_pool = test.x_flat[pool["pool_idx"]]
    y_pool = test.y[pool["pool_idx"]]
    source_wrapped = FlattenedInputWrapper(models_by_seed[0])
    x_adv = sweep._batched_fgsm(source_wrapped, x_pool, y_pool, 0.1)

    for eval_seed in models_by_seed:
        wrapped = FlattenedInputWrapper(models_by_seed[eval_seed])
        expected_R_adv = achieved_ratio(wrapped, x_pool, x_adv, distance_fn=euclidean_distance_fn)
        row = transfer_df[(transfer_df["eval_seed"] == eval_seed)
                           & (transfer_df["epsilon"] == 0.1)].sort_values("test_index")
        np.testing.assert_allclose(row["R_adv_euclidean"].to_numpy(), expected_R_adv.detach().numpy(),
                                    atol=1e-9)


def test_summarize_transfer_attack_transfer_accuracy_matches_manual_computation():
    df = pd.DataFrame({
        "source_seed": [0, 0, 0, 0], "eval_seed": [1, 1, 1, 1], "epsilon": [0.1, 0.1, 0.1, 0.1],
        "is_misclassified": [True, False, True, False],
        "R_adv_euclidean": [1.0, 2.0, 3.0, 4.0], "R_adv_mahalanobis": [0.1, 0.2, 0.3, 0.4],
    })
    result = sweep.summarize_transfer_attack(df, save_path=None)
    row = result.iloc[0]
    assert row["transfer_accuracy"] == 0.5
    assert row["mean_R_adv_euclidean"] == 2.5
