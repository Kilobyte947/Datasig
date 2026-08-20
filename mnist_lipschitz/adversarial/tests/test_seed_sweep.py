import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mnist_lipschitz.data import load_mnist
from mnist_lipschitz.estimators import euclidean_distance_fn
from mnist_lipschitz.adversarial.run_experiment import (
    build_pixel_mahalanobis_distance_fn,
    run_cnn_adversarial_width_sweep_with_distance_fn,
)
from mnist_lipschitz.adversarial.seed_sweep import (
    run_single_seed_width,
    run_seed_sweep,
    summarize_seed_sweep,
    run_attack_seed_variance_decomposition,
    train_or_load_checkpoint,
)

TINY_KWARGS = dict(epochs=2, train_subset_size=500, n_query_points=15, n_train_norm_points=60,
                    n_pool_points=100, n_attack_points=20, epsilons=(0.15,),
                    pgd_num_steps=3, pgd_num_restarts=1, max_pairs=None)


# --- run_seed_sweep reproduces the existing single-seed width sweep exactly ---

def test_run_seed_sweep_single_seed_width16_matches_existing_width_sweep_exactly(tmp_path):
    train = load_mnist(train=True)
    maha_fn = build_pixel_mahalanobis_distance_fn(train.x_flat[:500], epsilon=0.01)

    combined_eucl, _, _ = run_cnn_adversarial_width_sweep_with_distance_fn(
        euclidean_distance_fn, widths=(16,), seed=0, verbose=False, save_path=None, **TINY_KWARGS)
    combined_maha, _, _ = run_cnn_adversarial_width_sweep_with_distance_fn(
        maha_fn, widths=(16,), seed=0, verbose=False, save_path=None, **TINY_KWARGS)

    sweep_df = run_seed_sweep(
        seeds=range(1), widths=(16,), maha_fit_size=500, maha_epsilon=0.01,
        checkpoint_dir=tmp_path / "ckpts", verbose=False, save_path=None, **TINY_KWARGS)

    eucl_row = sweep_df[(sweep_df["metric"] == "Euclidean") & (sweep_df["method"] == "PGD")].iloc[0]
    maha_row = sweep_df[(sweep_df["metric"] == "Mahalanobis") & (sweep_df["method"] == "PGD")].iloc[0]

    assert abs(eucl_row["L_full_estimated"] - combined_eucl["L_full_estimated"].iloc[0]) < 1e-9
    assert abs(eucl_row["max_R_adv"] - combined_eucl["max_R_adv_pgd"].iloc[0]) < 1e-9
    assert abs(maha_row["L_full_estimated"] - combined_maha["L_full_estimated"].iloc[0]) < 1e-9
    assert abs(maha_row["max_R_adv"] - combined_maha["max_R_adv_pgd"].iloc[0]) < 1e-9


# --- output shape / completeness ---

def test_run_seed_sweep_output_shape_and_no_nulls(tmp_path):
    n_seeds, widths = 2, (16, 32)
    epsilons = TINY_KWARGS["epsilons"]
    n_methods, n_metrics = 2, 2

    df = run_seed_sweep(
        seeds=range(n_seeds), widths=widths, maha_fit_size=500, maha_epsilon=0.01,
        checkpoint_dir=tmp_path / "ckpts", verbose=False, save_path=None, **TINY_KWARGS)

    expected_rows = n_seeds * len(widths) * len(epsilons) * n_methods * n_metrics
    assert len(df) == expected_rows

    required_cols = [
        "seed", "width", "metric", "epsilon", "method", "mean_R_adv", "max_R_adv",
        "pct_misclassified", "L_full_estimated", "product_bound", "L_head_exact",
        "L_extractor_estimated", "looseness_ratio", "L_margin_estimated",
        "mean_logit_norm", "std_logit_norm", "mean_top2_margin", "std_top2_margin",
        "p5_top2_margin", "p10_top2_margin", "n_flipped", "n_evaluated",
        "mean_cosine_alignment", "std_cosine_alignment",
    ]
    for col in required_cols:
        assert col in df.columns, col
        assert df[col].isnull().sum() == 0, col

    assert set(df["metric"].unique()) == {"Euclidean", "Mahalanobis"}
    assert set(df["width"].unique()) == set(widths)
    assert set(df["seed"].unique()) == set(range(n_seeds))


# --- L_head_exact metric-independence, checked directly on the sweep's own output ---

def test_run_seed_sweep_L_head_exact_identical_across_metrics_within_group(tmp_path):
    df = run_seed_sweep(
        seeds=range(1), widths=(16,), maha_fit_size=500, maha_epsilon=0.01,
        checkpoint_dir=tmp_path / "ckpts", verbose=False, save_path=None, **TINY_KWARGS)

    for (seed, width), group in df.groupby(["seed", "width"]):
        vals = group["L_head_exact"].values
        assert (abs(vals - vals[0]) < 1e-9).all(), (seed, width, vals)


# --- summarize_seed_sweep (Checkpoint 4) ---

_MECHANISM_COLS = ("mean_logit_norm", "mean_top2_margin", "L_full_estimated",
                    "L_margin_estimated", "mean_cosine_alignment")


def test_summarize_seed_sweep_per_config_correct_mean_and_std():
    """Synthetic frame with known values, single (width, epsilon, method, metric) group."""
    values = [10.0, 20.0, 30.0]
    rows = []
    for seed, val in zip((0, 1, 2), values):
        row = {"seed": seed, "width": 16, "epsilon": 0.1, "method": "PGD", "metric": "Euclidean",
               "mean_R_adv": val}
        row.update({col: 1.0 for col in _MECHANISM_COLS})
        rows.append(row)
    df = pd.DataFrame(rows)

    result = summarize_seed_sweep(df, width_pairs=(), save_dir=None)
    per_config = result["per_config"]
    assert len(per_config) == 1
    row = per_config.iloc[0]

    expected_series = pd.Series(values)
    assert abs(row["mean_R_adv_mean"] - expected_series.mean()) < 1e-9
    assert abs(row["mean_R_adv_std"] - expected_series.std()) < 1e-9
    assert row["mean_R_adv_min"] == 10.0
    assert row["mean_R_adv_max"] == 30.0


def test_summarize_seed_sweep_sign_counts_on_mixed_sign_hand_built_frame():
    """5 seeds, width=16 vs width=32, misclassification_rate diff = [+0.1, 0.0, +0.1, -0.2, +0.05]
    (seed 0..4) -> sign(diff) = [+1, 0, +1, -1, +1]. reference_seed=0 -> reference_sign=+1, and
    exactly 3 of the 5 seeds (0, 2, 4) share that sign -- k_matching_sign must be 3, not 5 (a
    naive "any nonzero same-direction count" bug) and not 4 (mustn't count the zero-diff seed as
    a match)."""
    width16_misclass = [0.5, 0.4, 0.6, 0.3, 0.55]
    width32_misclass = [0.4, 0.4, 0.5, 0.5, 0.5]
    rows = []
    for seed in range(5):
        for width, misclass in ((16, width16_misclass[seed]), (32, width32_misclass[seed])):
            row = {"seed": seed, "width": width, "epsilon": 0.25, "method": "PGD", "metric": "Euclidean",
                   "misclassification_rate": misclass, "L_full_estimated": 1.0, "max_R_adv": 1.0,
                   "mean_top2_margin": 1.0, "mean_logit_norm": 1.0, "L_margin_estimated": 1.0,
                   "mean_cosine_alignment": 1.0}
            rows.append(row)
    df = pd.DataFrame(rows)

    result = summarize_seed_sweep(df, width_pairs=((16, 32),), reference_seed=0, save_dir=None)
    paired = result["paired_differences"]
    row = paired[(paired["width_low"] == 16) & (paired["width_high"] == 32)
                 & (paired["quantity"] == "misclassification_rate")].iloc[0]

    assert row["reference_sign"] == 1
    assert row["n_seeds"] == 5
    assert row["k_matching_sign"] == 3
    expected_diffs = [a - b for a, b in zip(width16_misclass, width32_misclass)]
    assert abs(row["mean_diff"] - (sum(expected_diffs) / 5)) < 1e-9


# --- run_attack_seed_variance_decomposition (Checkpoint 6, optional) ---

def test_run_attack_seed_variance_decomposition_reuses_cached_checkpoint(tmp_path):
    """train_seed is fixed while attack_seed varies -- pre-populate the checkpoint cache once,
    then confirm every (width, attack_seed) row reports the SAME train_acc/test_acc (the model
    was loaded from the shared checkpoint, not independently retrained per attack_seed)."""
    checkpoint_dir = tmp_path / "ckpts"
    train_or_load_checkpoint(train_seed=0, width=16, epochs=2, train_subset_size=500,
                              checkpoint_dir=checkpoint_dir, verbose=False)

    df = run_attack_seed_variance_decomposition(
        widths=(16,), attack_seeds=(0, 1), train_seed=0, maha_fit_size=500, maha_epsilon=0.01,
        checkpoint_dir=checkpoint_dir, verbose=False, save_path=None, **TINY_KWARGS)

    assert set(df["train_seed"].unique()) == {0}
    assert set(df["attack_seed"].unique()) == {0, 1}
    assert df["train_acc"].nunique() == 1
    assert df["test_acc"].nunique() == 1

    expected_rows = len(df["attack_seed"].unique()) * len(TINY_KWARGS["epsilons"]) * 2 * 2  # methods x metrics
    assert len(df) == expected_rows
