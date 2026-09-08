"""Multi-seed confirmation sweep for the CNN-width adversarial comparison, checking whether the
width-32/width-64 misclassification-rate inversion found in run_experiment.py survives reseeding,
and which mechanism explains it: logit scale, margin Lipschitz constant, or attack-direction alignment.
"""

from pathlib import Path

import pandas as pd
import torch

from mnist_example.data import load_mnist, get_dev_subset, make_loader
from mnist_example.models import SmallCNN, FlattenedInputWrapper, train_classifier
from mnist_example.estimators import euclidean_distance_fn
from mnist_example.adversarial_run_experiment import (
    RESULTS_DIR,
    DEFAULT_EPSILONS,
    MAHALANOBIS_EPSILON,
    build_pixel_mahalanobis_distance_fn,
    compute_bounds_with_distance_fn,
    run_epsilon_sweep,
    summarize_epsilon_sweep,
    adversarial_accuracy,
    clean_logit_stats,
    margin_lipschitz_estimate,
    flip_direction_alignment,
)

CHECKPOINT_DIR = RESULTS_DIR / "seed_sweep_checkpoints"


def _checkpoint_path(checkpoint_dir, train_seed, width):
    """Checkpoint file path for (train_seed, width), or None if checkpoint_dir is None."""
    if checkpoint_dir is None:
        return None
    return Path(checkpoint_dir) / f"train_seed{train_seed}_width{width}.pt"


def train_or_load_checkpoint(train_seed, width, epochs=6, train_subset_size=5000,
                              checkpoint_dir=CHECKPOINT_DIR, verbose=True):
    """Loads a cached SmallCNN checkpoint for (train_seed, width) if one exists, else trains and 
    saves one. Takes train_seed only — attack_seed never influences training. 
    Returns (model, train, test, train_acc, test_acc)."""
    path = _checkpoint_path(checkpoint_dir, train_seed, width)
    train = load_mnist(train=True)
    test = load_mnist(train=False)

    if path is not None and path.exists():
        state = torch.load(path, weights_only=True)
        model = SmallCNN(conv_channels=(width, 2 * width))
        model.load_state_dict(state["model_state_dict"])
        train_acc, test_acc = state["train_acc"], state["test_acc"]
        if verbose:
            print(f"  [checkpoint] loaded train_seed={train_seed} width={width} from {path} "
                  f"(train_acc={train_acc:.4f}  test_acc={test_acc:.4f})")
        return model, train, test, train_acc, test_acc

    dev = get_dev_subset(train, n=train_subset_size, seed=train_seed)
    train_loader = make_loader(dev.x_image, dev.y, batch_size=128, shuffle=True, seed=train_seed)
    test_loader = make_loader(test.x_image, test.y, batch_size=1000, shuffle=False)

    torch.manual_seed(train_seed)
    model, train_acc, test_acc = train_classifier(
        SmallCNN(conv_channels=(width, 2 * width)), train_loader, test_loader,
        epochs=epochs, lr=1e-3, verbose=False)
    if verbose:
        print(f"  [trained] train_seed={train_seed} width={width}  "
              f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "train_acc": train_acc, "test_acc": test_acc}, path)

    return model, train, test, train_acc, test_acc


def run_single_seed_width(train_seed, attack_seed, width, distance_fn,
                           epochs=6, train_subset_size=5000, n_query_points=40,
                           n_train_norm_points=500, n_pool_points=2000, n_attack_points=500,
                           epsilons=DEFAULT_EPSILONS, pgd_alpha_frac=0.25, pgd_num_steps=20,
                           pgd_num_restarts=5, max_pairs=None, margin_estimator="pairwise",
                           checkpoint_dir=CHECKPOINT_DIR, metric_name="Euclidean", verbose=True):
    """Runs the full measurement battery for one (train_seed, attack_seed, width, distance_fn)
    combination: trains or loads the checkpoint, runs the epsilon x method attack grid, and computes 
    every bound, R_adv, and mechanism diagnostic (logit stats, margin Lipschitz estimate, direction alignment).
    
    train_seed controls only model init and training-data order; attack_seed controls only
    query/pool sampling and attack randomness — the two are never derived from each other. distance_fn 
    must already be fitted; it's never refit here.
    
    Returns one row per (epsilon, method), covering both the bound/R_adv columns and the mechanism diagnostics."""
    model, train, test, train_acc, test_acc = train_or_load_checkpoint(
        train_seed, width, epochs=epochs, train_subset_size=train_subset_size,
        checkpoint_dir=checkpoint_dir, verbose=verbose)
    wrapped_model = FlattenedInputWrapper(model)

    generator = torch.Generator().manual_seed(attack_seed)
    query_idx = torch.randperm(len(test), generator=generator)[:n_query_points]
    x_query, y_query = test.x_flat[query_idx], test.y[query_idx]
    norm_idx = torch.randperm(len(train), generator=generator)[:n_train_norm_points]
    x_train_for_norm = train.x_flat[norm_idx]

    pool_mask = torch.ones(len(test), dtype=torch.bool)
    pool_mask[query_idx] = False
    remaining_idx = pool_mask.nonzero(as_tuple=True)[0]
    pool_idx = remaining_idx[torch.randperm(len(remaining_idx), generator=generator)[:n_pool_points]]
    x_pool, y_pool = test.x_flat[pool_idx], test.y[pool_idx]

    bound_result = compute_bounds_with_distance_fn(
        model, x_query, y_query, distance_fn, x_train_for_norm,
        max_pairs=max_pairs, seed=attack_seed, verbose=verbose)
    L_full_estimated = bound_result["L_full_estimated"]
    product_bound = bound_result["product"]

    sweep_results = run_epsilon_sweep(
        wrapped_model, x_pool, y_pool, epsilons=epsilons, pgd_alpha_frac=pgd_alpha_frac,
        pgd_num_steps=pgd_num_steps, pgd_num_restarts=pgd_num_restarts, n_points=n_attack_points,
        distance_fn=distance_fn, seed=attack_seed, verbose=verbose)
    summary_df = summarize_epsilon_sweep(sweep_results, L_full_estimated, product_bound, verbose=verbose)

    x_eval, y_eval = sweep_results["x_eval"], sweep_results["y_eval"]
    L_margin_estimated = margin_lipschitz_estimate(
        wrapped_model, x_query, y_query, estimator=margin_estimator, distance_fn=distance_fn,
        max_pairs=max_pairs, seed=attack_seed)
    logit_stats = clean_logit_stats(wrapped_model, x_eval)

    rows = []
    for _, summary_row in summary_df.iterrows():
        epsilon, method = summary_row["epsilon"], summary_row["method"]
        x_adv = sweep_results["per_case"][(epsilon, method)]["x_adv"]
        acc_stats = adversarial_accuracy(wrapped_model, x_eval, x_adv, y_eval)
        align_stats = flip_direction_alignment(wrapped_model, x_eval, x_adv, y_eval)

        rows.append({
            "train_seed": train_seed, "attack_seed": attack_seed, "width": width,
            "metric": metric_name, "train_acc": train_acc, "test_acc": test_acc,
            **summary_row.to_dict(),
            "L_head_exact": bound_result["L_head_exact"],
            "L_extractor_estimated": bound_result["L_extractor_estimated"],
            "looseness_ratio": bound_result["looseness_ratio"],
            "L_margin_estimated": L_margin_estimated,
            **logit_stats,
            "misclassification_rate": acc_stats["misclassification_rate"],
            "n_flipped": acc_stats["n_flipped"],
            "n_evaluated": acc_stats["n_evaluated"],
            "mean_cosine_alignment": align_stats["mean_cosine_alignment"],
            "std_cosine_alignment": align_stats["std_cosine_alignment"],
        })

    return pd.DataFrame(rows)


def run_seed_sweep(seeds=range(5), widths=(16, 32, 64), maha_fit_size=60000,
                    maha_epsilon=MAHALANOBIS_EPSILON, epochs=6, train_subset_size=5000,
                    n_query_points=40, n_train_norm_points=500, n_pool_points=2000,
                    n_attack_points=500, epsilons=DEFAULT_EPSILONS, pgd_alpha_frac=0.25,
                    pgd_num_steps=20, pgd_num_restarts=5, max_pairs=None,
                    margin_estimator="pairwise", checkpoint_dir=CHECKPOINT_DIR,
                    verbose=True, save_path=RESULTS_DIR / "seed_sweep_raw.csv"):
    """Runs the full measurement battery for one (train_seed, attack_seed, width, distance_fn) combination: 
    trains or loads the checkpoint, runs the epsilon x method attack grid, and computes 
    every bound, R_adv, and mechanism diagnostic (logit stats, margin Lipschitz estimate, direction alignment).
    
    train_seed controls only model init and training-data order; attack_seed controls only 
    query/pool sampling and attack randomness. The two are never derived from one another.
    
    Returns one row per (epsilon, method), covering both the bound/R_adv columns and the mechanism diagnostics."""
    train = load_mnist(train=True)
    if verbose:
        print(f"Fitting Mahalanobis precision matrix once from {maha_fit_size} training points...")
    maha_distance_fn = build_pixel_mahalanobis_distance_fn(train.x_flat[:maha_fit_size], epsilon=maha_epsilon)

    all_dfs = []
    for s in seeds:
        for width in widths:
            if verbose:
                print(f"=== seed={s} width={width} ===")
            for distance_fn, metric_name in ((euclidean_distance_fn, "Euclidean"),
                                              (maha_distance_fn, "Mahalanobis")):
                df = run_single_seed_width(
                    train_seed=s, attack_seed=s, width=width, distance_fn=distance_fn,
                    epochs=epochs, train_subset_size=train_subset_size, n_query_points=n_query_points,
                    n_train_norm_points=n_train_norm_points, n_pool_points=n_pool_points,
                    n_attack_points=n_attack_points, epsilons=epsilons, pgd_alpha_frac=pgd_alpha_frac,
                    pgd_num_steps=pgd_num_steps, pgd_num_restarts=pgd_num_restarts, max_pairs=max_pairs,
                    margin_estimator=margin_estimator, checkpoint_dir=checkpoint_dir,
                    metric_name=metric_name, verbose=verbose)
                all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.rename(columns={"train_seed": "seed"}).drop(columns=["attack_seed"])

    if save_path is not None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        combined.to_csv(save_path, index=False)
        if verbose:
            print(f"\nSaved seed-sweep raw results to {save_path}")

    return combined

PAIRED_DIFF_QUANTITIES = ("misclassification_rate", "L_full_estimated", "max_R_adv",
                           "mean_top2_margin", "mean_logit_norm")

MECHANISM_TABLE_COLUMNS = ("mean_logit_norm", "mean_top2_margin", "L_full_estimated",
                            "L_margin_estimated", "mean_cosine_alignment")

DEFAULT_WIDTH_PAIRS = ((16, 32), (32, 64), (16, 64))


def _sign(value):
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def summarize_seed_sweep(df, width_pairs=DEFAULT_WIDTH_PAIRS, reference_seed=None, save_dir=RESULTS_DIR):
    """Aggregates run_seed_sweep's raw output into three tables: per_config, paired_differences, 
    and mechanism_table. reference_seed defaults to the smallest seed present. Also saved as CSV if save_dir is set.
    
    Note on paired_differences: pairing is by seed index only. A shared seed gives two widths the 
    same data order and attack randomness, but not the same model initialisation, since SmallCNN's
    random init also depends on width — so this is a partial pairing, not a true matched-pairs
    design. Report replication as "consistent in k of n seeds," never as statistically significant."""

    if reference_seed is None:
        reference_seed = int(df["seed"].min())

    group_cols = ["width", "epsilon", "method", "metric"]
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    agg_cols = [c for c in numeric_cols if c not in group_cols and c != "seed"]
    per_config = df.groupby(group_cols)[agg_cols].agg(["mean", "std", "min", "max"])
    per_config.columns = [f"{col}_{stat}" for col, stat in per_config.columns]
    per_config = per_config.reset_index()

    paired_rows = []
    for (epsilon, method, metric), sub in df.groupby(["epsilon", "method", "metric"]):
        for width_low, width_high in width_pairs:
            low = sub[sub["width"] == width_low].set_index("seed")
            high = sub[sub["width"] == width_high].set_index("seed")
            common_seeds = sorted(set(low.index) & set(high.index))
            if not common_seeds:
                continue
            for quantity in PAIRED_DIFF_QUANTITIES:
                diffs = low.loc[common_seeds, quantity] - high.loc[common_seeds, quantity]
                signs = diffs.apply(_sign)
                reference_sign = signs.loc[reference_seed] if reference_seed in signs.index else None
                k_matching_sign = int((signs == reference_sign).sum()) if reference_sign is not None else None
                paired_rows.append({
                    "epsilon": epsilon, "method": method, "metric": metric,
                    "width_low": width_low, "width_high": width_high, "quantity": quantity,
                    "mean_diff": diffs.mean(), "std_diff": diffs.std(),
                    "n_seeds": len(diffs), "reference_seed": reference_seed,
                    "reference_sign": reference_sign, "k_matching_sign": k_matching_sign,
                })
    paired_differences = pd.DataFrame(paired_rows)

    mechanism_rows = []
    for (width, metric), sub in df.groupby(["width", "metric"]):
        row = {"width": width, "metric": metric}
        for col in MECHANISM_TABLE_COLUMNS:
            per_seed = sub.groupby("seed")[col].mean()
            row[f"{col}_mean"] = per_seed.mean()
            row[f"{col}_std"] = per_seed.std()
        mechanism_rows.append(row)
    mechanism_table = pd.DataFrame(mechanism_rows)

    result = {
        "per_config": per_config,
        "paired_differences": paired_differences,
        "mechanism_table": mechanism_table,
    }

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, table in result.items():
            table.to_csv(save_dir / f"seed_sweep_{name}.csv", index=False)

    return result


def run_attack_seed_variance_decomposition(
        widths=(16, 32, 64), attack_seeds=(0, 1, 2), train_seed=0,
        maha_fit_size=60000, maha_epsilon=MAHALANOBIS_EPSILON, epochs=6, train_subset_size=5000,
        n_query_points=40, n_train_norm_points=500, n_pool_points=2000, n_attack_points=500,
        epsilons=DEFAULT_EPSILONS, pgd_alpha_frac=0.25, pgd_num_steps=20, pgd_num_restarts=5,
        max_pairs=None, margin_estimator="pairwise", checkpoint_dir=CHECKPOINT_DIR, verbose=True,
        save_path=RESULTS_DIR / "seed_sweep_attack_seed_variance.csv"):
    """Isolates attack/estimator noise from model-to-model variation: holds train_seed fixed and 
    varies only attack_seed, reusing the same trained checkpoint at each width. Returns a long-format
    DataFrame with train_seed and attack_seed as separate columns, also saved to save_path as CSV."""
    train = load_mnist(train=True)
    if verbose:
        print(f"Fitting Mahalanobis precision matrix once from {maha_fit_size} training points...")
    maha_distance_fn = build_pixel_mahalanobis_distance_fn(train.x_flat[:maha_fit_size], epsilon=maha_epsilon)

    all_dfs = []
    for width in widths:
        for attack_seed in attack_seeds:
            if verbose:
                print(f"=== train_seed={train_seed} (fixed) width={width} attack_seed={attack_seed} ===")
            for distance_fn, metric_name in ((euclidean_distance_fn, "Euclidean"),
                                              (maha_distance_fn, "Mahalanobis")):
                df = run_single_seed_width(
                    train_seed=train_seed, attack_seed=attack_seed, width=width, distance_fn=distance_fn,
                    epochs=epochs, train_subset_size=train_subset_size, n_query_points=n_query_points,
                    n_train_norm_points=n_train_norm_points, n_pool_points=n_pool_points,
                    n_attack_points=n_attack_points, epsilons=epsilons, pgd_alpha_frac=pgd_alpha_frac,
                    pgd_num_steps=pgd_num_steps, pgd_num_restarts=pgd_num_restarts, max_pairs=max_pairs,
                    margin_estimator=margin_estimator, checkpoint_dir=checkpoint_dir,
                    metric_name=metric_name, verbose=verbose)
                all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    if save_path is not None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        combined.to_csv(save_path, index=False)
        if verbose:
            print(f"\nSaved attack-seed variance decomposition to {save_path}")

    return combined
