"""Multi-seed StrongCNN sweep comparing achieved adversarial sensitivity against theoretical
Lipschitz bounds, under Euclidean and Mahalanobis distance, across five independently trained
checkpoints.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from mnist_example.data import load_mnist, make_loader
from mnist_example.models import StrongCNN, STRONG_CNN_CONFIG, FlattenedInputWrapper, train_classifier
from mnist_example.augmentation import random_affine_augment
from mnist_example.estimators import euclidean_distance_fn
from mnist_example.attacks import fgsm_attack
from mnist_example.adversarial_run_experiment import (
    RESULTS_DIR,
    MAHALANOBIS_EPSILON,
    build_pixel_mahalanobis_distance_fn,
    achieved_ratio,
)
from mnist_example.adversarial_strong_cnn import compute_strong_cnn_bounds

torch.set_default_dtype(torch.float64)

SEEDS = (0, 1, 2, 3, 4)
ATTACK_SEED = 0
EPSILONS = (0.1, 0.2)
MAHALANOBIS_EPSILONS_SWEEP = (MAHALANOBIS_EPSILON, 0.1)

POOL_PATH = RESULTS_DIR / "strong_cnn_seed_sweep_common_pool.pt"
TRAINING_SUMMARY_PATH = RESULTS_DIR / "strong_cnn_seed_sweep_training_summary.csv"
R_ADV_TABLE_PATH = RESULTS_DIR / "strong_cnn_seed_sweep_r_adv_table.csv"
REALIZED_NORM_DIAGNOSTIC_PATH = RESULTS_DIR / "strong_cnn_seed_sweep_realized_norm_diagnostic.csv"
SHRINKAGE_TABLE_PATH = RESULTS_DIR / "strong_cnn_seed_sweep_shrinkage_sensitivity.csv"
BOUNDS_TABLE_PATH = RESULTS_DIR / "strong_cnn_seed_sweep_bounds_per_seed.csv"
SUMMARY_TABLE_PATH = RESULTS_DIR / "strong_cnn_seed_sweep_project1_summary.csv"


# ---------------------------------------------------------------------------
# Checkpoint 1 -- per-seed training and caching
# ---------------------------------------------------------------------------

def _checkpoint_path(seed, checkpoint_dir=RESULTS_DIR):
    """Checkpoint file path for one seed."""
    return Path(checkpoint_dir) / f"strong_cnn_state_dict_seed{seed}.pt"


@torch.no_grad()
def _eval_mode_mean_loss(model, x_image, y, batch_size=1000):
    """Mean cross-entropy loss over (x_image, y) in eval mode, batched."""
    model.eval()
    loader = make_loader(x_image, y, batch_size=batch_size, shuffle=False)
    total_loss, n = 0.0, 0
    for xb, yb in loader:
        loss = F.cross_entropy(model(xb), yb, reduction="sum")
        total_loss += loss.item()
        n += xb.shape[0]
    return total_loss / n


def train_or_load_strong_cnn(seed, train, test, force_retrain=False, checkpoint_dir=RESULTS_DIR, verbose=True):
    """Trains or loads a cached StrongCNN for one seed. Returns a dict with the eval-mode model,
    train/test accuracy, final train loss, wall-clock time, and whether it was loaded from cache."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _checkpoint_path(seed, checkpoint_dir)

    torch.manual_seed(seed)
    train_loader = make_loader(train.x_image, train.y, batch_size=STRONG_CNN_CONFIG["batch_size"],
                                shuffle=True, seed=seed)
    test_loader = make_loader(test.x_image, test.y, batch_size=1000, shuffle=False)
    model = StrongCNN(dropout_conv=STRONG_CNN_CONFIG["dropout_conv"], dropout_fc=STRONG_CNN_CONFIG["dropout_fc"])

    if checkpoint_path.exists() and not force_retrain:
        state = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        train_acc, test_acc = state["train_acc"], state["test_acc"]
        final_train_loss = state["final_train_loss"]
        wall_clock_seconds = 0.0
        loaded_from_cache = True
        if verbose:
            print(f"[checkpoint] loaded seed={seed} from {checkpoint_path} "
                  f"(train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  "
                  f"final_train_loss={final_train_loss:.4f})")
    else:
        augment_generator = torch.Generator().manual_seed(seed)
        augment_fn = lambda x: random_affine_augment(
            x, degrees=STRONG_CNN_CONFIG["augment_degrees"],
            translate=STRONG_CNN_CONFIG["augment_translate"], generator=augment_generator)
        lr_scheduler_fn = lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=STRONG_CNN_CONFIG["lr_scheduler_t_max"], eta_min=STRONG_CNN_CONFIG["lr_scheduler_eta_min"])

        t0 = time.time()
        model, train_acc, test_acc = train_classifier(
            model, train_loader, test_loader, epochs=STRONG_CNN_CONFIG["epochs"],
            lr=STRONG_CNN_CONFIG["lr"], verbose=verbose, augment_fn=augment_fn,
            lr_scheduler_fn=lr_scheduler_fn)
        wall_clock_seconds = time.time() - t0
        model.eval()
        final_train_loss = _eval_mode_mean_loss(model, train.x_image, train.y)
        loaded_from_cache = False

        torch.save({
            "model_state_dict": model.state_dict(), "train_acc": train_acc, "test_acc": test_acc,
            "final_train_loss": final_train_loss, "wall_clock_seconds": wall_clock_seconds,
        }, checkpoint_path)
        if verbose:
            print(f"[trained] seed={seed}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  "
                  f"final_train_loss={final_train_loss:.4f}  wall_clock={wall_clock_seconds:.1f}s")

    model.eval()
    return {
        "model": model, "train_acc": train_acc, "test_acc": test_acc,
        "final_train_loss": final_train_loss, "wall_clock_seconds": wall_clock_seconds,
        "loaded_from_cache": loaded_from_cache,
    }


def check_accuracy_premise(summary_df, max_dev_pp=0.5):
    """Checks that test accuracy across seeds stays within max_dev_pp percentage points of the mean.
    Returns (ok, message)."""
    test_accs = summary_df["test_acc"].to_numpy() * 100.0
    spread = test_accs.max() - test_accs.min()
    mean_dev = np.abs(test_accs - test_accs.mean()).max()
    ok = mean_dev <= max_dev_pp
    message = (f"test_acc range=[{test_accs.min():.3f}, {test_accs.max():.3f}]% "
               f"(spread={spread:.3f}pp, max deviation from mean={mean_dev:.3f}pp, "
               f"threshold={max_dev_pp}pp) -- {'OK' if ok else 'PREMISE VIOLATED'}")
    return ok, message


def run_all_seed_trainings(seeds=SEEDS, force_retrain=False, checkpoint_dir=RESULTS_DIR,
                            save_path=TRAINING_SUMMARY_PATH, verbose=True):
    """Trains or loads all five seeds and builds the training summary table. 
    Returns (models_by_seed, test, train, summary_df)."""
    train = load_mnist(train=True)
    test = load_mnist(train=False)

    models_by_seed = {}
    rows = []
    for seed in seeds:
        if verbose:
            print(f"=== seed={seed} ===")
        result = train_or_load_strong_cnn(seed, train, test, force_retrain=force_retrain,
                                           checkpoint_dir=checkpoint_dir, verbose=verbose)
        models_by_seed[seed] = result["model"]
        rows.append({"seed": seed, "train_acc": result["train_acc"], "test_acc": result["test_acc"],
                     "final_train_loss": result["final_train_loss"],
                     "wall_clock_seconds": result["wall_clock_seconds"],
                     "loaded_from_cache": result["loaded_from_cache"]})

    summary_df = pd.DataFrame(rows)
    if save_path is not None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(save_path, index=False)

    ok, message = check_accuracy_premise(summary_df)
    print(f"[accuracy premise check] {message}")

    return models_by_seed, test, train, summary_df


# ---------------------------------------------------------------------------
# Checkpoint 2 -- common evaluation pool
# ---------------------------------------------------------------------------

@torch.no_grad()
def _clean_predictions_and_margin(model, test):
    """Full-test-set (10000-point) clean predictions and top-2 logit margin for one model.
    Returns (preds: (10000,) LongTensor, margin: (10000,) FloatTensor).
    """
    wrapped = FlattenedInputWrapper(model)
    logits = wrapped(test.x_flat)
    preds = logits.argmax(dim=1)
    top2 = logits.topk(2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    return preds, margin


def build_common_pool(models_by_seed, test, save_path=POOL_PATH, verbose=True):
    """Finds the test images correctly classified by all five seeds. 
    Returns and saves {"pool_idx", "margins_by_seed", "test_acc_full"}."""
    correct_masks = {}
    margins_by_seed = {}
    test_acc_full = {}
    for seed, model in models_by_seed.items():
        preds, margin = _clean_predictions_and_margin(model, test)
        correct = preds == test.y
        correct_masks[seed] = correct
        margins_by_seed[seed] = margin
        test_acc_full[seed] = correct.float().mean().item()

    combined_mask = torch.ones(len(test), dtype=torch.bool)
    for mask in correct_masks.values():
        combined_mask &= mask
    pool_idx = combined_mask.nonzero(as_tuple=True)[0]

    for seed, model in models_by_seed.items():
        wrapped = FlattenedInputWrapper(model)
        with torch.no_grad():
            preds_on_pool = wrapped(test.x_flat[pool_idx]).argmax(dim=1)
        assert torch.equal(preds_on_pool, test.y[pool_idx]), \
            f"seed={seed} misclassifies at least one supposedly-common-pool index"

    if verbose:
        print(f"[common pool] size={pool_idx.shape[0]} / {len(test)} "
              f"({pool_idx.shape[0] / len(test):.4f} of the full test set); "
              f"per-seed full test_acc: {test_acc_full}")

    pool = {"pool_idx": pool_idx, "margins_by_seed": margins_by_seed, "test_acc_full": test_acc_full}
    if save_path is not None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(pool, save_path)
    return pool


def load_common_pool(path=POOL_PATH):
    """Loads a `build_common_pool`-persisted pool -- every downstream stage (Checkpoints 3-6) MUST
    call this rather than recomputing the pool itself, per the plan's Checkpoint-2 gate."""
    return torch.load(path, weights_only=False)


# ---------------------------------------------------------------------------
# Checkpoint 3 -- attacks and R_adv computation
# ---------------------------------------------------------------------------

def _batched_fgsm(model, x, y, epsilon, batch_size=2000):
    """FGSM over the full pool, processed in chunks to bound peak memory. FGSM has no randomness, so 
    this is safe to simply rerun rather than cache."""
    chunks = []
    for i in range(0, x.shape[0], batch_size):
        chunks.append(fgsm_attack(model, x[i:i + batch_size], y[i:i + batch_size], epsilon))
    return torch.cat(chunks, dim=0)


def fit_shared_mahalanobis_distance_fn(train, epsilon=MAHALANOBIS_EPSILON):
    """Fits one Mahalanobis precision matrix from the training set, shared across every seed."""
    return build_pixel_mahalanobis_distance_fn(train.x_flat, epsilon=epsilon)


def build_r_adv_table(models_by_seed, test, pool, mahalanobis_distance_fn, epsilons=EPSILONS,
                       batch_size=2000, verbose=True):
    """Builds the per-example table of realized distance, misclassification, and R_adv under both 
    Euclidean and Mahalanobis distance, per seed and epsilon. 
    Returns (df, ratio_diagnostic_df)."""
    pool_idx = pool["pool_idx"]
    x_pool = test.x_flat[pool_idx]
    y_pool = test.y[pool_idx]
    nominal_norm = {eps: eps * (784.0 ** 0.5) for eps in epsilons}

    table_chunks = []
    diagnostic_rows = []
    for seed, model in models_by_seed.items():
        wrapped = FlattenedInputWrapper(model)
        margin_pool = pool["margins_by_seed"][seed][pool_idx]
        for epsilon in epsilons:
            x_adv = _batched_fgsm(wrapped, x_pool, y_pool, epsilon, batch_size=batch_size)
            with torch.no_grad():
                preds_adv = wrapped(x_adv).argmax(dim=1)
            is_misclassified = preds_adv != y_pool
            realized_norm = euclidean_distance_fn(x_pool, x_adv)
            R_adv_euclidean = achieved_ratio(wrapped, x_pool, x_adv, distance_fn=euclidean_distance_fn)
            R_adv_mahalanobis = achieved_ratio(wrapped, x_pool, x_adv, distance_fn=mahalanobis_distance_fn)

            mean_ratio = (realized_norm / nominal_norm[epsilon]).mean().item()
            diagnostic_rows.append({"seed": seed, "epsilon": epsilon,
                                     "mean_realized_over_nominal": mean_ratio})
            if verbose:
                print(f"  seed={seed}  epsilon={epsilon:g}  "
                      f"pct_misclassified={is_misclassified.float().mean().item():.4f}  "
                      f"mean_realized_norm={realized_norm.mean().item():.4f}  "
                      f"(nominal={nominal_norm[epsilon]:.4f}, ratio={mean_ratio:.4f})")

            table_chunks.append(pd.DataFrame({
                "seed": seed, "epsilon": epsilon,
                "pool_position": np.arange(pool_idx.shape[0]),
                "test_index": pool_idx.numpy(),
                "realized_norm": realized_norm.detach().numpy(),
                "is_misclassified": is_misclassified.detach().numpy(),
                "clean_margin": margin_pool.detach().numpy(),
                "R_adv_euclidean": R_adv_euclidean.detach().numpy(),
                "R_adv_mahalanobis": R_adv_mahalanobis.detach().numpy(),
            }))

    df = pd.concat(table_chunks, ignore_index=True)
    ratio_diagnostic_df = pd.DataFrame(diagnostic_rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(R_ADV_TABLE_PATH, index=False)
    ratio_diagnostic_df.to_csv(REALIZED_NORM_DIAGNOSTIC_PATH, index=False)

    return df, ratio_diagnostic_df


def extract_extreme_example(model, test, pool, df, seed, epsilon, which, distance_fn=euclidean_distance_fn,
                             metric_name="Euclidean"):
    """Recovers the (x, x_adv) image pair for the largest- or smallest-R_adv example in one 
    (seed, epsilon) case. which is "most_sensitive" or "least_sensitive". Returns a dict for the
    example-pair plotting functions."""
    R_adv_col = "R_adv_euclidean" if metric_name == "Euclidean" else "R_adv_mahalanobis"
    case = df[(df["seed"] == seed) & (df["epsilon"] == epsilon)]
    row = case.loc[case[R_adv_col].idxmax()] if which == "most_sensitive" else case.loc[case[R_adv_col].idxmin()]

    test_index = int(row["test_index"])
    x = test.x_flat[test_index:test_index + 1]
    y = test.y[test_index:test_index + 1]
    wrapped = FlattenedInputWrapper(model)
    x_adv = _batched_fgsm(wrapped, x, y, epsilon)

    with torch.no_grad():
        pred_clean = wrapped(x).argmax(dim=1).item()
        pred_adv = wrapped(x_adv).argmax(dim=1).item()

    return {
        "x": x.squeeze(0), "x_adv": x_adv.squeeze(0), "y_true": int(test.y[test_index].item()),
        "pred_clean": pred_clean, "pred_adv": pred_adv, "R_adv": row[R_adv_col],
        "epsilon": epsilon, "method": "FGSM",
        "pixel_distance": distance_fn(x, x_adv).item(),
    }


# ---------------------------------------------------------------------------
# Checkpoint 4 -- Mahalanobis shrinkage sensitivity (Goal A)
# ---------------------------------------------------------------------------

def run_shrinkage_sensitivity(models_by_seed, test, train, pool, df, epsilons_maha=MAHALANOBIS_EPSILONS_SWEEP,
                               attack_epsilons=EPSILONS, batch_size=2000, verbose=True):
    """Recomputes Mahalanobis R_adv at each candidate shrinkage epsilon. 
    Returns a long-format DataFrame, also saved to disk."""
    pool_idx = pool["pool_idx"]
    x_pool = test.x_flat[pool_idx]
    y_pool = test.y[pool_idx]

    rows = []
    for maha_epsilon in epsilons_maha:
        if verbose:
            print(f"[shrinkage sweep] fitting Mahalanobis precision at epsilon={maha_epsilon:g}...")
        distance_fn = fit_shared_mahalanobis_distance_fn(train, epsilon=maha_epsilon)
        for seed, model in models_by_seed.items():
            wrapped = FlattenedInputWrapper(model)
            for attack_epsilon in attack_epsilons:
                x_adv = _batched_fgsm(wrapped, x_pool, y_pool, attack_epsilon, batch_size=batch_size)
                R_adv = achieved_ratio(wrapped, x_pool, x_adv, distance_fn=distance_fn)
                rows.append(pd.DataFrame({
                    "seed": seed, "attack_epsilon": attack_epsilon, "maha_epsilon": maha_epsilon,
                    "pool_position": np.arange(pool_idx.shape[0]), "test_index": pool_idx.numpy(),
                    "R_adv_mahalanobis": R_adv.detach().numpy(),
                }))

    result = pd.concat(rows, ignore_index=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(SHRINKAGE_TABLE_PATH, index=False)
    return result


SHRINKAGE_BOUNDS_TABLE_PATH = RESULTS_DIR / "strong_cnn_seed_sweep_shrinkage_bounds.csv"


def shrinkage_bounds_per_seed(models_by_seed, train, test, pool, epsilons_maha=MAHALANOBIS_EPSILONS_SWEEP,
                               n_query=1000, n_norm=1000, sampling_seed=ATTACK_SEED, verbose=True):
    """Recomputes L_full_estimated and related bounds at each Mahalanobis shrinkage epsilon. 
    Returns a DataFrame, also saved to disk."""
    rows = []
    for maha_epsilon in epsilons_maha:
        distance_fn = fit_shared_mahalanobis_distance_fn(train, epsilon=maha_epsilon)
        bounds_df = per_seed_bounds(models_by_seed, train, test, pool,
                                     {f"Mahalanobis_eps{maha_epsilon:g}": distance_fn},
                                     n_query=n_query, n_norm=n_norm, sampling_seed=sampling_seed,
                                     verbose=verbose)
        bounds_df["maha_epsilon"] = maha_epsilon
        rows.append(bounds_df)

    result = pd.concat(rows, ignore_index=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(SHRINKAGE_BOUNDS_TABLE_PATH, index=False)
    return result


# ---------------------------------------------------------------------------
# Checkpoint 5 -- statistics
# ---------------------------------------------------------------------------

def primary_r_adv_stats(df, distance_col, metric_name):
    """Median, p95, p99 and max R_adv over the common pool, per seed and epsilon."""
    grouped = df.groupby(["seed", "epsilon"])[distance_col].agg(
        median="median", p95=lambda s: s.quantile(0.95), p99=lambda s: s.quantile(0.99), max="max")
    grouped = grouped.reset_index()
    grouped["metric"] = metric_name
    return grouped


def attack_success_rate(df):
    """Attack success rate (mean misclassification), per seed and epsilon."""
    return df.groupby(["seed", "epsilon"])["is_misclassified"].mean().reset_index(
        name="attack_success_rate")


def by_outcome_split(df, distance_col, metric_name):
    """R_adv and clean margin split by attack outcome, per seed and epsilon."""
    grouped = df.groupby(["seed", "epsilon", "is_misclassified"])[distance_col].agg(
        median="median", p95=lambda s: s.quantile(0.95), n="count")
    grouped = grouped.reset_index()
    margin_grouped = df.groupby(["seed", "epsilon", "is_misclassified"])["clean_margin"].mean().reset_index(
        name="mean_clean_margin")
    grouped = grouped.merge(margin_grouped, on=["seed", "epsilon", "is_misclassified"])
    grouped["metric"] = metric_name
    return grouped


def common_success_set(df, epsilon, min_reliable_size=50):
    """Images misclassified by all five seeds at one epsilon. Returns test indices and per-seed R_adv 
    restricted to that subset, with a reliable flag for small sets."""
    sub = df[df["epsilon"] == epsilon]
    seeds = sorted(sub["seed"].unique())
    misclassified_sets = {}
    for seed in seeds:
        seed_sub = sub[sub["seed"] == seed]
        misclassified_sets[seed] = set(seed_sub.loc[seed_sub["is_misclassified"], "test_index"])

    common = set.intersection(*misclassified_sets.values()) if misclassified_sets else set()
    common_indices = sorted(common)
    n = len(common_indices)
    reliable = n >= min_reliable_size

    r_adv_euclidean_by_seed, r_adv_mahalanobis_by_seed = {}, {}
    for seed in seeds:
        seed_sub = sub[(sub["seed"] == seed) & (sub["test_index"].isin(common_indices))]
        seed_sub = seed_sub.set_index("test_index").loc[common_indices]
        r_adv_euclidean_by_seed[seed] = seed_sub["R_adv_euclidean"].to_numpy()
        r_adv_mahalanobis_by_seed[seed] = seed_sub["R_adv_mahalanobis"].to_numpy()

    return {
        "epsilon": epsilon, "test_indices": common_indices, "n": n, "reliable": reliable,
        "r_adv_euclidean_by_seed": r_adv_euclidean_by_seed,
        "r_adv_mahalanobis_by_seed": r_adv_mahalanobis_by_seed,
    }


def per_seed_bounds(models_by_seed, train, test, pool, distance_fns, n_query=1000, n_norm=1000,
                     sampling_seed=ATTACK_SEED, verbose=True):
    """L_full_estimated, product_bound and looseness_ratio per seed, under each given distance 
    function. Returns a DataFrame, also saved to disk."""
    pool_idx_set = set(pool["pool_idx"].tolist())
    generator = torch.Generator().manual_seed(sampling_seed)
    all_idx = torch.randperm(len(test), generator=generator)
    non_pool_idx = torch.tensor([i.item() for i in all_idx if i.item() not in pool_idx_set])

    query_idx = non_pool_idx[:n_query]
    x_query, y_query = test.x_flat[query_idx], test.y[query_idx]

    train_norm_generator = torch.Generator().manual_seed(sampling_seed)
    norm_idx = torch.randperm(len(train), generator=train_norm_generator)[:n_norm]
    x_train_for_norm = train.x_flat[norm_idx]

    rows = []
    for seed, model in models_by_seed.items():
        for metric_name, distance_fn in distance_fns.items():
            if verbose:
                print(f"[bounds] seed={seed}  metric={metric_name}")
            result = compute_strong_cnn_bounds(
                model, x_query, y_query, distance_fn, x_train_for_norm,
                seed=sampling_seed, verbose=verbose)
            rows.append({"seed": seed, "metric": metric_name, **result})

    bounds_df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bounds_df.to_csv(BOUNDS_TABLE_PATH, index=False)
    return bounds_df


def build_project1_summary_table(training_summary_df, df, bounds_df, epsilons=EPSILONS,
                                  headline_epsilon=None, save_path=SUMMARY_TABLE_PATH):
    """Per-seed summary table: clean and adversarial accuracy, Lipschitz bounds, and headline p99 
    R_adv, under both metrics. Returns a DataFrame, also saved to disk."""
    if headline_epsilon is None:
        headline_epsilon = max(epsilons)

    success = attack_success_rate(df)
    primary_by_metric = {
        "euclidean": primary_r_adv_stats(df, "R_adv_euclidean", "Euclidean"),
        "mahalanobis": primary_r_adv_stats(df, "R_adv_mahalanobis", "Mahalanobis"),
    }

    rows = []
    for seed in training_summary_df["seed"]:
        train_row = training_summary_df[training_summary_df["seed"] == seed].iloc[0]
        row = {"seed": seed, "train_acc": train_row["train_acc"], "test_acc": train_row["test_acc"]}

        for epsilon in epsilons:
            succ_row = success[(success["seed"] == seed) & (success["epsilon"] == epsilon)]
            row[f"adv_acc_eps{epsilon:g}"] = 1.0 - succ_row["attack_success_rate"].iloc[0]

        for metric_key, metric_label in (("euclidean", "Euclidean"), ("mahalanobis", "Mahalanobis")):
            bound_row = bounds_df[(bounds_df["seed"] == seed) & (bounds_df["metric"] == metric_label)]
            if len(bound_row):
                b = bound_row.iloc[0]
                row[f"L_full_estimated_{metric_key}"] = b["L_full_estimated"]
                row[f"product_bound_{metric_key}"] = b["product"]
                row[f"looseness_ratio_{metric_key}"] = b["looseness_ratio"]

            primary_row = primary_by_metric[metric_key]
            primary_row = primary_row[(primary_row["seed"] == seed) & (primary_row["epsilon"] == headline_epsilon)]
            if len(primary_row):
                row[f"p99_R_adv_{metric_key}_eps{headline_epsilon:g}"] = primary_row.iloc[0]["p99"]

        rows.append(row)

    result = pd.DataFrame(rows)
    if save_path is not None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        result.to_csv(save_path, index=False)
    return result


# ---------------------------------------------------------------------------
# Transferability check
# ---------------------------------------------------------------------------

TRANSFER_TABLE_PATH = RESULTS_DIR / "strong_cnn_seed_sweep_transfer_table.csv"
TRANSFER_SUMMARY_PATH = RESULTS_DIR / "strong_cnn_seed_sweep_transfer_summary.csv"


def run_transfer_attack(models_by_seed, test, pool, mahalanobis_distance_fn, source_seed=0,
                         epsilons=EPSILONS, batch_size=2000, verbose=True):
    """Evaluates every seed's model on adversarial examples crafted against one source_seed model.
    Returns one row per (source_seed, eval_seed, epsilon, pool position), also saved to disk."""
    pool_idx = pool["pool_idx"]
    x_pool = test.x_flat[pool_idx]
    y_pool = test.y[pool_idx]

    source_wrapped = FlattenedInputWrapper(models_by_seed[source_seed])

    table_chunks = []
    for epsilon in epsilons:
        x_adv = _batched_fgsm(source_wrapped, x_pool, y_pool, epsilon, batch_size=batch_size)
        for eval_seed, model in models_by_seed.items():
            wrapped = FlattenedInputWrapper(model)
            with torch.no_grad():
                preds_adv = wrapped(x_adv).argmax(dim=1)
            is_misclassified = preds_adv != y_pool
            R_adv_euclidean = achieved_ratio(wrapped, x_pool, x_adv, distance_fn=euclidean_distance_fn)
            R_adv_mahalanobis = achieved_ratio(wrapped, x_pool, x_adv, distance_fn=mahalanobis_distance_fn)

            if verbose:
                print(f"  source_seed={source_seed}  eval_seed={eval_seed}  epsilon={epsilon:g}  "
                      f"transfer_pct_misclassified={is_misclassified.float().mean().item():.4f}")

            table_chunks.append(pd.DataFrame({
                "source_seed": source_seed, "eval_seed": eval_seed, "epsilon": epsilon,
                "pool_position": np.arange(pool_idx.shape[0]), "test_index": pool_idx.numpy(),
                "is_misclassified": is_misclassified.detach().numpy(),
                "R_adv_euclidean": R_adv_euclidean.detach().numpy(),
                "R_adv_mahalanobis": R_adv_mahalanobis.detach().numpy(),
            }))

    result = pd.concat(table_chunks, ignore_index=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(TRANSFER_TABLE_PATH, index=False)
    return result


def summarize_transfer_attack(transfer_df, save_path=TRANSFER_SUMMARY_PATH):
    """Aggregates run_transfer_attack's results into transfer accuracy and R_adv statistics per
    (source_seed, eval_seed, epsilon). Returns a DataFrame, also saved to disk."""
    grouped = transfer_df.groupby(["source_seed", "eval_seed", "epsilon"]).agg(
        transfer_accuracy=("is_misclassified", lambda s: 1.0 - s.mean()),
        mean_R_adv_euclidean=("R_adv_euclidean", "mean"),
        p99_R_adv_euclidean=("R_adv_euclidean", lambda s: s.quantile(0.99)),
        mean_R_adv_mahalanobis=("R_adv_mahalanobis", "mean"),
        p99_R_adv_mahalanobis=("R_adv_mahalanobis", lambda s: s.quantile(0.99)),
    ).reset_index()

    if save_path is not None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        grouped.to_csv(save_path, index=False)
    return grouped


# ---------------------------------------------------------------------------
# Full pipeline driver
# ---------------------------------------------------------------------------

def run_full_sweep(seeds=SEEDS, epsilons=EPSILONS, maha_epsilon=MAHALANOBIS_EPSILON,
                    shrinkage_epsilons=MAHALANOBIS_EPSILONS_SWEEP, force_retrain=False,
                    checkpoint_dir=RESULTS_DIR, n_query=1000, n_norm=1000, verbose=True):
    """Runs the full seed-sweep pipeline: training, common pool, attacks, shrinkage sensitivity, and
    summary statistics. Returns a dict of every result table."""
    if verbose:
        print("=== Checkpoint 1: per-seed training ===")
    models_by_seed, test, train, training_summary_df = run_all_seed_trainings(
        seeds=seeds, force_retrain=force_retrain, checkpoint_dir=checkpoint_dir, verbose=verbose)

    if verbose:
        print("=== Checkpoint 2: common evaluation pool ===")
    pool = build_common_pool(models_by_seed, test, verbose=verbose)

    if verbose:
        print("=== Checkpoint 3: attacks + R_adv table ===")
    mahalanobis_distance_fn = fit_shared_mahalanobis_distance_fn(train, epsilon=maha_epsilon)
    r_adv_df, realized_norm_diagnostic_df = build_r_adv_table(
        models_by_seed, test, pool, mahalanobis_distance_fn, epsilons=epsilons, verbose=verbose)

    if verbose:
        print("=== Checkpoint 4: Mahalanobis shrinkage sensitivity ===")
    shrinkage_df = run_shrinkage_sensitivity(
        models_by_seed, test, train, pool, r_adv_df, epsilons_maha=shrinkage_epsilons,
        attack_epsilons=epsilons, verbose=verbose)
    shrinkage_bounds_df = shrinkage_bounds_per_seed(
        models_by_seed, train, test, pool, epsilons_maha=shrinkage_epsilons,
        n_query=n_query, n_norm=n_norm, verbose=verbose)

    if verbose:
        print("=== Checkpoint 5: statistics ===")
    primary_stats_euclidean = primary_r_adv_stats(r_adv_df, "R_adv_euclidean", "Euclidean")
    primary_stats_mahalanobis = primary_r_adv_stats(r_adv_df, "R_adv_mahalanobis", "Mahalanobis")
    success_rate_df = attack_success_rate(r_adv_df)
    by_outcome_euclidean = by_outcome_split(r_adv_df, "R_adv_euclidean", "Euclidean")
    by_outcome_mahalanobis = by_outcome_split(r_adv_df, "R_adv_mahalanobis", "Mahalanobis")
    common_success_sets = {epsilon: common_success_set(r_adv_df, epsilon) for epsilon in epsilons}
    bounds_df = per_seed_bounds(
        models_by_seed, train, test, pool,
        {"Euclidean": euclidean_distance_fn, "Mahalanobis": mahalanobis_distance_fn},
        n_query=n_query, n_norm=n_norm, verbose=verbose)
    project1_summary_df = build_project1_summary_table(training_summary_df, r_adv_df, bounds_df,
                                                        epsilons=epsilons)

    if verbose:
        print("=== run_full_sweep complete ===")
        print(project1_summary_df.to_string(index=False))

    return {
        "models_by_seed": models_by_seed, "train": train, "test": test,
        "training_summary_df": training_summary_df, "pool": pool,
        "mahalanobis_distance_fn": mahalanobis_distance_fn, "r_adv_df": r_adv_df,
        "realized_norm_diagnostic_df": realized_norm_diagnostic_df, "shrinkage_df": shrinkage_df,
        "shrinkage_bounds_df": shrinkage_bounds_df, "primary_stats_euclidean": primary_stats_euclidean,
        "primary_stats_mahalanobis": primary_stats_mahalanobis, "success_rate_df": success_rate_df,
        "by_outcome_euclidean": by_outcome_euclidean, "by_outcome_mahalanobis": by_outcome_mahalanobis,
        "common_success_sets": common_success_sets, "bounds_df": bounds_df,
        "project1_summary_df": project1_summary_df,
    }
