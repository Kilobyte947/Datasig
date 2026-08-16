"""This file contains the main driver for the MNIST Lipschitz experiment, including the ratio-distribution analysis (Steps 2b/4b)."""

import json
import os
from pathlib import Path
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from mnist_lipschitz.data import load_mnist, get_dev_subset, make_loader, stratified_subset_idx
from mnist_lipschitz.models import (
    LogisticRegressionModel, SmallMLP, SmallCNN, FlattenedInputWrapper,
    train_classifier, margin_fn, DEVICE,
)
from mnist_lipschitz.estimators import (
    euclidean_distance_fn,
    pairwise_lipschitz,
    pairwise_lipschitz_all,
    ratio_and_components_for_pairs,
    local_perturbation_lipschitz,
    gradient_norm_estimate,
)
from mnist_lipschitz.distance import (
    svd_ridge_precision, make_mahalanobis_distance_fn, covariance_eigenvalues,
    sweep_epsilon,
)
from mnist_lipschitz.plots import MODEL_ORDER

torch.set_default_dtype(torch.float64)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SEED = 0

# ---------------------------------------------------------------------------
# Epsilon selection
# ---------------------------------------------------------------------------

def epsilon_stability_check(model, dataset, epsilon_values, n_subsamples=5, subsample_frac=0.8,
                             n_points=100, seed=SEED, verbose=True):
    """No-ground-truth substitute for validating against a true L*: for each
    epsilon, draw independent random subsamples of `dataset`, fit Sigma on
    each, and compute the mean Mahalanobis gradient-norm estimate. Reports
    the coefficient of variation (std/mean) across subsamples -- a stable
    epsilon reproduces its estimate across resamples; an unstable one is
    fitting noise in that subsample's covariance.

    Uses the MEAN gradient-norm estimate, not pairwise_lipschitz's max:
    empirically, max-based estimates are dominated by extreme-value sampling
    noise (cv 0.04-0.26, no clean trend) while the mean gives a clean,
    interpretable trend (cv 0.01-0.04).

    Uses a fixed reference model's margin_fn throughout, since epsilon
    selection only needs a consistent yardstick, not the final model.
    """
    generator = torch.Generator().manual_seed(seed)
    N = len(dataset)
    n_sub = int(round(N * subsample_frac))

    results = {}
    for eps in epsilon_values:
        L_hats = []
        for _ in range(n_subsamples):
            idx = torch.randperm(N, generator=generator)[:n_sub]
            x_sub, y_sub = dataset.x_flat[idx], dataset.y[idx]

            precision = svd_ridge_precision(x_sub, eps)

            pt_idx = torch.randperm(x_sub.shape[0], generator=generator)[:n_points]
            vals = gradient_norm_estimate(model, x_sub[pt_idx], y_sub[pt_idx], margin_fn, precision=precision)
            L_hats.append(vals.mean().item())

        L_hats_t = torch.tensor(L_hats)
        mean = L_hats_t.mean().item()
        std = L_hats_t.std().item()
        cv = std / mean if mean > 1e-12 else float("inf")
        results[eps] = {"L_hats": L_hats, "mean": mean, "std": std, "cv": cv}
        if verbose:
            print(f"  epsilon={eps:<10g} mean L_hat={mean:.4f}  std={std:.4f}  cv={cv:.4f}")

    return results


def select_epsilon(epsilon_values, cond_numbers, cv_values, max_cond=1e4, max_cv=0.15, verbose=True):
    """Smallest epsilon meeting both a condition-number and a
    stability (cv) bound. Falls back to the lowest-cv epsilon, with a
    warning, if none qualify."""
    candidates = [eps for eps, cond, cv in zip(epsilon_values, cond_numbers, cv_values)
                  if cond <= max_cond and cv <= max_cv]
    if candidates:
        chosen = min(candidates)
        if verbose:
            print(f"selected epsilon={chosen:g} (smallest meeting cond<={max_cond:g}, cv<={max_cv})")
        return chosen

    best_idx = min(range(len(cv_values)), key=lambda i: cv_values[i])
    chosen = epsilon_values[best_idx]
    if verbose:
        print(f"WARNING: no epsilon met both cond<={max_cond:g} and cv<={max_cv}; "
              f"falling back to epsilon={chosen:g} (lowest cv={cv_values[best_idx]:.4f})")
    return chosen

# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def _build_models_and_data(seed):
    """Loads train/test MNIST and wraps each in flat and image-shaped loaders."""
    train = load_mnist(train=True)
    test = load_mnist(train=False)

    train_flat = make_loader(train.x_flat, train.y, batch_size=256, shuffle=True, seed=seed)
    test_flat = make_loader(test.x_flat, test.y, batch_size=1000, shuffle=False)
    train_img = make_loader(train.x_image, train.y, batch_size=256, shuffle=True, seed=seed)
    test_img = make_loader(test.x_image, test.y, batch_size=1000, shuffle=False)

    return train, test, train_flat, test_flat, train_img, test_img

def _run_estimators_for_model(model, x_query, y_query, distance_fn, precision=None,
                               local_radius=1.0, n_directions=20, seed=SEED):
    """Runs all three Lipschitz sub-methods for one model under one distance
    metric. `model` must accept flat (N, 784) input. Returns scalar
    summaries plus i_pair/j_pair (the argmax pair's indices) and the full
    per-point local/gradient arrays."""
    L_pairwise, i_pair, j_pair = pairwise_lipschitz(model, x_query, y_query, margin_fn, distance_fn)

    local_vals = local_perturbation_lipschitz(model, x_query, y_query, margin_fn, distance_fn,
                                               radius=local_radius, n_directions=n_directions, seed=seed)
    grad_vals = gradient_norm_estimate(model, x_query, y_query, margin_fn, precision=precision)

    return {
        "pairwise": L_pairwise,
        "i_pair": i_pair,
        "j_pair": j_pair,
        "local_vals": local_vals.tolist(),
        "local_max": local_vals.max().item(),
        "local_mean": local_vals.mean().item(),
        "grad_vals": grad_vals.tolist(),
        "grad_max": grad_vals.max().item(),
        "grad_mean": grad_vals.mean().item(),
    }



# ---------------------------------------------------------------------------
# Step 2b/4b: ratio distribution, all pairs vs. nearest neighbors
# ---------------------------------------------------------------------------

def run_ratio_distribution_analysis(model, model_name, metric_name, x_pool, y_pool, distance_fn,
                                     exclude_idx=None, n_points=1000, k_neighbors=5,
                                     max_pairs=None, top_k_images=6, seed=SEED, verbose=True):
    """Compares the full pairwise ratio distribution against ratios
    restricted to nearest-neighbor pairs in raw pixel space, on a
    stratified-by-class subset (disjoint from `exclude_idx`).

    Nearest neighbors are found in raw pixel space regardless of
    `distance_fn`, so the comparison isolates one question: do pairs a
    human would call visually similar show different ratios than the
    general pair population? The ratio itself always uses margin_fn/
    distance_fn, never raw pixel distance -- only pair *selection* uses
    raw pixels.

    Generic over model/distance_fn, so the same call covers any
    model/metric combination without new code.

    Returns dicts of tensors/arrays: ratios, pair indices, the subset and
    its predictions, top_near_neighbor_pairs, a scalar `summary`, and
    `arrays` (numpy, prefixed `{metric_name}_{model_name}_...` for saving).
    """

    subset_idx = stratified_subset_idx(y_pool, n_points, seed=seed, exclude_idx=exclude_idx)
    x_subset = x_pool[subset_idx]
    y_subset = y_pool[subset_idx]

    with torch.no_grad():
        preds_subset = model(x_subset).argmax(dim=1)

    all_pairs_ratio, all_ii, all_jj = pairwise_lipschitz_all(
        model, x_subset, y_subset, margin_fn, distance_fn, max_pairs=max_pairs, seed=seed)

    # sklearn's threaded kneighbors query can segfault alongside torch
    # (conflicting OpenMP runtimes) -- force single-threaded for this call only.
    _prev_omp_threads = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        nn = NearestNeighbors(n_neighbors=k_neighbors + 1)
        nn.fit(x_subset.numpy())
        _, neighbor_idx = nn.kneighbors(x_subset.numpy())
    finally:
        if _prev_omp_threads is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = _prev_omp_threads
    neighbor_idx = neighbor_idx[:, 1:]

    near_ii = torch.arange(x_subset.shape[0]).repeat_interleave(k_neighbors)
    near_jj = torch.as_tensor(neighbor_idx.reshape(-1), dtype=torch.long)
    near_ratio, near_dist, near_margin_diff = ratio_and_components_for_pairs(
        model, x_subset, y_subset, margin_fn, distance_fn, near_ii, near_jj)

    # Dedup by canonical (min(i,j), max(i,j)) -- mutual nearest neighbors
    # otherwise appear twice as a mirrored duplicate.
    sorted_idx = torch.argsort(near_ratio, descending=True)
    top_near_neighbor_pairs = []
    seen_canonical = set()
    for k in sorted_idx.tolist():
        i, j = near_ii[k].item(), near_jj[k].item()
        canonical = (min(i, j), max(i, j))
        if canonical in seen_canonical:
            continue
        seen_canonical.add(canonical)
        top_near_neighbor_pairs.append((
            x_subset[i].numpy(), x_subset[j].numpy(),
            y_subset[i].item(), preds_subset[i].item(),
            y_subset[j].item(), preds_subset[j].item(),
            near_ratio[k].item(),
            near_dist[k].item(),
            near_margin_diff[k].item(),
        ))
        if len(top_near_neighbor_pairs) >= top_k_images:
            break

    summary = {
        "all_pairs_mean": all_pairs_ratio.mean().item(), "all_pairs_max": all_pairs_ratio.max().item(),
        "near_neighbor_mean": near_ratio.mean().item(), "near_neighbor_max": near_ratio.max().item(),
        "n_all_pairs": int(all_pairs_ratio.shape[0]), "n_near_neighbor_pairs": int(near_ratio.shape[0]),
    }
    if verbose:
        print(f"  [{metric_name}/{model_name}] all-pairs ratio:        "
              f"mean={summary['all_pairs_mean']:.4f}  max={summary['all_pairs_max']:.4f}  n={summary['n_all_pairs']}")
        print(f"  [{metric_name}/{model_name}] near-neighbor ratio:     "
              f"mean={summary['near_neighbor_mean']:.4f}  max={summary['near_neighbor_max']:.4f}  n={summary['n_near_neighbor_pairs']}")

    prefix = f"{metric_name}_{model_name}"
    arrays = {
        f"{prefix}_all_pairs_ratio": all_pairs_ratio.numpy(),
        f"{prefix}_all_pairs_ii": all_ii.numpy(),
        f"{prefix}_all_pairs_jj": all_jj.numpy(),
        f"{prefix}_near_neighbor_ratio": near_ratio.numpy(),
        f"{prefix}_near_neighbor_ii": near_ii.numpy(),
        f"{prefix}_near_neighbor_jj": near_jj.numpy(),
        f"{prefix}_subset_idx": subset_idx.numpy(),
    }

    return {
        "subset_idx": subset_idx, "x_subset": x_subset, "y_subset": y_subset, "preds_subset": preds_subset,
        "all_pairs_ratio": all_pairs_ratio, "all_pairs_ii": all_ii, "all_pairs_jj": all_jj,
        "near_neighbor_ratio": near_ratio, "near_neighbor_ii": near_ii, "near_neighbor_jj": near_jj,
        "top_near_neighbor_pairs": top_near_neighbor_pairs,
        "summary": summary,
        "arrays": arrays,
    }


def run_mnist_experiment(
    epochs_lr=15, epochs_mlp=15, epochs_cnn=8, mlp_hidden_sizes=(128,),
    n_lipschitz_points=1000, local_radius=1.0, n_directions=20,
    epsilon_values=(1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    n_subsamples=10, subsample_frac=0.8, stability_n_points=100,
    max_cond=1e4, max_cv=0.05,
    n_ratio_points=1000, k_neighbors=5,
    seed=SEED, verbose=True,
):
    """Full pipeline:
    1. Train all three models (logistic regression, small MLP, small CNN) on MNIST.
    2. Run all three Lipschitz sub-methods (pairwise, local-perturbation, gradient-norm) on all three models under Euclidean distance.
    2b. Ratio-distribution analysis under Euclidean distance (all pairs vs. nearest neighbors).
    3. Pixel covariance + epsilon sweep/selection.
    4. Run all three Lipschitz sub-methods on all three models under Mahalanobis distance.
    4b. Ratio-distribution analysis under Mahalanobis distance (all pairs vs. nearest neighbors).
    5. Save results to results/.
    """

    torch.manual_seed(seed)
    train, test, train_flat, test_flat, train_img, test_img = _build_models_and_data(seed)

    # Step 1: train all three models, record train/test accuracies
    if verbose:
        print("=== Step 1: training models ===")
    lr_model, lr_train_acc, lr_test_acc = train_classifier(
        LogisticRegressionModel(), train_flat, test_flat, epochs=epochs_lr, lr=1e-3, verbose=verbose)
    mlp_model, mlp_train_acc, mlp_test_acc = train_classifier(
        SmallMLP(hidden_sizes=mlp_hidden_sizes), train_flat, test_flat, epochs=epochs_mlp, lr=1e-3, verbose=verbose)
    cnn_model_raw, cnn_train_acc, cnn_test_acc = train_classifier(
        SmallCNN(), train_img, test_img, epochs=epochs_cnn, lr=1e-3, verbose=verbose)
    cnn_model = FlattenedInputWrapper(cnn_model_raw)  # so it accepts flat (N,784) like the others

    accuracies = {
        "logistic_regression": {"train": lr_train_acc, "test": lr_test_acc},
        "mlp": {"train": mlp_train_acc, "test": mlp_test_acc},
        "cnn": {"train": cnn_train_acc, "test": cnn_test_acc},
    }
    if verbose:
        for name, acc in accuracies.items():
            print(f"  {name}: train_acc={acc['train']:.4f}  test_acc={acc['test']:.4f}")

    models = {"logistic_regression": lr_model, "mlp": mlp_model, "cnn": cnn_model}

    # Fixed held-out query set, shared across every model/metric.
    generator = torch.Generator().manual_seed(seed)
    query_idx = torch.randperm(len(test), generator=generator)[:n_lipschitz_points]
    x_query = test.x_flat[query_idx]
    y_query = test.y[query_idx]

    # Step 2: Euclidean-distance estimators on all three models
    if verbose:
        print("\n=== Step 2: Euclidean-distance estimators ===")
    euclidean_results = {}
    for name in MODEL_ORDER:
        model = models[name]
        euclidean_results[name] = _run_estimators_for_model(
            model, x_query, y_query, euclidean_distance_fn,
            local_radius=local_radius, n_directions=n_directions, seed=seed)
        if verbose:
            r = euclidean_results[name]
            print(f"  {name}: pairwise={r['pairwise']:.4f}  local_max={r['local_max']:.4f}  "
                  f"grad_max={r['grad_max']:.4f}  grad_mean={r['grad_mean']:.4f}")

    # Step 2b: Euclidean ratio-distribution analysis
    if verbose:
        print("\n=== Step 2b: Euclidean ratio-distribution analysis ===")
    ratio_dist_euclidean_results = {}
    for name in MODEL_ORDER:
        model = models[name]
        ratio_dist_euclidean_results[name] = run_ratio_distribution_analysis(
            model, name, "euclidean", test.x_flat, test.y, euclidean_distance_fn,
            exclude_idx=query_idx, n_points=n_ratio_points, k_neighbors=k_neighbors, seed=seed, verbose=verbose)

    # Logistic-regression/Euclidean argmax pair, pre-assembled for plot_image_pairs.
    lr_euclidean = euclidean_results["logistic_regression"]
    i_pair, j_pair = lr_euclidean["i_pair"], lr_euclidean["j_pair"]
    with torch.no_grad():
        preds_query = lr_model(x_query).argmax(dim=1)
    argmax_pair_lr_euclidean = [(
        x_query[i_pair].numpy(), x_query[j_pair].numpy(),
        y_query[i_pair].item(), preds_query[i_pair].item(),
        y_query[j_pair].item(), preds_query[j_pair].item(),
        lr_euclidean["pairwise"],
    )]

    # Step 3: epsilon selection (pixel covariance + ridge regularization)
    if verbose:
        print("\n === Step 3: epsilon selection (pixel covariance + ridge regularization) ===")
    eigenvalues = covariance_eigenvalues(train.x_flat)
    cond_numbers = sweep_epsilon(train.x_flat, list(epsilon_values))
    stability_results = epsilon_stability_check(
        lr_model, train, list(epsilon_values), n_subsamples=n_subsamples,
        subsample_frac=subsample_frac, n_points=stability_n_points, seed=seed, verbose=verbose)
    cv_values = [stability_results[eps]["cv"] for eps in epsilon_values]
    selected_epsilon = select_epsilon(list(epsilon_values), cond_numbers, cv_values,
                                       max_cond=max_cond, max_cv=max_cv, verbose=verbose)

    precision = svd_ridge_precision(train.x_flat, selected_epsilon)
    mahalanobis_distance_fn = make_mahalanobis_distance_fn(precision)

    # Step 4: Mahalanobis-distance estimators on all three models
    if verbose:
        print(f"\n=== Step 4: Mahalanobis-distance estimators (epsilon={selected_epsilon:g}) ===")
    mahalanobis_results = {}
    for name in MODEL_ORDER:
        model = models[name]
        mahalanobis_results[name] = _run_estimators_for_model(
            model, x_query, y_query, mahalanobis_distance_fn, precision=precision,
            local_radius=local_radius, n_directions=n_directions, seed=seed)
        if verbose:
            r = mahalanobis_results[name]
            print(f"  {name}: pairwise={r['pairwise']:.4f}  local_max={r['local_max']:.4f}  "
                  f"grad_max={r['grad_max']:.4f}  grad_mean={r['grad_mean']:.4f}")

    # Step 4b: Mahalanobis ratio-distribution analysis (reuses selected epsilon)
    if verbose:
        print(f"\n=== Step 4b: Mahalanobis ratio-distribution analysis (epsilon={selected_epsilon:g}, all models) ===")
    ratio_dist_mahalanobis_results = {}
    for name in MODEL_ORDER:
        model = models[name]
        ratio_dist_mahalanobis_results[name] = run_ratio_distribution_analysis(
            model, name, "mahalanobis", test.x_flat, test.y, mahalanobis_distance_fn,
            exclude_idx=query_idx, n_points=n_ratio_points, k_neighbors=k_neighbors, seed=seed, verbose=verbose)

    # Step 5: save results
    RESULTS_DIR.mkdir(exist_ok=True)

    summary = {
        "accuracies": accuracies,
        "epsilon_selection": {
            "epsilon_values": list(epsilon_values),
            "cond_numbers": cond_numbers,
            "cv_values": cv_values,
            "selected_epsilon": selected_epsilon,
            "max_cond": max_cond,
            "max_cv": max_cv,
        },
        "euclidean": {name: {k: v for k, v in r.items() if not k.endswith("_vals")}
                      for name, r in euclidean_results.items()},
        "mahalanobis": {name: {k: v for k, v in r.items() if not k.endswith("_vals")}
                        for name, r in mahalanobis_results.items()},
        "ratio_distribution_analysis": {
            **{f"euclidean_{name}": r["summary"] for name, r in ratio_dist_euclidean_results.items()},
            **{f"mahalanobis_{name}": r["summary"] for name, r in ratio_dist_mahalanobis_results.items()},
        },
        "config": {
            "n_lipschitz_points": n_lipschitz_points, "local_radius": local_radius,
            "n_directions": n_directions, "n_subsamples": n_subsamples,
            "subsample_frac": subsample_frac, "seed": seed,
            "n_ratio_points": n_ratio_points, "k_neighbors": k_neighbors,
        },
    }
    with open(RESULTS_DIR / "mnist_experiment_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    arrays = {}
    for name, r in euclidean_results.items():
        arrays[f"euclidean_{name}_local"] = np.array(r["local_vals"])
        arrays[f"euclidean_{name}_grad"] = np.array(r["grad_vals"])
    for name, r in mahalanobis_results.items():
        arrays[f"mahalanobis_{name}_local"] = np.array(r["local_vals"])
        arrays[f"mahalanobis_{name}_grad"] = np.array(r["grad_vals"])
    arrays["covariance_eigenvalues"] = eigenvalues.numpy()
    for r in ratio_dist_euclidean_results.values():
        arrays.update(r["arrays"])
    for r in ratio_dist_mahalanobis_results.values():
        arrays.update(r["arrays"])
    np.savez(RESULTS_DIR / "mnist_experiment_arrays.npz", **arrays)

    if verbose:
        print(f"\nSaved results to {RESULTS_DIR}")

    return {
        "accuracies": accuracies,
        "euclidean_results": euclidean_results,
        "mahalanobis_results": mahalanobis_results,
        "ratio_dist_euclidean_results": ratio_dist_euclidean_results,
        "ratio_dist_mahalanobis_results": ratio_dist_mahalanobis_results,
        "argmax_pair_lr_euclidean": argmax_pair_lr_euclidean,
        "selected_epsilon": selected_epsilon,
        "cond_numbers": cond_numbers,
        "cv_values": cv_values,
        "epsilon_values": list(epsilon_values),
        "covariance_eigenvalues": eigenvalues,
    }


def main():
    run_mnist_experiment()

if __name__ == "__main__":
    main()
