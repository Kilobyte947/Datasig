"""This file contains the main driver for the MNIST Lipschitz experiment, including the ratio-distribution analysis (Steps 2b/4b)."""

import json
import os
from pathlib import Path
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from mnist_lipschitz.data import load_mnist, get_dev_subset, make_loader, stratified_subset_idx
from mnist_lipschitz.models import (
    LogisticRegressionModel, SmallMLP, SmallCNN, StrongCNN, STRONG_CNN_CONFIG,
    FlattenedInputWrapper, train_classifier, margin_fn, DEVICE,
)
from mnist_lipschitz.augmentation import random_affine_augment
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
from mnist_lipschitz.embeddings import elementwise_embedding
from mnist_lipschitz.smoothing import smoothed_cross_terms_embedding, gaussian_blur_embedding
from mnist_lipschitz.plots import (
    MODEL_ORDER, plot_embedding_degree_sweep, plot_ratio_distribution, plot_image_pairs,
    plot_smoothing_gallery, plot_smoothing_stability_sweep, plot_smoothing_ratio_sweep,
)

torch.set_default_dtype(torch.float64)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SEED = 0

# ---------------------------------------------------------------------------
# Epsilon selection
# ---------------------------------------------------------------------------

def epsilon_stability_check(model, dataset, epsilon_values, n_subsamples=5, subsample_frac=0.8,
                             n_points=100, seed=SEED, verbose=True, embed_fn=None):
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

    If `embed_fn` is given (e.g. embeddings.py::elementwise_embedding), the
    precision matrix is fit on each subsample's *embedded* covariance
    (`svd_ridge_precision(embed_fn(x_sub), eps)`) instead of the raw pixel
    covariance, and `gradient_norm_estimate`'s embed_fn-aware pullback-metric
    path (see its docstring) supplies the correctly-dimensioned dual norm --
    this is what makes epsilon selection meaningful for an embedded space at
    all, rather than raising a shape mismatch. Leaving `embed_fn` unset (the
    default) leaves existing behavior exactly unchanged.
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

            x_for_cov = embed_fn(x_sub) if embed_fn is not None else x_sub
            precision = svd_ridge_precision(x_for_cov, eps)

            pt_idx = torch.randperm(x_sub.shape[0], generator=generator)[:n_points]
            vals = gradient_norm_estimate(model, x_sub[pt_idx], y_sub[pt_idx], margin_fn,
                                           precision=precision, embed_fn=embed_fn)
            L_hats.append(vals.mean().item())

        L_hats_t = torch.tensor(L_hats)
        mean = L_hats_t.mean().item()
        std = L_hats_t.std().item()
        cv = std / mean if mean > 1e-12 else float("inf")
        results[eps] = {"L_hats": L_hats, "mean": mean, "std": std, "cv": cv}
        if verbose:
            print(f"  epsilon={eps:<10g} mean L_hat={mean:.4f}  std={std:.4f}  cv={cv:.4f}", flush=True)

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


def _knn_label_purity(embedded, labels, k=5):
    """For each point, the fraction of its `k` nearest neighbors (Euclidean, in the given embedded
    space, excluding itself) that share its true label, averaged over every point -- a well-
    clustered-by-digit embedding scores well above the 10-class chance baseline (0.10); a
    scattered one sits close to it. Used by `run_smoothing_sweep` as a quantitative embedding-
    quality check per sigma, the same convention `umap_embedding.py::knn_label_purity` already
    established for the UMAP sub-experiment -- kept as a small private duplicate here rather than
    imported from `umap_embedding.py`, since that module is its own self-contained sub-experiment
    (see its docstring) and pulling the `umap-learn` dependency into this file's imports for one
    generic helper would work against that separation.

    `embedded`: (N, d) array/tensor of embedded coordinates. `labels`: (N,) integer true labels,
    same order.
    """
    embedded_np = embedded.detach().cpu().numpy() if hasattr(embedded, "detach") else np.asarray(embedded)
    labels_np = labels.detach().cpu().numpy() if hasattr(labels, "detach") else np.asarray(labels)

    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(embedded_np)
    _, neighbor_idx = nn.kneighbors(embedded_np)
    neighbor_idx = neighbor_idx[:, 1:]

    matches = (labels_np[neighbor_idx] == labels_np[:, None]).mean()
    return float(matches)

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


# ---------------------------------------------------------------------------
# Embedding-degree sweep (opt-in, not part of main()/run_mnist_experiment())
# ---------------------------------------------------------------------------

def run_embedding_degree_sweep(
    degrees=(1, 2, 3), lr_model=None, train=None, test=None, epsilon_pool=None,
    epsilon_values=(1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    epsilon_pool_size=3000, n_subsamples=10, subsample_frac=0.8, stability_n_points=100,
    max_cond=1e4, max_cv=0.05, n_ratio_points=1000, k_neighbors=5,
    seed=SEED, verbose=True,
):
    """Repeats epsilon selection + ratio-distribution analysis, once per degree in `degrees`,
    for `embeddings.py::elementwise_embedding` at that degree under Mahalanobis distance --
    exercising the embed_fn-aware path in `epsilon_stability_check` and
    `gradient_norm_estimate` (see their docstrings), which used to raise a dimension-mismatch
    error for any degree > 1: a precision matrix sized for raw 784-pixel space cannot pair with
    an embedded, higher-dimensional gradient.

    Matches the setup validated in exploratory work before being promoted here: epsilon is
    selected on a small, fixed `epsilon_pool_size`-point pool (`get_dev_subset`, not the full
    60k) -- cheap enough to sweep several candidate epsilons x several subsamples x every degree
    without retraining anything -- while the *final* precision matrix used for the
    ratio-distribution analysis is fit on the full `train` set, matching every other precision
    matrix in this file (e.g. `run_mnist_experiment`'s Mahalanobis step). Pass a pre-built
    `epsilon_pool` to pin exactly which points are used (e.g. for an apples-to-apples test against
    a raw-pixel baseline computed on the same pool); otherwise one is drawn via `get_dev_subset`.

    Uses logistic regression as the reference model throughout, matching
    `epsilon_stability_check`'s existing "one cheap, consistent yardstick" convention. `degree=1`
    is `elementwise_embedding`'s identity case, so its results should closely match the
    pre-existing raw-pixel (`embed_fn=None`) Mahalanobis pipeline on the same pool/model/seed --
    checked directly in
    `tests/test_epsilon_selection.py::test_run_embedding_degree_sweep_degree_1_matches_raw_pixel_baseline`,
    not just assumed, since that's the only degree with a raw-pixel result to compare against.

    If `lr_model`/`train`/`test` aren't given, this trains its own reference model and loads MNIST
    fresh -- self-contained like `toy_lipschitz`'s opt-in seed-averaged sweep, at the cost of
    retraining a model already trained by `run_mnist_experiment()` if that was also called. Pass
    an already-trained model (and/or `train`/`test`) to skip that retraining.

    This is markedly slower than `run_mnist_experiment()` alone -- fitting a precision matrix on
    the full 60k-point set at `degree=3` means an SVD of a `(60000, 2352)` matrix, repeated across
    every degree -- and is deliberately **not** called from `main()` (mirrors
    `toy_lipschitz.run_experiment.run_gap_N_sweep_seed_averaged`'s "opt-in, not in main()"
    convention); run it directly, e.g. from the notebook or the CLI.

    Returns a dict: `degree_results` (keyed by degree, each holding `selected_epsilon`,
    `epsilon_values`/`cond_numbers`/`cv_values` -- the full per-epsilon sweep --
    `cond_number_at_selected_epsilon`, and `ratio_summary`), `lr_model`/`train`/`test` (for reuse
    by a caller), and `figure` (`plots.plot_embedding_degree_sweep`'s output). Also saves a
    summary JSON, the merged ratio-distribution arrays, and the plot to `results/`.
    """
    torch.manual_seed(seed)
    if train is None:
        train = load_mnist(train=True)
    if test is None:
        test = load_mnist(train=False)
    if lr_model is None:
        train_flat = make_loader(train.x_flat, train.y, batch_size=256, shuffle=True, seed=seed)
        test_flat = make_loader(test.x_flat, test.y, batch_size=1000, shuffle=False)
        lr_model, lr_train_acc, lr_test_acc = train_classifier(
            LogisticRegressionModel(), train_flat, test_flat, epochs=15, lr=1e-3, verbose=verbose)
        if verbose:
            print(f"trained reference logistic-regression model: "
                  f"train_acc={lr_train_acc:.4f}  test_acc={lr_test_acc:.4f}")
    if epsilon_pool is None:
        epsilon_pool = get_dev_subset(train, epsilon_pool_size, seed=seed)

    degree_results = {}
    ratio_results = {}
    for degree in degrees:
        if verbose:
            print(f"\n=== embedding degree={degree} ===")
        embed_fn = lambda x: elementwise_embedding(x, degree)

        cond_numbers = sweep_epsilon(embed_fn(epsilon_pool.x_flat), list(epsilon_values))
        stability_results = epsilon_stability_check(
            lr_model, epsilon_pool, list(epsilon_values), n_subsamples=n_subsamples,
            subsample_frac=subsample_frac, n_points=stability_n_points, seed=seed, verbose=verbose,
            embed_fn=embed_fn)
        cv_values = [stability_results[eps]["cv"] for eps in epsilon_values]
        selected_epsilon = select_epsilon(list(epsilon_values), cond_numbers, cv_values,
                                           max_cond=max_cond, max_cv=max_cv, verbose=verbose)
        cond_at_selected = cond_numbers[list(epsilon_values).index(selected_epsilon)]

        precision = svd_ridge_precision(embed_fn(train.x_flat), selected_epsilon)
        mahalanobis_fn = make_mahalanobis_distance_fn(precision, embed_fn=embed_fn)

        ratio_result = run_ratio_distribution_analysis(
            lr_model, "logistic_regression", f"embedding_degree{degree}",
            test.x_flat, test.y, mahalanobis_fn,
            n_points=n_ratio_points, k_neighbors=k_neighbors, seed=seed, verbose=verbose)
        ratio_results[degree] = ratio_result

        degree_results[degree] = {
            "selected_epsilon": selected_epsilon,
            "cond_number_at_selected_epsilon": cond_at_selected,
            "epsilon_values": list(epsilon_values),
            "cond_numbers": cond_numbers,
            "cv_values": cv_values,
            "ratio_summary": ratio_result["summary"],
        }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "embedding_degree_sweep_results.json", "w") as f:
        json.dump({str(d): r for d, r in degree_results.items()}, f, indent=2)

    arrays = {}
    for r in ratio_results.values():
        arrays.update(r["arrays"])
    np.savez(RESULTS_DIR / "embedding_degree_sweep_arrays.npz", **arrays)

    fig = plot_embedding_degree_sweep(degree_results, save_path=RESULTS_DIR / "embedding_degree_sweep.png")

    if verbose:
        print(f"\nSaved results to {RESULTS_DIR}")

    return {"degree_results": degree_results, "ratio_results": ratio_results,
            "lr_model": lr_model, "train": train, "test": test, "figure": fig}


# ---------------------------------------------------------------------------
# Smoothing-strength sweep (opt-in, not part of main()/run_mnist_experiment())
# ---------------------------------------------------------------------------

def run_smoothing_sweep(
    sigmas=(0, 0.5, 1, 1.5, 2, 3), lr_model=None, train=None, test=None, epsilon_pool=None,
    epsilon_values=(1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    epsilon_pool_size=3000, n_subsamples=5, subsample_frac=0.8, stability_n_points=100,
    max_cond=1e4, max_cv=0.05, n_ratio_points=300, k_neighbors=5, n_purity_points=1000,
    gallery_digits=(0, 1, 3, 5, 7, 9), seed=SEED, verbose=True,
):
    """Repeats epsilon-selection + ratio-distribution analysis, once per Gaussian-blur strength
    `sigma` in `sigmas`, for `smoothing.py::smoothed_cross_terms_embedding` -- tests whether
    blurring the raw image before computing `embeddings.py::local_patch_cross_terms` fixes that
    embedding's categorical Mahalanobis epsilon-selection failure (`README.md`'s "Epsilon selection
    fails categorically for this embedding" section: cv 0.91-1.45 against a `cv<=0.05` bound at
    every epsilon tried on the *unblurred* embedding). `sigma=0` reproduces that exact unblurred
    case (`smoothed_cross_terms_embedding(x, 0)` is `local_patch_cross_terms(x)` unchanged), so this
    sweep's first row is a direct determinism cross-check against that already-documented result,
    not a fresh, unverifiable starting point.

    Follows `run_embedding_degree_sweep`'s established structure and defaults (same epsilon pool
    size/values, same `epsilon_stability_check` call, same `select_epsilon` bound), with two
    changes specific to this embedding:

    - **`n_ratio_points` defaults to 300, not 1000.** `local_patch_cross_terms`'s 3920-dimensional
      output already exhausted this machine's memory/swap at 1000 points (~499,500 gathered pairs)
      in the earlier Euclidean follow-up (`results/local_patch_cross_terms_euclidean_followup.md`)
      -- this sweep repeats that embedding's memory footprint up to 6 times (once per sigma), so it
      inherits that same reduced point count preemptively rather than discovering the same ceiling
      6 times over.
    - **Mahalanobis is only computed when epsilon selection actually passes both bounds at that
      sigma** (`cond<=max_cond` and `cv<=max_cv` for at least one candidate epsilon) -- unlike
      `run_embedding_degree_sweep`, which always has a stable epsilon to fall back to.
      `select_epsilon`'s fallback (lowest-cv epsilon among uniformly bad candidates, with its own
      warning) is exactly the mechanism that produced the *unreliable* `epsilon=1` fallback number
      documented in `README.md` for the unblurred embedding -- computing an expensive full-60k-point
      Mahalanobis ratio-distribution analysis on top of a fallback epsilon that never met the
      stability bound would just repeat that same caveat 6 times without adding information. Rows
      where this is skipped have `mahalanobis_ratio_summary=None`.

    Each sigma's row also gets a `knn_label_purity` embedding-quality number (same convention
    `umap_embedding.py::knn_label_purity` established for the UMAP sub-experiment, duplicated
    locally as `_knn_label_purity` above rather than imported -- see that helper's docstring) on a
    fixed, sigma-independent validation subset, and a visual gallery
    (`plots.py::plot_smoothing_gallery`) of a handful of sample digits blurred at that sigma, to
    directly check the risk this whole sweep exists to guard against: too much smoothing making
    different digits look alike.

    Returns a dict: `sigma_results` (keyed by sigma, each holding the epsilon sweep, `min_cv`,
    `stability_pass`, `selected_epsilon`, `knn_label_purity`, `euclidean_ratio_summary`,
    `mahalanobis_ratio_summary` -- `None` when skipped -- and `gallery_figure`), `stability_figure`,
    `ratio_figure`, plus `lr_model`/`train`/`test` for reuse by a caller. Also saves a summary JSON
    (figures excluded -- not JSON-serializable), the merged ratio-distribution arrays, and both
    summary plots plus one gallery PNG per sigma to `results/`.

    Deliberately **not** called from `main()` (mirrors `run_embedding_degree_sweep`'s own opt-in
    convention) -- run it directly, e.g. from `notebook_smoothing.ipynb`.
    """
    torch.manual_seed(seed)
    if train is None:
        train = load_mnist(train=True)
    if test is None:
        test = load_mnist(train=False)
    if lr_model is None:
        train_flat = make_loader(train.x_flat, train.y, batch_size=256, shuffle=True, seed=seed)
        test_flat = make_loader(test.x_flat, test.y, batch_size=1000, shuffle=False)
        lr_model, lr_train_acc, lr_test_acc = train_classifier(
            LogisticRegressionModel(), train_flat, test_flat, epochs=15, lr=1e-3, verbose=verbose)
        if verbose:
            print(f"trained reference logistic-regression model: "
                  f"train_acc={lr_train_acc:.4f}  test_acc={lr_test_acc:.4f}")
    if epsilon_pool is None:
        epsilon_pool = get_dev_subset(train, epsilon_pool_size, seed=seed)

    val_idx = stratified_subset_idx(test.y, n_purity_points, seed=seed)
    x_val, y_val = test.x_flat[val_idx], test.y[val_idx]

    gallery_idx = [(test.y == d).nonzero(as_tuple=True)[0][0].item() for d in gallery_digits]
    gallery_images = test.x_flat[gallery_idx]

    RESULTS_DIR.mkdir(exist_ok=True)

    sigma_results = {}
    ratio_arrays = {}
    for sigma in sigmas:
        if verbose:
            print(f"\n=== sigma={sigma} ===")
        embed_fn = lambda x, sigma=sigma: smoothed_cross_terms_embedding(x, sigma)

        cond_numbers = sweep_epsilon(embed_fn(epsilon_pool.x_flat), list(epsilon_values))
        stability_results = epsilon_stability_check(
            lr_model, epsilon_pool, list(epsilon_values), n_subsamples=n_subsamples,
            subsample_frac=subsample_frac, n_points=stability_n_points, seed=seed, verbose=verbose,
            embed_fn=embed_fn)
        cv_values = [stability_results[eps]["cv"] for eps in epsilon_values]
        min_cv = min(cv_values)

        candidates = [eps for eps, cond, cv in zip(epsilon_values, cond_numbers, cv_values)
                      if cond <= max_cond and cv <= max_cv]
        stability_pass = len(candidates) > 0
        selected_epsilon = select_epsilon(list(epsilon_values), cond_numbers, cv_values,
                                           max_cond=max_cond, max_cv=max_cv, verbose=verbose)
        cond_at_selected = cond_numbers[list(epsilon_values).index(selected_epsilon)]

        purity = _knn_label_purity(embed_fn(x_val), y_val, k=5)

        euclidean_embedded_fn = lambda x, y, embed_fn=embed_fn: euclidean_distance_fn(embed_fn(x), embed_fn(y))
        euclidean_ratio_result = run_ratio_distribution_analysis(
            lr_model, "logistic_regression", f"smoothing_sigma{sigma}_euclidean",
            test.x_flat, test.y, euclidean_embedded_fn,
            n_points=n_ratio_points, k_neighbors=k_neighbors, seed=seed, verbose=verbose)
        ratio_arrays.update(euclidean_ratio_result["arrays"])

        mahalanobis_ratio_result = None
        if stability_pass:
            precision = svd_ridge_precision(embed_fn(train.x_flat), selected_epsilon)
            mahalanobis_fn = make_mahalanobis_distance_fn(precision, embed_fn=embed_fn)
            mahalanobis_ratio_result = run_ratio_distribution_analysis(
                lr_model, "logistic_regression", f"smoothing_sigma{sigma}_mahalanobis",
                test.x_flat, test.y, mahalanobis_fn,
                n_points=n_ratio_points, k_neighbors=k_neighbors, seed=seed, verbose=verbose)
            ratio_arrays.update(mahalanobis_ratio_result["arrays"])
        elif verbose:
            print(f"  sigma={sigma}: epsilon stability did not pass (min cv={min_cv:.4f} > {max_cv}) "
                  f"-- skipping Mahalanobis ratio-distribution")

        blurred_gallery = gaussian_blur_embedding(gallery_images, sigma)
        gallery_samples = [
            {"digit": d, "original": gallery_images[i].numpy().reshape(28, 28),
             "blurred": blurred_gallery[i].numpy().reshape(28, 28)}
            for i, d in enumerate(gallery_digits)
        ]
        gallery_fig = plot_smoothing_gallery(
            gallery_samples, sigma, save_path=RESULTS_DIR / f"smoothing_gallery_sigma{sigma}.png")

        sigma_results[sigma] = {
            "epsilon_values": list(epsilon_values), "cond_numbers": cond_numbers, "cv_values": cv_values,
            "min_cv": min_cv, "stability_pass": stability_pass, "selected_epsilon": selected_epsilon,
            "cond_number_at_selected_epsilon": cond_at_selected,
            "knn_label_purity": purity,
            "euclidean_ratio_summary": euclidean_ratio_result["summary"],
            "mahalanobis_ratio_summary": mahalanobis_ratio_result["summary"] if mahalanobis_ratio_result else None,
            "gallery_figure": gallery_fig,
        }

    with open(RESULTS_DIR / "smoothing_sweep_results.json", "w") as f:
        json.dump({str(s): {k: v for k, v in r.items() if k != "gallery_figure"}
                    for s, r in sigma_results.items()}, f, indent=2)
    np.savez(RESULTS_DIR / "smoothing_sweep_arrays.npz", **ratio_arrays)

    stability_rows = [{"sigma": s, "min_cv": r["min_cv"]} for s, r in sigma_results.items()]
    stability_fig = plot_smoothing_stability_sweep(
        stability_rows, max_cv=max_cv, save_path=RESULTS_DIR / "smoothing_stability_sweep.png")

    ratio_rows = [{
        "sigma": s,
        "knn_label_purity": r["knn_label_purity"],
        "euclidean_near_over_all": r["euclidean_ratio_summary"]["near_neighbor_mean"] / r["euclidean_ratio_summary"]["all_pairs_mean"],
        "mahalanobis_near_over_all": (r["mahalanobis_ratio_summary"]["near_neighbor_mean"] / r["mahalanobis_ratio_summary"]["all_pairs_mean"])
                                      if r["mahalanobis_ratio_summary"] else None,
    } for s, r in sigma_results.items()]
    ratio_fig = plot_smoothing_ratio_sweep(ratio_rows, save_path=RESULTS_DIR / "smoothing_ratio_sweep.png")

    if verbose:
        print(f"\nSaved results to {RESULTS_DIR}")

    return {"sigma_results": sigma_results, "stability_figure": stability_fig, "ratio_figure": ratio_fig,
            "lr_model": lr_model, "train": train, "test": test}


def run_stronger_cnn_raw_mnist_experiment(
    n_lipschitz_points=1000, local_radius=1.0, n_directions=20,
    n_ratio_points=1000, k_neighbors=5,
    mahalanobis_epsilon=0.01,
    seed=SEED, verbose=True,
):
    """CNN-only counterpart to run_mnist_experiment(), for the higher-capacity
    `StrongCNN` (models.py) on raw (uncleaned, standard train/test split)
    MNIST -- a stronger baseline captured ahead of a later data-cleaning
    experiment. Logistic regression and MLP, and the original `SmallCNN`
    baseline, are untouched by this function and continue to live in
    results/ exactly as before; this saves to its own
    results/stronger_cnn_raw_mnist/ subfolder instead.

    Trains StrongCNN with STRONG_CNN_CONFIG's exact recipe (batch norm,
    dropout, light rotation/translation augmentation via
    `augmentation.random_affine_augment`, a cosine-annealed learning rate,
    and more epochs than SmallCNN's original 8) via train_classifier's
    augment_fn/lr_scheduler_fn parameters, then runs the same three
    Lipschitz sub-methods (pairwise, local-perturbation, gradient-norm) and
    the same ratio-distribution/near-neighbor analysis
    run_mnist_experiment() runs for the CNN, under both Euclidean and
    Mahalanobis distance, on the same-shaped query/ratio-distribution
    subsets (1000 points each, matching the existing safe-tested config --
    see README's "Pairwise sampling keeps N modest" design decision).

    Mahalanobis epsilon is *not* reselected here: raw MNIST's pixel
    covariance is exactly the same data run_mnist_experiment() already
    selected epsilon=0.01 for (see README's Epsilon selection section) --
    reselecting via epsilon_stability_check (which trains a fresh reference
    model and does several resampled SVDs) would just reproduce the same
    answer at real extra cost. Pass a different `mahalanobis_epsilon`
    explicitly if that assumption is ever revisited (e.g. once the
    data-cleaning step changes the pixel covariance itself).
    """
    torch.manual_seed(seed)
    train = load_mnist(train=True)
    test = load_mnist(train=False)
    train_img = make_loader(train.x_image, train.y, batch_size=STRONG_CNN_CONFIG["batch_size"],
                             shuffle=True, seed=seed)
    test_img = make_loader(test.x_image, test.y, batch_size=1000, shuffle=False)

    augment_generator = torch.Generator().manual_seed(seed)
    augment_fn = lambda x: random_affine_augment(
        x, degrees=STRONG_CNN_CONFIG["augment_degrees"],
        translate=STRONG_CNN_CONFIG["augment_translate"], generator=augment_generator)
    lr_scheduler_fn = lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=STRONG_CNN_CONFIG["lr_scheduler_t_max"], eta_min=STRONG_CNN_CONFIG["lr_scheduler_eta_min"])

    if verbose:
        print("=== Training StrongCNN on raw MNIST ===", flush=True)
    cnn_model_raw, cnn_train_acc, cnn_test_acc = train_classifier(
        StrongCNN(dropout_conv=STRONG_CNN_CONFIG["dropout_conv"], dropout_fc=STRONG_CNN_CONFIG["dropout_fc"]),
        train_img, test_img, epochs=STRONG_CNN_CONFIG["epochs"], lr=STRONG_CNN_CONFIG["lr"],
        verbose=verbose, augment_fn=augment_fn, lr_scheduler_fn=lr_scheduler_fn)
    cnn_model = FlattenedInputWrapper(cnn_model_raw)
    if verbose:
        print(f"StrongCNN: train_acc={cnn_train_acc:.4f}  test_acc={cnn_test_acc:.4f}", flush=True)

    query_generator = torch.Generator().manual_seed(seed)
    query_idx = torch.randperm(len(test), generator=query_generator)[:n_lipschitz_points]
    x_query, y_query = test.x_flat[query_idx], test.y[query_idx]

    if verbose:
        print("\n=== Euclidean-distance estimators (StrongCNN) ===", flush=True)
    euclidean_result = _run_estimators_for_model(
        cnn_model, x_query, y_query, euclidean_distance_fn,
        local_radius=local_radius, n_directions=n_directions, seed=seed)
    if verbose:
        r = euclidean_result
        print(f"  pairwise={r['pairwise']:.4f}  local_max={r['local_max']:.4f}  "
              f"grad_max={r['grad_max']:.4f}  grad_mean={r['grad_mean']:.4f}", flush=True)

    if verbose:
        print("\n=== Euclidean ratio-distribution / near-neighbor analysis (StrongCNN) ===", flush=True)
    ratio_euclidean = run_ratio_distribution_analysis(
        cnn_model, "cnn", "euclidean", test.x_flat, test.y, euclidean_distance_fn,
        exclude_idx=query_idx, n_points=n_ratio_points, k_neighbors=k_neighbors, seed=seed, verbose=verbose)

    precision = svd_ridge_precision(train.x_flat, mahalanobis_epsilon)
    mahalanobis_distance_fn = make_mahalanobis_distance_fn(precision)

    if verbose:
        print(f"\n=== Mahalanobis-distance estimators (StrongCNN, epsilon={mahalanobis_epsilon:g}) ===", flush=True)
    mahalanobis_result = _run_estimators_for_model(
        cnn_model, x_query, y_query, mahalanobis_distance_fn, precision=precision,
        local_radius=local_radius, n_directions=n_directions, seed=seed)
    if verbose:
        r = mahalanobis_result
        print(f"  pairwise={r['pairwise']:.4f}  local_max={r['local_max']:.4f}  "
              f"grad_max={r['grad_max']:.4f}  grad_mean={r['grad_mean']:.4f}", flush=True)

    if verbose:
        print("\n=== Mahalanobis ratio-distribution / near-neighbor analysis (StrongCNN) ===", flush=True)
    ratio_mahalanobis = run_ratio_distribution_analysis(
        cnn_model, "cnn", "mahalanobis", test.x_flat, test.y, mahalanobis_distance_fn,
        exclude_idx=query_idx, n_points=n_ratio_points, k_neighbors=k_neighbors, seed=seed, verbose=verbose)

    out_dir = RESULTS_DIR / "stronger_cnn_raw_mnist"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist trained weights alongside the results -- without this, later
    # inspection of specific flagged pairs (e.g. re-running the model to get
    # per-image softmax confidences) would require retraining from scratch.
    torch.save(cnn_model_raw.state_dict(), out_dir / "strong_cnn_state_dict.pt")

    def _labeled_pairs(top_pairs):
        # top_pairs entries: (img1, img2, true1, pred1, true2, pred2, ratio, dist, margin_diff)
        # -- images are dropped here (already saved as PNGs below); only the
        # labels/ratio/components (what's needed to answer "which digit
        # pairs get flagged") are kept in the JSON summary.
        return [
            {"true1": t1, "pred1": p1, "true2": t2, "pred2": p2,
             "ratio": ratio, "dist": dist, "margin_diff": margin_diff}
            for (_img1, _img2, t1, p1, t2, p2, ratio, dist, margin_diff) in top_pairs
        ]

    summary = {
        "architecture": "StrongCNN (see models.py docstring for the exact fixed layer spec)",
        "config": STRONG_CNN_CONFIG,
        "accuracies": {"cnn": {"train": cnn_train_acc, "test": cnn_test_acc}},
        "mahalanobis_epsilon": mahalanobis_epsilon,
        "mahalanobis_epsilon_note": "reused from mnist_experiment_results.json -- raw pixel covariance is unchanged",
        "euclidean": {"cnn": {k: v for k, v in euclidean_result.items() if not k.endswith("_vals")}},
        "mahalanobis": {"cnn": {k: v for k, v in mahalanobis_result.items() if not k.endswith("_vals")}},
        "ratio_distribution_analysis": {
            "euclidean_cnn": ratio_euclidean["summary"],
            "mahalanobis_cnn": ratio_mahalanobis["summary"],
        },
        "top_near_neighbor_pairs_euclidean": _labeled_pairs(ratio_euclidean["top_near_neighbor_pairs"]),
        "top_near_neighbor_pairs_mahalanobis": _labeled_pairs(ratio_mahalanobis["top_near_neighbor_pairs"]),
        "config_run": {
            "n_lipschitz_points": n_lipschitz_points, "local_radius": local_radius,
            "n_directions": n_directions, "n_ratio_points": n_ratio_points,
            "k_neighbors": k_neighbors, "seed": seed,
        },
    }
    with open(out_dir / "stronger_cnn_raw_mnist_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    arrays = {
        "euclidean_cnn_local": np.array(euclidean_result["local_vals"]),
        "euclidean_cnn_grad": np.array(euclidean_result["grad_vals"]),
        "mahalanobis_cnn_local": np.array(mahalanobis_result["local_vals"]),
        "mahalanobis_cnn_grad": np.array(mahalanobis_result["grad_vals"]),
    }
    arrays.update(ratio_euclidean["arrays"])
    arrays.update(ratio_mahalanobis["arrays"])
    np.savez(out_dir / "stronger_cnn_raw_mnist_arrays.npz", **arrays)

    plot_ratio_distribution({"cnn": ratio_euclidean}, metric_name="Euclidean",
                             save_path=out_dir / "ratio_distribution_euclidean.png")
    plot_ratio_distribution({"cnn": ratio_mahalanobis}, metric_name="Mahalanobis",
                             save_path=out_dir / "ratio_distribution_mahalanobis.png")
    plot_image_pairs(ratio_euclidean["top_near_neighbor_pairs"],
                      save_path=out_dir / "top_near_neighbor_pairs_euclidean.png")
    plot_image_pairs(ratio_mahalanobis["top_near_neighbor_pairs"],
                      save_path=out_dir / "top_near_neighbor_pairs_mahalanobis.png")

    if verbose:
        print(f"\nSaved results to {out_dir}", flush=True)

    return {
        "cnn_model": cnn_model, "train_acc": cnn_train_acc, "test_acc": cnn_test_acc,
        "euclidean_result": euclidean_result, "mahalanobis_result": mahalanobis_result,
        "ratio_euclidean": ratio_euclidean, "ratio_mahalanobis": ratio_mahalanobis,
        "train": train, "test": test,
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
