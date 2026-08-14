"""Driver: train the three classifiers, run the three Lipschitz estimators
under Euclidean distance, select a ridge epsilon for the Mahalanobis
distance, then re-run the three estimators under that distance. Saves
results to results/. All reusable logic lives in the sibling modules --
this file only wires them together, matching toy_lipschitz's convention.
"""

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
    pixel_covariance, ridge_precision, make_mahalanobis_distance_fn, covariance_eigenvalues,
)

torch.set_default_dtype(torch.float64)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SEED = 0


# ---------------------------------------------------------------------------
# Epsilon selection (Checkpoint 5)
# ---------------------------------------------------------------------------

def sweep_epsilon(Sigma, epsilon_values):
    """Condition number of Sigma + epsilon*I for each candidate epsilon.
    Returns a list of floats aligned with epsilon_values."""
    d = Sigma.shape[0]
    identity = torch.eye(d)
    return [torch.linalg.cond(Sigma + eps * identity).item() for eps in epsilon_values]


def epsilon_stability_check(model, dataset, epsilon_values, n_subsamples=5, subsample_frac=0.8,
                             n_points=100, seed=SEED, verbose=True):
    """For each epsilon: draw `n_subsamples` independent random subsamples
    of `dataset` (subsample_frac of it each), fit Sigma on each subsample,
    compute the MEAN Mahalanobis gradient-norm estimate (using `model`'s
    margin_fn) over `n_points` points from that same subsample under the
    resulting ridge precision, and report the spread (coefficient of
    variation = std/mean) of that mean across repeats.

    This is the no-ground-truth substitute for Experiment 1's
    accuracy-vs-L* sweep: MNIST has no closed-form L* to check against, so
    instead we check whether the ESTIMATE reproduces across independent
    resamples of the same underlying distribution. An epsilon whose
    estimate swings wildly between resamples is fitting noise in that
    particular subsample's covariance -- the high-dimensional analogue of
    Experiment 1's degree-5/6 condition-number blowup.

    Uses the mean gradient-norm estimate, not pairwise_lipschitz's max,
    deliberately: a max over a modest number of points/pairs is an
    extreme-value statistic, dominated by whichever single pair happens to
    be sampled near the current metric's steepest direction -- that adds a
    lot of irreducible sampling noise on top of (and easily swamping) the
    genuine metric-shape instability epsilon is meant to control, which
    was confirmed empirically (pairwise-max gave cv in the 0.04-0.26 range
    with no clean trend; the mean gradient-norm gives cv in the 0.01-0.04
    range with a clear decreasing-then-flat trend). See README's Design
    decisions section.

    Uses `model`'s margin_fn as the fixed scalar function throughout (in
    practice, the already-trained logistic regression model -- cheapest to
    evaluate, and epsilon selection only needs *a* consistent yardstick,
    not the final model being analyzed). Unlike Experiment 1's strict
    "no model involved" L_hat_data, there is no model-free scalar function
    of x on MNIST to fall back on.
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

            Sigma_sub = pixel_covariance(x_sub)
            precision = ridge_precision(Sigma_sub, eps)

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
    """Pick the smallest epsilon (least regularization, closest to the "true"
    Mahalanobis distance) whose condition number and subsample
    coefficient-of-variation both fall within the given bounds. Falls back
    to the epsilon with the lowest cv (with an explicit warning) if none
    qualify -- never silently returns a value outside the stated criteria."""
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
# Checkpoint 6: main comparison
# ---------------------------------------------------------------------------

MODEL_NAMES = ("logistic_regression", "mlp", "cnn")


def _build_models_and_data(seed):
    train = load_mnist(train=True)
    test = load_mnist(train=False)

    train_flat = make_loader(train.x_flat, train.y, batch_size=256, shuffle=True, seed=seed)
    test_flat = make_loader(test.x_flat, test.y, batch_size=1000, shuffle=False)
    train_img = make_loader(train.x_image, train.y, batch_size=256, shuffle=True, seed=seed)
    test_img = make_loader(test.x_image, test.y, batch_size=1000, shuffle=False)

    return train, test, train_flat, test_flat, train_img, test_img


def _run_estimators_for_model(model, x_query, y_query, distance_fn, precision=None,
                               local_radius=1.0, n_directions=20, seed=SEED):
    """Runs all three sub-methods for one model under one distance metric.
    `model` must already accept flat (N, 784) input (wrap the CNN with
    FlattenedInputWrapper first). Includes `i_pair`/`j_pair` (the indices
    into x_query/y_query of pairwise_lipschitz's argmax pair) in the
    returned dict, not just the scalar `pairwise` value, so a caller can
    look at *which* two points produced it (e.g. via
    plots.plot_image_pairs) rather than only the number."""
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
# Checkpoint 7 (in progress): ratio distribution, all pairs vs. nearest neighbors
# ---------------------------------------------------------------------------

def run_ratio_distribution_analysis(model, model_name, metric_name, x_pool, y_pool, distance_fn,
                                     exclude_idx=None, n_points=1000, k_neighbors=5,
                                     max_pairs=None, top_k_images=6, seed=SEED, verbose=True):
    """Compares the full pairwise ratio distribution against ratios
    restricted to actual nearest-neighbor pairs in raw pixel space, on a
    stratified-by-class subset of `n_points` drawn from `x_pool`/`y_pool`
    (disjoint from `exclude_idx`, e.g. run_mnist_experiment's `query_idx`).

    Deliberately generic over `model`/`distance_fn` (matching
    pairwise_lipschitz's own interface) so the same function can be called
    again for a different model or distance_fn without new code -- only
    `model_name`/`metric_name` need to change (they're only used to label
    the returned arrays), not the logic. run_mnist_experiment's Step 2b
    calls this once per model under Euclidean distance; the Mahalanobis
    distance_fn is a planned follow-up using the same loop.

    Nearest neighbors are found via sklearn's NearestNeighbors on RAW
    pixel space specifically -- that's "which points count as
    near-neighbors" defined independently of whatever distance_fn is
    under test, so the comparison answers: does a metric-consistent
    ratio, computed only on pairs a human would call visually similar,
    look different from the ratio computed over ALL pairs (most of which
    are visually unrelated digits)? The ratio itself is always computed
    via margin_fn/distance_fn (through `ratio_for_pairs`), never raw pixel
    distance -- only *which* pairs get selected uses raw pixel space.

    No plotting here (matches this file's convention: run_experiment.py
    never calls plots.py itself, the notebook does) -- `top_near_neighbor_pairs`
    is pre-assembled into the exact tuple shape plots.plot_image_pairs
    expects, so a caller can pass it straight through.

    Returns a dict of tensors/arrays (ratios, pair indices, the subset
    itself and its predictions, `top_near_neighbor_pairs`, a `summary` of
    scalars, and `arrays` -- the same data as plain numpy arrays keyed by
    `{metric_name}_{model_name}_...`, ready to merge into a caller's
    saved-results dict).
    """
    subset_idx = stratified_subset_idx(y_pool, n_points, seed=seed, exclude_idx=exclude_idx)
    x_subset = x_pool[subset_idx]
    y_subset = y_pool[subset_idx]

    with torch.no_grad():
        preds_subset = model(x_subset).argmax(dim=1)

    all_pairs_ratio, all_ii, all_jj = pairwise_lipschitz_all(
        model, x_subset, y_subset, margin_fn, distance_fn, max_pairs=max_pairs, seed=seed)

    # Raw pixel space on purpose -- see docstring. +1 neighbor since each
    # point is trivially its own nearest neighbor; dropped via [:, 1:].
    #
    # sklearn's threaded kneighbors query segfaults in-process alongside
    # torch on at least one dev machine (macOS; conflicting OpenMP
    # runtimes) -- confirmed directly with a minimal repro (bare
    # NearestNeighbors().kneighbors() call after `import torch`), not
    # assumed. Forcing single-threaded OpenMP for the duration of this
    # call only avoids it; restored immediately after (best-effort --
    # OpenMP may cache the thread count at first use, so this may not
    # perfectly undo any process-wide effect, but it doesn't hurt to try).
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

    # Deduplicated by canonical (min(i,j), max(i,j)): mutual nearest
    # neighbors produce both (i,j) and (j,i) in near_ii/near_jj with the
    # identical (symmetric) ratio, so without this a single pair of
    # points could occupy two of the top_k_images slots as a mirrored
    # duplicate -- confirmed this actually happens on real MNIST output,
    # not just a theoretical concern.
    sorted_idx = torch.argsort(near_ratio, descending=True)
    top_near_neighbor_pairs = []
    seen_canonical = set()
    for k in sorted_idx.tolist():
        i, j = near_ii[k].item(), near_jj[k].item()
        canonical = (min(i, j), max(i, j))
        if canonical in seen_canonical:
            continue
        seen_canonical.add(canonical)
        # Trailing (dist, margin_diff) fields beyond the 7 plot_image_pairs
        # needs -- it slices to the first 7 and ignores the rest, so this
        # stays compatible with every other caller that builds plain
        # 7-tuples (e.g. run_mnist_experiment's argmax_pair_lr_euclidean).
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
    """The main driver, in named steps:
    1. Train logistic regression, MLP, and CNN on full MNIST.
    2. Run all three Lipschitz estimators (Euclidean distance) on all three models.
    2b. Ratio-distribution analysis (Checkpoint 7), Euclidean distance, all
        three models.
    3. Compute full-training-set pixel covariance, sweep epsilon, select one.
    4. Re-run all three estimators (Mahalanobis distance, selected epsilon) on all three models.
    4b. Ratio-distribution analysis again, Mahalanobis distance (reusing the
        epsilon/precision from Step 3/4, not re-selected), all three models.
    5. Save everything to results/.
    """
    torch.manual_seed(seed)
    train, test, train_flat, test_flat, train_img, test_img = _build_models_and_data(seed)

    # --- Step 1: train all three models ---
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

    # Fixed set of held-out (test-set) query points, same points/labels for every model/metric.
    generator = torch.Generator().manual_seed(seed)
    query_idx = torch.randperm(len(test), generator=generator)[:n_lipschitz_points]
    x_query = test.x_flat[query_idx]
    y_query = test.y[query_idx]

    # --- Step 2: Euclidean estimators on all three models ---
    if verbose:
        print("\n=== Step 2: Euclidean-distance estimators ===")
    euclidean_results = {}
    for name, model in models.items():
        euclidean_results[name] = _run_estimators_for_model(
            model, x_query, y_query, euclidean_distance_fn,
            local_radius=local_radius, n_directions=n_directions, seed=seed)
        if verbose:
            r = euclidean_results[name]
            print(f"  {name}: pairwise={r['pairwise']:.4f}  local_max={r['local_max']:.4f}  "
                  f"grad_max={r['grad_max']:.4f}  grad_mean={r['grad_mean']:.4f}")

    # --- Step 2b: ratio-distribution analysis (Checkpoint 7) ---
    # Euclidean distance, all three models -- run_ratio_distribution_analysis
    # takes model/distance_fn as parameters (see its docstring), so this is
    # the same call repeated per model, matching Step 2's models.items()
    # loop; the Mahalanobis distance_fn is the same pattern again (Phase 2).
    if verbose:
        print("\n=== Step 2b: ratio-distribution analysis (Euclidean, all models) ===")
    ratio_dist_euclidean_results = {}
    for name, model in models.items():
        ratio_dist_euclidean_results[name] = run_ratio_distribution_analysis(
            model, name, "euclidean", test.x_flat, test.y, euclidean_distance_fn,
            exclude_idx=query_idx, n_points=n_ratio_points, k_neighbors=k_neighbors, seed=seed, verbose=verbose)

    # The pairwise-argmax pair from Step 2's logistic-regression/Euclidean
    # result, pre-assembled into the same tuple shape plot_image_pairs
    # expects (see plots.py) -- a single-pair list rather than the 6-pair
    # list run_ratio_distribution_analysis returns, since this is one
    # specific, already-identified pair, not a top-k selection.
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

    # --- Step 3: pixel covariance + epsilon sweep/selection ---
    if verbose:
        print("\n=== Step 3: epsilon selection ===")
    Sigma = pixel_covariance(train.x_flat)
    eigenvalues = covariance_eigenvalues(Sigma)
    cond_numbers = sweep_epsilon(Sigma, list(epsilon_values))
    stability_results = epsilon_stability_check(
        lr_model, train, list(epsilon_values), n_subsamples=n_subsamples,
        subsample_frac=subsample_frac, n_points=stability_n_points, seed=seed, verbose=verbose)
    cv_values = [stability_results[eps]["cv"] for eps in epsilon_values]
    selected_epsilon = select_epsilon(list(epsilon_values), cond_numbers, cv_values,
                                       max_cond=max_cond, max_cv=max_cv, verbose=verbose)

    precision = ridge_precision(Sigma, selected_epsilon)
    mahalanobis_distance_fn = make_mahalanobis_distance_fn(precision)

    # --- Step 4: Mahalanobis estimators on all three models ---
    if verbose:
        print(f"\n=== Step 4: Mahalanobis-distance estimators (epsilon={selected_epsilon:g}) ===")
    mahalanobis_results = {}
    for name, model in models.items():
        mahalanobis_results[name] = _run_estimators_for_model(
            model, x_query, y_query, mahalanobis_distance_fn, precision=precision,
            local_radius=local_radius, n_directions=n_directions, seed=seed)
        if verbose:
            r = mahalanobis_results[name]
            print(f"  {name}: pairwise={r['pairwise']:.4f}  local_max={r['local_max']:.4f}  "
                  f"grad_max={r['grad_max']:.4f}  grad_mean={r['grad_mean']:.4f}")

    # --- Step 4b: ratio-distribution analysis (Checkpoint 7, Mahalanobis) ---
    # Same three models, same stratified subset construction (same seed ->
    # same query_idx/exclude_idx and same stratified_subset_idx draw as
    # Step 2b), only the distance_fn changes -- reuses the epsilon/precision
    # already selected in Step 3, does not re-run epsilon selection.
    if verbose:
        print(f"\n=== Step 4b: ratio-distribution analysis (Mahalanobis, epsilon={selected_epsilon:g}, all models) ===")
    ratio_dist_mahalanobis_results = {}
    for name, model in models.items():
        ratio_dist_mahalanobis_results[name] = run_ratio_distribution_analysis(
            model, name, "mahalanobis", test.x_flat, test.y, mahalanobis_distance_fn,
            exclude_idx=query_idx, n_points=n_ratio_points, k_neighbors=k_neighbors, seed=seed, verbose=verbose)

    # --- Step 5: save results ---
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
        "Sigma": Sigma,
        "covariance_eigenvalues": eigenvalues,
    }


def main():
    run_mnist_experiment()


if __name__ == "__main__":
    main()
