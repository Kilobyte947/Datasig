"""Compares achieved adversarial sensitivity against the theoretical Lipschitz bounds from
layer_decomposition.py, for the same trained CNN checkpoint.

Bounds are computed on the full logit vector, not the margin function used elsewhere in this
project — see layer_decomposition.py. achieved_ratio is measured on the same representation so
the comparison is valid. max_R_adv is a lower bound on the network's true worst-case sensitivity,
not an exact value: FGSM/PGD maximise cross-entropy loss, not this ratio directly.
"""

from pathlib import Path

import pandas as pd
import torch

from mnist_example.data import load_mnist, get_dev_subset, make_loader
from mnist_example.models import (
    SmallCNN, FlattenedInputWrapper, train_classifier, margin_fn, train_or_load_small_cnn,
)
from mnist_example.estimators import (
    linear_layer_lipschitz,
    pairwise_lipschitz,
    local_perturbation_lipschitz,
    gradient_norm_estimate,
    euclidean_distance_fn,
)
from mnist_example.distance import svd_ridge_precision, make_mahalanobis_distance_fn
from mnist_example.layer_decomposition import (
    layer_decomposition_experiment,
    extractor_output_fn,
    full_logits_output_fn,
    fit_feature_normalizer,
    METHODS,
    _make_normalized_extractor_output_fn,
    _effective_head_lipschitz_exact,
)
from mnist_example.attacks import fgsm_attack, pgd_attack

RESULTS_DIR = Path(__file__).resolve().parent / "results"

torch.set_default_dtype(torch.float64)

DEFAULT_EPSILONS = (0.05, 0.1, 0.15, 0.2, 0.25)  # Goodfellow et al. 2015's MNIST range
DEFAULT_WIDTHS = (4, 8, 16, 32, 64)
MAHALANOBIS_EPSILON = 0.01


def build_pixel_mahalanobis_distance_fn(x_flat, epsilon=MAHALANOBIS_EPSILON):
    """Fits a ridge-regularised Mahalanobis distance on raw pixel data and returns it as a
    distance_fn(x, y) callable. x_flat should be the full training set, since the precision matrix
    is a property of the data, not of any one model."""
    precision = svd_ridge_precision(x_flat, epsilon)
    return make_mahalanobis_distance_fn(precision)


def filter_correctly_classified(model, x, y):
    """Restricts (x, y) to points model already classifies correctly. Returns (x_correct, y_correct,
    kept_fraction)."""
    with torch.no_grad():
        preds = model(x).argmax(dim=1)
    correct = preds == y
    return x[correct], y[correct], correct.float().mean().item()


def achieved_ratio(model, x, x_adv, distance_fn=euclidean_distance_fn):
    """R_adv = ||f(x) - f(x_adv)||_2 / distance_fn(x, x_adv), per example, where f is the model's
    full logit vector. Only the denominator is pluggable; the numerator is always Euclidean distance
    in logit space. Returns the (N,) tensor of ratios, with 0 where x and x_adv coincide."""
    with torch.no_grad():
        f_x = model(x)
        f_adv = model(x_adv)
    numerator = (f_x - f_adv).norm(p=2, dim=-1)
    denominator = distance_fn(x, x_adv)
    return torch.where(denominator > 1e-12, numerator / denominator.clamp_min(1e-12),
                        torch.zeros_like(denominator))


def run_epsilon_sweep(model, x_pool, y_pool, epsilons=DEFAULT_EPSILONS,
                       pgd_alpha_frac=0.25, pgd_num_steps=20, pgd_num_restarts=5,
                       n_points=500, distance_fn=euclidean_distance_fn, seed=0, verbose=True):
    """Runs FGSM and PGD at each epsilon against n_points correctly-classified points, and computes
    achieved_ratio and the post-attack misclassification rate for each. Filtering happens once, up
    front, so the same points are attacked at every epsilon. Returns a dict with the evaluated points
    and per-(epsilon, method) results."""
    generator = torch.Generator().manual_seed(seed)
    x_correct, y_correct, kept_frac = filter_correctly_classified(model, x_pool, y_pool)
    if verbose:
        print(f"filter_correctly_classified: kept {kept_frac:.4f} of the pool "
              f"({x_correct.shape[0]}/{x_pool.shape[0]} points)")

    n = min(n_points, x_correct.shape[0])
    idx = torch.randperm(x_correct.shape[0], generator=generator)[:n]
    x_eval, y_eval = x_correct[idx], y_correct[idx]

    per_case = {}
    for epsilon in epsilons:
        x_fgsm = fgsm_attack(model, x_eval, y_eval, epsilon)
        x_pgd = pgd_attack(model, x_eval, y_eval, epsilon=epsilon, alpha=pgd_alpha_frac * epsilon,
                            num_steps=pgd_num_steps, num_restarts=pgd_num_restarts, seed=seed)

        for method, x_adv in (("FGSM", x_fgsm), ("PGD", x_pgd)):
            R_adv = achieved_ratio(model, x_eval, x_adv, distance_fn=distance_fn)
            with torch.no_grad():
                preds_adv = model(x_adv).argmax(dim=1)
            is_misclassified = preds_adv != y_eval
            pct_misclassified = is_misclassified.float().mean().item()
            per_case[(epsilon, method)] = {
                "x_adv": x_adv, "R_adv": R_adv, "pct_misclassified": pct_misclassified,
                "is_misclassified": is_misclassified,
            }
            if verbose:
                print(f"  epsilon={epsilon:g}  {method:4s}  mean_R_adv={R_adv.mean().item():.4f}  "
                      f"max_R_adv={R_adv.max().item():.4f}  "
                      f"pct_misclassified={pct_misclassified:.4f}")

    return {"x_eval": x_eval, "y_eval": y_eval, "kept_frac": kept_frac, "per_case": per_case}


def summarize_epsilon_sweep(sweep_results, L_full_estimated, product_bound, verbose=True):
    """Builds the per-(epsilon, method) summary table, including each row's ratio to L_full_estimated
    and to product_bound. Warns (without raising) if max_R_adv exceeds L_full_estimated, since that
    bound should never be violated — a likely sign the bound itself under-sampled."""
    rows = []
    violations = []
    for (epsilon, method), case in sweep_results["per_case"].items():
        R = case["R_adv"]
        max_R = R.max().item()
        rows.append({
            "epsilon": epsilon,
            "method": method,
            "mean_R_adv": R.mean().item(),
            "median_R_adv": R.median().item(),
            "max_R_adv": max_R,
            "pct_misclassified": case["pct_misclassified"],
            "L_full_estimated": L_full_estimated,
            "product_bound": product_bound,
            "ratio_to_L_full": max_R / L_full_estimated if L_full_estimated > 1e-12 else float("inf"),
            "ratio_to_product_bound": max_R / product_bound if product_bound > 1e-12 else float("inf"),
        })
        if max_R > L_full_estimated * (1 + 1e-6):
            violations.append((epsilon, method, max_R))

    df = pd.DataFrame(rows).sort_values(["method", "epsilon"]).reset_index(drop=True)

    for epsilon, method, max_R in violations:
        print(f"WARNING: max_R_adv={max_R:.4f} > L_full_estimated={L_full_estimated:.4f} at "
              f"epsilon={epsilon:g}, method={method!r} ")

    if verbose:
        print(df.to_string(index=False))
    return df


def most_and_least_sensitive_examples(model, sweep_results, distance_fn=euclidean_distance_fn):
    """Finds the single example with the largest and smallest R_adv across every (epsilon, method)
    case. Returns (most_sensitive, least_sensitive) dicts with the input/adversarial pair, predictions,
    and pixel distance."""
    x_eval, y_eval = sweep_results["x_eval"], sweep_results["y_eval"]

    best = None   # (R_adv, epsilon, method, idx)
    worst = None
    for (epsilon, method), case in sweep_results["per_case"].items():
        R = case["R_adv"]
        idx_max = R.argmax().item()
        idx_min = R.argmin().item()
        if best is None or R[idx_max].item() > best[0]:
            best = (R[idx_max].item(), epsilon, method, idx_max)
        if worst is None or R[idx_min].item() < worst[0]:
            worst = (R[idx_min].item(), epsilon, method, idx_min)

    def _package(entry):
        R_adv, epsilon, method, idx = entry
        x = x_eval[idx]
        x_adv = sweep_results["per_case"][(epsilon, method)]["x_adv"][idx]
        with torch.no_grad():
            pred_clean = model(x.unsqueeze(0)).argmax(dim=1).item()
            pred_adv = model(x_adv.unsqueeze(0)).argmax(dim=1).item()
        return {
            "epsilon": epsilon, "method": method, "index": idx, "R_adv": R_adv,
            "x": x, "x_adv": x_adv, "pixel_distance": distance_fn(x, x_adv).item(),
            "y_true": y_eval[idx].item(),
            "pred_clean": pred_clean, "pred_adv": pred_adv,
        }

    return _package(best), _package(worst)


def find_examples_by_criteria(model, sweep_results, epsilon, method, R_adv_max=None,
                               misclassified_only=None, distance_fn=euclidean_distance_fn):
    """Finds every example in one (epsilon, method) case matching given thresholds: R_adv_max and/or
    misclassified_only. Returns a list of example dicts, each including the full clean and adversarial
    logit vectors."""
    case = sweep_results["per_case"][(epsilon, method)]
    x_eval, y_eval = sweep_results["x_eval"], sweep_results["y_eval"]
    R_adv, is_misclassified, x_adv_all = case["R_adv"], case["is_misclassified"], case["x_adv"]

    mask = torch.ones_like(R_adv, dtype=torch.bool)
    if R_adv_max is not None:
        mask &= R_adv < R_adv_max
    if misclassified_only is not None:
        mask &= is_misclassified if misclassified_only else ~is_misclassified

    examples = []
    for idx in mask.nonzero(as_tuple=True)[0].tolist():
        x, x_adv = x_eval[idx], x_adv_all[idx]
        with torch.no_grad():
            logits_clean = model(x.unsqueeze(0)).squeeze(0)
            logits_adv = model(x_adv.unsqueeze(0)).squeeze(0)
        examples.append({
            "epsilon": epsilon, "method": method, "index": idx, "R_adv": R_adv[idx].item(),
            "x": x, "x_adv": x_adv, "pixel_distance": distance_fn(x, x_adv).item(),
            "y_true": y_eval[idx].item(),
            "pred_clean": logits_clean.argmax().item(), "pred_adv": logits_adv.argmax().item(),
            "logits_clean": logits_clean, "logits_adv": logits_adv,
        })
    return examples


def head_layer_bound_check(model, example):
    """Checks the head layer's exact Lipschitz bound against its actual behaviour for one attacked
    example: feature-space distance, the bound it implies for logit movement, and the network's real
    logit-space distance. Returns a dict including head_bound_tightness, the ratio of actual to bound."""
    with torch.no_grad():
        x_image = example["x"].reshape(1, 1, 28, 28)
        x_adv_image = example["x_adv"].reshape(1, 1, 28, 28)
        features = model.extractor(x_image).squeeze(0)
        features_adv = model.extractor(x_adv_image).squeeze(0)
        logits = model.head(features)
        logits_adv = model.head(features_adv)

    feature_distance = (features - features_adv).norm(p=2).item()
    L_head_exact = linear_layer_lipschitz(model.head)
    head_bound = L_head_exact * feature_distance
    actual_logit_distance = (logits - logits_adv).norm(p=2).item()
    tightness = actual_logit_distance / head_bound if head_bound > 1e-12 else float("nan")

    return {
        "feature_distance": feature_distance,
        "L_head_exact": L_head_exact,
        "head_bound": head_bound,
        "actual_logit_distance": actual_logit_distance,
        "head_bound_tightness": tightness,
    }


# ---------------------------------------------------------------------------
# Per-run measurement helpers for the multi-seed confirmation sweep (seed_sweep.py).
# ---------------------------------------------------------------------------

def adversarial_accuracy(model, x, x_adv, y):
    """Clean and adversarial accuracy for one attacked batch, assumed already restricted to correctly-
    classified points. Returns clean_acc, adv_acc, misclassification_rate, and flip counts."""
    with torch.no_grad():
        pred_clean = model(x).argmax(dim=1)
        pred_adv = model(x_adv).argmax(dim=1)
    n = y.shape[0]
    n_flipped = (pred_adv != y).sum().item()
    return {
        "clean_acc": (pred_clean == y).float().mean().item(),
        "adv_acc": 1.0 - n_flipped / n,
        "misclassification_rate": n_flipped / n,
        "n_flipped": n_flipped,
        "n_evaluated": n,
    }


def clean_logit_stats(model, x):
    """Distribution of the clean model's logit norm and top-2 margin over x. 
    Returns mean/std of both, plus the 5th and 10th percentiles of the margin."""
    with torch.no_grad():
        logits = model(x)
    logit_norm = logits.norm(p=2, dim=-1)
    top2 = logits.topk(2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    return {
        "mean_logit_norm": logit_norm.mean().item(),
        "std_logit_norm": logit_norm.std().item(),
        "mean_top2_margin": margin.mean().item(),
        "std_top2_margin": margin.std().item(),
        "p5_top2_margin": margin.quantile(0.05).item(),
        "p10_top2_margin": margin.quantile(0.10).item(),
    }


def margin_lipschitz_estimate(model, x, y, estimator="pairwise", distance_fn=euclidean_distance_fn,
                               max_pairs=None, radius=1.0, n_directions=40, seed=0):
    """Empirical Lipschitz constant of the scalar margin function, via one of the three estimators in
    estimators.py ("pairwise", "grid", "gradient"). This is a different function from the full-logit
    bounds used elsewhere in this module — report it separately, never merged with L_full_estimated."""
    if estimator not in METHODS:
        raise ValueError(f"unknown estimator {estimator!r}, expected one of {METHODS}")
    if estimator == "pairwise":
        L_hat, _, _ = pairwise_lipschitz(model, x, y, margin_fn, distance_fn=distance_fn,
                                          max_pairs=max_pairs, seed=seed)
        return L_hat
    elif estimator == "grid":
        return local_perturbation_lipschitz(
            model, x, y, margin_fn, distance_fn=distance_fn,
            radius=radius, n_directions=n_directions, seed=seed).max().item()
    else:  # "gradient"
        return gradient_norm_estimate(model, x, y, margin_fn).max().item()


def flip_direction_alignment(model, x, x_adv, y):
    """For each attacked point, how well the logit movement aligns with the direction that would flip
    the prediction toward the clean runner-up class. Positive means aligned with flipping; points with
    zero movement get cosine 0. Returns mean and std of the cosine alignment."""
    with torch.no_grad():
        logits_clean = model(x)
        logits_adv = model(x_adv)
    dz = logits_adv - logits_clean

    masked = logits_clean.clone()
    masked.scatter_(1, y.unsqueeze(1), float("-inf"))
    k = masked.argmax(dim=1)

    dz_k = dz.gather(1, k.unsqueeze(1)).squeeze(1)
    dz_y = dz.gather(1, y.unsqueeze(1)).squeeze(1)
    numerator = dz_k - dz_y
    denom = dz.norm(p=2, dim=-1) * (2.0 ** 0.5)
    cosine = torch.where(denom > 1e-12, numerator / denom.clamp_min(1e-12), torch.zeros_like(denom))

    return {
        "mean_cosine_alignment": cosine.mean().item(),
        "std_cosine_alignment": cosine.std().item(),
    }


def run_bound_comparison(model, x_query, y_query, x_pool, y_pool, x_train_for_norm,
                          epsilons=DEFAULT_EPSILONS, bound_method="pairwise",
                          pgd_alpha_frac=0.25, pgd_num_steps=20, pgd_num_restarts=5,
                          n_points=500, normalize_features=True, seed=0, verbose=True):
    """Computes L_full_estimated and product_bound for model via layer_decomposition_experiment, then
    runs the FGSM/PGD epsilon sweep against the same checkpoint and compares achieved sensitivity to
    both bounds. Returns (summary_df, sweep_results)."""
    bound_result = layer_decomposition_experiment(
        model, x_query, y_query, x_train_for_norm=x_train_for_norm,
        method=bound_method, normalize_features=normalize_features, seed=seed, verbose=verbose)
    L_full_estimated = bound_result["L_full_estimated"]
    product_bound = bound_result["product"]

    wrapped_model = FlattenedInputWrapper(model)
    sweep_results = run_epsilon_sweep(
        wrapped_model, x_pool, y_pool, epsilons=epsilons, pgd_alpha_frac=pgd_alpha_frac,
        pgd_num_steps=pgd_num_steps, pgd_num_restarts=pgd_num_restarts, n_points=n_points,
        seed=seed, verbose=verbose)

    summary_df = summarize_epsilon_sweep(sweep_results, L_full_estimated, product_bound, verbose=verbose)
    return summary_df, sweep_results


def compute_bounds_with_distance_fn(model, x_query, y_query, distance_fn, x_train_for_norm,
                                     max_pairs=None, seed=0, verbose=True):
    """Recomputes L_head_exact, L_extractor_estimated, L_full_estimated and product_bound under an
    arbitrary pixel-space distance_fn — needed since layer_decomposition_experiment hardcodes
    Euclidean. L_head_exact is unaffected by distance_fn, since it operates in feature/logit space,
    not pixel space. Matches layer_decomposition_experiment's own numbers exactly when
    distance_fn=euclidean_distance_fn. Returns the same keys as layer_decomposition_experiment."""
    wrapped_model = FlattenedInputWrapper(model)
    mean, std = fit_feature_normalizer(model, x_train_for_norm)
    extractor_fn = _make_normalized_extractor_output_fn(mean, std)

    L_head_exact = _effective_head_lipschitz_exact(model.head, std)
    L_extractor_estimated, _, _ = pairwise_lipschitz(
        model, x_query, y_query, extractor_fn, distance_fn=distance_fn,
        max_pairs=max_pairs, seed=seed)
    L_full_estimated, _, _ = pairwise_lipschitz(
        wrapped_model, x_query, y_query, full_logits_output_fn, distance_fn=distance_fn,
        max_pairs=max_pairs, seed=seed)

    product = L_extractor_estimated * L_head_exact
    looseness_ratio = product / L_full_estimated if L_full_estimated > 1e-12 else float("inf")

    if looseness_ratio < 1.0 - 1e-6:
        print(f"WARNING: looseness_ratio={looseness_ratio:.4f} < 1 -- this violates the "
              f"theoretical submultiplicative bound (Szegedy et al. 2014) and most likely "
              f"indicates an estimator sampling issue (too few pairs/query points), not a real "
              f"result. Inspect before trusting.")
    elif verbose:
        print(f"  [distance_fn={getattr(distance_fn, '__name__', distance_fn)!r}] "
              f"L_head_exact={L_head_exact:.4f}  L_extractor_est={L_extractor_estimated:.4f}  "
              f"L_full_est={L_full_estimated:.4f}  looseness_ratio={looseness_ratio:.4f}")

    return {
        "L_head_exact": L_head_exact,
        "L_extractor_estimated": L_extractor_estimated,
        "L_full_estimated": L_full_estimated,
        "product": product,
        "looseness_ratio": looseness_ratio,
    }


def run_bound_comparison_with_distance_fn(model, x_query, y_query, x_pool, y_pool, x_train_for_norm,
                                           distance_fn, epsilons=DEFAULT_EPSILONS,
                                           pgd_alpha_frac=0.25, pgd_num_steps=20, pgd_num_restarts=5,
                                           n_points=500, max_pairs=None, seed=0, verbose=True):
    """Pluggable-distance_fn version of run_bound_comparison. Produces the same adversarial examples
    as run_bound_comparison for the same model/seed — only how sensitivity is measured differs.
    Returns (summary_df, sweep_results)."""
    bound_result = compute_bounds_with_distance_fn(
        model, x_query, y_query, distance_fn, x_train_for_norm,
        max_pairs=max_pairs, seed=seed, verbose=verbose)
    L_full_estimated = bound_result["L_full_estimated"]
    product_bound = bound_result["product"]

    wrapped_model = FlattenedInputWrapper(model)
    sweep_results = run_epsilon_sweep(
        wrapped_model, x_pool, y_pool, epsilons=epsilons, pgd_alpha_frac=pgd_alpha_frac,
        pgd_num_steps=pgd_num_steps, pgd_num_restarts=pgd_num_restarts, n_points=n_points,
        distance_fn=distance_fn, seed=seed, verbose=verbose)

    summary_df = summarize_epsilon_sweep(sweep_results, L_full_estimated, product_bound, verbose=verbose)
    return summary_df, sweep_results


def run_cnn_adversarial_width_sweep(widths=DEFAULT_WIDTHS, epochs=6, train_subset_size=5000,
                                     n_query_points=40, n_train_norm_points=500,
                                     n_pool_points=2000, n_attack_points=500,
                                     epsilons=DEFAULT_EPSILONS, bound_method="pairwise",
                                     pgd_alpha_frac=0.25, pgd_num_steps=20, pgd_num_restarts=5,
                                     normalize_features=True, seed=0, verbose=True,
                                     save_path=RESULTS_DIR / "adversarial_width_sweep.csv"):
    """Repeats run_bound_comparison across CNN widths, to see whether the gap between achieved
    adversarial sensitivity and the theoretical bounds narrows or widens with model capacity. Trains
    a fresh SmallCNN per width. Returns (combined_df, per_width_summary_dfs, per_width_extremes)."""
    train = load_mnist(train=True)
    test = load_mnist(train=False)
    dev = get_dev_subset(train, n=train_subset_size, seed=seed)
    train_loader = make_loader(dev.x_image, dev.y, batch_size=128, shuffle=True, seed=seed)
    test_loader = make_loader(test.x_image, test.y, batch_size=1000, shuffle=False)

    generator = torch.Generator().manual_seed(seed)
    query_idx = torch.randperm(len(test), generator=generator)[:n_query_points]
    x_query, y_query = test.x_flat[query_idx], test.y[query_idx]
    norm_idx = torch.randperm(len(train), generator=generator)[:n_train_norm_points]
    x_train_for_norm = train.x_flat[norm_idx]

    pool_mask = torch.ones(len(test), dtype=torch.bool)
    pool_mask[query_idx] = False
    remaining_idx = pool_mask.nonzero(as_tuple=True)[0]
    pool_idx = remaining_idx[torch.randperm(len(remaining_idx), generator=generator)[:n_pool_points]]
    x_pool, y_pool = test.x_flat[pool_idx], test.y[pool_idx]

    max_epsilon = max(epsilons)
    rows = []
    per_width_summary_dfs = {}
    per_width_extremes = {}
    for width in widths:
        if verbose:
            print(f"=== width={width} (conv_channels=({width}, {2 * width})) ===")
        torch.manual_seed(seed)
        model, train_acc, test_acc = train_classifier(
            SmallCNN(conv_channels=(width, 2 * width)), train_loader, test_loader,
            epochs=epochs, lr=1e-3, verbose=False)
        if verbose:
            print(f"  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

        summary_df, sweep_results = run_bound_comparison(
            model, x_query, y_query, x_pool, y_pool, x_train_for_norm,
            epsilons=epsilons, bound_method=bound_method, pgd_alpha_frac=pgd_alpha_frac,
            pgd_num_steps=pgd_num_steps, pgd_num_restarts=pgd_num_restarts,
            n_points=n_attack_points, normalize_features=normalize_features,
            seed=seed, verbose=verbose)
        per_width_summary_dfs[width] = summary_df

        most_sensitive, least_sensitive = most_and_least_sensitive_examples(
            FlattenedInputWrapper(model), sweep_results)
        most_sensitive.update(head_layer_bound_check(model, most_sensitive))
        least_sensitive.update(head_layer_bound_check(model, least_sensitive))
        per_width_extremes[width] = (most_sensitive, least_sensitive)

        row = {
            "width": width, "train_acc": train_acc, "test_acc": test_acc,
            "L_full_estimated": summary_df["L_full_estimated"].iloc[0],
            "product_bound": summary_df["product_bound"].iloc[0],
        }
        for method in ("FGSM", "PGD"):
            m = summary_df[(summary_df["method"] == method) & (summary_df["epsilon"] == max_epsilon)].iloc[0]
            suffix = method.lower()
            row[f"max_R_adv_{suffix}"] = m["max_R_adv"]
            row[f"ratio_to_L_full_{suffix}"] = m["ratio_to_L_full"]
            row[f"ratio_to_product_bound_{suffix}"] = m["ratio_to_product_bound"]
        rows.append(row)

    combined_df = pd.DataFrame(rows)

    if save_path is not None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(save_path, index=False)
        for width, df in per_width_summary_dfs.items():
            df.to_csv(RESULTS_DIR / f"adversarial_epsilon_sweep_width{width}.csv", index=False)
        if verbose:
            print(f"\nSaved width-sweep results to {save_path} "
                  f"(+ one adversarial_epsilon_sweep_width{{w}}.csv per width)")

    return combined_df, per_width_summary_dfs, per_width_extremes


def run_cnn_adversarial_width_sweep_with_distance_fn(
        distance_fn, widths=DEFAULT_WIDTHS, epochs=6, train_subset_size=5000,
        n_query_points=40, n_train_norm_points=500, n_pool_points=2000, n_attack_points=500,
        epsilons=DEFAULT_EPSILONS, pgd_alpha_frac=0.25, pgd_num_steps=20, pgd_num_restarts=5,
        max_pairs=None, seed=0, verbose=True,
        save_path=RESULTS_DIR / "adversarial_width_sweep_distance_fn.csv"):
    """Pluggable-distance_fn version of run_cnn_adversarial_width_sweep. Reproduces bit-identical
    checkpoints to that function for the same seed, so both sweeps compare the same trained models
    under different distance metrics. Returns the same shape as run_cnn_adversarial_width_sweep."""
    train = load_mnist(train=True)
    test = load_mnist(train=False)
    dev = get_dev_subset(train, n=train_subset_size, seed=seed)
    train_loader = make_loader(dev.x_image, dev.y, batch_size=128, shuffle=True, seed=seed)
    test_loader = make_loader(test.x_image, test.y, batch_size=1000, shuffle=False)

    generator = torch.Generator().manual_seed(seed)
    query_idx = torch.randperm(len(test), generator=generator)[:n_query_points]
    x_query, y_query = test.x_flat[query_idx], test.y[query_idx]
    norm_idx = torch.randperm(len(train), generator=generator)[:n_train_norm_points]
    x_train_for_norm = train.x_flat[norm_idx]

    pool_mask = torch.ones(len(test), dtype=torch.bool)
    pool_mask[query_idx] = False
    remaining_idx = pool_mask.nonzero(as_tuple=True)[0]
    pool_idx = remaining_idx[torch.randperm(len(remaining_idx), generator=generator)[:n_pool_points]]
    x_pool, y_pool = test.x_flat[pool_idx], test.y[pool_idx]

    max_epsilon = max(epsilons)
    rows = []
    per_width_summary_dfs = {}
    per_width_extremes = {}
    for width in widths:
        if verbose:
            print(f"=== width={width} (conv_channels=({width}, {2 * width})) ===")
        torch.manual_seed(seed)
        model, train_acc, test_acc = train_classifier(
            SmallCNN(conv_channels=(width, 2 * width)), train_loader, test_loader,
            epochs=epochs, lr=1e-3, verbose=False)
        if verbose:
            print(f"  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

        summary_df, sweep_results = run_bound_comparison_with_distance_fn(
            model, x_query, y_query, x_pool, y_pool, x_train_for_norm, distance_fn,
            epsilons=epsilons, pgd_alpha_frac=pgd_alpha_frac,
            pgd_num_steps=pgd_num_steps, pgd_num_restarts=pgd_num_restarts,
            n_points=n_attack_points, max_pairs=max_pairs, seed=seed, verbose=verbose)
        per_width_summary_dfs[width] = summary_df

        most_sensitive, least_sensitive = most_and_least_sensitive_examples(
            FlattenedInputWrapper(model), sweep_results, distance_fn=distance_fn)
        most_sensitive.update(head_layer_bound_check(model, most_sensitive))
        least_sensitive.update(head_layer_bound_check(model, least_sensitive))
        per_width_extremes[width] = (most_sensitive, least_sensitive)

        row = {
            "width": width, "train_acc": train_acc, "test_acc": test_acc,
            "L_full_estimated": summary_df["L_full_estimated"].iloc[0],
            "product_bound": summary_df["product_bound"].iloc[0],
        }
        for method in ("FGSM", "PGD"):
            m = summary_df[(summary_df["method"] == method) & (summary_df["epsilon"] == max_epsilon)].iloc[0]
            suffix = method.lower()
            row[f"max_R_adv_{suffix}"] = m["max_R_adv"]
            row[f"ratio_to_L_full_{suffix}"] = m["ratio_to_L_full"]
            row[f"ratio_to_product_bound_{suffix}"] = m["ratio_to_product_bound"]
        rows.append(row)

    combined_df = pd.DataFrame(rows)

    if save_path is not None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(save_path, index=False)
        for width, df in per_width_summary_dfs.items():
            df.to_csv(RESULTS_DIR / f"adversarial_epsilon_sweep_width{width}_distance_fn.csv", index=False)
        if verbose:
            print(f"\nSaved width-sweep results to {save_path} "
                  f"(+ one adversarial_epsilon_sweep_width{{w}}_distance_fn.csv per width)")

    return combined_df, per_width_summary_dfs, per_width_extremes


def main(seed=0, verbose=True):
    """Baseline run: trains the project's default SmallCNN on full MNIST, then runs the full epsilon
    sweep and bound comparison. At seed=0, reuses the shared checkpoint used elsewhere in this
    project. Returns (summary_df, sweep_results)."""
    train = load_mnist(train=True)
    test = load_mnist(train=False)

    if seed == 0:
        model, train_acc, test_acc = train_or_load_small_cnn(seed=seed, verbose=verbose)
    else:
        torch.manual_seed(seed)
        train_loader = make_loader(train.x_image, train.y, batch_size=256, shuffle=True, seed=seed)
        test_loader = make_loader(test.x_image, test.y, batch_size=1000, shuffle=False)
        model, train_acc, test_acc = train_classifier(
            SmallCNN(), train_loader, test_loader, epochs=8, lr=1e-3, verbose=verbose)
    if verbose:
        print(f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    generator = torch.Generator().manual_seed(seed)
    query_idx = torch.randperm(len(test), generator=generator)[:200]
    x_query, y_query = test.x_flat[query_idx], test.y[query_idx]

    pool_mask = torch.ones(len(test), dtype=torch.bool)
    pool_mask[query_idx] = False
    remaining_idx = pool_mask.nonzero(as_tuple=True)[0]
    pool_idx = remaining_idx[torch.randperm(len(remaining_idx), generator=generator)[:2000]]
    x_pool, y_pool = test.x_flat[pool_idx], test.y[pool_idx]

    norm_idx = torch.randperm(len(train), generator=generator)[:1000]
    x_train_for_norm = train.x_flat[norm_idx]

    summary_df, sweep_results = run_bound_comparison(
        model, x_query, y_query, x_pool, y_pool, x_train_for_norm, seed=seed, verbose=verbose)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(RESULTS_DIR / "adversarial_epsilon_sweep_baseline.csv", index=False)

    return summary_df, sweep_results


def main_with_distance_fn(distance_fn, seed=0, verbose=True):
    """Pluggable-distance_fn version of main, using the same SmallCNN configuration and seed
    convention. Returns (summary_df, sweep_results)."""
    train = load_mnist(train=True)
    test = load_mnist(train=False)

    if seed == 0:
        model, train_acc, test_acc = train_or_load_small_cnn(seed=seed, verbose=verbose)
    else:
        torch.manual_seed(seed)
        train_loader = make_loader(train.x_image, train.y, batch_size=256, shuffle=True, seed=seed)
        test_loader = make_loader(test.x_image, test.y, batch_size=1000, shuffle=False)
        model, train_acc, test_acc = train_classifier(
            SmallCNN(), train_loader, test_loader, epochs=8, lr=1e-3, verbose=verbose)
    if verbose:
        print(f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    generator = torch.Generator().manual_seed(seed)
    query_idx = torch.randperm(len(test), generator=generator)[:200]
    x_query, y_query = test.x_flat[query_idx], test.y[query_idx]

    pool_mask = torch.ones(len(test), dtype=torch.bool)
    pool_mask[query_idx] = False
    remaining_idx = pool_mask.nonzero(as_tuple=True)[0]
    pool_idx = remaining_idx[torch.randperm(len(remaining_idx), generator=generator)[:2000]]
    x_pool, y_pool = test.x_flat[pool_idx], test.y[pool_idx]

    norm_idx = torch.randperm(len(train), generator=generator)[:1000]
    x_train_for_norm = train.x_flat[norm_idx]

    summary_df, sweep_results = run_bound_comparison_with_distance_fn(
        model, x_query, y_query, x_pool, y_pool, x_train_for_norm, distance_fn,
        seed=seed, verbose=verbose)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(RESULTS_DIR / "adversarial_epsilon_sweep_baseline_distance_fn.csv", index=False)

    return summary_df, sweep_results
