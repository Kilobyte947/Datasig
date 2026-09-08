"""StrongCNN variant of the adversarial-vs-Lipschitz-bound comparison."""
import torch

from mnist_example.data import load_mnist, make_loader
from mnist_example.models import (
    StrongCNN, STRONG_CNN_CONFIG, FlattenedInputWrapper, train_classifier, train_or_load_strong_cnn,
)
from mnist_example.augmentation import random_affine_augment
from mnist_example.estimators import pairwise_lipschitz, linear_layer_lipschitz, euclidean_distance_fn
from mnist_example.layer_decomposition import (
    full_logits_output_fn,
    _effective_head_lipschitz_exact,
)
from mnist_example.adversarial_run_experiment import (
    RESULTS_DIR,
    DEFAULT_EPSILONS,
    run_epsilon_sweep,
    summarize_epsilon_sweep,
)

torch.set_default_dtype(torch.float64)


def _require_eval_mode(model, fn_name):
    """Raises ValueError if model is in train() mode."""
    if model.training:
        raise ValueError(
            f"{fn_name}: model is in train() mode. StrongCNN has BatchNorm1d/Dropout2d/Dropout, "
            f"so every quantity this function computes would depend on which other points share "
            f"a batch (BatchNorm) or would be non-deterministic (Dropout) unless model.eval() is "
            f"active. Call model.eval() before calling {fn_name}.")


def strong_cnn_extractor_fn(model, x, y):
    """output_fn for everything in StrongCNN up to (not including) the final linear layer. Returns (N, 256)."""
    _require_eval_mode(model, "strong_cnn_extractor_fn")
    features = model.features(x.reshape(x.shape[0], 1, 28, 28))
    return model.classifier[:4](features)


def strong_cnn_head_module(model):
    """The final nn.Linear layer of model.classifier."""
    return model.classifier[4]


def full_logits_fn(model, x, y):
    """output_fn for the full network's logits."""
    return full_logits_output_fn(model, x, y)


def fit_strong_cnn_feature_normalizer(model, x_train, relative_floor=1e-2, absolute_floor=1e-8):
    """Fits per-dimension mean and std on the extractor's output. Returns (mean, std), each shape (256,)."""
    with torch.no_grad():
        features = strong_cnn_extractor_fn(model, x_train, None)
    mean = features.mean(dim=0)
    raw_std = features.std(dim=0)
    floor = max(relative_floor * raw_std.median().item(), absolute_floor)
    std = raw_std.clamp_min(floor)
    return mean, std


def _make_normalized_strong_cnn_extractor_fn(mean, std):
    """Wraps strong_cnn_extractor_fn to standardize its output by (mean, std)."""
    def _fn(model, x, y):
        return (strong_cnn_extractor_fn(model, x, y) - mean) / std
    return _fn


def compute_strong_cnn_bounds(model, x_query, y_query, distance_fn, x_train_for_norm,
                               normalize_features=True, max_pairs=None, seed=0, verbose=True):
    """L_head_exact, L_extractor_estimated, L_full_estimated, product and looseness_ratio for a 
    trained StrongCNN, under a given pixel-space distance function. Returns a dict of those five values."""
    _require_eval_mode(model, "compute_strong_cnn_bounds")
    wrapped_model = FlattenedInputWrapper(model)

    std = None
    extractor_fn = strong_cnn_extractor_fn
    if normalize_features:
        mean, std = fit_strong_cnn_feature_normalizer(model, x_train_for_norm)
        extractor_fn = _make_normalized_strong_cnn_extractor_fn(mean, std)

    head = strong_cnn_head_module(model)
    L_head_exact = _effective_head_lipschitz_exact(head, std)

    L_extractor_estimated, _, _ = pairwise_lipschitz(
        model, x_query, y_query, extractor_fn, distance_fn=distance_fn,
        max_pairs=max_pairs, seed=seed)
    L_full_estimated, _, _ = pairwise_lipschitz(
        wrapped_model, x_query, y_query, full_logits_fn, distance_fn=distance_fn,
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


def strong_cnn_head_layer_bound_check(model, example):
    """Compares the final linear layer's exact Lipschitz bound against its actual behaviour for one 
    attacked example. 
    Returns {"feature_distance", "L_head_exact", "head_bound", "actual_logit_distance", "head_bound_tightness"}."""
    _require_eval_mode(model, "strong_cnn_head_layer_bound_check")
    with torch.no_grad():
        x_image = example["x"].reshape(1, 1, 28, 28)
        x_adv_image = example["x_adv"].reshape(1, 1, 28, 28)
        features = model.classifier[:4](model.features(x_image)).squeeze(0)
        features_adv = model.classifier[:4](model.features(x_adv_image)).squeeze(0)
        head = strong_cnn_head_module(model)
        logits = head(features)
        logits_adv = head(features_adv)

    feature_distance = (features - features_adv).norm(p=2).item()
    L_head_exact = linear_layer_lipschitz(head)
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


def strong_cnn_bound_comparison(model, x_query, y_query, x_pool, y_pool, x_train_for_norm,
                                 distance_fn, epsilons=DEFAULT_EPSILONS, pgd_alpha_frac=0.25,
                                 pgd_num_steps=20, pgd_num_restarts=5, n_points=500,
                                 normalize_features=True, max_pairs=None, seed=0, verbose=True):
    """Computes L_full_estimated and product_bound, then runs the FGSM/PGD epsilon sweep with R_adv 
    measured under the same distance function. Returns (summary_df, sweep_results)."""
    _require_eval_mode(model, "strong_cnn_bound_comparison")
    bound_result = compute_strong_cnn_bounds(
        model, x_query, y_query, distance_fn, x_train_for_norm,
        normalize_features=normalize_features, max_pairs=max_pairs, seed=seed, verbose=verbose)
    L_full_estimated = bound_result["L_full_estimated"]
    product_bound = bound_result["product"]

    wrapped_model = FlattenedInputWrapper(model)
    sweep_results = run_epsilon_sweep(
        wrapped_model, x_pool, y_pool, epsilons=epsilons, pgd_alpha_frac=pgd_alpha_frac,
        pgd_num_steps=pgd_num_steps, pgd_num_restarts=pgd_num_restarts, n_points=n_points,
        distance_fn=distance_fn, seed=seed, verbose=verbose)

    summary_df = summarize_epsilon_sweep(sweep_results, L_full_estimated, product_bound, verbose=verbose)
    return summary_df, sweep_results


def main(distance_fn=None, seed=0, verbose=True):
    """Single-checkpoint StrongCNN baseline: trains (or loads the shared seed=0 checkpoint), then 
    runs the full epsilon sweep and bound comparison. distance_fn defaults to Euclidean. 
    Returns (summary_df, sweep_results, model)."""
    if distance_fn is None:
        distance_fn = euclidean_distance_fn

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train = load_mnist(train=True)
    test = load_mnist(train=False)

    if seed == 0:
        model, train_acc, test_acc = train_or_load_strong_cnn(seed=seed, verbose=verbose)
    else:
        torch.manual_seed(seed)
        train_loader = make_loader(train.x_image, train.y, batch_size=STRONG_CNN_CONFIG["batch_size"],
                                    shuffle=True, seed=seed)
        test_loader = make_loader(test.x_image, test.y, batch_size=1000, shuffle=False)
        model = StrongCNN(dropout_conv=STRONG_CNN_CONFIG["dropout_conv"],
                           dropout_fc=STRONG_CNN_CONFIG["dropout_fc"])
        augment_generator = torch.Generator().manual_seed(seed)
        augment_fn = lambda x: random_affine_augment(
            x, degrees=STRONG_CNN_CONFIG["augment_degrees"],
            translate=STRONG_CNN_CONFIG["augment_translate"], generator=augment_generator)
        lr_scheduler_fn = lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=STRONG_CNN_CONFIG["lr_scheduler_t_max"],
            eta_min=STRONG_CNN_CONFIG["lr_scheduler_eta_min"])
        model, train_acc, test_acc = train_classifier(
            model, train_loader, test_loader, epochs=STRONG_CNN_CONFIG["epochs"],
            lr=STRONG_CNN_CONFIG["lr"], verbose=verbose, augment_fn=augment_fn,
            lr_scheduler_fn=lr_scheduler_fn)
        if verbose:
            print(f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    model.eval()

    generator = torch.Generator().manual_seed(seed)
    query_idx = torch.randperm(len(test), generator=generator)[:1000]
    x_query, y_query = test.x_flat[query_idx], test.y[query_idx]

    pool_mask = torch.ones(len(test), dtype=torch.bool)
    pool_mask[query_idx] = False
    remaining_idx = pool_mask.nonzero(as_tuple=True)[0]
    pool_idx = remaining_idx[torch.randperm(len(remaining_idx), generator=generator)[:2000]]
    x_pool, y_pool = test.x_flat[pool_idx], test.y[pool_idx]

    norm_idx = torch.randperm(len(train), generator=generator)[:1000]
    x_train_for_norm = train.x_flat[norm_idx]

    summary_df, sweep_results = strong_cnn_bound_comparison(
        model, x_query, y_query, x_pool, y_pool, x_train_for_norm, distance_fn,
        seed=seed, verbose=verbose)

    metric_name = getattr(distance_fn, "__name__", "mahalanobis")
    suffix = "euclidean" if metric_name == "euclidean_distance_fn" else "mahalanobis"
    summary_df.to_csv(RESULTS_DIR / f"strong_cnn_epsilon_sweep_baseline_{suffix}.csv", index=False)

    return summary_df, sweep_results, model
