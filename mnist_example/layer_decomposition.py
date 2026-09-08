"""Layer-decomposed Lipschitz sub-experiment for the CNN: splits f = head o extractor and compares
the exact and estimated Lipschitz constant of each layer against the whole network's, to test the
submultiplicative bound L_extractor * L_head >= L_full (Szegedy et al., 2014).

L_full here is measured on the full logit vector, not the scalar margin used elsewhere in this
project — the bound only applies to the function the product actually bounds.
"""

from pathlib import Path
import pandas as pd
import torch
from mnist_example.data import load_mnist, get_dev_subset, make_loader
from mnist_example.estimators import (
    euclidean_distance_fn,
    pairwise_lipschitz,
    local_perturbation_lipschitz,
    gradient_norm_estimate,
    linear_layer_lipschitz,
)
from mnist_example.models import SmallCNN, FlattenedInputWrapper, train_classifier

RESULTS_DIR = Path(__file__).resolve().parent / "results"

METHODS = ("pairwise", "grid", "gradient")

def extractor_output_fn(model, x, y):
    """output_fn for model.extractor: flat (N, 784) pixel input. Returns (N, feat_dim)."""
    return model.extractor(x.reshape(x.shape[0], 1, 28, 28))


def head_output_fn(model, x, y):
    """output_fn for model.head: (N, feat_dim) already-extracted features. Returns (N, num_classes) logits."""
    return model.head(x)


def full_logits_output_fn(model, x, y):
    """output_fn for the full network: flat (N, 784) pixel input. Returns (N, num_classes) logits."""
    return model(x)


def fit_feature_normalizer(model, x_train, relative_floor=1e-2, absolute_floor=1e-8):
    """Fits a per-dimension mean/std normalizer on the extractor's output features, to be used in
    `layer_decomposition_experiment` when `normalize_features=True`. Returns (mean, std) tensors, 
    each of shape (feat_dim,). `std` is clamped to a floor to avoid near-zero 
    scales dominating the Lipschitz estimate. 
    The floor is set to max(relative_floor * median(std), absolute_floor)."""
    with torch.no_grad():
        features = extractor_output_fn(model, x_train, None)
    mean = features.mean(dim=0)
    raw_std = features.std(dim=0)
    floor = max(relative_floor * raw_std.median().item(), absolute_floor)
    std = raw_std.clamp_min(floor)
    return mean, std


def _make_normalized_extractor_output_fn(mean, std):
    """Wraps extractor_output_fn to standardize its output by (mean, std)."""
    def _fn(model, x, y):
        return (extractor_output_fn(model, x, y) - mean) / std
    return _fn


def _make_normalized_head_output_fn(mean, std):
    """Wraps head_output_fn to un-standardize its input first, so the composition of the normalized
    extractor and this function still equals the original model exactly."""
    def _fn(model, z, y):
        return head_output_fn(model, z * std + mean, y)
    return _fn


def _effective_head_lipschitz_exact(head, std=None):
    """Returns the exact spectral norm of the head's weight matrix, adjusted for feature standardization
    if `std` is provided (the standardizer's per-dimension std, shape (feat_dim,))."""
    if std is None:
        return linear_layer_lipschitz(head)
    W_effective = head.weight.detach() * std.unsqueeze(0)
    return torch.linalg.matrix_norm(W_effective, ord=2).item()


def layer_decomposition_experiment(model, x_query, y_query, x_train_for_norm=None,
                                    method="pairwise", normalize_features=True,
                                    radius=1.0, n_directions=40, max_pairs=None, seed=0, verbose=True):
    """Computes L_head (exact and estimated), L_extractor (estimated), L_full (estimated), and the
    resulting looseness ratio, for a trained SmallCNN. method is "pairwise", "grid" or "gradient".
    normalize_features standardizes extractor features before estimation.
    
    Warns (without raising) if looseness_ratio falls below 1, since that violates the theoretical
    bound and most likely reflects estimator under-sampling. looseness_ratio_estimated is reported
    separately and is expected to fall below 1 more often, since it uses the estimated (not exact)L_head.

    Returns a dict with L_head_exact, L_head_estimated, L_extractor_estimated, L_full_estimated,
    product, looseness_ratio, product_estimated, and looseness_ratio_estimated."""

    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}, expected one of {METHODS}")
    if normalize_features and x_train_for_norm is None:
        raise ValueError("normalize_features=True requires x_train_for_norm to fit the standardizer")

    wrapped_model = FlattenedInputWrapper(model)

    mean, std = (None, None)
    extractor_fn = extractor_output_fn
    head_fn = head_output_fn
    if normalize_features:
        mean, std = fit_feature_normalizer(model, x_train_for_norm)
        extractor_fn = _make_normalized_extractor_output_fn(mean, std)
        head_fn = _make_normalized_head_output_fn(mean, std)

    with torch.no_grad():
        x_head_query = extractor_fn(model, x_query, y_query)

    L_head_exact = _effective_head_lipschitz_exact(model.head, std)

    if method == "pairwise":
        L_head_estimated, _, _ = pairwise_lipschitz(
            model, x_head_query, y_query, head_fn, max_pairs=max_pairs, seed=seed)
        L_extractor_estimated, _, _ = pairwise_lipschitz(
            model, x_query, y_query, extractor_fn, max_pairs=max_pairs, seed=seed)
        L_full_estimated, _, _ = pairwise_lipschitz(
            wrapped_model, x_query, y_query, full_logits_output_fn, max_pairs=max_pairs, seed=seed)
    elif method == "grid":
        L_head_estimated = local_perturbation_lipschitz(
            model, x_head_query, y_query, head_fn, radius=radius, n_directions=n_directions, seed=seed).max().item()
        L_extractor_estimated = local_perturbation_lipschitz(
            model, x_query, y_query, extractor_fn, radius=radius, n_directions=n_directions, seed=seed).max().item()
        L_full_estimated = local_perturbation_lipschitz(
            wrapped_model, x_query, y_query, full_logits_output_fn,
            radius=radius, n_directions=n_directions, seed=seed).max().item()
    else:  # "gradient"
        L_head_estimated = gradient_norm_estimate(model, x_head_query, y_query, head_fn).max().item()
        L_extractor_estimated = gradient_norm_estimate(model, x_query, y_query, extractor_fn).max().item()
        L_full_estimated = gradient_norm_estimate(
            wrapped_model, x_query, y_query, full_logits_output_fn).max().item()

    product = L_extractor_estimated * L_head_exact
    looseness_ratio = product / L_full_estimated if L_full_estimated > 1e-12 else float("inf")

    product_estimated = L_extractor_estimated * L_head_estimated
    looseness_ratio_estimated = product_estimated / L_full_estimated if L_full_estimated > 1e-12 else float("inf")

    if looseness_ratio < 1.0 - 1e-6:
        print(f"WARNING: looseness_ratio={looseness_ratio:.4f} < 1 for method={method!r} -- this "
              f"violates the theoretical submultiplicative bound (Szegedy et al. 2014) and most likely "
              f"indicates an estimator sampling issue (too few pairs/directions/query points), not a "
              f"real result. Inspect before trusting.")
    elif verbose:
        print(f"  [{method}] L_head_exact={L_head_exact:.4f}  L_head_est={L_head_estimated:.4f}  "
              f"L_extractor_est={L_extractor_estimated:.4f}  L_full_est={L_full_estimated:.4f}  "
              f"looseness_ratio={looseness_ratio:.4f}  looseness_ratio_estimated={looseness_ratio_estimated:.4f}")

    return {
        "L_head_exact": L_head_exact,
        "L_head_estimated": L_head_estimated,
        "L_extractor_estimated": L_extractor_estimated,
        "L_full_estimated": L_full_estimated,
        "product": product,
        "looseness_ratio": looseness_ratio,
        "product_estimated": product_estimated,
        "looseness_ratio_estimated": looseness_ratio_estimated,
    }


# ---------------------------------------------------------------------------
# CNN-width capacity sweep
# ---------------------------------------------------------------------------

def run_cnn_width_sweep(widths=(4, 8, 16, 32, 64), epochs=6, train_subset_size=5000,
                         n_query_points=40, n_train_norm_points=500,
                         method="pairwise", normalize_features=True, seed=0, verbose=True,
                         save_path=RESULTS_DIR / "layer_decomposition_width_sweep.csv"):
    """Trains a fresh SmallCNN at each width and runs layer_decomposition_experiment on each. 
    Returns a DataFrame with one row per width, also saved to save_path as CSV."""
    train = load_mnist(train=True)
    test = load_mnist(train=False)
    dev = get_dev_subset(train, n=train_subset_size, seed=seed)
    train_loader = make_loader(dev.x_image, dev.y, batch_size=128, shuffle=True, seed=seed)
    test_loader = make_loader(test.x_image, test.y, batch_size=1000, shuffle=False)

    generator = torch.Generator().manual_seed(seed)
    query_idx = torch.randperm(len(test), generator=generator)[:n_query_points]
    x_query = test.x_flat[query_idx]
    y_query = test.y[query_idx]
    norm_idx = torch.randperm(len(train), generator=generator)[:n_train_norm_points]
    x_train_for_norm = train.x_flat[norm_idx]

    rows = []
    for width in widths:
        if verbose:
            print(f"=== width={width} (conv_channels=({width}, {2 * width})) ===")
        torch.manual_seed(seed)
        model, train_acc, test_acc = train_classifier(
            SmallCNN(conv_channels=(width, 2 * width)), train_loader, test_loader,
            epochs=epochs, lr=1e-3, verbose=False)
        if verbose:
            print(f"  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

        result = layer_decomposition_experiment(
            model, x_query, y_query, x_train_for_norm=x_train_for_norm,
            method=method, normalize_features=normalize_features, seed=seed, verbose=verbose)

        rows.append({"width": width, "train_acc": train_acc, "test_acc": test_acc, **result})

    df = pd.DataFrame(rows)

    if save_path is not None:
        RESULTS_DIR.mkdir(exist_ok=True)
        df.to_csv(save_path, index=False)
        if verbose:
            print(f"\nSaved width-sweep results to {save_path}")

    violations = df[df["looseness_ratio"] < 1.0 - 1e-6]
    if len(violations) > 0:
        print(f"WARNING: {len(violations)}/{len(df)} width-sweep entries have looseness_ratio < 1 "
              f"(widths: {violations['width'].tolist()}) -- see layer_decomposition_experiment's warning "
              f"for each, printed above.")

    return df
