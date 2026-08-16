"""Layer-decomposed Lipschitz sub-experiment for the CNN.

Splits `f = head o extractor` (SmallCNN's conv/relu/pool stack, then its
final `nn.Linear`) and compares:

- `L_head` -- the Lipschitz constant of the final linear layer, both
  **exact** (closed-form spectral norm, `estimators.linear_layer_lipschitz`)
  and **estimated** (via the same pairwise/local-perturbation/gradient-norm
  machinery used everywhere else in this project, as a calibration check
  on the estimators themselves).
- `L_extractor` -- the empirical Lipschitz constant of the feature
  extractor, `||extractor(x1)-extractor(x2)|| / ||x1-x2||` in raw pixel
  space.
- `L_full` -- the empirical Lipschitz constant of the whole network.

**Important: `L_full` here is measured on the full 10-d logit vector, not
the scalar margin `run_experiment.py` uses elsewhere.** `f = head o
extractor` (this module's Context) is logit-valued, and the
submultiplicative per-layer bound this module checks --
`L_extractor * L_head >= L_full` (Szegedy et al., 2014, "Intriguing
Properties of Neural Networks") -- only rigorously applies to the same
function `L_extractor * L_head` actually bounds. The margin function
(`models.margin_fn`) is itself Lipschitz w.r.t. the logit vector (with a
small constant, at most sqrt(2) -- it's a min of sqrt(2)-Lipschitz
coordinate differences), but is a *different*, generally smaller quantity
than the raw logit vector's own Lipschitz constant, so reusing it here
would not actually test the cited bound. The estimator *machinery*
(pairwise/local-perturbation/gradient-norm) is reused as-is; only the
target function changes, from margin to full logits.
"""

from pathlib import Path

import pandas as pd
import torch

from mnist_lipschitz.data import load_mnist, get_dev_subset, make_loader
from mnist_lipschitz.estimators import (
    euclidean_distance_fn,
    pairwise_lipschitz,
    local_perturbation_lipschitz,
    gradient_norm_estimate,
    linear_layer_lipschitz,
)
from mnist_lipschitz.models import SmallCNN, FlattenedInputWrapper, train_classifier

RESULTS_DIR = Path(__file__).resolve().parent / "results"

METHODS = ("pairwise", "grid", "gradient")
# "grid" here is this task's requested name for the finite-difference
# method -- this project's own name for it is local_perturbation_lipschitz
# (toy_lipschitz's "_grid" suffix means something different there:
# evaluated over many query points at once, not a distinct estimation
# method). Mapped to the existing function rather than introducing a
# second, redundant "grid" method that doesn't otherwise exist here.


def extractor_output_fn(model, x, y):
    """output_fn wrapper for `model.extractor`: flat (N, 784) pixel input
    (matching every other estimator call site's convention in this
    project), reshaped to (N,1,28,28) before the conv stack -- same
    reshape `FlattenedInputWrapper` does for the full model. Returns
    flattened features, (N, feat_dim). `y` is unused (the extractor
    doesn't depend on the label) but still accepted, so this matches the
    (model, x, y) -> Tensor convention every estimator in estimators.py
    expects.
    """
    return model.extractor(x.reshape(x.shape[0], 1, 28, 28))


def head_output_fn(model, x, y):
    """output_fn wrapper for `model.head`: `x` here is already-extracted
    RAW (un-normalized) features, (N, feat_dim) -- distinct from
    `extractor_output_fn`/`margin_fn`, which both take pixel input. Returns
    the full (N, num_classes) logit vector (not a margin). `y` unused.
    """
    return model.head(x)


def full_logits_output_fn(model, x, y):
    """output_fn wrapper for the full network `f = head o extractor`:
    flat (N, 784) pixel input, full (N, num_classes) logit output. `model`
    must already accept flat input (wrap with `FlattenedInputWrapper`
    first, as `layer_decomposition_experiment` does internally). This is
    the function `L_extractor * L_head` is supposed to bound -- see this
    module's docstring for why it's the full logit vector, not the margin.
    """
    return model(x)


def fit_feature_normalizer(model, x_train, relative_floor=1e-2, absolute_floor=1e-8):
    """Fits a per-dimension (mean, std) on `model.extractor(x_train)`,
    once -- meant to be called a single time per trained model and reused
    across every subsequent evaluation of that model (refitting per query
    batch would make L_head/L_extractor depend on which batch happened to
    be used to normalize, not just on the model itself, contaminating any
    comparison across models/widths).

    `std` is floored at `max(relative_floor * median(std), absolute_floor)`,
    not just `absolute_floor` alone -- checked directly on a real trained
    CNN, not assumed: dead ReLU/MaxPool units (a channel that never
    activates on the normalization sample) are common (92/1568 features on
    one trained SmallCNN checkpoint) and land std exactly at whatever floor
    is used. An `absolute_floor` alone (e.g. 1e-8) barely constrains
    anything -- dividing by 1e-8 amplifies any perturbation-induced change
    in that dimension by up to 1e8x, which single-handedly dominated
    `L_extractor`'s finite-difference (`"grid"`) estimate in practice
    (inflating it into the tens of millions, an artifact of the
    normalization floor, not a real property of the network). Tying the
    floor to the *other* features' typical scale (their median std) instead
    keeps a dead dimension from silently exploding the estimate while still
    letting genuinely small-but-real variation through.

    Returns (mean, std), each shape (feat_dim,).
    """
    with torch.no_grad():
        features = extractor_output_fn(model, x_train, None)
    mean = features.mean(dim=0)
    raw_std = features.std(dim=0)
    floor = max(relative_floor * raw_std.median().item(), absolute_floor)
    std = raw_std.clamp_min(floor)
    return mean, std


def _make_normalized_extractor_output_fn(mean, std):
    def _fn(model, x, y):
        return (extractor_output_fn(model, x, y) - mean) / std
    return _fn


def _make_normalized_head_output_fn(mean, std):
    """output_fn(model, z, y) = model.head(z*std + mean): treats `z` as
    STANDARDIZED features and un-standardizes before applying the actual
    (trained-on-raw-features) head weights, so this function's INPUT
    domain matches the normalized extractor's OUTPUT domain exactly --
    composing them, `normalized_head(normalized_extractor(x))`, gives back
    `model.head(model.extractor(x)) = model(x)` exactly (the
    standardization cancels), so the decomposition still bounds the same
    original function regardless of `normalize_features`.
    """
    def _fn(model, z, y):
        return head_output_fn(model, z * std + mean, y)
    return _fn


def _effective_head_lipschitz_exact(head, std=None):
    """Exact closed-form Lipschitz constant of `head`, w.r.t. whichever
    input scale `L_head_estimated` is actually measured on -- **always
    comparable to L_head_estimated, unlike unconditionally calling
    `linear_layer_lipschitz(head)` would be**.

    If `std` is None (normalize_features=False): exactly
    `linear_layer_lipschitz(head)` on the raw weights.

    If `std` is given (normalize_features=True): accounts for the
    standardization folded into the head's input (see
    `_make_normalized_head_output_fn`). `standardized_head(z) =
    head(z*std + mean) = (W @ diag(std)) @ z + (W@mean + b)` -- still
    linear in `z`, with effective weight `W @ diag(std)` (scaling column j
    of W by std[j]) -- so the exact spectral norm is computed on *that*
    effective weight, not the raw one. Without this adjustment,
    `L_head_exact` and `L_head_estimated` would silently measure the head
    at two different input scales whenever normalize_features=True,
    making them incomparable (and the looseness_ratio check meaningless)
    -- exactly the misuse this function exists to prevent.
    """
    if std is None:
        return linear_layer_lipschitz(head)
    W_effective = head.weight.detach() * std.unsqueeze(0)
    return torch.linalg.matrix_norm(W_effective, ord=2).item()


def layer_decomposition_experiment(model, x_query, y_query, x_train_for_norm=None,
                                    method="pairwise", normalize_features=True,
                                    radius=1.0, n_directions=40, max_pairs=None, seed=0, verbose=True):
    """Computes L_head (exact + estimated), L_extractor (estimated), and
    L_full (estimated) for a trained CNN, and the resulting looseness
    ratio of the per-layer Lipschitz bound.

    `model`: a trained `SmallCNN` (raw, i.e. with `.extractor`/`.head`
    directly accessible -- NOT pre-wrapped in `FlattenedInputWrapper`;
    wrapped internally here only for the L_full computation, where it's
    needed).
    `x_query`, `y_query`: flat (N, 784) pixel points / (N,) labels to
    evaluate the Lipschitz estimates on.
    `x_train_for_norm`: flat (M, 784) pixel points to fit the feature
    standardizer on (required if `normalize_features=True`; ignored
    otherwise). Should be a training-set sample, not `x_query` itself --
    fitting normalization on the same points being evaluated would leak
    query-set statistics into the "calibration."
    `method`: "pairwise", "grid" (finite-difference,
    `local_perturbation_lipschitz`), or "gradient" (autograd; the "gradient"
    branch uses the exact per-example Jacobian spectral norm for
    L_head/L_extractor, since both are vector-valued -- see
    `estimators.gradient_norm_estimate`'s docstring).
    `normalize_features`: standardize extractor features (per-dimension
    mean/std, fit once via `x_train_for_norm`) before computing L_extractor
    and L_head, so no single badly-scaled feature dimension dominates the
    estimate. `L_head_exact` is adjusted to match (see
    `_effective_head_lipschitz_exact`) rather than silently comparing two
    different input scales.

    Returns:
        {
            "L_head_exact": float,
            "L_head_estimated": float,
            "L_extractor_estimated": float,
            "L_full_estimated": float,
            "product": float,                      # L_extractor_estimated * L_head_exact
            "looseness_ratio": float,               # product / L_full_estimated
            "product_estimated": float,             # L_extractor_estimated * L_head_estimated
            "looseness_ratio_estimated": float,     # product_estimated / L_full_estimated
        }

    If `looseness_ratio < 1` (violating the theoretical submultiplicative
    bound -- Szegedy et al. 2014), prints a warning rather than raising:
    this is far more likely to indicate an estimator under-sampling issue
    (too few pairs/directions/query points) than a genuine violation of
    the bound, and should surface for inspection, not silently fail or
    silently pass.

    `looseness_ratio_estimated` (using `L_head_estimated` in the product
    instead of the exact spectral norm) is reported alongside but is a
    *different* quantity, not a drop-in substitute: `L_head_estimated` is
    itself only a lower bound on the true `L_head` (same reasoning as
    `L_extractor_estimated`/`L_full_estimated`), so `looseness_ratio_estimated
    <= looseness_ratio` always, and it dropping below 1 is expected/normal
    rather than a bug to flag -- no warning is attached to it.
    """
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

    # All-estimated counterpart: product/looseness_ratio using L_head_ESTIMATED
    # instead of the exact spectral norm. This is NOT the quantity the
    # Szegedy et al. submultiplicative bound guarantees >= 1 -- L_head_estimated
    # is itself only a lower bound on the true L_head (same Cauchy-Schwarz
    # reasoning as L_extractor_estimated/L_full_estimated), so
    # product_estimated <= product and looseness_ratio_estimated <=
    # looseness_ratio always. Falling below 1 here is expected/normal, not
    # a sampling-issue flag the way looseness_ratio < 1 is -- reported
    # plainly, no warning attached.
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
#
# No existing capacity sweep exists in mnist_lipschitz to extend (unlike
# toy_lipschitz's MLP-width sweep, run_experiment.sweep_over_capacity) --
# this builds one specifically for the layer-decomposition comparison,
# self-contained in this module rather than woven into run_experiment.py's
# main driver, matching this module's own scope. Same CSV/dataframe
# convention (one row per width, all requirements 1-5's outputs as
# columns) so this can be saved/loaded like any other results/ artifact.
# ---------------------------------------------------------------------------

def run_cnn_width_sweep(widths=(4, 8, 16, 32, 64), epochs=6, train_subset_size=5000,
                         n_query_points=40, n_train_norm_points=500,
                         method="pairwise", normalize_features=True, seed=0, verbose=True,
                         save_path=RESULTS_DIR / "layer_decomposition_width_sweep.csv"):
    """Trains a fresh `SmallCNN` at each width in `widths`
    (`conv_channels=(width, 2*width)`, preserving SmallCNN's own default
    16/32 1:2 channel ratio -- width=16 reproduces the project's default
    CNN exactly), runs `layer_decomposition_experiment` at each, and
    returns a `pandas.DataFrame` with one row per width.

    Trains on a `train_subset_size`-point dev subset (not the full 60k
    training set) to keep the sweep fast across several widths -- this is
    a diagnostics check of how the looseness ratio moves with capacity,
    not an accuracy benchmark (see `models.SmallCNN`'s own docstring for
    the same reasoning applied to the project's single default CNN).
    `method="pairwise"` is the default specifically for sweep speed
    (the `"gradient"` method's exact per-example Jacobian is
    correspondingly slower per width -- see
    `estimators.gradient_norm_estimate`'s docstring -- fine for a single
    model, slower multiplied across a whole sweep).

    The same held-out query points and normalization-fitting points are
    reused across every width (fixed by `seed`), so differences between
    rows reflect the model, not which points happened to be sampled.

    Columns: width, train_acc, test_acc, L_head_exact, L_head_estimated,
    L_extractor_estimated, L_full_estimated, product, looseness_ratio.

    Saves the dataframe to `save_path` as CSV (set to None to skip saving).
    """
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
