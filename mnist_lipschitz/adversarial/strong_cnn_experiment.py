"""StrongCNN variant of the adversarial-vs-Lipschitz-bound comparison (`run_experiment.py`).

`StrongCNN` (`models.py`) has no `.extractor`/`.head` submodule split (unlike `SmallCNN`) --
`layer_decomposition.py`'s own docstring explicitly excludes it: "this model isn't part of" that
sub-experiment. But `StrongCNN.classifier[4]` (the FINAL layer of its classifier) IS a plain
`nn.Linear` with nothing nonlinear after it, so the same tight/loose Lipschitz-bound comparison
`layer_decomposition.py` performs for `SmallCNN` is still possible here via an EXTERNALLY
constructed extractor/head split (`model.features` + `model.classifier[:4]` as "extractor",
`model.classifier[4]` as "head"), without modifying `models.py` or `layer_decomposition.py` --
following the same "reuse, never edit `layer_decomposition.py`" precedent this package's own
`compute_bounds_with_distance_fn` already established for the Mahalanobis generalization (see its
docstring).

**Scope**: baseline only (one trained checkpoint, Euclidean + Mahalanobis, FGSM/PGD, WITH the
tight/loose bound comparison) -- no CNN-width sweep (`StrongCNN`'s conv channels are hardcoded,
not a constructor parameter, unlike `SmallCNN`'s) and no multi-seed pilot
(`seed_sweep.py`'s pattern). Both deliberately out of scope for this module.

**Eval-mode discipline (the one genuinely new correctness concern vs. `SmallCNN`)**: `SmallCNN`
has no `BatchNorm`/`Dropout`, so nothing in this codebase has ever needed to care about
train/eval mode -- every existing function silently assumes deterministic, batch-composition-
independent behavior, which is only true for `SmallCNN`. `StrongCNN` has `BatchNorm1d`/
`Dropout2d`/`Dropout`, so every function below that forwards data through the model REQUIRES
`model.eval()` to already be active, and RAISES `ValueError` rather than silently calling
`.eval()` itself if it isn't -- a caller who forgot to set eval mode has a bug worth surfacing
loudly (BatchNorm in train mode uses per-BATCH statistics, so a bound/attack computation would
silently depend on which other points happen to share a batch -- exactly the kind of numerical
footgun this whole sub-experiment exists to catch, not paper over). `attacks.py`
(`fgsm_attack`/`pgd_attack`) never touches train/eval mode itself, so `main()`'s single explicit
`.eval()` call (right after training) is what keeps every downstream computation correct.

**Weaker checkpoint-gating than the SmallCNN machinery, stated explicitly**: `compute_bounds_with_
distance_fn`'s central correctness checkpoint (`tests/test_mahalanobis.py`) proves it reduces to
EXACTLY `layer_decomposition_experiment`'s own independently-existing numbers. No such
independently-existing reference exists for `StrongCNN` (`layer_decomposition.py` was never
extended to it). `tests/test_strong_cnn_experiment.py`'s parity test is therefore only a
SELF-consistency check (does `compute_strong_cnn_bounds` reduce to a from-scratch manual
`pairwise_lipschitz` call using the same closures) -- necessary, but not as strong a checkpoint as
the SmallCNN precedent.
"""

import torch

from mnist_lipschitz.data import load_mnist, make_loader
from mnist_lipschitz.models import StrongCNN, STRONG_CNN_CONFIG, FlattenedInputWrapper, train_classifier
from mnist_lipschitz.augmentation import random_affine_augment
from mnist_lipschitz.estimators import pairwise_lipschitz, linear_layer_lipschitz, euclidean_distance_fn
from mnist_lipschitz.layer_decomposition import (
    full_logits_output_fn,
    # Underscore-prefixed ("module-private") in layer_decomposition.py, but imported here
    # deliberately -- same precedent adversarial/run_experiment.py's compute_bounds_with_
    # distance_fn already established: this helper only needs an nn.Linear + optional std, is
    # architecture-agnostic, and re-deriving it independently would risk silent drift.
    _effective_head_lipschitz_exact,
)
from mnist_lipschitz.adversarial.run_experiment import (
    RESULTS_DIR,
    DEFAULT_EPSILONS,
    run_epsilon_sweep,
    summarize_epsilon_sweep,
)

torch.set_default_dtype(torch.float64)


def _require_eval_mode(model, fn_name):
    if model.training:
        raise ValueError(
            f"{fn_name}: model is in train() mode. StrongCNN has BatchNorm1d/Dropout2d/Dropout, "
            f"so every quantity this function computes would depend on which other points share "
            f"a batch (BatchNorm) or would be non-deterministic (Dropout) unless model.eval() is "
            f"active. Call model.eval() before calling this function -- not done silently here, "
            f"since a caller who forgot is a bug worth surfacing, not hiding.")


def strong_cnn_extractor_fn(model, x, y):
    """output_fn wrapper for everything in `StrongCNN` up to (not including) the final linear
    layer: flat (N, 784) pixel input, reshaped to (N,1,28,28), through `model.features` (the
    conv/BatchNorm2d/ReLU/MaxPool2d/Dropout2d stack) then `model.classifier[:4]`
    (`Linear(3136,256) -> BatchNorm1d -> ReLU -> Dropout`, i.e. `model.classifier` minus its
    final `nn.Linear`). Returns (N, 256). `y` unused, accepted only to match every other
    `output_fn`'s `(model, x, y)` convention in this project.

    Requires `model.eval()` already active -- see this module's docstring.
    """
    _require_eval_mode(model, "strong_cnn_extractor_fn")
    features = model.features(x.reshape(x.shape[0], 1, 28, 28))
    return model.classifier[:4](features)


def strong_cnn_head_module(model):
    """The final `nn.Linear(256, num_classes)` of `model.classifier` -- the "head", analogous to
    `SmallCNN.head`. `model.classifier[4]` is the LAST layer with nothing nonlinear after it
    (`model.classifier` is `[Linear, BatchNorm1d, ReLU, Dropout, Linear]`), so
    `model(x) == model.classifier[4](strong_cnn_extractor_fn(model, x, y))` exactly -- checked
    directly in `tests/test_strong_cnn_experiment.py`, not just asserted.
    """
    return model.classifier[4]


def full_logits_fn(model, x, y):
    """output_fn wrapper for the full network: flat (N, 784) input, full (N, num_classes) logit
    output. `model` must already accept flat input (`FlattenedInputWrapper`-wrapped). Identical
    role to `layer_decomposition.full_logits_output_fn` -- re-exported under this module's own
    name for a self-contained import list, but delegates to the same architecture-agnostic
    function (it only ever calls `model(x)`, no StrongCNN-specific logic needed).
    """
    return full_logits_output_fn(model, x, y)


def fit_strong_cnn_feature_normalizer(model, x_train, relative_floor=1e-2, absolute_floor=1e-8):
    """Fits a per-dimension (mean, std) on `strong_cnn_extractor_fn(model, x_train, None)`, once
    -- same role and floor formula as `layer_decomposition.fit_feature_normalizer`
    (`max(relative_floor * median(raw_std), absolute_floor)`, regression-tested there against a
    real dead-ReLU-unit bug on `SmallCNN`), applied here to `StrongCNN`'s 256-d penultimate layer
    instead of `SmallCNN`'s 1568-d one. Re-implemented (not imported) because the original calls
    `extractor_output_fn`, which is `SmallCNN`-specific.

    Requires `model.eval()` already active (enforced inside `strong_cnn_extractor_fn`).

    Returns (mean, std), each shape (256,).
    """
    with torch.no_grad():
        features = strong_cnn_extractor_fn(model, x_train, None)
    mean = features.mean(dim=0)
    raw_std = features.std(dim=0)
    floor = max(relative_floor * raw_std.median().item(), absolute_floor)
    std = raw_std.clamp_min(floor)
    return mean, std


def _make_normalized_strong_cnn_extractor_fn(mean, std):
    def _fn(model, x, y):
        return (strong_cnn_extractor_fn(model, x, y) - mean) / std
    return _fn


def compute_strong_cnn_bounds(model, x_query, y_query, distance_fn, x_train_for_norm,
                               normalize_features=True, max_pairs=None, seed=0, verbose=True):
    """StrongCNN analogue of `run_experiment.compute_bounds_with_distance_fn`: computes
    `L_head_exact`/`L_extractor_estimated`/`L_full_estimated`/`product`/`looseness_ratio` for a
    trained `StrongCNN`, under an arbitrary pixel-space `distance_fn`.

    Requires `model.eval()` already active -- raises `ValueError` otherwise (see this module's
    docstring).

    `model`: a trained, raw `StrongCNN` (not `FlattenedInputWrapper`-wrapped -- wrapped
    internally here for the `L_full_estimated` computation, matching
    `compute_bounds_with_distance_fn`'s own convention).
    `x_train_for_norm`: training-set sample to fit the feature standardizer on, kept separate
    from `x_query` (same reasoning as `layer_decomposition_experiment`'s own docstring).

    Returns {"L_head_exact", "L_extractor_estimated", "L_full_estimated", "product",
    "looseness_ratio"} -- same keys `compute_bounds_with_distance_fn`/
    `layer_decomposition_experiment` use, so this can be handed to the same downstream code.
    """
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
    """StrongCNN analogue of `run_experiment.head_layer_bound_check`: for one attacked example,
    compares the Euclidean distance between the (raw, un-standardized) penultimate features of
    the clean vs. adversarial input against the final linear layer's own exact Lipschitz bound.

    `model`: the raw, trained `StrongCNN` (not `FlattenedInputWrapper`-wrapped). Requires
    `model.eval()` already active -- raises `ValueError` otherwise.
    `example`: one of `most_and_least_sensitive_examples`'s returned dicts (needs flat `(784,)`
    `x`/`x_adv` pixel tensors).

    Returns {"feature_distance", "L_head_exact", "head_bound", "actual_logit_distance",
    "head_bound_tightness"} -- identical keys to `head_layer_bound_check`'s return value.
    """
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
    """StrongCNN analogue of `run_experiment.run_bound_comparison_with_distance_fn`: computes
    `L_full_estimated`/`product_bound` via `compute_strong_cnn_bounds`, then runs the FGSM/PGD
    epsilon sweep (`run_epsilon_sweep`, imported UNCHANGED -- fully architecture-agnostic) with
    `R_adv` measured under the SAME `distance_fn`.

    `model`: raw, trained `StrongCNN`. Requires `model.eval()` already active -- raises
    `ValueError` otherwise.

    Returns (summary_df, sweep_results), same shape as `run_bound_comparison_with_distance_fn`'s.
    """
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
    """Single-checkpoint StrongCNN baseline: trains via `STRONG_CNN_CONFIG`'s exact recipe
    (mirrors `mnist_lipschitz.run_experiment.run_stronger_cnn_raw_mnist_experiment`'s wiring --
    full 60k MNIST, `augment_fn` built from `augmentation.random_affine_augment`,
    `lr_scheduler_fn` built from `torch.optim.lr_scheduler.CosineAnnealingLR`), then runs the full
    epsilon sweep and tight/loose bound comparison against it -- mirrors `run_experiment.main()`/
    `main_with_distance_fn()`'s two-call pattern (call once per metric).

    `distance_fn` defaults to plain Euclidean (`estimators.euclidean_distance_fn`) if left `None`.
    Pass a Mahalanobis `distance_fn` (via `run_experiment.build_pixel_mahalanobis_distance_fn`)
    for the Mahalanobis baseline.

    Calls `model.eval()` explicitly right after training -- defensively, not relying on
    `train_classifier`'s internal `evaluate_accuracy` call happening to leave the model in eval
    mode (see this module's docstring for why this matters for `StrongCNN` specifically).

    Saves the checkpoint (`strong_cnn_state_dict.pt`, shared across both metric calls with the
    same `seed` -- training doesn't depend on `distance_fn`) and a per-metric summary CSV under
    `results/`, both prefixed `strong_cnn_` (matching this package's own flat `results/`
    convention, e.g. `adversarial_epsilon_sweep_baseline.csv`, rather than the root
    `mnist_lipschitz` package's `results/stronger_cnn_raw_mnist/` subfolder convention).

    Returns (summary_df, sweep_results, model) -- `model` (the raw, trained, eval-mode
    `StrongCNN`) is returned in addition to `run_experiment.main()`'s own
    `(summary_df, sweep_results)` shape, so a second call with a different `distance_fn` can
    reuse the SAME trained weights instead of retraining (see notebook usage).
    """
    if distance_fn is None:
        distance_fn = euclidean_distance_fn

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = RESULTS_DIR / "strong_cnn_state_dict.pt"

    torch.manual_seed(seed)
    train = load_mnist(train=True)
    test = load_mnist(train=False)
    train_loader = make_loader(train.x_image, train.y, batch_size=STRONG_CNN_CONFIG["batch_size"],
                                shuffle=True, seed=seed)
    test_loader = make_loader(test.x_image, test.y, batch_size=1000, shuffle=False)

    model = StrongCNN(dropout_conv=STRONG_CNN_CONFIG["dropout_conv"],
                       dropout_fc=STRONG_CNN_CONFIG["dropout_fc"])

    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        train_acc, test_acc = state["train_acc"], state["test_acc"]
        if verbose:
            print(f"[checkpoint] loaded StrongCNN from {checkpoint_path} "
                  f"(train_acc={train_acc:.4f}  test_acc={test_acc:.4f})")
    else:
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
        torch.save({"model_state_dict": model.state_dict(), "train_acc": train_acc, "test_acc": test_acc},
                    checkpoint_path)
        if verbose:
            print(f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    model.eval()  # defensive -- see module docstring

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

    summary_df, sweep_results = strong_cnn_bound_comparison(
        model, x_query, y_query, x_pool, y_pool, x_train_for_norm, distance_fn,
        seed=seed, verbose=verbose)

    metric_name = getattr(distance_fn, "__name__", "mahalanobis")
    suffix = "euclidean" if metric_name == "euclidean_distance_fn" else "mahalanobis"
    summary_df.to_csv(RESULTS_DIR / f"strong_cnn_epsilon_sweep_baseline_{suffix}.csv", index=False)

    return summary_df, sweep_results, model
