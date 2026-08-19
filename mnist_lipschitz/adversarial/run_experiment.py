"""Adversarial-perturbation vs. Lipschitz-bound comparison, for the CNN.

`layer_decomposition.py` (sibling module, one level up) computes two theoretical upper bounds on
how much a trained CNN's full 10-d logit vector can move per unit of input change:

- `L_full_estimated` -- the TIGHT bound: the network's own empirical Lipschitz constant, measured
  directly on `f = head o extractor`.
- `product_bound` (`L_extractor_estimated * L_head_exact`) -- the LOOSE bound: the
  submultiplicative per-layer bound (Szegedy et al. 2014), which `layer_decomposition_experiment`
  already finds to be loose by a factor of roughly 2.4x-155x depending on estimator/width.

This module asks a practical question the looseness-ratio number alone can't answer: does that
looseness actually matter? It generates real FGSM/PGD adversarial examples against the SAME
checkpoint and checks whether their achieved sensitivity ratio (`achieved_ratio` below) sits close
to the tight bound, close to the loose bound, or well below both.

**This module reuses `layer_decomposition.layer_decomposition_experiment` directly to get
L_full_estimated/product_bound for whatever checkpoint it evaluates, rather than recomputing them
independently.** Recomputing them here with different query points/estimator settings would risk
the two numbers silently drifting out of sync with layer_decomposition.py's own results, making any
"how close did the attack get" comparison meaningless.

**IMPORTANT -- there are TWO different "L_full"-like quantities in this project.**
`mnist_lipschitz/run_experiment.py` (one directory up) measures Lipschitz estimates on the
scalar MARGIN function (`models.margin_fn`, `logit[y_true] - max(logit[j], j != y_true)`) -- that
is the project's main robustness measure everywhere else. `layer_decomposition.py`, and this
module, instead measure on the FULL logit vector (see layer_decomposition.py's own docstring for
why: the submultiplicative bound only applies to the function `L_extractor * L_head` actually
bounds). `achieved_ratio` below is deliberately computed on the full logit vector to match
`L_full_estimated`'s convention -- using the margin here would compare two different functions'
sensitivities and make ratio_to_L_full/ratio_to_product_bound meaningless.

**`max_R_adv` is a LOWER BOUND on the network's true worst-case sensitivity, not an upper bound or
exact value.** FGSM/PGD maximize cross-entropy loss (see attacks.py), not `achieved_ratio`
directly -- a ratio-maximizing search could in principle find a larger value. Report results as
"achieved under this attack," never as "the" worst case.
"""

from pathlib import Path

import pandas as pd
import torch

from mnist_lipschitz.data import load_mnist, get_dev_subset, make_loader
from mnist_lipschitz.models import SmallCNN, FlattenedInputWrapper, train_classifier
from mnist_lipschitz.estimators import linear_layer_lipschitz
from mnist_lipschitz.layer_decomposition import layer_decomposition_experiment
from mnist_lipschitz.adversarial.attacks import fgsm_attack, pgd_attack

RESULTS_DIR = Path(__file__).resolve().parent / "results"

torch.set_default_dtype(torch.float64)

DEFAULT_EPSILONS = (0.05, 0.1, 0.15, 0.2, 0.25)  # Goodfellow et al. 2015's MNIST range

# Matches layer_decomposition.run_cnn_width_sweep's own default widths exactly, so results from
# the two sweeps are directly comparable width-for-width.
DEFAULT_WIDTHS = (4, 8, 16, 32, 64)


def filter_correctly_classified(model, x, y):
    """Restrict (x, y) to points `model` already classifies correctly.

    Attacking an already-misclassified point isn't meaningful for this experiment -- the question
    being asked ("how easily can a small input change flip a *correct* prediction, and how does
    that compare to the Lipschitz bounds") presupposes the model got the clean point right in the
    first place. Including misclassified points would silently mix in a different quantity ("how
    far can an already-wrong prediction's logits move") into the same R_adv statistics, and this
    filtering must happen before sampling/attacking, not after, so a fixed downstream sample size
    (`n_points` in `run_epsilon_sweep`) is drawn entirely from the intended population.

    `model` must accept the same input shape as `x` (flat (N, 784) -- wrap a raw SmallCNN in
    `models.FlattenedInputWrapper` first, as every call site in this module does).

    Returns (x_correct, y_correct, kept_fraction) -- kept_fraction (the clean accuracy on this
    pool) is informational, not required by callers, but useful to sanity-check the pool wasn't
    pathologically small after filtering.
    """
    with torch.no_grad():
        preds = model(x).argmax(dim=1)
    correct = preds == y
    return x[correct], y[correct], correct.float().mean().item()


def achieved_ratio(model, x, x_adv):
    """R_adv = ||f(x) - f(x_adv)||_2 / ||x - x_adv||_2, per example.

    `f(x)` here is `model(x)`, the FULL (N, num_classes) logit vector (pre-softmax) -- NOT the
    scalar margin (`models.margin_fn`) used elsewhere in this project for the main robustness
    measure, and NOT softmax probabilities. This must match the output representation
    `layer_decomposition.py`'s L_full_estimated was computed on (see this module's top docstring),
    or the ratio_to_L_full/ratio_to_product_bound comparison in `summarize_epsilon_sweep` is
    meaningless -- comparing a ratio computed on one function against a Lipschitz bound computed
    on a different function is not a valid comparison, even though both are scalars in a similar
    numeric range.

    `model` must accept the same input shape as `x`/`x_adv` (flat (N, 784) --
    FlattenedInputWrapper-wrapped, as elsewhere in this module).

    Points where `x` and `x_adv` are numerically identical (e.g. epsilon=0, or an attack step that
    made no progress) get ratio 0, not a division-by-zero NaN/inf.

    Returns the (N,) tensor of per-example ratios.
    """
    with torch.no_grad():
        f_x = model(x)
        f_adv = model(x_adv)
    numerator = (f_x - f_adv).norm(p=2, dim=-1)
    denominator = (x - x_adv).norm(p=2, dim=-1)
    return torch.where(denominator > 1e-12, numerator / denominator.clamp_min(1e-12),
                        torch.zeros_like(denominator))


def run_epsilon_sweep(model, x_pool, y_pool, epsilons=DEFAULT_EPSILONS,
                       pgd_alpha_frac=0.25, pgd_num_steps=20, pgd_num_restarts=5,
                       n_points=500, seed=0, verbose=True):
    """For each (epsilon, method) pair in epsilons x {"FGSM", "PGD"}, generates adversarial
    examples for up to `n_points` correctly-classified points sampled from `x_pool`/`y_pool`, and
    computes `achieved_ratio` and the post-attack misclassification rate for each.

    `model`: FlattenedInputWrapper-wrapped trained SmallCNN (flat (N, 784) input, raw logits out)
    -- matches `achieved_ratio`'s/`filter_correctly_classified`'s expected convention.

    Filtering (Requirement 2) happens ONCE up front on the full x_pool/y_pool, before any sampling
    or attacking -- which points are fair game to attack depends only on the model's clean
    predictions, not on which attack is later run against them. The same n_points-point
    correctly-classified sample is then reused across every epsilon and method, so differences
    across rows of the resulting table reflect the attack, not which points happened to be sampled.

    PGD's step size is set to `pgd_alpha_frac * epsilon` per epsilon (default epsilon/4), not one
    fixed alpha across the whole sweep -- a step size tuned for the smallest epsilon would take
    many more steps than necessary to usefully traverse a much larger ball at the largest epsilon.

    Returns a dict: {"x_eval", "y_eval" (the sampled correctly-classified points actually
    attacked), "kept_frac" (Requirement 2's filtering diagnostic), "per_case": {(epsilon, method):
    {"x_adv", "R_adv" (the (n_points,) achieved_ratio tensor), "pct_misclassified"}}}.
    """
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
            R_adv = achieved_ratio(model, x_eval, x_adv)
            with torch.no_grad():
                preds_adv = model(x_adv).argmax(dim=1)
            pct_misclassified = (preds_adv != y_eval).float().mean().item()
            per_case[(epsilon, method)] = {
                "x_adv": x_adv, "R_adv": R_adv, "pct_misclassified": pct_misclassified,
            }
            if verbose:
                print(f"  epsilon={epsilon:g}  {method:4s}  mean_R_adv={R_adv.mean().item():.4f}  "
                      f"max_R_adv={R_adv.max().item():.4f}  "
                      f"pct_misclassified={pct_misclassified:.4f}")

    return {"x_eval": x_eval, "y_eval": y_eval, "kept_frac": kept_frac, "per_case": per_case}


def summarize_epsilon_sweep(sweep_results, L_full_estimated, product_bound, verbose=True):
    """Builds the per-(epsilon, method) summary table: columns epsilon, method, mean_R_adv,
    median_R_adv, max_R_adv, pct_misclassified, L_full_estimated, product_bound, ratio_to_L_full
    (= max_R_adv / L_full_estimated), ratio_to_product_bound (= max_R_adv / product_bound).

    `L_full_estimated`/`product_bound` are constants for the whole table (the same checkpoint's
    bounds, from `layer_decomposition_experiment`) -- repeated on every row rather than stored
    once elsewhere, so the table is self-contained and directly comparable row to row.

    **Sanity check (Requirement 5), not just described but enforced here:** `max_R_adv` is only
    ever a LOWER BOUND on the network's true worst-case sensitivity (see this module's top
    docstring), so it should never exceed `L_full_estimated`, which by definition upper-bounds
    every achieved ratio. A violation almost certainly means `L_full_estimated` itself
    under-sampled (too few pairs/directions/query points in `layer_decomposition_experiment`) --
    not a real bound violation -- so it's flagged with a printed warning, never a crash, so it can
    be investigated rather than silently reported as a result.
    """
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
              f"epsilon={epsilon:g}, method={method!r} -- L_full is supposed to upper-bound any "
              f"achieved ratio by definition; this almost certainly indicates L_full_estimated "
              f"under-sampled (too few pairs/directions/query points in "
              f"layer_decomposition_experiment), not a genuine bound violation. Inspect before "
              f"trusting the comparison at this row.")

    if verbose:
        print(df.to_string(index=False))
    return df


def most_and_least_sensitive_examples(model, sweep_results):
    """Across EVERY (epsilon, method) case in a `run_epsilon_sweep` result, finds the single
    attacked example that achieved the LARGEST R_adv and the single one that achieved the
    SMALLEST R_adv -- not per epsilon/method, but the overall extremes for this checkpoint, so
    "the attack that required the biggest/smallest R_adv" has one unambiguous answer per model.

    `model`: FlattenedInputWrapper-wrapped (matching `sweep_results`'s own convention) -- used
    only to record the clean/adversarial predicted class for display; `R_adv` itself is already
    computed and carried over from `sweep_results`, not recomputed here.

    "Smallest R_adv" is still an ATTACKED point (drawn from the same correctly-classified pool
    every other point in the sweep came from, see `filter_correctly_classified`) -- it means "of
    the points this attack was run against, this one's logits moved the least per unit of pixel
    change," not "an arbitrary unperturbed point."

    Returns (most_sensitive, least_sensitive), each a dict:
        {"epsilon": float, "method": str, "index": int, "R_adv": float,
         "x": (784,) Tensor, "x_adv": (784,) Tensor, "pixel_distance": float,
         "y_true": int, "pred_clean": int, "pred_adv": int}

    `pixel_distance` is `||x - x_adv||_2` -- the raw, INITIAL Euclidean distance between the two
    images, in the original 784-d pixel space, before either goes through the extractor. This is
    exactly `R_adv`'s denominator (`achieved_ratio`'s `||x - x_adv||_2`), so
    `R_adv == (logit-space distance) / pixel_distance` holds for this example -- see
    `head_layer_bound_check`'s `actual_logit_distance`, which is that same numerator.
    """
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
            "x": x, "x_adv": x_adv, "pixel_distance": (x - x_adv).norm(p=2).item(),
            "y_true": y_eval[idx].item(),
            "pred_clean": pred_clean, "pred_adv": pred_adv,
        }

    return _package(best), _package(worst)


def head_layer_bound_check(model, example):
    """For one attacked example (an entry from `most_and_least_sensitive_examples`), computes the
    Euclidean distance between the extracted features -- `model.extractor`'s output, right BEFORE
    the final linear head -- for the clean vs. adversarial input, and compares it against the
    head layer's own exact Lipschitz bound: for a linear map `head(a) = W@a + b`,
    `||head(a) - head(b)||_2 <= L_head_exact * ||a - b||_2` (`L_head_exact` = the spectral norm of
    `W`, `estimators.linear_layer_lipschitz`), with equality attainable along `W`'s top
    right-singular direction -- see `layer_decomposition.py`'s docstring for the same identity.

    `model`: the raw, trained `SmallCNN` (`.extractor`/`.head` directly accessible -- NOT
    FlattenedInputWrapper-wrapped). Uses the RAW (unnormalized) extractor features and the raw
    head weights throughout -- not the standardized features `layer_decomposition_experiment`'s
    `normalize_features=True` path uses internally for estimation purposes -- since
    `model.head(model.extractor(x)) == model(x)` exactly only for the raw, unstandardized
    composition (see `layer_decomposition.py`'s `_make_normalized_head_output_fn` docstring); this
    function checks the bound for the network's ACTUAL forward computation, not an internal
    estimation convenience.
    `example`: one of `most_and_least_sensitive_examples`'s returned dicts (needs flat `(784,)`
    `x`/`x_adv` pixel tensors).

    Returns:
        {"feature_distance": ||extractor(x) - extractor(x_adv)||_2,
         "L_head_exact": the head's exact spectral-norm Lipschitz constant (raw weights),
         "head_bound": L_head_exact * feature_distance -- the bound's prediction for how far the
             logits can move given this much feature-space movement,
         "actual_logit_distance": ||head(extractor(x)) - head(extractor(x_adv))||_2, the network's
             REAL logit-space distance for this pair (equals ||f(x)-f(x_adv)||_2,
             achieved_ratio's numerator for this example, since f = head o extractor exactly),
         "head_bound_tightness": actual_logit_distance / head_bound -- always <= 1 (up to floating
             point) for a linear layer, by Cauchy-Schwarz; how close to 1 shows whether this
             specific pair of features happened to differ mostly along the head's most-sensitive
             (top-singular-value) direction, or a much less sensitive one.}
    """
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


def run_bound_comparison(model, x_query, y_query, x_pool, y_pool, x_train_for_norm,
                          epsilons=DEFAULT_EPSILONS, bound_method="pairwise",
                          pgd_alpha_frac=0.25, pgd_num_steps=20, pgd_num_restarts=5,
                          n_points=500, normalize_features=True, seed=0, verbose=True):
    """Full single-checkpoint pipeline: computes L_full_estimated/product_bound for `model` via
    `layer_decomposition.layer_decomposition_experiment` (reused, not recomputed independently --
    see this module's top docstring), then runs the FGSM/PGD epsilon sweep against the SAME
    checkpoint and compares achieved sensitivity to those two bounds.

    `model`: a trained, raw `SmallCNN` (`.extractor`/`.head` directly accessible, NOT
    FlattenedInputWrapper-wrapped) -- wrapped internally here, matching
    `layer_decomposition_experiment`'s own convention, since both the bound computation and the
    attacks need the flat-input wrapped version.

    `x_query`/`y_query`: held-out points for the Lipschitz bound estimate (passed straight through
    to `layer_decomposition_experiment`). `x_pool`/`y_pool`: a separate, larger pool the attack's
    evaluation points are sampled from (after correctness-filtering) -- kept disjoint from
    x_query/y_query so the bound estimate and the attack results are independent measurements of
    the same model, not double use of the same points. `x_train_for_norm`: passed straight through
    to `layer_decomposition_experiment` (feature-standardizer fitting).

    Returns (summary_df, sweep_results): `summary_df` is `summarize_epsilon_sweep`'s
    per-(epsilon, method) table; `sweep_results` is `run_epsilon_sweep`'s raw per-point data (for
    plotting distributions, see plots.py).
    """
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


def run_cnn_adversarial_width_sweep(widths=DEFAULT_WIDTHS, epochs=6, train_subset_size=5000,
                                     n_query_points=40, n_train_norm_points=500,
                                     n_pool_points=2000, n_attack_points=500,
                                     epsilons=DEFAULT_EPSILONS, bound_method="pairwise",
                                     pgd_alpha_frac=0.25, pgd_num_steps=20, pgd_num_restarts=5,
                                     normalize_features=True, seed=0, verbose=True,
                                     save_path=RESULTS_DIR / "adversarial_width_sweep.csv"):
    """Repeats `run_bound_comparison` across the same CNN widths as
    `layer_decomposition.run_cnn_width_sweep` (`DEFAULT_WIDTHS` matches its default exactly), to
    see whether the gap between achieved adversarial sensitivity and the theoretical bounds
    narrows or widens with model capacity -- the practical-consequences extension of that sweep's
    looseness-vs-width finding.

    No checkpoints from `layer_decomposition.run_cnn_width_sweep` are saved to disk (it only
    persists its results CSV, not the trained models) -- this trains a FRESH `SmallCNN` at each
    width instead, using the identical training configuration (`conv_channels=(width, 2*width)`,
    same `epochs`/`train_subset_size`/`seed` defaults) so results are still comparable width-for-
    width, just not from literally the same weights.

    The same held-out query points (for the Lipschitz bound), training-sample points (for feature
    normalization), and attack pool are reused across every width (fixed by `seed`) -- exactly
    `layer_decomposition.run_cnn_width_sweep`'s own reasoning for doing this -- so differences
    between rows reflect the model, not which points happened to be sampled this time. The attack
    pool is drawn from the test points NOT used as Lipschitz-bound query points, so the two
    measurements stay independent (see `run_bound_comparison`'s docstring).

    Saves one per-(epsilon, method) summary CSV per width (`adversarial_epsilon_sweep_width{w}.csv`)
    plus a single combined CSV (`save_path`) with one row per width. The combined CSV extends
    `layer_decomposition.run_cnn_width_sweep`'s own column convention (`width`, `train_acc`,
    `test_acc`, ...) by APPENDING new columns rather than introducing a parallel format:
    `L_full_estimated`, `product_bound` (shared across methods for that width), and, for each of
    FGSM/PGD, `max_R_adv_{method}`/`ratio_to_L_full_{method}`/`ratio_to_product_bound_{method}`
    evaluated at the LARGEST swept epsilon (the strongest, most informative attack condition for
    the width-vs-bound-closeness comparison -- see `plots.py::plot_bound_closeness_vs_width`).

    Also records, per width, the single attacked example achieving the LARGEST and the SMALLEST
    R_adv across every (epsilon, method) case (`most_and_least_sensitive_examples`), each then
    extended in place with `head_layer_bound_check`'s feature-space-distance-vs-head-Lipschitz-
    bound comparison for that specific example -- both computed here, while that width's trained
    model is still in scope, rather than reconstructed later from `per_width_sweep_results` alone
    (which carries the raw (x, x_adv) tensors but not predictions or the trained model itself,
    since it isn't retained after this function returns).

    Returns (combined_df, per_width_summary_dfs, per_width_extremes): combined_df is the
    one-row-per-width DataFrame described above; per_width_summary_dfs is a dict
    {width: summary_df} with the full per-(epsilon, method) table for each width (same as
    `run_bound_comparison`'s summary_df); per_width_extremes is a dict
    {width: (most_sensitive, least_sensitive)}, each element `most_and_least_sensitive_examples`'s
    return value for that width's checkpoint, with `head_layer_bound_check`'s keys
    (`feature_distance`, `L_head_exact`, `head_bound`, `actual_logit_distance`,
    `head_bound_tightness`) merged in.
    """
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


def main(seed=0, verbose=True):
    """Single-checkpoint baseline run: the project's default `SmallCNN` (`conv_channels=(16,32)`,
    matching every other CNN trained elsewhere in this project), trained on full MNIST, then the
    full epsilon sweep and bound comparison against it. Mirrors
    `layer_decomposition.py`'s notebook default -- NOT the width sweep
    (`run_cnn_adversarial_width_sweep`), which is opt-in and called directly, matching this
    project's convention that markedly-slower sweep functions aren't wired into `main()`.
    """
    torch.manual_seed(seed)
    train = load_mnist(train=True)
    test = load_mnist(train=False)
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
