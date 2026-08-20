"""Multi-seed confirmation sweep for the CNN-width adversarial comparison.

`run_experiment.run_cnn_adversarial_width_sweep`'s single-seed run found an apparent inversion:
width=64 has a larger `L_full_estimated` than width=32 and PGD achieves more absolute logit
movement against it, yet width=32 shows a HIGHER misclassification rate at matched epsilon. This
module tests whether that inversion survives reseeding, and captures the diagnostics needed to
identify WHICH mechanism produces it -- logit scale (`clean_logit_stats`), the margin functional's
own Lipschitz constant (`margin_lipschitz_estimate`), or attack-direction alignment
(`flip_direction_alignment`), all three added to `run_experiment.py` for exactly this purpose.

**Additive only**: nothing here changes any existing function's signature or behavior in
`run_experiment.py`, `attacks.py`, `plots.py`, or `layer_decomposition.py`.

**Known limitation** (state this in any write-up of the results): `run_seed_sweep` reseeds
`train_seed` and `attack_seed` TOGETHER (both set to the run index `s`), so a null result from
this sweep alone cannot distinguish "the inversion doesn't replicate because the MODEL differs
across seeds" from "...because the ATTACK/ESTIMATOR sampling differs across seeds."
`run_attack_seed_variance_decomposition` (Checkpoint 6, optional, run last) recovers that
decomposition cheaply by holding `train_seed` fixed and varying only `attack_seed`, reusing the
already-trained checkpoints.

**Statistical framing**: with 5 seeds, a paired sign test reaches at best p ~ 0.06 -- report
findings as "consistent in k of 5 seeds," never as "significant."

**Compute**: 5 seeds x 3 widths = 15 trainings, each with a 5-epsilon x 2-method attack grid,
measured under 2 metrics, plus one Mahalanobis SVD -- roughly an order of magnitude above the
main notebook's ~10 minutes on CPU. Intended to run on a GPU cluster (e.g. via a SLURM array job
over `(train_seed, width)`, with aggregation as a separate serial step over the written CSVs), not
inline in a notebook cell.
"""

from pathlib import Path

import pandas as pd
import torch

from mnist_lipschitz.data import load_mnist, get_dev_subset, make_loader
from mnist_lipschitz.models import SmallCNN, FlattenedInputWrapper, train_classifier
from mnist_lipschitz.estimators import euclidean_distance_fn
from mnist_lipschitz.adversarial.run_experiment import (
    RESULTS_DIR,
    DEFAULT_EPSILONS,
    MAHALANOBIS_EPSILON,
    build_pixel_mahalanobis_distance_fn,
    compute_bounds_with_distance_fn,
    run_epsilon_sweep,
    summarize_epsilon_sweep,
    adversarial_accuracy,
    clean_logit_stats,
    margin_lipschitz_estimate,
    flip_direction_alignment,
)

CHECKPOINT_DIR = RESULTS_DIR / "seed_sweep_checkpoints"


def _checkpoint_path(checkpoint_dir, train_seed, width):
    """`None` in, `None` out -- the "don't touch disk, always retrain" escape hatch used by the
    seeding-discipline tests in `tests/test_mahalanobis.py`, which need two independent trainings
    with NO caching in between to prove determinism isn't coming from a stale cache hit."""
    if checkpoint_dir is None:
        return None
    return Path(checkpoint_dir) / f"train_seed{train_seed}_width{width}.pt"


def train_or_load_checkpoint(train_seed, width, epochs=6, train_subset_size=5000,
                              checkpoint_dir=CHECKPOINT_DIR, verbose=True):
    """Loads a cached `SmallCNN(conv_channels=(width, 2*width))` checkpoint keyed by
    `(train_seed, width)` from `checkpoint_dir` if present, else trains one and saves it.

    This is the ONLY place training happens in this module, and it takes `train_seed` alone --
    no `attack_seed` parameter exists here at all, so "attack_seed cannot influence training" is
    true by construction, not just by convention (see `run_single_seed_width`'s seeding-discipline
    docstring, and the direct checks in `tests/test_mahalanobis.py`).

    Lets `run_seed_sweep` call `run_single_seed_width` once per `(train_seed, width)` for BOTH
    metrics without retraining the second time (see `run_single_seed_width`'s docstring for why
    this still gives bit-identical adversarial examples across metrics despite not literally
    caching `x_adv`), and lets an interrupted `run_seed_sweep` resume without redoing
    already-completed trainings -- this project's `SmallCNN` trainings are the dominant cost of
    the whole sweep (see this module's top docstring).

    Data loading (`train`/`test`) always happens regardless of cache hit/miss -- cheap relative to
    training, and needed either way for the caller's query/pool-point sampling.

    Returns `(model, train, test, train_acc, test_acc)`.
    """
    path = _checkpoint_path(checkpoint_dir, train_seed, width)
    train = load_mnist(train=True)
    test = load_mnist(train=False)

    if path is not None and path.exists():
        state = torch.load(path, weights_only=True)
        model = SmallCNN(conv_channels=(width, 2 * width))
        model.load_state_dict(state["model_state_dict"])
        train_acc, test_acc = state["train_acc"], state["test_acc"]
        if verbose:
            print(f"  [checkpoint] loaded train_seed={train_seed} width={width} from {path} "
                  f"(train_acc={train_acc:.4f}  test_acc={test_acc:.4f})")
        return model, train, test, train_acc, test_acc

    dev = get_dev_subset(train, n=train_subset_size, seed=train_seed)
    train_loader = make_loader(dev.x_image, dev.y, batch_size=128, shuffle=True, seed=train_seed)
    test_loader = make_loader(test.x_image, test.y, batch_size=1000, shuffle=False)

    torch.manual_seed(train_seed)
    model, train_acc, test_acc = train_classifier(
        SmallCNN(conv_channels=(width, 2 * width)), train_loader, test_loader,
        epochs=epochs, lr=1e-3, verbose=False)
    if verbose:
        print(f"  [trained] train_seed={train_seed} width={width}  "
              f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "train_acc": train_acc, "test_acc": test_acc}, path)

    return model, train, test, train_acc, test_acc


def run_single_seed_width(train_seed, attack_seed, width, distance_fn,
                           epochs=6, train_subset_size=5000, n_query_points=40,
                           n_train_norm_points=500, n_pool_points=2000, n_attack_points=500,
                           epsilons=DEFAULT_EPSILONS, pgd_alpha_frac=0.25, pgd_num_steps=20,
                           pgd_num_restarts=5, max_pairs=None, margin_estimator="pairwise",
                           checkpoint_dir=CHECKPOINT_DIR, metric_name="Euclidean", verbose=True):
    """One `(train_seed, attack_seed, width, distance_fn)` run of the full measurement battery.

    Trains (or loads a cached checkpoint for) one `SmallCNN` at `width` via
    `train_or_load_checkpoint`, runs the epsilon x method attack grid against it, and returns one
    row per `(epsilon, method)` carrying every existing bound/`R_adv` column (matching
    `run_bound_comparison_with_distance_fn`'s own columns) PLUS the mechanism diagnostics added to
    `run_experiment.py` for this sweep (`adversarial_accuracy`, `clean_logit_stats`,
    `margin_lipschitz_estimate`, `flip_direction_alignment`).

    **Seeding discipline (load-bearing)**: `train_seed` controls ONLY the model's initialization
    and training-data shuffling (via `train_or_load_checkpoint`, which doesn't even accept an
    attack_seed parameter); `attack_seed` controls ONLY query/pool/norm-point sampling and the
    attack/estimator randomness (PGD restarts, `pairwise_lipschitz`'s `max_pairs` subsampling).
    The two are NEVER derived from one another. This decomposition is load-bearing for this
    project's existing determinism guarantee (`tests/test_mahalanobis.py`'s retraining-determinism
    check, and the design decision that the Mahalanobis half reuses bit-identical checkpoints and
    adversarial examples, see `run_experiment.py`'s adversarial README) -- collapsing the two
    seeds back into one would break it. Checked directly in `tests/test_mahalanobis.py`'s
    seeding-discipline tests, not just asserted here.

    `distance_fn`: a single ALREADY-FITTED distance function (plain Euclidean, or a Mahalanobis
    `distance_fn` from `run_experiment.build_pixel_mahalanobis_distance_fn`) -- fit ONCE by the
    caller (`run_seed_sweep`) and passed in, never refit here: the Mahalanobis precision matrix is
    a property of the DATA, not of this run, and refitting it per `(seed, width)` call would be
    the dominant avoidable cost of the whole sweep. `metric_name`: only labels the returned rows
    (`"Euclidean"`/`"Mahalanobis"`), never selects behavior.

    **Model/attack reuse across metrics**: calling this function twice with the same
    `train_seed`/`width` but different `distance_fn` loads the SAME cached weights the second time
    (via `train_or_load_checkpoint`) rather than retraining. Since attacks
    (`fgsm_attack`/`pgd_attack`, via `run_epsilon_sweep`) depend only on the model and
    `attack_seed` -- never on `distance_fn` -- the two calls also generate BIT-IDENTICAL `x_adv`
    tensors, exactly the same determinism `run_bound_comparison_with_distance_fn` already relies
    on; only how sensitivity is MEASURED differs. `x_adv` is recomputed (not literally cached)
    across the two calls, since PGD/FGSM on ~500 points is cheap relative to training a CNN --
    the caching effort goes toward the expensive part (training), not the cheap deterministic one.

    Returns a `pd.DataFrame`, one row per `(epsilon, method)`, with columns: `train_seed`,
    `attack_seed`, `width`, `metric`, `train_acc`, `test_acc`, plus everything
    `summarize_epsilon_sweep` returns (`epsilon`, `method`, `mean_R_adv`, `median_R_adv`,
    `max_R_adv`, `pct_misclassified`, `L_full_estimated`, `product_bound`, `ratio_to_L_full`,
    `ratio_to_product_bound`), `L_head_exact`, `L_extractor_estimated`, `looseness_ratio`,
    `L_margin_estimated` (kept as its own column, never merged with `L_full_estimated` -- see
    `margin_lipschitz_estimate`'s docstring), the checkpoint-level `clean_logit_stats` columns
    (repeated across every row of this checkpoint, same convention `summarize_epsilon_sweep`
    already uses for `L_full_estimated`/`product_bound`), and the per-`(epsilon, method)`
    `n_flipped`/`n_evaluated`/`mean_cosine_alignment`/`std_cosine_alignment` columns
    (`pct_misclassified` from `summarize_epsilon_sweep` already equals
    `adversarial_accuracy`'s `misclassification_rate` for the same points, so only its extra
    columns are added here, not a duplicate).
    """
    model, train, test, train_acc, test_acc = train_or_load_checkpoint(
        train_seed, width, epochs=epochs, train_subset_size=train_subset_size,
        checkpoint_dir=checkpoint_dir, verbose=verbose)
    wrapped_model = FlattenedInputWrapper(model)

    generator = torch.Generator().manual_seed(attack_seed)
    query_idx = torch.randperm(len(test), generator=generator)[:n_query_points]
    x_query, y_query = test.x_flat[query_idx], test.y[query_idx]
    norm_idx = torch.randperm(len(train), generator=generator)[:n_train_norm_points]
    x_train_for_norm = train.x_flat[norm_idx]

    pool_mask = torch.ones(len(test), dtype=torch.bool)
    pool_mask[query_idx] = False
    remaining_idx = pool_mask.nonzero(as_tuple=True)[0]
    pool_idx = remaining_idx[torch.randperm(len(remaining_idx), generator=generator)[:n_pool_points]]
    x_pool, y_pool = test.x_flat[pool_idx], test.y[pool_idx]

    bound_result = compute_bounds_with_distance_fn(
        model, x_query, y_query, distance_fn, x_train_for_norm,
        max_pairs=max_pairs, seed=attack_seed, verbose=verbose)
    L_full_estimated = bound_result["L_full_estimated"]
    product_bound = bound_result["product"]

    sweep_results = run_epsilon_sweep(
        wrapped_model, x_pool, y_pool, epsilons=epsilons, pgd_alpha_frac=pgd_alpha_frac,
        pgd_num_steps=pgd_num_steps, pgd_num_restarts=pgd_num_restarts, n_points=n_attack_points,
        distance_fn=distance_fn, seed=attack_seed, verbose=verbose)
    summary_df = summarize_epsilon_sweep(sweep_results, L_full_estimated, product_bound, verbose=verbose)

    x_eval, y_eval = sweep_results["x_eval"], sweep_results["y_eval"]
    L_margin_estimated = margin_lipschitz_estimate(
        wrapped_model, x_query, y_query, estimator=margin_estimator, distance_fn=distance_fn,
        max_pairs=max_pairs, seed=attack_seed)
    logit_stats = clean_logit_stats(wrapped_model, x_eval)

    rows = []
    for _, summary_row in summary_df.iterrows():
        epsilon, method = summary_row["epsilon"], summary_row["method"]
        x_adv = sweep_results["per_case"][(epsilon, method)]["x_adv"]
        acc_stats = adversarial_accuracy(wrapped_model, x_eval, x_adv, y_eval)
        align_stats = flip_direction_alignment(wrapped_model, x_eval, x_adv, y_eval)

        rows.append({
            "train_seed": train_seed, "attack_seed": attack_seed, "width": width,
            "metric": metric_name, "train_acc": train_acc, "test_acc": test_acc,
            **summary_row.to_dict(),
            "L_head_exact": bound_result["L_head_exact"],
            "L_extractor_estimated": bound_result["L_extractor_estimated"],
            "looseness_ratio": bound_result["looseness_ratio"],
            "L_margin_estimated": L_margin_estimated,
            **logit_stats,
            # Alias of summary_row["pct_misclassified"] under the name summarize_seed_sweep's
            # paired_differences (Checkpoint 4) expects -- both are adversarial_accuracy's/
            # summarize_epsilon_sweep's identical flip-rate quantity for the same points, kept as
            # two names deliberately rather than requiring downstream aggregation code to know
            # they're the same column.
            "misclassification_rate": acc_stats["misclassification_rate"],
            "n_flipped": acc_stats["n_flipped"],
            "n_evaluated": acc_stats["n_evaluated"],
            "mean_cosine_alignment": align_stats["mean_cosine_alignment"],
            "std_cosine_alignment": align_stats["std_cosine_alignment"],
        })

    return pd.DataFrame(rows)


def run_seed_sweep(seeds=range(5), widths=(16, 32, 64), maha_fit_size=60000,
                    maha_epsilon=MAHALANOBIS_EPSILON, epochs=6, train_subset_size=5000,
                    n_query_points=40, n_train_norm_points=500, n_pool_points=2000,
                    n_attack_points=500, epsilons=DEFAULT_EPSILONS, pgd_alpha_frac=0.25,
                    pgd_num_steps=20, pgd_num_restarts=5, max_pairs=None,
                    margin_estimator="pairwise", checkpoint_dir=CHECKPOINT_DIR,
                    verbose=True, save_path=RESULTS_DIR / "seed_sweep_raw.csv"):
    """Repeats `run_single_seed_width` across every `(seed, width, metric)` combination -- the
    reseeding check for the single-seed width-32/width-64 misclassification-rate inversion (see
    this module's top docstring).

    The Mahalanobis precision matrix is fit ONCE, from `maha_fit_size` training points (default
    the full 60k training set, matching `build_pixel_mahalanobis_distance_fn`'s own "fit on the
    full training set" convention), not per `(seed, width)` -- see `run_single_seed_width`'s
    docstring for why refitting it repeatedly would dominate the sweep's cost.

    Both `train_seed` and `attack_seed` are set to the SAME run index `s`, for every `s` in
    `seeds` (see this module's top-level "Known limitation" docstring for what this does and
    doesn't let you conclude from a null result).

    Returns a long-format `pd.DataFrame` keyed by `(seed, width, epsilon, method, metric)`: the
    concatenation of every `run_single_seed_width` call's rows, with `train_seed` renamed to
    `seed` (since `train_seed == attack_seed` throughout this sweep, by construction above) and
    `attack_seed` dropped as redundant. Also written to `save_path` as CSV (`None` to skip saving).
    """
    train = load_mnist(train=True)
    if verbose:
        print(f"Fitting Mahalanobis precision matrix once from {maha_fit_size} training points...")
    maha_distance_fn = build_pixel_mahalanobis_distance_fn(train.x_flat[:maha_fit_size], epsilon=maha_epsilon)

    all_dfs = []
    for s in seeds:
        for width in widths:
            if verbose:
                print(f"=== seed={s} width={width} ===")
            for distance_fn, metric_name in ((euclidean_distance_fn, "Euclidean"),
                                              (maha_distance_fn, "Mahalanobis")):
                df = run_single_seed_width(
                    train_seed=s, attack_seed=s, width=width, distance_fn=distance_fn,
                    epochs=epochs, train_subset_size=train_subset_size, n_query_points=n_query_points,
                    n_train_norm_points=n_train_norm_points, n_pool_points=n_pool_points,
                    n_attack_points=n_attack_points, epsilons=epsilons, pgd_alpha_frac=pgd_alpha_frac,
                    pgd_num_steps=pgd_num_steps, pgd_num_restarts=pgd_num_restarts, max_pairs=max_pairs,
                    margin_estimator=margin_estimator, checkpoint_dir=checkpoint_dir,
                    metric_name=metric_name, verbose=verbose)
                all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.rename(columns={"train_seed": "seed"}).drop(columns=["attack_seed"])

    if save_path is not None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        combined.to_csv(save_path, index=False)
        if verbose:
            print(f"\nSaved seed-sweep raw results to {save_path}")

    return combined


# ---------------------------------------------------------------------------
# Aggregation (Checkpoint 4): turns run_seed_sweep's long, per-(seed, width, epsilon, method,
# metric) raw frame into the three tables the width-32/width-64 inversion question actually needs
# answered: does it hold up on average (per_config), does its SIGN replicate across seeds
# (paired_differences), and which mechanism explains it (mechanism_table)?
# ---------------------------------------------------------------------------

PAIRED_DIFF_QUANTITIES = ("misclassification_rate", "L_full_estimated", "max_R_adv",
                           "mean_top2_margin", "mean_logit_norm")

MECHANISM_TABLE_COLUMNS = ("mean_logit_norm", "mean_top2_margin", "L_full_estimated",
                            "L_margin_estimated", "mean_cosine_alignment")

DEFAULT_WIDTH_PAIRS = ((16, 32), (32, 64), (16, 64))


def _sign(value):
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def summarize_seed_sweep(df, width_pairs=DEFAULT_WIDTH_PAIRS, reference_seed=None, save_dir=RESULTS_DIR):
    """Aggregates `run_seed_sweep`'s long-format raw frame into three tables.

    `reference_seed`: which seed's sign counts as "the single-seed finding" `paired_differences`
    checks replication against -- defaults to the smallest seed present in `df` (this project's
    convention is `seed=0` as the default/original single-seed run everywhere else, so the
    smallest seed in a `run_seed_sweep(seeds=range(...))` call is exactly that original run).

    Returns `{"per_config", "paired_differences", "mechanism_table"}`:

    - **`per_config`**: mean/std/min/max across seeds of every numeric column (except `seed`
      itself), grouped by `(width, epsilon, method, metric)` -- one row per config, columns named
      `{original_column}_{mean,std,min,max}`.

    - **`paired_differences`**: for each `(epsilon, method, metric)` and each `(width_low,
      width_high)` pair in `width_pairs`, and each quantity in `PAIRED_DIFF_QUANTITIES`
      (`misclassification_rate`, `L_full_estimated`, `max_R_adv`, `mean_top2_margin`,
      `mean_logit_norm`): the per-seed difference `value(width_low) - value(width_high)`,
      summarized as `mean_diff`, `std_diff`, `n_seeds`, `reference_sign` (the sign of that
      difference at `reference_seed`), and `k_matching_sign` (how many of the `n_seeds` seeds --
      INCLUDING `reference_seed` itself -- have that same sign). Report findings as "consistent
      in k of n_seeds seeds," per this module's top-docstring statistical-framing note, never as
      "significant." Pairing is by seed index only -- a shared seed gives different widths the
      SAME data ordering/attack randomness, but NOT the same model initialization (SmallCNN's
      random init depends on `width` too), so this pairing is partial, not a true matched-pairs
      design; stated here so it isn't overclaimed downstream.

    - **`mechanism_table`**: for each `(width, metric)`, the across-seed mean +/- std of
      `MECHANISM_TABLE_COLUMNS` (`mean_logit_norm`, `mean_top2_margin`, `L_full_estimated`,
      `L_margin_estimated`, `mean_cosine_alignment`) -- the table that adjudicates between the
      three candidate mechanisms (logit scale, margin-functional Lipschitz constant, attack-
      direction alignment). Each column is first averaged WITHIN a seed across that seed's
      `(epsilon, method)` rows (a no-op for the checkpoint-level columns, which are already
      constant within a seed -- see `run_single_seed_width`'s docstring -- and a genuine average
      over attack conditions for `mean_cosine_alignment`, which is not checkpoint-level), then
      mean/std is taken ACROSS seeds of that per-seed value.
    """
    if reference_seed is None:
        reference_seed = int(df["seed"].min())

    group_cols = ["width", "epsilon", "method", "metric"]
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    agg_cols = [c for c in numeric_cols if c not in group_cols and c != "seed"]
    per_config = df.groupby(group_cols)[agg_cols].agg(["mean", "std", "min", "max"])
    per_config.columns = [f"{col}_{stat}" for col, stat in per_config.columns]
    per_config = per_config.reset_index()

    paired_rows = []
    for (epsilon, method, metric), sub in df.groupby(["epsilon", "method", "metric"]):
        for width_low, width_high in width_pairs:
            low = sub[sub["width"] == width_low].set_index("seed")
            high = sub[sub["width"] == width_high].set_index("seed")
            common_seeds = sorted(set(low.index) & set(high.index))
            if not common_seeds:
                continue
            for quantity in PAIRED_DIFF_QUANTITIES:
                diffs = low.loc[common_seeds, quantity] - high.loc[common_seeds, quantity]
                signs = diffs.apply(_sign)
                reference_sign = signs.loc[reference_seed] if reference_seed in signs.index else None
                k_matching_sign = int((signs == reference_sign).sum()) if reference_sign is not None else None
                paired_rows.append({
                    "epsilon": epsilon, "method": method, "metric": metric,
                    "width_low": width_low, "width_high": width_high, "quantity": quantity,
                    "mean_diff": diffs.mean(), "std_diff": diffs.std(),
                    "n_seeds": len(diffs), "reference_seed": reference_seed,
                    "reference_sign": reference_sign, "k_matching_sign": k_matching_sign,
                })
    paired_differences = pd.DataFrame(paired_rows)

    mechanism_rows = []
    for (width, metric), sub in df.groupby(["width", "metric"]):
        row = {"width": width, "metric": metric}
        for col in MECHANISM_TABLE_COLUMNS:
            per_seed = sub.groupby("seed")[col].mean()
            row[f"{col}_mean"] = per_seed.mean()
            row[f"{col}_std"] = per_seed.std()
        mechanism_rows.append(row)
    mechanism_table = pd.DataFrame(mechanism_rows)

    result = {
        "per_config": per_config,
        "paired_differences": paired_differences,
        "mechanism_table": mechanism_table,
    }

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, table in result.items():
            table.to_csv(save_dir / f"seed_sweep_{name}.csv", index=False)

    return result


# ---------------------------------------------------------------------------
# Checkpoint 6 (OPTIONAL, run LAST) -- variance decomposition.
#
# NOT wired into run_seed_sweep or any main()-equivalent -- deliberately opt-in, matching this
# project's convention that markedly slower/exploratory sweep functions are called directly, not
# auto-included in a driver (see toy_lipschitz.run_experiment.run_gap_N_sweep_seed_averaged for
# the same convention applied elsewhere in this codebase). Only worth running if
# summarize_seed_sweep's paired_differences/per_config output (Checkpoint 4) shows the
# width-32/width-64 inversion is inconsistent across seeds, or the spread on L_full_estimated
# looks large relative to the width-32/width-64 gap being tested.
# ---------------------------------------------------------------------------

def run_attack_seed_variance_decomposition(
        widths=(16, 32, 64), attack_seeds=(0, 1, 2), train_seed=0,
        maha_fit_size=60000, maha_epsilon=MAHALANOBIS_EPSILON, epochs=6, train_subset_size=5000,
        n_query_points=40, n_train_norm_points=500, n_pool_points=2000, n_attack_points=500,
        epsilons=DEFAULT_EPSILONS, pgd_alpha_frac=0.25, pgd_num_steps=20, pgd_num_restarts=5,
        max_pairs=None, margin_estimator="pairwise", checkpoint_dir=CHECKPOINT_DIR, verbose=True,
        save_path=RESULTS_DIR / "seed_sweep_attack_seed_variance.csv"):
    """Decomposes `run_seed_sweep`'s total across-seed spread into measurement/attack noise vs.
    model-to-model variation (see this module's top-docstring "Known limitation").

    `run_seed_sweep` reseeds `train_seed` and `attack_seed` TOGETHER, so its
    `per_config`/`paired_differences` spread conflates two different sources of variation: the
    MODEL changing across seeds (different init/training-data order), and the ATTACK/ESTIMATOR
    sampling changing across seeds (different query/pool points, different PGD restarts) even for
    an IDENTICAL model. This function isolates the second source alone: `train_seed` is held FIXED
    (default 0) across every call here, so every run at a given width should reuse the SAME
    already-trained checkpoint (via `train_or_load_checkpoint`'s cache -- genuinely "no
    retraining" only if that checkpoint already exists on disk under `checkpoint_dir` from a prior
    `run_seed_sweep` call with the SAME `train_seed`/training config; a cache miss falls back to
    training it, just without the intended cost savings), while `attack_seed` varies over
    `attack_seeds`.

    Returns a long-format `pd.DataFrame`, same column shape as `run_seed_sweep`'s output (`width`,
    `metric`, `epsilon`, `method`, ...) but with `attack_seed` as the varying dimension and
    `train_seed` constant throughout (both kept as separate columns here, unlike
    `run_seed_sweep`'s collapsed `seed` column, since the two are deliberately NOT equal in this
    sweep). Also written to `save_path` as CSV (`None` to skip saving).
    """
    train = load_mnist(train=True)
    if verbose:
        print(f"Fitting Mahalanobis precision matrix once from {maha_fit_size} training points...")
    maha_distance_fn = build_pixel_mahalanobis_distance_fn(train.x_flat[:maha_fit_size], epsilon=maha_epsilon)

    all_dfs = []
    for width in widths:
        for attack_seed in attack_seeds:
            if verbose:
                print(f"=== train_seed={train_seed} (fixed) width={width} attack_seed={attack_seed} ===")
            for distance_fn, metric_name in ((euclidean_distance_fn, "Euclidean"),
                                              (maha_distance_fn, "Mahalanobis")):
                df = run_single_seed_width(
                    train_seed=train_seed, attack_seed=attack_seed, width=width, distance_fn=distance_fn,
                    epochs=epochs, train_subset_size=train_subset_size, n_query_points=n_query_points,
                    n_train_norm_points=n_train_norm_points, n_pool_points=n_pool_points,
                    n_attack_points=n_attack_points, epsilons=epsilons, pgd_alpha_frac=pgd_alpha_frac,
                    pgd_num_steps=pgd_num_steps, pgd_num_restarts=pgd_num_restarts, max_pairs=max_pairs,
                    margin_estimator=margin_estimator, checkpoint_dir=checkpoint_dir,
                    metric_name=metric_name, verbose=verbose)
                all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    if save_path is not None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        combined.to_csv(save_path, index=False)
        if verbose:
            print(f"\nSaved attack-seed variance decomposition to {save_path}")

    return combined
