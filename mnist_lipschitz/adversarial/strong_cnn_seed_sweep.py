"""Multi-seed confirmation sweep for the StrongCNN adversarial-vs-Lipschitz-bound comparison
(`strong_cnn_experiment.py`, sibling module).

The single `seed=0` `StrongCNN` run (`strong_cnn_experiment.py`, see its README section) found
that under Euclidean distance, `max_R_adv` exceeded `L_full_estimated` at 9/10 epsilon/method
combinations, but that gap shrank dramatically under Mahalanobis distance -- the OPPOSITE
direction from the `SmallCNN` width sweep (`seed_sweep.py`), where Mahalanobis-based estimates
came out consistently lower than Euclidean. This module asks two questions with five
independently-trained `StrongCNN` checkpoints, identical architecture/recipe, only the training
seed varying:

- **Goal A**: does the Euclidean-vs-Mahalanobis flip reproduce across seeds, and is it robust to
  the Mahalanobis shrinkage parameter (epsilon)?
- **Goal B**: given clean test accuracy is matched by construction across the five seeds (checked
  directly, not assumed -- see `run_all_seed_trainings`'s accuracy-premise check), how much does
  adversarial sensitivity vary seed-to-seed? This is the cleanest instantiation of this project's
  "similar accuracy, different extension behavior" claim, since any observed difference cannot be
  attributed to an accuracy or capacity confound. A null result (all five seeds behaving
  near-identically) is a valid, reportable answer, not a failed run -- see Goodfellow et al.
  (2015)'s ensemble-of-twelve-maxout-networks finding that independently-seeded models trained on
  the same task learn similar functions, which a null result here would be consistent with.

**Out of scope**: cross-architecture comparison (StrongCNN vs. SmallCNN) -- those models don't
have matched clean accuracy, so that comparison would be confounded with respect to Goal B.

**Additive only**: nothing here changes any existing function's signature or behavior in
`strong_cnn_experiment.py`, `run_experiment.py`, `attacks.py`, `models.py`, or `plots.py`. Per-seed
checkpoints are written to their own path (`strong_cnn_state_dict_seed{k}.pt`) -- the
pre-existing single-checkpoint path (`strong_cnn_state_dict.pt`, `strong_cnn_experiment.main`'s
own file) is never read or overwritten by this module.

**Determinism caveat (state this in any write-up of results, per this module's own README
section)**: `torch.manual_seed`/data-loader seeding is set per training run, but
`torch.use_deterministic_algorithms(True)` is deliberately NOT forced (would cost real
determinism-checking value against total training-run variation, which is itself part of what's
being measured here -- see this module's `README`/findings section). The measured quantity is
*training-run* variation (init + data order + whatever residual CPU-arithmetic nondeterminism
exists), not *initialization-seed* variation in isolation. This run's environment: CPU only (no
CUDA available), recorded via `torch.__version__`/platform at the top of the driver notebook.

**FGSM only** (not PGD) -- keeps five full-pool attacks per seed cheap (a single deterministic
gradient step per point, batched, no random restarts), matching this sweep's own compute budget
concerns; PGD's own robustness across seeds is a candidate follow-up, not attempted here.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from mnist_lipschitz.data import load_mnist, make_loader
from mnist_lipschitz.models import StrongCNN, STRONG_CNN_CONFIG, FlattenedInputWrapper, train_classifier
from mnist_lipschitz.augmentation import random_affine_augment
from mnist_lipschitz.estimators import euclidean_distance_fn
from mnist_lipschitz.adversarial.attacks import fgsm_attack
from mnist_lipschitz.adversarial.run_experiment import (
    RESULTS_DIR,
    MAHALANOBIS_EPSILON,
    build_pixel_mahalanobis_distance_fn,
    achieved_ratio,
)
from mnist_lipschitz.adversarial.strong_cnn_experiment import compute_strong_cnn_bounds

torch.set_default_dtype(torch.float64)

SEEDS = (0, 1, 2, 3, 4)

# Held fixed across all five training seeds (see this module's top docstring, section 2.1 of the
# plan this implements): FGSM itself has no randomness (see attacks.fgsm_attack -- a single
# deterministic gradient-sign step, no generator involved at all), so this only actually matters
# for query/pool/norm-point SAMPLING below, not for the attack itself.
ATTACK_SEED = 0

EPSILONS = (0.1, 0.2)

# Goal A's shrinkage-sensitivity check (Checkpoint 4): the original single-seed run used
# MAHALANOBIS_EPSILON=0.01 (run_experiment.py's own established choice); 0.1 is an order of
# magnitude more heavily regularized, to see whether the flip's direction depends on this choice.
MAHALANOBIS_EPSILONS_SWEEP = (MAHALANOBIS_EPSILON, 0.1)

POOL_PATH = RESULTS_DIR / "strong_cnn_seed_sweep_common_pool.pt"
TRAINING_SUMMARY_PATH = RESULTS_DIR / "strong_cnn_seed_sweep_training_summary.csv"
R_ADV_TABLE_PATH = RESULTS_DIR / "strong_cnn_seed_sweep_r_adv_table.csv"
REALIZED_NORM_DIAGNOSTIC_PATH = RESULTS_DIR / "strong_cnn_seed_sweep_realized_norm_diagnostic.csv"
SHRINKAGE_TABLE_PATH = RESULTS_DIR / "strong_cnn_seed_sweep_shrinkage_sensitivity.csv"
BOUNDS_TABLE_PATH = RESULTS_DIR / "strong_cnn_seed_sweep_bounds_per_seed.csv"
SUMMARY_TABLE_PATH = RESULTS_DIR / "strong_cnn_seed_sweep_project1_summary.csv"


# ---------------------------------------------------------------------------
# Checkpoint 1 -- per-seed training and caching
# ---------------------------------------------------------------------------

def _checkpoint_path(seed, checkpoint_dir=RESULTS_DIR):
    """Per plan section 3.2: a NEW path per seed, never the existing single-checkpoint
    `strong_cnn_state_dict.pt` `strong_cnn_experiment.main()` reads/writes -- even seed=0
    reproducing that file's weights exactly is fine, but it is written here under its own name."""
    return Path(checkpoint_dir) / f"strong_cnn_state_dict_seed{seed}.pt"


@torch.no_grad()
def _eval_mode_mean_loss(model, x_image, y, batch_size=1000):
    """Mean cross-entropy loss over `(x_image, y)`, in eval mode, batched. Used as this module's
    "final train loss" (Checkpoint 1's requirement) -- deliberately NOT the raw per-epoch training
    running_loss `models.train_classifier` prints internally (that quantity is dropout-noisy and
    depends on batch order/augmentation, and isn't returned by `train_classifier` at all -- adding
    a return value there would be a signature change to a shared module, out of scope here per
    this module's "additive only" docstring). This is a well-defined, deterministic, reproducible
    alternative: eval-mode (no dropout, BatchNorm running stats) mean CE loss over the full
    training set, computed once right after training (or reloaded from the cached checkpoint)."""
    model.eval()
    loader = make_loader(x_image, y, batch_size=batch_size, shuffle=False)
    total_loss, n = 0.0, 0
    for xb, yb in loader:
        loss = F.cross_entropy(model(xb), yb, reduction="sum")
        total_loss += loss.item()
        n += xb.shape[0]
    return total_loss / n


def train_or_load_strong_cnn(seed, train, test, force_retrain=False, checkpoint_dir=RESULTS_DIR, verbose=True):
    """Trains (or loads a cached) `StrongCNN` for one `seed`, via `STRONG_CNN_CONFIG`'s exact
    recipe (same wiring as `strong_cnn_experiment.main`: full 60k MNIST, augmentation, cosine LR)
    -- the only difference from that function is the per-seed checkpoint path and the additional
    `final_train_loss`/`wall_clock_seconds` bookkeeping this sweep's Checkpoint 1 requires.

    `train`/`test`: already-loaded `data.MNISTData` (loaded once by the caller and reused across
    all five seeds, rather than reloaded from disk five times).

    `force_retrain=True` bypasses the cache even if a checkpoint file already exists (Checkpoint
    1's "re-running with the cache present retrains nothing" gate needs the OPPOSITE call --
    `force_retrain=False`, the default -- to be checked twice in a row and confirmed to skip
    training the second time).

    Returns a dict: {"model" (eval-mode), "train_acc", "test_acc", "final_train_loss",
    "wall_clock_seconds" (0.0 on a cache hit -- no training happened), "loaded_from_cache" (bool)}.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _checkpoint_path(seed, checkpoint_dir)

    torch.manual_seed(seed)
    train_loader = make_loader(train.x_image, train.y, batch_size=STRONG_CNN_CONFIG["batch_size"],
                                shuffle=True, seed=seed)
    test_loader = make_loader(test.x_image, test.y, batch_size=1000, shuffle=False)
    model = StrongCNN(dropout_conv=STRONG_CNN_CONFIG["dropout_conv"], dropout_fc=STRONG_CNN_CONFIG["dropout_fc"])

    if checkpoint_path.exists() and not force_retrain:
        state = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        train_acc, test_acc = state["train_acc"], state["test_acc"]
        final_train_loss = state["final_train_loss"]
        wall_clock_seconds = 0.0
        loaded_from_cache = True
        if verbose:
            print(f"[checkpoint] loaded seed={seed} from {checkpoint_path} "
                  f"(train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  "
                  f"final_train_loss={final_train_loss:.4f})")
    else:
        augment_generator = torch.Generator().manual_seed(seed)
        augment_fn = lambda x: random_affine_augment(
            x, degrees=STRONG_CNN_CONFIG["augment_degrees"],
            translate=STRONG_CNN_CONFIG["augment_translate"], generator=augment_generator)
        lr_scheduler_fn = lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=STRONG_CNN_CONFIG["lr_scheduler_t_max"], eta_min=STRONG_CNN_CONFIG["lr_scheduler_eta_min"])

        t0 = time.time()
        model, train_acc, test_acc = train_classifier(
            model, train_loader, test_loader, epochs=STRONG_CNN_CONFIG["epochs"],
            lr=STRONG_CNN_CONFIG["lr"], verbose=verbose, augment_fn=augment_fn,
            lr_scheduler_fn=lr_scheduler_fn)
        wall_clock_seconds = time.time() - t0
        model.eval()
        final_train_loss = _eval_mode_mean_loss(model, train.x_image, train.y)
        loaded_from_cache = False

        torch.save({
            "model_state_dict": model.state_dict(), "train_acc": train_acc, "test_acc": test_acc,
            "final_train_loss": final_train_loss, "wall_clock_seconds": wall_clock_seconds,
        }, checkpoint_path)
        if verbose:
            print(f"[trained] seed={seed}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  "
                  f"final_train_loss={final_train_loss:.4f}  wall_clock={wall_clock_seconds:.1f}s")

    model.eval()
    return {
        "model": model, "train_acc": train_acc, "test_acc": test_acc,
        "final_train_loss": final_train_loss, "wall_clock_seconds": wall_clock_seconds,
        "loaded_from_cache": loaded_from_cache,
    }


def check_accuracy_premise(summary_df, max_dev_pp=0.5):
    """Checkpoint 1's required check: "this establishes the similar-accuracy premise of Goal B and
    must be verified, not assumed." Returns (ok: bool, message: str) -- non-fatal by design (see
    this module's docstring on gate reporting): a caller decides whether to proceed on a failure,
    rather than this function raising and potentially discarding a completed 90-minute training run.

    `max_dev_pp`: max allowed deviation, in PERCENTAGE POINTS, of any seed's test_acc from the
    across-seed mean, before flagging the premise as violated (plan default: 0.5pp).
    """
    test_accs = summary_df["test_acc"].to_numpy() * 100.0
    spread = test_accs.max() - test_accs.min()
    mean_dev = np.abs(test_accs - test_accs.mean()).max()
    ok = mean_dev <= max_dev_pp
    message = (f"test_acc range=[{test_accs.min():.3f}, {test_accs.max():.3f}]% "
               f"(spread={spread:.3f}pp, max deviation from mean={mean_dev:.3f}pp, "
               f"threshold={max_dev_pp}pp) -- {'OK' if ok else 'PREMISE VIOLATED'}")
    return ok, message


def run_all_seed_trainings(seeds=SEEDS, force_retrain=False, checkpoint_dir=RESULTS_DIR,
                            save_path=TRAINING_SUMMARY_PATH, verbose=True):
    """Trains (or loads) all five seeds, and builds/saves the Checkpoint-1 training summary table.

    Loads MNIST train/test ONCE (not once per seed) and reuses the same `MNISTData` objects across
    every `train_or_load_strong_cnn` call.

    Returns (models_by_seed: {seed: eval-mode StrongCNN}, test: MNISTData, train: MNISTData,
    summary_df: pd.DataFrame with columns seed, train_acc, test_acc, final_train_loss,
    wall_clock_seconds, loaded_from_cache).
    """
    train = load_mnist(train=True)
    test = load_mnist(train=False)

    models_by_seed = {}
    rows = []
    for seed in seeds:
        if verbose:
            print(f"=== seed={seed} ===")
        result = train_or_load_strong_cnn(seed, train, test, force_retrain=force_retrain,
                                           checkpoint_dir=checkpoint_dir, verbose=verbose)
        models_by_seed[seed] = result["model"]
        rows.append({"seed": seed, "train_acc": result["train_acc"], "test_acc": result["test_acc"],
                     "final_train_loss": result["final_train_loss"],
                     "wall_clock_seconds": result["wall_clock_seconds"],
                     "loaded_from_cache": result["loaded_from_cache"]})

    summary_df = pd.DataFrame(rows)
    if save_path is not None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(save_path, index=False)

    ok, message = check_accuracy_premise(summary_df)
    print(f"[accuracy premise check] {message}")

    return models_by_seed, test, train, summary_df


# ---------------------------------------------------------------------------
# Checkpoint 2 -- common evaluation pool
# ---------------------------------------------------------------------------

@torch.no_grad()
def _clean_predictions_and_margin(model, test):
    """Full-test-set (10000-point) clean predictions and top-2 logit margin for one model.

    `top2_margin` here follows `run_experiment.clean_logit_stats`'s existing project convention:
    top-1 logit minus runner-up logit BY VALUE (not true-class-vs-runner-up -- that's
    `models.margin_fn`'s different, TRUE-class-anchored quantity) -- kept consistent with the rest
    of this sub-experiment rather than introducing a second, differently-defined "margin".

    Returns (preds: (10000,) LongTensor, margin: (10000,) FloatTensor).
    """
    wrapped = FlattenedInputWrapper(model)
    logits = wrapped(test.x_flat)
    preds = logits.argmax(dim=1)
    top2 = logits.topk(2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    return preds, margin


def build_common_pool(models_by_seed, test, save_path=POOL_PATH, verbose=True):
    """Checkpoint 2: the common evaluation pool -- test images correctly classified by ALL five
    seeds, so cross-seed R_adv comparisons aren't confounded by each seed misclassifying a
    different subset of clean images (see this module's top docstring / plan section 4).

    Explicitly asserts (not just infers from set construction) that every returned pool index is
    correctly classified under every one of the five models, per the plan's Checkpoint-2 gate.

    Returns and persists {"pool_idx": (P,) LongTensor of test-set indices, "margins_by_seed":
    {seed: (10000,) FloatTensor}, "test_acc_full": {seed: float}}.
    """
    correct_masks = {}
    margins_by_seed = {}
    test_acc_full = {}
    for seed, model in models_by_seed.items():
        preds, margin = _clean_predictions_and_margin(model, test)
        correct = preds == test.y
        correct_masks[seed] = correct
        margins_by_seed[seed] = margin
        test_acc_full[seed] = correct.float().mean().item()

    combined_mask = torch.ones(len(test), dtype=torch.bool)
    for mask in correct_masks.values():
        combined_mask &= mask
    pool_idx = combined_mask.nonzero(as_tuple=True)[0]

    # Explicit programmatic check (plan Checkpoint 2 gate), not inferred from the intersection
    # construction above -- recomputes correctness on exactly the returned pool_idx.
    for seed, model in models_by_seed.items():
        wrapped = FlattenedInputWrapper(model)
        with torch.no_grad():
            preds_on_pool = wrapped(test.x_flat[pool_idx]).argmax(dim=1)
        assert torch.equal(preds_on_pool, test.y[pool_idx]), \
            f"seed={seed} misclassifies at least one supposedly-common-pool index"

    if verbose:
        print(f"[common pool] size={pool_idx.shape[0]} / {len(test)} "
              f"({pool_idx.shape[0] / len(test):.4f} of the full test set); "
              f"per-seed full test_acc: {test_acc_full}")

    pool = {"pool_idx": pool_idx, "margins_by_seed": margins_by_seed, "test_acc_full": test_acc_full}
    if save_path is not None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(pool, save_path)
    return pool


def load_common_pool(path=POOL_PATH):
    """Loads a `build_common_pool`-persisted pool -- every downstream stage (Checkpoints 3-6) MUST
    call this rather than recomputing the pool itself, per the plan's Checkpoint-2 gate."""
    return torch.load(path, weights_only=False)


# ---------------------------------------------------------------------------
# Checkpoint 3 -- attacks and R_adv computation
# ---------------------------------------------------------------------------

def _batched_fgsm(model, x, y, epsilon, batch_size=2000):
    """FGSM over the full pool, chunked to bound peak memory -- `fgsm_attack` itself is a single
    deterministic gradient-sign step with NO random state (no generator, unlike PGD), so this is
    bit-identical to (and cheap enough to just re-run instead of caching) calling it on the whole
    pool at once; chunking exists purely for memory headroom, not correctness."""
    chunks = []
    for i in range(0, x.shape[0], batch_size):
        chunks.append(fgsm_attack(model, x[i:i + batch_size], y[i:i + batch_size], epsilon))
    return torch.cat(chunks, dim=0)


def fit_shared_mahalanobis_distance_fn(train, epsilon=MAHALANOBIS_EPSILON):
    """Fits ONE Mahalanobis precision matrix from the full 60k training set (via
    `run_experiment.build_pixel_mahalanobis_distance_fn`, reused not re-derived), shared across
    every seed -- the precision matrix is a property of the DATA distribution, not of any one
    trained model (plan section 5.3), so fitting it once here and passing the resulting
    `distance_fn` closure to every seed's R_adv computation is what makes it seed-independent BY
    CONSTRUCTION (a single shared closure, not five independently-fit ones) rather than something
    that merely needs to be checked for equality afterward.
    """
    return build_pixel_mahalanobis_distance_fn(train.x_flat, epsilon=epsilon)


def build_r_adv_table(models_by_seed, test, pool, mahalanobis_distance_fn, epsilons=EPSILONS,
                       batch_size=2000, verbose=True):
    """Checkpoint 3: the single tidy per-example table every downstream statistic (Checkpoint 5)
    and plot (Checkpoint 6) derives from -- one row per (seed, epsilon, pool position).

    For each (seed, epsilon), FGSM is run ONCE against the common pool; both the Euclidean and the
    Mahalanobis R_adv are then computed from that SAME `x_adv` tensor (via `achieved_ratio`, which
    only changes the denominator's `distance_fn` -- see its own docstring), which makes "the same
    adversarial examples are used for both metrics" true BY CONSTRUCTION here, not something
    verified after the fact by hashing two independently-generated tensors.

    R_adv's numerator/denominator (`achieved_ratio`) already use the REALIZED `||x - x_adv||`
    distance, not a nominal `epsilon * sqrt(784)` value -- confirmed directly by inspecting
    `achieved_ratio`'s implementation (its denominator is `distance_fn(x, x_adv)`, called on the
    actual post-clipping tensors) before writing this function, per plan section 5.1's concern.
    This function additionally reports the realized/nominal norm ratio as a diagnostic (see
    `ratio_diagnostic_df`'s return), confirming FGSM does clip substantially at these epsilons
    (most MNIST pixels are saturated at 0 or 1) without that clipping actually biasing R_adv itself.

    Returns (df, ratio_diagnostic_df):
    - `df` columns: seed, epsilon, pool_position, test_index, realized_norm, is_misclassified,
      clean_margin, R_adv_euclidean, R_adv_mahalanobis.
    - `ratio_diagnostic_df` columns: seed, epsilon, mean_realized_over_nominal (nominal =
      epsilon * sqrt(784), the L_inf-ball's max possible L2 excursion before clipping).
    """
    pool_idx = pool["pool_idx"]
    x_pool = test.x_flat[pool_idx]
    y_pool = test.y[pool_idx]
    nominal_norm = {eps: eps * (784.0 ** 0.5) for eps in epsilons}

    table_chunks = []
    diagnostic_rows = []
    for seed, model in models_by_seed.items():
        wrapped = FlattenedInputWrapper(model)
        margin_pool = pool["margins_by_seed"][seed][pool_idx]
        for epsilon in epsilons:
            x_adv = _batched_fgsm(wrapped, x_pool, y_pool, epsilon, batch_size=batch_size)
            with torch.no_grad():
                preds_adv = wrapped(x_adv).argmax(dim=1)
            is_misclassified = preds_adv != y_pool
            realized_norm = euclidean_distance_fn(x_pool, x_adv)
            R_adv_euclidean = achieved_ratio(wrapped, x_pool, x_adv, distance_fn=euclidean_distance_fn)
            R_adv_mahalanobis = achieved_ratio(wrapped, x_pool, x_adv, distance_fn=mahalanobis_distance_fn)

            mean_ratio = (realized_norm / nominal_norm[epsilon]).mean().item()
            diagnostic_rows.append({"seed": seed, "epsilon": epsilon,
                                     "mean_realized_over_nominal": mean_ratio})
            if verbose:
                print(f"  seed={seed}  epsilon={epsilon:g}  "
                      f"pct_misclassified={is_misclassified.float().mean().item():.4f}  "
                      f"mean_realized_norm={realized_norm.mean().item():.4f}  "
                      f"(nominal={nominal_norm[epsilon]:.4f}, ratio={mean_ratio:.4f})")

            table_chunks.append(pd.DataFrame({
                "seed": seed, "epsilon": epsilon,
                "pool_position": np.arange(pool_idx.shape[0]),
                "test_index": pool_idx.numpy(),
                "realized_norm": realized_norm.detach().numpy(),
                "is_misclassified": is_misclassified.detach().numpy(),
                "clean_margin": margin_pool.detach().numpy(),
                "R_adv_euclidean": R_adv_euclidean.detach().numpy(),
                "R_adv_mahalanobis": R_adv_mahalanobis.detach().numpy(),
            }))

    df = pd.concat(table_chunks, ignore_index=True)
    ratio_diagnostic_df = pd.DataFrame(diagnostic_rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(R_ADV_TABLE_PATH, index=False)
    ratio_diagnostic_df.to_csv(REALIZED_NORM_DIAGNOSTIC_PATH, index=False)

    return df, ratio_diagnostic_df


def extract_extreme_example(model, test, pool, df, seed, epsilon, which, distance_fn=euclidean_distance_fn,
                             metric_name="Euclidean"):
    """Recovers the actual (x, x_adv) image pair for the largest- or smallest-R_adv row of one
    (seed, epsilon) case in `df` -- `df` itself only stores scalars (R_adv, realized_norm, ...),
    not the (784,)-dim image tensors, so this re-runs FGSM for that single (seed, epsilon) case
    (cheap -- see `_batched_fgsm`'s docstring on why FGSM is safe to just recompute rather than
    cache) and indexes out the one row `df` identifies as extreme.

    `which`: "most_sensitive" (largest R_adv_euclidean) or "least_sensitive" (smallest).
    `distance_fn`/`metric_name`: which R_adv column ranks the examples and what `pixel_distance`
    is computed under -- pass Mahalanobis's `distance_fn` and `metric_name="Mahalanobis"` for the
    Mahalanobis-side extreme example (ranking still uses the Euclidean-computed rows currently in
    `df`'s R_adv_euclidean/R_adv_mahalanobis columns, selected by `distance_fn`'s matching column
    below).

    Returns a dict compatible with `plots.plot_extreme_examples`/`plot_example_pair`: {"x",
    "x_adv", "y_true", "pred_clean", "pred_adv", "R_adv", "epsilon", "method", "pixel_distance"}.
    """
    R_adv_col = "R_adv_euclidean" if metric_name == "Euclidean" else "R_adv_mahalanobis"
    case = df[(df["seed"] == seed) & (df["epsilon"] == epsilon)]
    row = case.loc[case[R_adv_col].idxmax()] if which == "most_sensitive" else case.loc[case[R_adv_col].idxmin()]

    test_index = int(row["test_index"])
    x = test.x_flat[test_index:test_index + 1]
    y = test.y[test_index:test_index + 1]
    wrapped = FlattenedInputWrapper(model)
    x_adv = _batched_fgsm(wrapped, x, y, epsilon)

    with torch.no_grad():
        pred_clean = wrapped(x).argmax(dim=1).item()
        pred_adv = wrapped(x_adv).argmax(dim=1).item()

    return {
        "x": x.squeeze(0), "x_adv": x_adv.squeeze(0), "y_true": int(test.y[test_index].item()),
        "pred_clean": pred_clean, "pred_adv": pred_adv, "R_adv": row[R_adv_col],
        "epsilon": epsilon, "method": "FGSM",
        "pixel_distance": distance_fn(x, x_adv).item(),
    }


# ---------------------------------------------------------------------------
# Checkpoint 4 -- Mahalanobis shrinkage sensitivity (Goal A)
# ---------------------------------------------------------------------------

def run_shrinkage_sensitivity(models_by_seed, test, train, pool, df, epsilons_maha=MAHALANOBIS_EPSILONS_SWEEP,
                               attack_epsilons=EPSILONS, batch_size=2000, verbose=True):
    """Checkpoint 4: refits the Mahalanobis precision matrix at each candidate `epsilons_maha`
    value and recomputes R_adv's (Mahalanobis) denominator only -- NO retraining, NO re-attacking
    (FGSM's `x_adv` is fully determined by `(model, x, y, attack_epsilon)`, independent of the
    Mahalanobis shrinkage parameter, which only ever changes how distance is MEASURED afterward;
    see `build_r_adv_table`'s docstring for the same "recompute rather than cache" reasoning).

    Returns a long-format `pd.DataFrame`: seed, attack_epsilon, maha_epsilon, pool_position,
    test_index, R_adv_mahalanobis. Also saved to `SHRINKAGE_TABLE_PATH`.
    """
    pool_idx = pool["pool_idx"]
    x_pool = test.x_flat[pool_idx]
    y_pool = test.y[pool_idx]

    rows = []
    for maha_epsilon in epsilons_maha:
        if verbose:
            print(f"[shrinkage sweep] fitting Mahalanobis precision at epsilon={maha_epsilon:g}...")
        distance_fn = fit_shared_mahalanobis_distance_fn(train, epsilon=maha_epsilon)
        for seed, model in models_by_seed.items():
            wrapped = FlattenedInputWrapper(model)
            for attack_epsilon in attack_epsilons:
                x_adv = _batched_fgsm(wrapped, x_pool, y_pool, attack_epsilon, batch_size=batch_size)
                R_adv = achieved_ratio(wrapped, x_pool, x_adv, distance_fn=distance_fn)
                rows.append(pd.DataFrame({
                    "seed": seed, "attack_epsilon": attack_epsilon, "maha_epsilon": maha_epsilon,
                    "pool_position": np.arange(pool_idx.shape[0]), "test_index": pool_idx.numpy(),
                    "R_adv_mahalanobis": R_adv.detach().numpy(),
                }))

    result = pd.concat(rows, ignore_index=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(SHRINKAGE_TABLE_PATH, index=False)
    return result


SHRINKAGE_BOUNDS_TABLE_PATH = RESULTS_DIR / "strong_cnn_seed_sweep_shrinkage_bounds.csv"


def shrinkage_bounds_per_seed(models_by_seed, train, test, pool, epsilons_maha=MAHALANOBIS_EPSILONS_SWEEP,
                               n_query=1000, n_norm=1000, sampling_seed=ATTACK_SEED, verbose=True):
    """Companion to `run_shrinkage_sensitivity`: Goal A asks whether the flip -- the GAP between
    achieved R_adv and `L_full_estimated` narrowing under Mahalanobis distance -- is robust to the
    shrinkage parameter, which requires `L_full_estimated` recomputed at EACH `epsilons_maha` value
    too, not just R_adv's denominator (`run_shrinkage_sensitivity` only varies the latter). Without
    this, "does the flip survive a different epsilon" can only be answered for R_adv in isolation,
    not for the ratio-to-bound comparison Goal A is actually about.

    Thin wrapper around `per_seed_bounds`, called once per `epsilons_maha` value with a
    freshly-fit Mahalanobis `distance_fn` at that epsilon (labeled `f"Mahalanobis_eps{epsilon:g}"`
    in the returned `metric` column, so it stays distinguishable from `per_seed_bounds`'s own
    default `"Mahalanobis"` label at `MAHALANOBIS_EPSILON`).

    Returns a `pd.DataFrame` (concatenation across `epsilons_maha`), also saved to
    `SHRINKAGE_BOUNDS_TABLE_PATH`.
    """
    rows = []
    for maha_epsilon in epsilons_maha:
        distance_fn = fit_shared_mahalanobis_distance_fn(train, epsilon=maha_epsilon)
        bounds_df = per_seed_bounds(models_by_seed, train, test, pool,
                                     {f"Mahalanobis_eps{maha_epsilon:g}": distance_fn},
                                     n_query=n_query, n_norm=n_norm, sampling_seed=sampling_seed,
                                     verbose=verbose)
        bounds_df["maha_epsilon"] = maha_epsilon
        rows.append(bounds_df)

    result = pd.concat(rows, ignore_index=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(SHRINKAGE_BOUNDS_TABLE_PATH, index=False)
    return result


# ---------------------------------------------------------------------------
# Checkpoint 5 -- statistics
# ---------------------------------------------------------------------------

def primary_r_adv_stats(df, distance_col, metric_name):
    """Unconditional R_adv on the full common pool, per seed x epsilon -- the ONLY genuinely
    apples-to-apples cross-seed statistic (plan section 7.1), since every seed is evaluated on
    identical images by construction (Checkpoint 2). Reports median/p95/p99/max; **p99 is the
    headline** (plan section 7.1) -- max is a single-sample statistic dominated by whichever one
    image happened to land in the tail for that seed, not a stable cross-seed comparison.
    """
    grouped = df.groupby(["seed", "epsilon"])[distance_col].agg(
        median="median", p95=lambda s: s.quantile(0.95), p99=lambda s: s.quantile(0.99), max="max")
    grouped = grouped.reset_index()
    grouped["metric"] = metric_name
    return grouped


def attack_success_rate(df):
    """Attack success rate (mean `is_misclassified`), per seed x epsilon -- a SEPARATE scalar,
    reported once (not per metric): `is_misclassified` depends only on the model's prediction on
    the SAME `x_adv` used for both R_adv columns (see `build_r_adv_table`'s docstring), so it is
    metric-independent by construction, not by coincidence (plan section 7.2)."""
    return df.groupby(["seed", "epsilon"])["is_misclassified"].mean().reset_index(
        name="attack_success_rate")


def by_outcome_split(df, distance_col, metric_name):
    """WITHIN-seed diagnostic only (plan section 7.3) -- explicitly NOT a cross-seed comparison
    statistic, since the attack succeeds on a different image subset per seed and success at a
    given epsilon correlates with small clean margin, so outcome-conditional stats would be
    partly compositional across seeds. Reports median/p95 R_adv AND mean clean_margin per
    (seed, epsilon, outcome) bucket, so the margin confound is visible alongside the split rather
    than silently contaminating its interpretation.
    """
    grouped = df.groupby(["seed", "epsilon", "is_misclassified"])[distance_col].agg(
        median="median", p95=lambda s: s.quantile(0.95), n="count")
    grouped = grouped.reset_index()
    margin_grouped = df.groupby(["seed", "epsilon", "is_misclassified"])["clean_margin"].mean().reset_index(
        name="mean_clean_margin")
    grouped = grouped.merge(margin_grouped, on=["seed", "epsilon", "is_misclassified"])
    grouped["metric"] = metric_name
    return grouped


def common_success_set(df, epsilon, min_reliable_size=50):
    """Intersects the SUCCESS sets (images misclassified by ALL five seeds) at one epsilon --
    plan section 7.4: the only way to compare success-conditional R_adv on fixed images across
    seeds, at the cost of a small, fragility-biased subset. Misclassification is metric-
    independent (see `attack_success_rate`'s docstring), so this uses `is_misclassified` directly,
    not a per-metric version of it.

    Returns {"epsilon", "test_indices" (LongTensor), "n", "reliable" (bool, False if
    `n < min_reliable_size` -- treat as illustrative only per the plan), "r_adv_euclidean_by_seed",
    "r_adv_mahalanobis_by_seed"} (each a dict {seed: (n,) np.ndarray} of that quantity restricted
    to the common-success test indices, for a caller to compute conditional stats on directly).
    """
    sub = df[df["epsilon"] == epsilon]
    seeds = sorted(sub["seed"].unique())
    misclassified_sets = {}
    for seed in seeds:
        seed_sub = sub[sub["seed"] == seed]
        misclassified_sets[seed] = set(seed_sub.loc[seed_sub["is_misclassified"], "test_index"])

    common = set.intersection(*misclassified_sets.values()) if misclassified_sets else set()
    common_indices = sorted(common)
    n = len(common_indices)
    reliable = n >= min_reliable_size

    r_adv_euclidean_by_seed, r_adv_mahalanobis_by_seed = {}, {}
    for seed in seeds:
        seed_sub = sub[(sub["seed"] == seed) & (sub["test_index"].isin(common_indices))]
        seed_sub = seed_sub.set_index("test_index").loc[common_indices]
        r_adv_euclidean_by_seed[seed] = seed_sub["R_adv_euclidean"].to_numpy()
        r_adv_mahalanobis_by_seed[seed] = seed_sub["R_adv_mahalanobis"].to_numpy()

    return {
        "epsilon": epsilon, "test_indices": common_indices, "n": n, "reliable": reliable,
        "r_adv_euclidean_by_seed": r_adv_euclidean_by_seed,
        "r_adv_mahalanobis_by_seed": r_adv_mahalanobis_by_seed,
    }


def per_seed_bounds(models_by_seed, train, test, pool, distance_fns, n_query=1000, n_norm=1000,
                     sampling_seed=ATTACK_SEED, verbose=True):
    """L_full_estimated / product_bound / looseness_ratio per seed, under EACH `distance_fns`
    entry -- plan section 7.5: attack-independent (so free of every pool/outcome confound above),
    but seed-dependent (each trained model has its own), and Goal A is specifically a statement
    about the GAP between `max_R_adv` and `L_full_estimated`, so both sides must be recomputed
    under Mahalanobis, not just the achieved-ratio side.

    Query/pool-exclusion points are sampled from the test set EXCLUDING the common evaluation pool
    (`pool["pool_idx"]`), matching this project's existing convention (see
    `strong_cnn_experiment.main`'s own query/pool separation) of keeping the bound estimate
    independent from the attack evaluation points, not double-using the same images for both.

    `distance_fns`: {metric_name: distance_fn} -- e.g. {"Euclidean": euclidean_distance_fn,
    "Mahalanobis": maha_distance_fn}.

    Returns a `pd.DataFrame`: seed, metric, L_head_exact, L_extractor_estimated,
    L_full_estimated, product, looseness_ratio.
    """
    pool_idx_set = set(pool["pool_idx"].tolist())
    generator = torch.Generator().manual_seed(sampling_seed)
    all_idx = torch.randperm(len(test), generator=generator)
    non_pool_idx = torch.tensor([i.item() for i in all_idx if i.item() not in pool_idx_set])

    query_idx = non_pool_idx[:n_query]
    x_query, y_query = test.x_flat[query_idx], test.y[query_idx]

    train_norm_generator = torch.Generator().manual_seed(sampling_seed)
    norm_idx = torch.randperm(len(train), generator=train_norm_generator)[:n_norm]
    x_train_for_norm = train.x_flat[norm_idx]

    rows = []
    for seed, model in models_by_seed.items():
        for metric_name, distance_fn in distance_fns.items():
            if verbose:
                print(f"[bounds] seed={seed}  metric={metric_name}")
            result = compute_strong_cnn_bounds(
                model, x_query, y_query, distance_fn, x_train_for_norm,
                seed=sampling_seed, verbose=verbose)
            rows.append({"seed": seed, "metric": metric_name, **result})

    bounds_df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bounds_df.to_csv(BOUNDS_TABLE_PATH, index=False)
    return bounds_df


def build_project1_summary_table(training_summary_df, df, bounds_df, epsilons=EPSILONS,
                                  headline_epsilon=None, save_path=SUMMARY_TABLE_PATH):
    """Checkpoint 6, point 5: the Project-1 punchline figure -- one row per seed, combining clean
    accuracy (`training_summary_df`), adversarial accuracy at each epsilon (from
    `attack_success_rate`, metric-independent -- see that function's docstring), the tight/loose
    Lipschitz bounds and looseness ratio under both metrics (`bounds_df`, from `per_seed_bounds`),
    and the p99 R_adv headline statistic (`primary_r_adv_stats`) under both metrics.

    `headline_epsilon`: which epsilon's p99(R_adv) to report as the single headline number per
    seed/metric -- defaults to the LARGEST epsilon in `epsilons`, matching this project's existing
    convention (`run_experiment.run_cnn_adversarial_width_sweep`'s own "evaluated at the largest
    swept epsilon" choice) that the strongest attack condition is the most informative one for a
    single summary column; adversarial accuracy is still reported at EVERY epsilon (a separate
    column each), since that quantity is cheap to show in full and directly interpretable.

    Returns a `pd.DataFrame`, one row per seed, also saved to `save_path` (`None` to skip saving).
    """
    if headline_epsilon is None:
        headline_epsilon = max(epsilons)

    success = attack_success_rate(df)
    primary_by_metric = {
        "euclidean": primary_r_adv_stats(df, "R_adv_euclidean", "Euclidean"),
        "mahalanobis": primary_r_adv_stats(df, "R_adv_mahalanobis", "Mahalanobis"),
    }

    rows = []
    for seed in training_summary_df["seed"]:
        train_row = training_summary_df[training_summary_df["seed"] == seed].iloc[0]
        row = {"seed": seed, "train_acc": train_row["train_acc"], "test_acc": train_row["test_acc"]}

        for epsilon in epsilons:
            succ_row = success[(success["seed"] == seed) & (success["epsilon"] == epsilon)]
            row[f"adv_acc_eps{epsilon:g}"] = 1.0 - succ_row["attack_success_rate"].iloc[0]

        for metric_key, metric_label in (("euclidean", "Euclidean"), ("mahalanobis", "Mahalanobis")):
            bound_row = bounds_df[(bounds_df["seed"] == seed) & (bounds_df["metric"] == metric_label)]
            if len(bound_row):
                b = bound_row.iloc[0]
                row[f"L_full_estimated_{metric_key}"] = b["L_full_estimated"]
                row[f"product_bound_{metric_key}"] = b["product"]
                row[f"looseness_ratio_{metric_key}"] = b["looseness_ratio"]

            primary_row = primary_by_metric[metric_key]
            primary_row = primary_row[(primary_row["seed"] == seed) & (primary_row["epsilon"] == headline_epsilon)]
            if len(primary_row):
                row[f"p99_R_adv_{metric_key}_eps{headline_epsilon:g}"] = primary_row.iloc[0]["p99"]

        rows.append(row)

    result = pd.DataFrame(rows)
    if save_path is not None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        result.to_csv(save_path, index=False)
    return result


# ---------------------------------------------------------------------------
# Transferability check (opt-in -- NOT part of run_full_sweep, matching this project's convention
# that additional exploratory drivers are called directly rather than auto-included, see
# toy_lipschitz.run_experiment.run_gap_N_sweep_seed_averaged for the same precedent).
#
# Every quantity above (Goal A, Goal B) attacks each seed's OWN model and measures sensitivity on
# ITS OWN logits -- it never asks whether an adversarial example crafted for ONE model also fools
# a DIFFERENT, independently-trained one. That's the classic adversarial-transferability question
# (Goodfellow et al. 2015's ensemble-of-twelve-maxout-networks result is exactly a transferability
# finding), and the notebook's Goal-B discussion of that paper only engaged with it indirectly (via
# the common-success-set overlap of five INDEPENDENTLY-crafted attacks). This section asks it
# directly: fix ONE seed's adversarial examples, then evaluate every seed's clean-vs-adversarial
# accuracy and R_adv on those SAME fixed images.
# ---------------------------------------------------------------------------

TRANSFER_TABLE_PATH = RESULTS_DIR / "strong_cnn_seed_sweep_transfer_table.csv"
TRANSFER_SUMMARY_PATH = RESULTS_DIR / "strong_cnn_seed_sweep_transfer_summary.csv"


def run_transfer_attack(models_by_seed, test, pool, mahalanobis_distance_fn, source_seed=0,
                         epsilons=EPSILONS, batch_size=2000, verbose=True):
    """Generates FGSM adversarial examples against ONE `source_seed` model, then evaluates EVERY
    seed's model on those SAME fixed `(x, x_adv)` pairs. `x_adv` is computed ONCE per epsilon
    (attacking `source_seed` only) and then reused unchanged for every `eval_seed` -- so any
    cross-seed difference in the results reflects how differently-trained models respond to an
    IDENTICAL perturbation, not a difference in how each was attacked.

    For `eval_seed == source_seed` this reduces EXACTLY to `build_r_adv_table`'s own
    `is_misclassified`/`R_adv_euclidean`/`R_adv_mahalanobis` values for that seed/epsilon (checked
    directly in `tests/test_strong_cnn_seed_sweep.py`, not just asserted) -- attacking a model and
    then evaluating it on its own adversarial examples is just the ordinary, non-transfer attack;
    every OTHER `eval_seed` row is the genuinely new transfer measurement.

    `mahalanobis_distance_fn`: an already-fit closure (e.g. `fit_shared_mahalanobis_distance_fn`'s
    output) -- not refit here, same reasoning as `build_r_adv_table`'s own parameter.

    Returns a tidy `pd.DataFrame`, one row per `(source_seed, eval_seed, epsilon, pool_position)`:
    `test_index`, `is_misclassified` (under `eval_seed`'s OWN model), `R_adv_euclidean`,
    `R_adv_mahalanobis` (both computed from `eval_seed`'s own logits on the fixed `x`/`x_adv`
    pair -- i.e. `achieved_ratio(eval_seed_model, x, x_adv, ...)`). Also saved to
    `TRANSFER_TABLE_PATH`.
    """
    pool_idx = pool["pool_idx"]
    x_pool = test.x_flat[pool_idx]
    y_pool = test.y[pool_idx]

    source_wrapped = FlattenedInputWrapper(models_by_seed[source_seed])

    table_chunks = []
    for epsilon in epsilons:
        x_adv = _batched_fgsm(source_wrapped, x_pool, y_pool, epsilon, batch_size=batch_size)
        for eval_seed, model in models_by_seed.items():
            wrapped = FlattenedInputWrapper(model)
            with torch.no_grad():
                preds_adv = wrapped(x_adv).argmax(dim=1)
            is_misclassified = preds_adv != y_pool
            R_adv_euclidean = achieved_ratio(wrapped, x_pool, x_adv, distance_fn=euclidean_distance_fn)
            R_adv_mahalanobis = achieved_ratio(wrapped, x_pool, x_adv, distance_fn=mahalanobis_distance_fn)

            if verbose:
                print(f"  source_seed={source_seed}  eval_seed={eval_seed}  epsilon={epsilon:g}  "
                      f"transfer_pct_misclassified={is_misclassified.float().mean().item():.4f}")

            table_chunks.append(pd.DataFrame({
                "source_seed": source_seed, "eval_seed": eval_seed, "epsilon": epsilon,
                "pool_position": np.arange(pool_idx.shape[0]), "test_index": pool_idx.numpy(),
                "is_misclassified": is_misclassified.detach().numpy(),
                "R_adv_euclidean": R_adv_euclidean.detach().numpy(),
                "R_adv_mahalanobis": R_adv_mahalanobis.detach().numpy(),
            }))

    result = pd.concat(table_chunks, ignore_index=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(TRANSFER_TABLE_PATH, index=False)
    return result


def summarize_transfer_attack(transfer_df, save_path=TRANSFER_SUMMARY_PATH):
    """Aggregates `run_transfer_attack`'s per-example table into one row per
    `(source_seed, eval_seed, epsilon)`: `transfer_accuracy` (`1 - mean(is_misclassified)` -- the
    fraction of the pool `eval_seed`'s model STILL classifies correctly despite the
    `source_seed`-crafted perturbation; at `eval_seed == source_seed` this is exactly the ordinary
    adversarial accuracy, i.e. `1 - attack_success_rate` from Step 3), plus mean/p99 `R_adv` under
    both metrics (`eval_seed`'s own logits, matching `primary_r_adv_stats`'s p99-is-the-headline
    convention).

    Returns a `pd.DataFrame`, also saved to `save_path` (`None` to skip saving).
    """
    grouped = transfer_df.groupby(["source_seed", "eval_seed", "epsilon"]).agg(
        transfer_accuracy=("is_misclassified", lambda s: 1.0 - s.mean()),
        mean_R_adv_euclidean=("R_adv_euclidean", "mean"),
        p99_R_adv_euclidean=("R_adv_euclidean", lambda s: s.quantile(0.99)),
        mean_R_adv_mahalanobis=("R_adv_mahalanobis", "mean"),
        p99_R_adv_mahalanobis=("R_adv_mahalanobis", lambda s: s.quantile(0.99)),
    ).reset_index()

    if save_path is not None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        grouped.to_csv(save_path, index=False)
    return grouped


# ---------------------------------------------------------------------------
# Full pipeline driver -- ties Checkpoints 1-6's data-side functions together into one call, so
# `notebook_strongcnn_seed_sweep.ipynb` (Checkpoint 7) stays a thin visualization driver with no
# logic of its own, per this project's convention (see mnist_lipschitz/README.md's "notebooks are
# thin drivers" note).
# ---------------------------------------------------------------------------

def run_full_sweep(seeds=SEEDS, epsilons=EPSILONS, maha_epsilon=MAHALANOBIS_EPSILON,
                    shrinkage_epsilons=MAHALANOBIS_EPSILONS_SWEEP, force_retrain=False,
                    checkpoint_dir=RESULTS_DIR, n_query=1000, n_norm=1000, verbose=True):
    """Runs Checkpoints 1-6's data/statistics pipeline end to end (everything except the plots
    themselves, which the notebook calls directly on this function's return value).

    Returns a dict with every artifact a caller/plot needs: {"models_by_seed", "train", "test",
    "training_summary_df", "pool", "mahalanobis_distance_fn", "r_adv_df", "realized_norm_diagnostic_df",
    "shrinkage_df", "shrinkage_bounds_df" (L_full_estimated etc. at each shrinkage epsilon -- the
    companion `shrinkage_df` itself lacks, needed to assess whether Goal A's ratio-to-bound
    narrowing survives a different Mahalanobis epsilon, not just R_adv in isolation),
    "primary_stats_euclidean", "primary_stats_mahalanobis", "success_rate_df",
    "by_outcome_euclidean", "by_outcome_mahalanobis", "common_success_sets" (dict keyed by epsilon),
    "bounds_df", "project1_summary_df"}.
    """
    if verbose:
        print("=== Checkpoint 1: per-seed training ===")
    models_by_seed, test, train, training_summary_df = run_all_seed_trainings(
        seeds=seeds, force_retrain=force_retrain, checkpoint_dir=checkpoint_dir, verbose=verbose)

    if verbose:
        print("=== Checkpoint 2: common evaluation pool ===")
    pool = build_common_pool(models_by_seed, test, verbose=verbose)

    if verbose:
        print("=== Checkpoint 3: attacks + R_adv table ===")
    mahalanobis_distance_fn = fit_shared_mahalanobis_distance_fn(train, epsilon=maha_epsilon)
    r_adv_df, realized_norm_diagnostic_df = build_r_adv_table(
        models_by_seed, test, pool, mahalanobis_distance_fn, epsilons=epsilons, verbose=verbose)

    if verbose:
        print("=== Checkpoint 4: Mahalanobis shrinkage sensitivity ===")
    shrinkage_df = run_shrinkage_sensitivity(
        models_by_seed, test, train, pool, r_adv_df, epsilons_maha=shrinkage_epsilons,
        attack_epsilons=epsilons, verbose=verbose)
    shrinkage_bounds_df = shrinkage_bounds_per_seed(
        models_by_seed, train, test, pool, epsilons_maha=shrinkage_epsilons,
        n_query=n_query, n_norm=n_norm, verbose=verbose)

    if verbose:
        print("=== Checkpoint 5: statistics ===")
    primary_stats_euclidean = primary_r_adv_stats(r_adv_df, "R_adv_euclidean", "Euclidean")
    primary_stats_mahalanobis = primary_r_adv_stats(r_adv_df, "R_adv_mahalanobis", "Mahalanobis")
    success_rate_df = attack_success_rate(r_adv_df)
    by_outcome_euclidean = by_outcome_split(r_adv_df, "R_adv_euclidean", "Euclidean")
    by_outcome_mahalanobis = by_outcome_split(r_adv_df, "R_adv_mahalanobis", "Mahalanobis")
    common_success_sets = {epsilon: common_success_set(r_adv_df, epsilon) for epsilon in epsilons}
    bounds_df = per_seed_bounds(
        models_by_seed, train, test, pool,
        {"Euclidean": euclidean_distance_fn, "Mahalanobis": mahalanobis_distance_fn},
        n_query=n_query, n_norm=n_norm, verbose=verbose)
    project1_summary_df = build_project1_summary_table(training_summary_df, r_adv_df, bounds_df,
                                                        epsilons=epsilons)

    if verbose:
        print("=== run_full_sweep complete ===")
        print(project1_summary_df.to_string(index=False))

    return {
        "models_by_seed": models_by_seed, "train": train, "test": test,
        "training_summary_df": training_summary_df, "pool": pool,
        "mahalanobis_distance_fn": mahalanobis_distance_fn, "r_adv_df": r_adv_df,
        "realized_norm_diagnostic_df": realized_norm_diagnostic_df, "shrinkage_df": shrinkage_df,
        "shrinkage_bounds_df": shrinkage_bounds_df, "primary_stats_euclidean": primary_stats_euclidean,
        "primary_stats_mahalanobis": primary_stats_mahalanobis, "success_rate_df": success_rate_df,
        "by_outcome_euclidean": by_outcome_euclidean, "by_outcome_mahalanobis": by_outcome_mahalanobis,
        "common_success_sets": common_success_sets, "bounds_df": bounds_df,
        "project1_summary_df": project1_summary_df,
    }
