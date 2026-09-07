"""Headline P90 local-Lipschitz-estimate analysis: point estimates, bootstrap
confidence intervals, and FGSM-vs-PGD comparison, across SmallCNN and
StrongCNN, Method B (winning 16h+0v/depth=2 config) and Method C (Hilbert).

This is the exact analysis named in the project brief's stated research
angle: "A nice final plot would show clean accuracy, adversarial accuracy,
and a high quantile of the local Lipschitz estimate across models. The
punchline could be: models with similar test accuracy may have very
different extension behaviour." (Plotting itself lives in `plots.py` -
`plot_headline_punchline`, `plot_headline_ci` - this module only computes
the underlying data.)

Assembles existing results rather than reimplementing anything: Method B's
winning configuration is evaluated via `method_b_sweep.run_stage_b_validation`
(unmodified, single finalist), Method C via
`adversarial_eval.run_hilbert_adversarial_eval` (unmodified) - both
already-existing, already-validated FGSM drivers - plus
`adversarial_eval.run_pgd_comparison` for the PGD side.

Two choices worth stating plainly rather than leaving implicit:

- **Quantile = 90th percentile.** Matches the operating-point convention
  already established elsewhere in this project (`distances.
  run_per_line_auc_diagnostic`'s default `tpr_target=0.90`), rather than
  introducing a new one.
- **"Clean" vs "adversarial" Lipschitz estimate = the ratio_control vs
  ratio_adv distributions already computed by the existing adversarial-
  eval drivers**, at the primary epsilon (0.03, the epsilon already used
  as the representative one elsewhere in this project's visualizations).
  There is no notion of a Lipschitz *ratio* without some perturbation (the
  numerator is a margin change under a perturbation) - `ratio_control`
  (margin change under undirected, magnitude-matched random noise) is
  this project's existing stand-in for "the local Lipschitz estimate on
  clean data", and `ratio_adv` (margin change under FGSM/PGD) is its
  adversarially-directed counterpart. Computed over the FULL per-path
  ratio distribution (every image x every line/segment, not just the
  genuinely-flipped subset used for the fold-ratio numbers reported
  elsewhere) - this is a different, complementary statistic: the general
  local-sensitivity distribution at a given perturbation magnitude, not
  one conditioned on attack success.
- **Method B's headline quantile excludes its 2 structurally border-adjacent
  lines** (`distances.METHOD_B_BORDER_LINE_INDICES` - the first/last
  horizontal line of the 16h+0v winning geometry, which sit exactly on the
  image border and have near-zero baseline signature distance regardless
  of perturbation) - a high quantile is exactly where such outliers would
  otherwise dominate. This is a DIFFERENT subset than the all-16-line
  mean-fold numbers reported elsewhere - stated explicitly here rather
  than left to cause confusion between the two. Method C's quantile uses
  all 16 segments, unchanged from how its own numbers were already
  reported (Stage A found no structurally degenerate segment). Every
  `collect_*` function below also reports an all-16-line Method B variant
  (`method_b_all16`) alongside the informative-lines headline one, closing
  the Method B (14) vs. Method C (16) subset-count asymmetry for this
  metric specifically, computed from the exact same already-fetched ratio
  tensors (no extra training/attack cost) - see `compare_line_counts`
  below.
- **Resampling unit (bootstrap CIs): IMAGES, not individual (image, line)
  ratio values.** Method B/C's per-image ratios (12 or 16 lines/segments
  per image) are correlated within an image - same underlying perturbation,
  same image content - not independent draws. Treating the flattened
  2400/3200 values as independent would understate the true sampling
  uncertainty. Each bootstrap resample instead draws 200 image indices WITH
  replacement, keeps each drawn image's full row of line/segment ratios,
  and recomputes the P90 quantile over the resulting (200, n_lines) array -
  this reflects the actual independent sample size (n=200 images), not the
  larger but correlated flattened count.
- **CI level: 90%**, matching the P90 quantile itself and the
  `tpr_target=0.90` operating-point convention already used elsewhere in
  this project, rather than introducing an unrelated new number.
"""

from pathlib import Path

import torch

from signature_distance import adversarial_eval as ae
from signature_distance import method_b_sweep as sweep
from signature_distance.distances import METHOD_B_INFORMATIVE_LINE_INDICES

torch.set_default_dtype(torch.float64)

RESULTS_DIR = Path(__file__).parent / "results"

PRIMARY_EPS = 0.03
QUANTILE = 0.90
N_BOOTSTRAP = 1000
CI_LEVEL = 0.90

WINNER_FINALIST = {
    "name": "best_combo_16h0v_depth2",
    "angles_deg": sweep.GEOMETRY_VARIANTS["16h+0v"][0],
    "counts": sweep.GEOMETRY_VARIANTS["16h+0v"][1],
    "points_per_line": 32, "depth": 2, "interpolation": "linear",
}


# ---------------------------------------------------------------------------
# Point estimates (FGSM)
# ---------------------------------------------------------------------------


def collect_headline_data(n_per_class: int = 20, epsilons=(0.02, 0.03, 0.05), primary_eps: float = PRIMARY_EPS,
                           quantile: float = QUANTILE, seed: int = 0, verbose: bool = True) -> dict:
    """Runs Method B's winning-config FGSM evaluation
    (`method_b_sweep.run_stage_b_validation`, unmodified) and Method C's
    FGSM evaluation (`adversarial_eval.run_hilbert_adversarial_eval`,
    unmodified) and assembles, per model: clean test accuracy,
    FGSM adversarial accuracy at every swept epsilon, and the `quantile`
    of the ratio_control/ratio_adv distributions at `primary_eps` for
    both methods. Both underlying drivers load the shared canonical
    checkpoint (no training)."""
    stage_b = sweep.run_stage_b_validation(
        finalists=[WINNER_FINALIST], n_per_class=n_per_class, epsilons=epsilons,
        seed=seed, verbose=verbose,
    )
    method_b = stage_b["results"][WINNER_FINALIST["name"]]

    method_c = ae.run_hilbert_adversarial_eval(
        depth=3, n_per_class=n_per_class, epsilons=epsilons, seed=seed, verbose=verbose,
    )

    b_idx = torch.tensor(METHOD_B_INFORMATIVE_LINE_INDICES)
    data = {"epsilons": list(epsilons), "primary_eps": primary_eps, "quantile": quantile, "models": {}}

    for mname in method_b["models"]:
        eb = method_b["models"][mname]
        ec = method_c["models"][mname]
        assert abs(eb["test_acc"] - ec["test_acc"]) < 1e-6, (
            f"{mname}: Method B and Method C test accuracies differ - expected identical "
            f"given matched seed/params (same deterministic training pipeline)."
        )

        adv_acc_by_eps = {}
        for eps in epsilons:
            fb = eb["eps"][eps]["flip_mask"]
            fc = ec["eps"][eps]["flip_mask"]
            assert torch.equal(fb, fc), (
                f"{mname} eps={eps}: Method B and Method C flip masks differ - expected "
                f"identical (same model, same FGSM attack, matched seed/params)."
            )
            adv_acc_by_eps[eps] = 1.0 - fb.float().mean().item()

        pe_b = eb["eps"][primary_eps]
        pe_c = ec["eps"][primary_eps]

        ratio_control_b = pe_b["ratio_control"][:, b_idx].flatten()
        ratio_adv_b = pe_b["ratio_adv"][:, b_idx].flatten()
        # All 16 lines, no border exclusion - same underlying tensors as
        # above, just not sliced to the informative subset first.
        ratio_control_b_all16 = pe_b["ratio_control"].flatten()
        ratio_adv_b_all16 = pe_b["ratio_adv"].flatten()
        ratio_control_c = pe_c["ratio_control"].flatten()
        ratio_adv_c = pe_c["ratio_adv"].flatten()

        data["models"][mname] = {
            "clean_test_acc": eb["test_acc"],
            "adv_acc_by_eps": adv_acc_by_eps,
            "method_b": {
                "clean_quantile": torch.quantile(ratio_control_b, quantile).item(),
                "adv_quantile": torch.quantile(ratio_adv_b, quantile).item(),
            },
            "method_b_all16": {
                "clean_quantile": torch.quantile(ratio_control_b_all16, quantile).item(),
                "adv_quantile": torch.quantile(ratio_adv_b_all16, quantile).item(),
            },
            "method_c": {
                "clean_quantile": torch.quantile(ratio_control_c, quantile).item(),
                "adv_quantile": torch.quantile(ratio_adv_c, quantile).item(),
            },
        }

    return data


def compare_line_counts(data: dict) -> dict:
    """Method B's P90 quantile using its informative lines vs. all 16
    (no border exclusion), per model and condition - same underlying ratio
    arrays as `collect_headline_data` already computed, so this is a pure
    aggregation-level comparison, no extra computation. Answers directly
    whether including the known-degenerate border lines actually shifts
    a tail statistic the way excluding them was meant to prevent (rather
    than just assuming it would)."""
    out = {}
    for mname, entry in data["models"].items():
        out[mname] = {}
        for cond in ("clean_quantile", "adv_quantile"):
            informative = entry["method_b"][cond]
            sixteen = entry["method_b_all16"][cond]
            out[mname][cond] = {
                "informative_line": informative, "all_16_line": sixteen,
                "delta": sixteen - informative,
                "pct_change": (sixteen - informative) / informative * 100 if informative else float("nan"),
            }
    return out


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals (FGSM)
# ---------------------------------------------------------------------------


def bootstrap_quantile_ci(ratio_matrix: torch.Tensor, quantile: float = QUANTILE,
                           n_bootstrap: int = N_BOOTSTRAP, ci_level: float = CI_LEVEL,
                           seed: int = 0) -> dict:
    """ratio_matrix: (n_images, n_lines) - one row per image, its per-
    line/segment ratios. Resamples IMAGES (rows) with replacement
    `n_bootstrap` times, recomputes the `quantile` quantile over each
    resample's full (n_images, n_lines) array, and returns the
    `ci_level`-level percentile interval of the resulting bootstrap
    distribution, plus the point estimate on the original (unresampled)
    data."""
    n_images = ratio_matrix.shape[0]
    generator = torch.Generator().manual_seed(seed)

    point_estimate = torch.quantile(ratio_matrix.flatten(), quantile).item()

    boot_estimates = torch.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = torch.randint(0, n_images, (n_images,), generator=generator)
        boot_estimates[b] = torch.quantile(ratio_matrix[idx].flatten(), quantile)

    alpha = 1.0 - ci_level
    lo = torch.quantile(boot_estimates, alpha / 2).item()
    hi = torch.quantile(boot_estimates, 1.0 - alpha / 2).item()

    return {
        "point_estimate": point_estimate, "ci_low": lo, "ci_high": hi,
        "ci_level": ci_level, "n_bootstrap": n_bootstrap, "n_images": n_images,
        "boot_std": boot_estimates.std().item(),
    }


def collect_headline_bootstrap(n_per_class: int = 20, epsilons=(0.02, 0.03, 0.05),
                                primary_eps: float = PRIMARY_EPS, quantile: float = QUANTILE,
                                seed: int = 0, n_bootstrap: int = N_BOOTSTRAP, ci_level: float = CI_LEVEL,
                                verbose: bool = True) -> dict:
    """Reruns the same two FGSM drivers `collect_headline_data` calls, with
    identical parameters, to get the raw per-image ratio arrays, then
    computes a bootstrap CI for each of the 12 P90 values (2 models x 2
    conditions x 3 subsets: Method B's 12-informative-line headline
    convention, Method B's all-16-line variant - closing the subset-count
    asymmetry with Method C, no extra training/attack cost, same already-
    fetched tensors - and Method C's own all-16-segment convention)."""
    stage_b = sweep.run_stage_b_validation(
        finalists=[WINNER_FINALIST], n_per_class=n_per_class, epsilons=epsilons,
        seed=seed, verbose=verbose,
    )
    method_b = stage_b["results"][WINNER_FINALIST["name"]]

    method_c = ae.run_hilbert_adversarial_eval(
        depth=3, n_per_class=n_per_class, epsilons=epsilons, seed=seed, verbose=verbose,
    )

    b_idx = torch.tensor(METHOD_B_INFORMATIVE_LINE_INDICES)
    out = {
        "primary_eps": primary_eps, "quantile": quantile, "ci_level": ci_level,
        "n_bootstrap": n_bootstrap, "models": {},
    }

    for model_num, mname in enumerate(method_b["models"]):
        eb = method_b["models"][mname]["eps"][primary_eps]
        ec = method_c["models"][mname]["eps"][primary_eps]

        # Distinct but fully deterministic bootstrap seed per (model, method,
        # condition) so the whole run is reproducible end to end, and no two
        # of the 12 resampling runs accidentally share a random stream.
        base = seed * 1000 + model_num * 10
        out["models"][mname] = {
            "method_b": {
                "clean": bootstrap_quantile_ci(eb["ratio_control"][:, b_idx], quantile, n_bootstrap, ci_level, seed=base + 1),
                "adv": bootstrap_quantile_ci(eb["ratio_adv"][:, b_idx], quantile, n_bootstrap, ci_level, seed=base + 2),
            },
            "method_b_all16": {
                "clean": bootstrap_quantile_ci(eb["ratio_control"], quantile, n_bootstrap, ci_level, seed=base + 5),
                "adv": bootstrap_quantile_ci(eb["ratio_adv"], quantile, n_bootstrap, ci_level, seed=base + 6),
            },
            "method_c": {
                "clean": bootstrap_quantile_ci(ec["ratio_control"], quantile, n_bootstrap, ci_level, seed=base + 3),
                "adv": bootstrap_quantile_ci(ec["ratio_adv"], quantile, n_bootstrap, ci_level, seed=base + 4),
            },
        }

    return out


def check_overlap(data: dict) -> dict:
    """For each method/condition pair, checks whether SmallCNN's and
    StrongCNN's bootstrap CIs overlap. Non-overlapping intervals is
    reasonably strong evidence the gap is not sampling noise; overlapping
    intervals means this specific check cannot rule out sampling noise as
    the explanation - it does NOT by itself prove there is no real gap."""
    small = data["models"]["SmallCNN"]
    strong = data["models"]["StrongCNN"]
    out = {}
    for method_key in ("method_b", "method_b_all16", "method_c"):
        if method_key not in small:
            continue
        out[method_key] = {}
        for cond in ("clean", "adv"):
            s = small[method_key][cond]
            g = strong[method_key][cond]
            overlap = not (s["ci_high"] < g["ci_low"] or g["ci_high"] < s["ci_low"])
            out[method_key][cond] = {
                "small_point": s["point_estimate"], "small_ci": (s["ci_low"], s["ci_high"]),
                "strong_point": g["point_estimate"], "strong_ci": (g["ci_low"], g["ci_high"]),
                "overlap": overlap,
            }
    return out


# ---------------------------------------------------------------------------
# PGD counterpart - built on identical infrastructure (same statistic, same
# bootstrap methodology, same CI level, same three subsets) so FGSM and PGD
# headline numbers are directly, apples-to-apples comparable.
#
# EFFICIENCY NOTE: unlike the FGSM collectors above (which each independently
# retrain models, since this codebase has no result caching), this pair
# computes the quantile point estimates AND their bootstrap CIs from a
# SINGLE call to `adversarial_eval.run_pgd_comparison`, which itself already
# trains once and evaluates both Method B and Method C against literally the
# same PGD-perturbed images. Not a methodology change (the bootstrap CI
# computation itself is unmodified) - just a cheaper way to get both outputs
# from data that's already fully computed, since bootstrap resampling itself
# costs nothing beyond the one real training/attack run.
# ---------------------------------------------------------------------------


def collect_pgd_headline(n_per_class: int = 20, epsilons=(0.02, 0.03, 0.05),
                          primary_eps: float = PRIMARY_EPS, quantile: float = QUANTILE,
                          seed: int = 0, pgd_steps: int = 10, n_bootstrap: int = N_BOOTSTRAP,
                          ci_level: float = CI_LEVEL, verbose: bool = True) -> dict:
    """Single PGD run (`adversarial_eval.run_pgd_comparison`, unmodified,
    canonical checkpoint - no training) -> P90 quantile point estimate AND
    bootstrap CI for Method B (12 informative lines, and all 16 with no
    border exclusion) and Method C (always all 16 segments), at
    `primary_eps` - 6 CIs per model (2 subsets x 2 conditions for Method B,
    1 subset x 2 conditions for Method C), same structure/keys as
    `collect_headline_bootstrap`'s output so the two are directly
    comparable and can share `check_overlap` unmodified."""
    results = ae.run_pgd_comparison(
        n_per_class=n_per_class, epsilons=epsilons, seed=seed,
        pgd_steps=pgd_steps, verbose=verbose,
    )

    b_idx = torch.tensor(METHOD_B_INFORMATIVE_LINE_INDICES)
    out = {
        "primary_eps": primary_eps, "quantile": quantile, "ci_level": ci_level,
        "n_bootstrap": n_bootstrap, "pgd_steps": pgd_steps, "models": {},
    }

    model_names = list(results["method_b"]["models"])
    for model_num, mname in enumerate(model_names):
        eb = results["method_b"]["models"][mname]["eps"][primary_eps]
        ec = results["method_c"]["models"][mname]["eps"][primary_eps]

        # Same deterministic-seed-per-(model,subset,condition) discipline as
        # collect_headline_bootstrap, extended to 6 streams per model so
        # none of them accidentally coincide.
        base = seed * 1000 + model_num * 10
        out["models"][mname] = {
            "test_acc": results["method_b"]["models"][mname]["test_acc"],
            "flip_fraction": eb["flip_fraction"],
            "fgsm_flip_fraction": eb["fgsm_flip_fraction"],
            "method_b": {
                "clean": bootstrap_quantile_ci(eb["ratio_control"][:, b_idx], quantile, n_bootstrap, ci_level, seed=base + 1),
                "adv": bootstrap_quantile_ci(eb["ratio_adv"][:, b_idx], quantile, n_bootstrap, ci_level, seed=base + 2),
            },
            "method_b_all16": {
                "clean": bootstrap_quantile_ci(eb["ratio_control"], quantile, n_bootstrap, ci_level, seed=base + 5),
                "adv": bootstrap_quantile_ci(eb["ratio_adv"], quantile, n_bootstrap, ci_level, seed=base + 6),
            },
            "method_c": {
                "clean": bootstrap_quantile_ci(ec["ratio_control"], quantile, n_bootstrap, ci_level, seed=base + 3),
                "adv": bootstrap_quantile_ci(ec["ratio_adv"], quantile, n_bootstrap, ci_level, seed=base + 4),
            },
        }

    return out


def compare_attacks(fgsm_data: dict, pgd_data: dict) -> dict:
    """Side-by-side FGSM vs. PGD comparison of the point estimate (and CI)
    for every (model, subset, condition) combination both `collect_*`
    functions share - lets the two attacks be read off one table instead
    of two separately-printed ones. `fgsm_data` is
    `collect_headline_bootstrap`'s output; `pgd_data` is
    `collect_pgd_headline`'s output - both use identical (model, subset,
    condition) keys by construction."""
    out = {}
    for mname in pgd_data["models"]:
        out[mname] = {}
        for subset in ("method_b", "method_b_all16", "method_c"):
            out[mname][subset] = {}
            for cond in ("clean", "adv"):
                f = fgsm_data["models"][mname][subset][cond]
                p = pgd_data["models"][mname][subset][cond]
                out[mname][subset][cond] = {
                    "fgsm_point": f["point_estimate"], "fgsm_ci": (f["ci_low"], f["ci_high"]),
                    "pgd_point": p["point_estimate"], "pgd_ci": (p["ci_low"], p["ci_high"]),
                    "pgd_stronger": p["point_estimate"] > f["point_estimate"],
                }
    return out
