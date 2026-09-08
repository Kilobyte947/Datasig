"""Local-Lipschitz-estimate analysis: point estimates, bootstrap confidence
intervals, and FGSM-vs-PGD comparison, across SmallCNN and StrongCNN, Method B, and Method C.

Bootstrap resampling draws images (not individual per-line ratio values) with replacement, since
a single image's per-line ratios are correlated, not independent draws — resampling flattened
values would understate the true sampling uncertainty. Quantile is the 90th percentile, matching
this project's other operating-point conventions; CI level is 90%.
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
    """Runs Method B's and Method C's FGSM evaluations and assembles, per model: clean test accuracy,
    adversarial accuracy at every swept epsilon, and the P90 quantile of the control and adversarial
    ratio distributions at the primary epsilon, for both methods."""
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
    """Method B's P90 quantile using its informative lines vs all 16 lines, per model and condition,
    from the same underlying ratio arrays."""
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
    """Bootstrap confidence interval for a quantile of ratio_matrix (n_images, n_lines): resamples
    images with replacement n_bootstrap times, recomputing the quantile each time. Returns the point
    estimate, ci_low, ci_high, and the bootstrap distribution's std."""
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
    """Bootstrap CI for each P90 value collect_headline_data computes: Method B's informative-line
    and all-16-line subsets, and Method C's all-16-segment convention, per model and condition."""
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
    """For each method/condition pair, checks whether SmallCNN's and StrongCNN's bootstrap CIs
    overlap. Non-overlapping intervals are reasonably strong evidence the gap isn't sampling noise;
    overlapping intervals mean this check can't rule sampling noise out."""
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
# headline numbers are directly comparable.
#
# NOTE: Unlike the FGSM collectors above (which each independently
# retrain models, since this codebase has no result caching), this pair
# computes the quantile point estimates and their bootstrap CIs from a
# single call to `adversarial_eval.run_pgd_comparison`, which itself already
# trains once and evaluates both Method B and Method C against literally the
# same PGD-perturbed images.
# ---------------------------------------------------------------------------


def collect_pgd_headline(n_per_class: int = 20, epsilons=(0.02, 0.03, 0.05),
                          primary_eps: float = PRIMARY_EPS, quantile: float = QUANTILE,
                          seed: int = 0, pgd_steps: int = 10, n_bootstrap: int = N_BOOTSTRAP,
                          ci_level: float = CI_LEVEL, verbose: bool = True) -> dict:
    """Same statistic and bootstrap methodology as collect_headline_bootstrap, computed from a single
    PGD run instead of FGSM."""
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
    """Side-by-side FGSM vs PGD comparison of the point estimate and CI for every (model, subset,
    condition) combination."""
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
