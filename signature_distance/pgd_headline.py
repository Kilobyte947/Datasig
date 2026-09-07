"""PGD counterpart to `headline_plot.py` + `headline_bootstrap.py`, built on
identical infrastructure - same statistic (P90 quantile of the full
`ratio_control`/`ratio_adv` distribution at the primary epsilon), same
bootstrap methodology (`headline_bootstrap.bootstrap_quantile_ci`: resample
IMAGES with replacement, recompute the quantile over each resample's full
(n_images, n_lines) array, take the percentile interval), same CI level,
and the same three subsets - Method B's 12-line informative-only headline
convention, Method B's all-16-line variant, and Method C's own all-16-
segment convention (it has no border exclusion of its own) - so FGSM and
PGD headline numbers are directly, apples-to-apples comparable, and so
Method B (12 vs. 16 lines) and Method C (always 16 segments) can be
compared on equal footing under both attacks.

EFFICIENCY NOTE: unlike `headline_plot.py`/`headline_bootstrap.py` (two
separate scripts that each independently retrain models, since this
codebase has no result caching), this module computes the quantile point
estimates AND their bootstrap CIs from a SINGLE call to
`pgd_adversarial_eval.run_pgd_comparison`, which itself already trains once
and evaluates both Method B and Method C against literally the same
PGD-perturbed images. This is not a methodology change (the bootstrap CI
computation itself is unmodified, reused directly from
`headline_bootstrap.bootstrap_quantile_ci`) - just a cheaper way to get
both outputs from data that's already fully computed, since bootstrap
resampling itself costs nothing beyond the one real training/attack run.
"""

from pathlib import Path

import torch

from signature_distance import pgd_adversarial_eval as pgd
from signature_distance.headline_bootstrap import CI_LEVEL, N_BOOTSTRAP, bootstrap_quantile_ci
from signature_distance.headline_plot import METHOD_B_INFORMATIVE_LINE_INDICES, PRIMARY_EPS, QUANTILE

RESULTS_DIR = Path(__file__).parent / "results"


def collect_pgd_headline(n_per_class: int = 20, epsilons=(0.02, 0.03, 0.05),
                          primary_eps: float = PRIMARY_EPS, quantile: float = QUANTILE,
                          seed: int = 0, cnn_epochs: int = 3, strong_epochs: int = 3,
                          pgd_steps: int = 10, n_bootstrap: int = N_BOOTSTRAP,
                          ci_level: float = CI_LEVEL, verbose: bool = True) -> dict:
    """Single PGD run (`pgd_adversarial_eval.run_pgd_comparison`, unmodified)
    -> P90 quantile point estimate AND bootstrap CI for Method B (12
    informative lines, and all 16 with no border exclusion) and Method C
    (always all 16 segments), at `primary_eps` - 6 CIs per model (2
    subsets x 2 conditions for Method B, 1 subset x 2 conditions for
    Method C), same structure/keys as
    `headline_bootstrap.collect_headline_bootstrap`'s output so the two are
    directly comparable and can share `headline_bootstrap.check_overlap`
    unmodified."""
    results = pgd.run_pgd_comparison(
        n_per_class=n_per_class, epsilons=epsilons, seed=seed,
        cnn_epochs=cnn_epochs, strong_epochs=strong_epochs, pgd_steps=pgd_steps, verbose=verbose,
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
        # headline_bootstrap.collect_headline_bootstrap, extended to 6
        # streams per model so none of them accidentally coincide.
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
    `headline_bootstrap.collect_headline_bootstrap`'s output;
    `pgd_data` is this module's `collect_pgd_headline`'s output - both use
    identical (model, subset, condition) keys by construction."""
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
