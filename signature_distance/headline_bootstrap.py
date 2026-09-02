"""Bootstrap confidence intervals on the headline plot's P90 local
Lipschitz estimates (`headline_plot.py`) - checks whether the
StrongCNN > SmallCNN gap reported there is statistically solid or within
sampling noise, given P90 is estimated from n=200 images (dominated by
roughly the top 20 values of each 200-image sample).

Reuses the SAME two unmodified FGSM drivers `headline_plot.collect_headline_data`
itself calls (`method_b_sweep.run_stage_b_validation`,
`hilbert_stream.run_hilbert_adversarial_eval`) with identical parameters/
seed - no new data generation or model training beyond what that function
already does (this codebase has no result caching, so "reusing" the
existing ratio distributions means rerunning the same deterministic
pipeline, not reading a cache; verified elsewhere in this project to be
bit-for-bit reproducible given a fixed seed). This module only adds the
resampling and interval computation on top of the resulting ratio arrays.

**Resampling unit: IMAGES, not individual (image, line) ratio values.**
Method B/C's per-image ratios (12 or 16 lines/segments per image) are
correlated within an image - same underlying perturbation, same image
content - not independent draws. Treating the flattened 2400/3200 values
as independent would understate the true sampling uncertainty. Each
bootstrap resample instead draws 200 image indices WITH replacement,
keeps each drawn image's full row of line/segment ratios, and recomputes
the P90 quantile over the resulting (200, n_lines) array - this reflects
the actual independent sample size (n=200 images), not the larger but
correlated flattened count.

CI level: 90%, matching the P90 quantile itself and the
`tpr_target=0.90` operating-point convention already used elsewhere in
this project, rather than introducing an unrelated new number.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from signature_distance import hilbert_stream
from signature_distance import method_b_sweep as sweep
from signature_distance.headline_plot import (
    METHOD_B_INFORMATIVE_LINE_INDICES,
    PRIMARY_EPS,
    QUANTILE,
    WINNER_FINALIST,
)

RESULTS_DIR = Path(__file__).parent / "results"

N_BOOTSTRAP = 1000
CI_LEVEL = 0.90


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
                                seed: int = 0, cnn_epochs: int = 3, strong_epochs: int = 3,
                                n_bootstrap: int = N_BOOTSTRAP, ci_level: float = CI_LEVEL,
                                verbose: bool = True) -> dict:
    """Reruns the same two FGSM drivers `headline_plot.collect_headline_data`
    calls, with identical parameters, to get the raw per-image ratio
    arrays, then computes a bootstrap CI for each of the 8 P90 values (2
    models x 2 conditions x 2 methods) already reported in the headline
    table."""
    stage_b = sweep.run_stage_b_validation(
        finalists=[WINNER_FINALIST], n_per_class=n_per_class, epsilons=epsilons,
        seed=seed, cnn_epochs=cnn_epochs, strong_epochs=strong_epochs, verbose=verbose,
    )
    method_b = stage_b["results"][WINNER_FINALIST["name"]]

    method_c = hilbert_stream.run_hilbert_adversarial_eval(
        depth=3, n_per_class=n_per_class, epsilons=epsilons, seed=seed,
        cnn_epochs=cnn_epochs, strong_epochs=strong_epochs, verbose=verbose,
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
        # of the 8 resampling runs accidentally share a random stream.
        base = seed * 1000 + model_num * 10
        out["models"][mname] = {
            "method_b": {
                "clean": bootstrap_quantile_ci(eb["ratio_control"][:, b_idx], quantile, n_bootstrap, ci_level, seed=base + 1),
                "adv": bootstrap_quantile_ci(eb["ratio_adv"][:, b_idx], quantile, n_bootstrap, ci_level, seed=base + 2),
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
    for method_key in ("method_b", "method_c"):
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


def plot_headline_ci(data: dict, title: str = None, save_path=None):
    """Two-panel figure (clean, adversarial), each a grouped bar chart of
    Method B's P90 point estimate +/- bootstrap CI for SmallCNN vs.
    StrongCNN, with Method C's point estimate +/- CI overlaid as an error-
    barred marker - same visual language (bars = Method B, markers =
    Method C) as `headline_plot.plot_headline_punchline`'s third panel,
    now with the sampling uncertainty made visible."""
    models = list(data["models"].keys())
    ci_pct = int(round(data["ci_level"] * 100))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, cond, cond_label in zip(axes, ("clean", "adv"), ("clean", "adversarial")):
        b_point = [data["models"][m]["method_b"][cond]["point_estimate"] for m in models]
        b_lo = [data["models"][m]["method_b"][cond]["ci_low"] for m in models]
        b_hi = [data["models"][m]["method_b"][cond]["ci_high"] for m in models]
        b_err = [[p - lo for p, lo in zip(b_point, b_lo)], [hi - p for p, hi in zip(b_point, b_hi)]]

        c_point = [data["models"][m]["method_c"][cond]["point_estimate"] for m in models]
        c_lo = [data["models"][m]["method_c"][cond]["ci_low"] for m in models]
        c_hi = [data["models"][m]["method_c"][cond]["ci_high"] for m in models]
        c_err = [[p - lo for p, lo in zip(c_point, c_lo)], [hi - p for p, hi in zip(c_point, c_hi)]]

        x = range(len(models))
        ax.bar(x, b_point, yerr=b_err, capsize=6, color="tab:blue", alpha=0.75, label="Method B")
        ax.errorbar(x, c_point, yerr=c_err, fmt="D", color="darkred", capsize=6, label="Method C")
        ax.set_xticks(list(x))
        ax.set_xticklabels(models)
        ax.set_ylabel("P90 local Lipschitz ratio")
        ax.set_title(f"{cond_label} (eps={data['primary_eps']})")
        ax.legend(fontsize=8)

    fig.suptitle(title or f"P90 local Lipschitz estimate with {ci_pct}% bootstrap CI ({data['n_bootstrap']} resamples)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
