"""Headline punchline plot: clean accuracy, adversarial accuracy, and a
high quantile of the local Lipschitz estimate, across SmallCNN and
StrongCNN.

This is the exact plot named in the project brief's stated research
angle: "A nice final plot would show clean accuracy, adversarial
accuracy, and a high quantile of the local Lipschitz estimate across
models. The punchline could be: models with similar test accuracy may
have very different extension behaviour."

Assembles existing results rather than reimplementing anything: Method
B's winning configuration is evaluated via
`method_b_sweep.run_stage_b_validation` (unmodified, single finalist),
Method C via `hilbert_stream.run_hilbert_adversarial_eval` (unmodified) -
both already-existing, already-validated FGSM drivers. No PGD dependency
(out of scope for this task; a PGD version can follow once wanted). This
module only adds the quantile computation, the accuracy/quantile table
assembly, and the plot itself.

Two choices worth stating plainly rather than leaving implicit:

- **Quantile = 90th percentile.** Matches the operating-point convention
  already established elsewhere in this project (`per_line_diagnostics.
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
  clean data", and `ratio_adv` (margin change under FGSM) is its
  adversarially-directed counterpart. Computed over the FULL per-path
  ratio distribution (every image x every line/segment, not just the
  genuinely-flipped subset used for the fold-ratio numbers reported
  elsewhere) - this is a different, complementary statistic: the general
  local-sensitivity distribution at a given perturbation magnitude, not
  one conditioned on attack success.
- **Method B's quantile excludes its 4 structurally border-adjacent lines
  (indices 0, 11, 12, 15 for the 12h+4v winning geometry - the first/last
  horizontal and vertical line, which sit exactly on the image border and
  have near-zero baseline signature distance regardless of perturbation).**
  This matches the reasoning already used elsewhere in this project for
  excluding the analogous 4 border lines of the original 8h+8v geometry
  (`per_path_adversarial_eval.BORDER_LINE_INDICES`/`spike_analysis`'s own
  note that their near-zero-denominator ratios are "numerically
  degenerate ... not meaningful") - a high quantile is exactly where such
  outliers would otherwise dominate. This is a DIFFERENT subset than the
  all-16-line mean-fold numbers already reported for Method B elsewhere
  (13.53x FGSM / 16.97x PGD) - stated explicitly here rather than left to
  cause confusion between the two. Method C's quantile uses all 16
  segments, unchanged from how its own numbers were already reported
  (Stage A found no structurally degenerate segment).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from signature_distance import hilbert_stream
from signature_distance import method_b_sweep as sweep

RESULTS_DIR = Path(__file__).parent / "results"

PRIMARY_EPS = 0.03
QUANTILE = 0.90

WINNER_FINALIST = {
    "name": "best_combo_12h4v_depth2",
    "angles_deg": sweep.GEOMETRY_VARIANTS["12h+4v"][0],
    "counts": sweep.GEOMETRY_VARIANTS["12h+4v"][1],
    "points_per_line": 32, "depth": 2, "interpolation": "linear",
}
METHOD_B_BORDER_LINE_INDICES = (0, 11, 12, 15)
METHOD_B_INFORMATIVE_LINE_INDICES = tuple(i for i in range(16) if i not in METHOD_B_BORDER_LINE_INDICES)


def collect_headline_data(n_per_class: int = 20, epsilons=(0.02, 0.03, 0.05), primary_eps: float = PRIMARY_EPS,
                           quantile: float = QUANTILE, seed: int = 0, cnn_epochs: int = 3, strong_epochs: int = 3,
                           verbose: bool = True) -> dict:
    """Runs Method B's winning-config FGSM evaluation
    (`method_b_sweep.run_stage_b_validation`, unmodified) and Method C's
    FGSM evaluation (`hilbert_stream.run_hilbert_adversarial_eval`,
    unmodified) and assembles, per model: clean test accuracy,
    FGSM adversarial accuracy at every swept epsilon, and the `quantile`
    of the ratio_control/ratio_adv distributions at `primary_eps` for
    both methods."""
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
        ratio_control_c = pe_c["ratio_control"].flatten()
        ratio_adv_c = pe_c["ratio_adv"].flatten()

        data["models"][mname] = {
            "clean_test_acc": eb["test_acc"],
            "adv_acc_by_eps": adv_acc_by_eps,
            "method_b": {
                "clean_quantile": torch.quantile(ratio_control_b, quantile).item(),
                "adv_quantile": torch.quantile(ratio_adv_b, quantile).item(),
            },
            "method_c": {
                "clean_quantile": torch.quantile(ratio_control_c, quantile).item(),
                "adv_quantile": torch.quantile(ratio_adv_c, quantile).item(),
            },
        }

    return data


def plot_headline_punchline(data: dict, title: str = None, save_path=None):
    """Three-panel presentation figure: clean test accuracy, FGSM
    adversarial accuracy at `data['primary_eps']`, and the
    `data['quantile']`-quantile local Lipschitz estimate (Method B bars,
    Method C as an overlaid marker), grouped by model - all three panels
    share the model x-axis so the two models line up across panels.
    Every bar/marker is labeled with its actual value."""
    models = list(data["models"].keys())
    eps = data["primary_eps"]
    q_pct = int(round(data["quantile"] * 100))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: clean accuracy
    clean_acc = [data["models"][m]["clean_test_acc"] * 100 for m in models]
    bars = axes[0].bar(models, clean_acc, color="tab:green")
    for bar, v in zip(bars, clean_acc):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=10)
    axes[0].set_ylabel("accuracy (%)")
    axes[0].set_title("Clean test accuracy")
    axes[0].set_ylim(0, 105)

    # Panel 2: adversarial accuracy at the primary epsilon
    adv_acc = [data["models"][m]["adv_acc_by_eps"][eps] * 100 for m in models]
    bars = axes[1].bar(models, adv_acc, color="tab:red")
    for bar, v in zip(bars, adv_acc):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=10)
    axes[1].set_ylabel("accuracy (%)")
    axes[1].set_title(f"FGSM adversarial accuracy (eps={eps})")
    axes[1].set_ylim(0, 105)

    # Panel 3: high-quantile local Lipschitz estimate (Method B bars, Method C markers)
    x = range(len(models))
    width = 0.35
    clean_q = [data["models"][m]["method_b"]["clean_quantile"] for m in models]
    adv_q = [data["models"][m]["method_b"]["adv_quantile"] for m in models]
    clean_q_c = [data["models"][m]["method_c"]["clean_quantile"] for m in models]
    adv_q_c = [data["models"][m]["method_c"]["adv_quantile"] for m in models]

    bars_clean = axes[2].bar([i - width / 2 for i in x], clean_q, width, label="clean (Method B)", color="tab:blue")
    bars_adv = axes[2].bar([i + width / 2 for i in x], adv_q, width, label="adversarial (Method B)", color="tab:orange")
    for bar, v in zip(bars_clean, clean_q):
        axes[2].text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    for bar, v in zip(bars_adv, adv_q):
        axes[2].text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    axes[2].scatter([i - width / 2 for i in x], clean_q_c, marker="D", color="navy", zorder=3, label="clean (Method C)")
    axes[2].scatter([i + width / 2 for i in x], adv_q_c, marker="D", color="darkred", zorder=3, label="adversarial (Method C)")

    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(models)
    axes[2].set_ylabel(f"P{q_pct} local Lipschitz ratio")
    axes[2].set_title(f"P{q_pct} local Lipschitz estimate (eps={eps})")
    axes[2].legend(fontsize=7, loc="upper left")

    fig.suptitle(title or "Clean accuracy, adversarial accuracy, and high-quantile local Lipschitz estimate")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
