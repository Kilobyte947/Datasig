"""Plotting for the adversarial-perturbation vs. Lipschitz-bound comparison.

Reuses `mnist_lipschitz.plots._maybe_save` (save-or-not-and-return-the-figure convention) rather
than duplicating it, and follows that module's palette/style: `tab:blue`/`tab:orange` for the two
things being compared (here, FGSM vs. PGD, mirroring its Euclidean-vs-Mahalanobis convention), and
a dashed reference line at the theoretical value a quantity is bounded by/against (mirroring
`plot_epsilon_sweep`'s `selected_epsilon` line and `plot_layer_decomposition_sweep`'s
`looseness_ratio=1` floor).
"""

import matplotlib.pyplot as plt
import numpy as np

from mnist_lipschitz.plots import _maybe_save

METHOD_COLORS = {"FGSM": "tab:blue", "PGD": "tab:orange"}


def plot_R_adv_distribution(sweep_results, L_full_estimated, product_bound,
                             metric_name="Euclidean", save_path=None):
    """Per-epsilon distribution of the achieved sensitivity ratio R_adv, FGSM vs. PGD side by
    side (box plots, one pair of boxes per epsilon), against the two Lipschitz bounds
    `run_bound_comparison`/`run_bound_comparison_with_distance_fn` computed for this same
    checkpoint: `L_full_estimated` (tight) and `product_bound` (loose), each a horizontal
    reference line -- the same "measured distribution vs. a theoretical reference value" layout
    used throughout this project's other plots.

    `sweep_results`: `run_experiment.run_epsilon_sweep`'s return value (specifically its
    `"per_case"` dict, keyed by `(epsilon, method) -> {"R_adv": Tensor, ...}`).

    `metric_name` (default `"Euclidean"`, matching `mnist_lipschitz.plots.plot_ratio_distribution`'s
    own parameter of the same name) only affects the title/axis wording -- pass `"Mahalanobis"`
    when `sweep_results` was computed with a Mahalanobis `distance_fn` (see
    `run_experiment.build_pixel_mahalanobis_distance_fn`), so the plot doesn't mislabel the
    distance actually used in the denominator.
    """
    per_case = sweep_results["per_case"]
    epsilons = sorted({eps for eps, _method in per_case.keys()})
    positions = list(range(len(epsilons)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    for offset, method in ((-width / 2, "FGSM"), (width / 2, "PGD")):
        data = [per_case[(eps, method)]["R_adv"].detach().cpu().numpy() for eps in epsilons]
        pos = [p + offset for p in positions]
        bp = ax.boxplot(data, positions=pos, widths=width * 0.9, patch_artist=True,
                         manage_ticks=False, showfliers=False)
        for box in bp["boxes"]:
            box.set_facecolor(METHOD_COLORS[method])
            box.set_alpha(0.6)
        for median in bp["medians"]:
            median.set_color("black")
        ax.plot([], [], color=METHOD_COLORS[method], alpha=0.6, linewidth=8, label=method)  # legend proxy

    ax.axhline(L_full_estimated, color="tab:red", linestyle="--", label="L_full_estimated (tight bound)")
    #ax.axhline(product_bound, color="black", linestyle=":", label="product_bound (loose bound)")

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{e:g}" for e in epsilons])
    ax.set_xlabel("epsilon (L_inf attack budget)")
    ax.set_ylabel(f"R_adv = ||f(x) - f(x_adv)||_2 / {metric_name.lower()}_distance(x, x_adv)")
    ax.set_title(f"Achieved adversarial sensitivity vs. the layer-decomposition Lipschitz bounds "
                 f"({metric_name} distance)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_bound_closeness_vs_width(combined_df, metric_name="Euclidean", save_path=None):
    """Two-panel summary across the CNN-width sweep (mirrors
    `mnist_lipschitz.plots.plot_layer_decomposition_sweep`'s two-panel layout for the same width
    axis): left panel is `max_R_adv / L_full_estimated` (the tight bound) vs. width, with a
    reference line at 1.0 -- the theoretical ceiling any achieved ratio must stay at or below,
    by definition (`summarize_epsilon_sweep`'s Requirement-5 sanity check enforces this
    directly). Right panel is the same normalized quantity against `product_bound` (the loose,
    submultiplicative bound), which has no such 1.0 ceiling in general -- shown for comparison,
    without a reference line, since the loose bound is not itself guaranteed to be approached
    the same way.

    `combined_df`: `run_experiment.run_cnn_adversarial_width_sweep`'s (or
    `run_cnn_adversarial_width_sweep_with_distance_fn`'s) combined (one-row-per-width) DataFrame
    -- needs `width`, `ratio_to_L_full_fgsm`/`_pgd`, and `ratio_to_product_bound_fgsm`/`_pgd`
    columns.

    `metric_name` (default `"Euclidean"`) only affects the suptitle wording -- pass
    `"Mahalanobis"` when `combined_df` came from a Mahalanobis-`distance_fn` width sweep.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.axhline(1.0, color="black", linestyle="--", label="L_full_estimated (ratio=1)")
    ax1.plot(combined_df["width"], combined_df["ratio_to_L_full_fgsm"], marker="o",
              color=METHOD_COLORS["FGSM"], label="FGSM")
    ax1.plot(combined_df["width"], combined_df["ratio_to_L_full_pgd"], marker="s",
              color=METHOD_COLORS["PGD"], label="PGD")
    ax1.set_xlabel("CNN width")
    ax1.set_ylabel("max_R_adv / L_full_estimated")
    ax1.set_title("Closeness to the tight bound (L_full) vs. width")
    ax1.legend(fontsize=8)

    ax2.plot(combined_df["width"], combined_df["ratio_to_product_bound_fgsm"], marker="o",
              color=METHOD_COLORS["FGSM"], label="FGSM")
    ax2.plot(combined_df["width"], combined_df["ratio_to_product_bound_pgd"], marker="s",
              color=METHOD_COLORS["PGD"], label="PGD")
    ax2.set_xlabel("CNN width")
    ax2.set_ylabel("max_R_adv / product_bound")
    ax2.set_title("Closeness to the loose bound (product) vs. width")
    ax2.legend(fontsize=8)

    fig.suptitle(f"Adversarial achieved sensitivity vs. capacity: closeness to the Lipschitz "
                 f"bounds ({metric_name} distance)")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_extreme_examples(most_sensitive, least_sensitive, width=None, metric_name="Euclidean",
                           save_path=None):
    """Visualizes the two examples from `run_experiment.most_and_least_sensitive_examples`: the
    single attacked point that achieved the LARGEST R_adv (top row) and the single one that
    achieved the SMALLEST R_adv (bottom row) for one checkpoint. Each row shows the clean image,
    the adversarial image, and their pixel-space absolute difference, labeled with the true
    class, the model's prediction on each, and the epsilon/method/R_adv that produced it.

    The left column gets an x-axis label with `pixel_distance` (`distance_fn(x, x_adv)` --
    `||x - x_adv||_2` if Euclidean, whatever `distance_fn` was passed to
    `most_and_least_sensitive_examples` otherwise -- the INITIAL distance in raw pixel space,
    before either image reaches the extractor) if present on the example dict. If
    `head_layer_bound_check`'s keys (`feature_distance`, `L_head_exact`, `head_bound`,
    `actual_logit_distance`, `head_bound_tightness`) are also present (merged in by
    `run_cnn_adversarial_width_sweep`/`run_cnn_adversarial_width_sweep_with_distance_fn`), the
    middle column gets an x-axis label comparing the Euclidean distance between the two
    extracted-feature vectors (right after the pixel-space distance shown on the left, right
    before the head -- ALWAYS Euclidean regardless of `metric_name`, see
    `run_experiment.compute_bounds_with_distance_fn`'s docstring for why) against the head
    layer's own exact Lipschitz bound on how far that distance can push the logits -- together
    the two labels trace the full pixel -> feature -> logit distance chain for this one example.
    Both are omitted gracefully (no crash) if a caller passes examples without that extra info.

    `most_sensitive`/`least_sensitive`: dicts with at least the keys
    `most_and_least_sensitive_examples` returns (`x`, `x_adv`, `y_true`, `pred_clean`,
    `pred_adv`, `epsilon`, `method`, `R_adv`) -- no model call happens here, purely display of
    already-computed values, matching this module's other plotting functions.

    `metric_name` (default `"Euclidean"`) only affects the title wording -- pass `"Mahalanobis"`
    when these examples came from a Mahalanobis-`distance_fn` sweep, so the (always-Euclidean)
    `feature_distance` label in the middle column isn't confused with the (metric-dependent)
    `pixel_distance` label on the left.
    """
    fig, axes = plt.subplots(2, 3, figsize=(9, 7.5))

    for row, (row_label, example) in enumerate((("Largest R_adv", most_sensitive),
                                                  ("Smallest R_adv", least_sensitive))):
        x = example["x"].detach().cpu().numpy().reshape(28, 28)
        x_adv = example["x_adv"].detach().cpu().numpy().reshape(28, 28)
        diff = np.abs(x_adv - x)

        axes[row, 0].imshow(x, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"clean\ntrue={example['y_true']} pred={example['pred_clean']}", fontsize=9)
        axes[row, 1].imshow(x_adv, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(f"adversarial\npred={example['pred_adv']}", fontsize=9)
        axes[row, 2].imshow(diff, cmap="gray")
        axes[row, 2].set_title("|x_adv - x|", fontsize=9)

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(
            f"{row_label}\nR_adv={example['R_adv']:.3f}\n"
            f"eps={example['epsilon']:g} {example['method']}", fontsize=8)

        if "pixel_distance" in example:
            axes[row, 0].set_xlabel(f"pixel dist.={example['pixel_distance']:.3f}", fontsize=7)

        if "feature_distance" in example:
            axes[row, 1].set_xlabel(
                f"feature dist.={example['feature_distance']:.3f}  L_head={example['L_head_exact']:.3f}\n"
                f"head_bound={example['head_bound']:.3f}  actual={example['actual_logit_distance']:.3f} "
                f"({example['head_bound_tightness']:.1%} of bound)", fontsize=7)

    title = f"Most/least sensitive attacked example ({metric_name} distance)"
    if width is not None:
        title += f", width={width}"
    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.6, top=0.90)
    _maybe_save(fig, save_path)
    return fig


def plot_euclidean_vs_mahalanobis_R_adv(euclidean_summary_df, mahalanobis_summary_df,
                                         stat="mean_R_adv", save_path=None):
    """Grouped-bar comparison of an R_adv summary statistic (default `mean_R_adv`) across the
    epsilon sweep, Euclidean vs. Mahalanobis distance, one panel per attack method (FGSM/PGD) --
    mirrors `mnist_lipschitz.plots.plot_euclidean_vs_mahalanobis`'s own grouped-bar-per-submethod
    convention (`tab:blue`=Euclidean, `tab:orange`=Mahalanobis), applied here to this
    sub-experiment's `(epsilon, method)` summary tables instead of that module's three
    Lipschitz-estimator sub-methods.

    `euclidean_summary_df`/`mahalanobis_summary_df`: `summarize_epsilon_sweep`'s per-(epsilon,
    method) tables (from `run_bound_comparison`/`run_bound_comparison_with_distance_fn`
    respectively) for the SAME checkpoint -- must share the same set of epsilons for the grouped
    bars to line up meaningfully.
    `stat`: which summary column to plot (`"mean_R_adv"`, `"median_R_adv"`, or `"max_R_adv"`).
    """
    methods = ("FGSM", "PGD")
    fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 5))

    for ax, method in zip(axes, methods):
        eucl = euclidean_summary_df[euclidean_summary_df["method"] == method].sort_values("epsilon")
        maha = mahalanobis_summary_df[mahalanobis_summary_df["method"] == method].sort_values("epsilon")
        epsilons = eucl["epsilon"].tolist()
        x = range(len(epsilons))
        width = 0.35

        ax.bar([xi - width / 2 for xi in x], eucl[stat], width, label="Euclidean", color="tab:blue")
        ax.bar([xi + width / 2 for xi in x], maha[stat], width, label="Mahalanobis", color="tab:orange")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{e:g}" for e in epsilons])
        ax.set_xlabel("epsilon (L_inf attack budget)")
        ax.set_ylabel(stat)
        ax.set_title(method)
        ax.legend(fontsize=8)

    fig.suptitle(f"Achieved adversarial sensitivity ({stat}): Euclidean vs. Mahalanobis distance")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_euclidean_vs_mahalanobis_bounds_vs_width(euclidean_combined_df, mahalanobis_combined_df,
                                                    save_path=None):
    """Extends `plot_bound_closeness_vs_width`'s two-panel (tight bound / loose bound) layout with
    a second dimension: solid lines for Euclidean, dashed for Mahalanobis (the same
    solid/dashed-for-a-second-dimension convention `mnist_lipschitz.plots.plot_layer_decomposition_
    sweep` uses for exact-vs-estimated), so the SAME question `plot_bound_closeness_vs_width` asks
    per metric ("does the gap between achieved sensitivity and the theoretical bound narrow or
    widen with capacity?") can be compared directly ACROSS metrics on one pair of axes.

    `euclidean_combined_df`/`mahalanobis_combined_df`: `run_cnn_adversarial_width_sweep`'s and
    `run_cnn_adversarial_width_sweep_with_distance_fn`'s combined (one-row-per-width) DataFrames
    respectively, for the SAME widths/checkpoints -- both need `width`, `ratio_to_L_full_fgsm`/
    `_pgd`, and `ratio_to_product_bound_fgsm`/`_pgd` columns.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.axhline(1.0, color="black", linestyle="--", label="ratio=1", linewidth=1)
    for df, metric_label, linestyle in ((euclidean_combined_df, "Euclidean", "-"),
                                         (mahalanobis_combined_df, "Mahalanobis", "--")):
        ax1.plot(df["width"], df["ratio_to_L_full_fgsm"], marker="o", color=METHOD_COLORS["FGSM"],
                  linestyle=linestyle, label=f"FGSM ({metric_label})")
        ax1.plot(df["width"], df["ratio_to_L_full_pgd"], marker="s", color=METHOD_COLORS["PGD"],
                  linestyle=linestyle, label=f"PGD ({metric_label})")
    ax1.set_xlabel("CNN width")
    ax1.set_ylabel("max_R_adv / L_full_estimated")
    ax1.set_title("Closeness to the tight bound (L_full) vs. width")
    ax1.legend(fontsize=7)

    for df, metric_label, linestyle in ((euclidean_combined_df, "Euclidean", "-"),
                                         (mahalanobis_combined_df, "Mahalanobis", "--")):
        ax2.plot(df["width"], df["ratio_to_product_bound_fgsm"], marker="o", color=METHOD_COLORS["FGSM"],
                  linestyle=linestyle, label=f"FGSM ({metric_label})")
        ax2.plot(df["width"], df["ratio_to_product_bound_pgd"], marker="s", color=METHOD_COLORS["PGD"],
                  linestyle=linestyle, label=f"PGD ({metric_label})")
    ax2.set_xlabel("CNN width")
    ax2.set_ylabel("max_R_adv / product_bound")
    ax2.set_title("Closeness to the loose bound (product) vs. width")
    ax2.legend(fontsize=7)

    fig.suptitle("Adversarial achieved sensitivity vs. capacity: Euclidean vs. Mahalanobis distance")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig
