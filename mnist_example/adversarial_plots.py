"""Plotting for the adversarial-perturbation vs. Lipschitz-bound comparison."""

import matplotlib.pyplot as plt
import numpy as np

from mnist_example.plots import _maybe_save

METHOD_COLORS = {"FGSM": "blue", "PGD": "orange"}


def plot_R_adv_distribution(sweep_results, L_full_estimated, product_bound,
                             metric_name="Euclidean", save_path=None):
    """Plot the distribution of R_adv across the epsilon sweep, one boxplot per epsilon, one boxplot
    per method (FGSM/PGD) per epsilon, with a horizontal line for the tight bound (L_full_estimated)
    and a horizontal line for the loose bound (product_bound)."""

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

    ax.axhline(L_full_estimated, color="red", linestyle="--", label="L_full_estimated (tight bound)")

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


def plot_R_adv_histogram_by_outcome(sweep_results, epsilon, method="FGSM", metric_name="Euclidean",
                                     bins=30, save_path=None):
    """Plot the distribution of R_adv for one (epsilon, method) pair, split by whether the attack
    succeeded in misclassifying the image or not (red = misclassified, green = not misclassified),
    with a single histogram binning across both outcomes (so the two histograms are stacked, not
    overlapping)."""
    case = sweep_results["per_case"][(epsilon, method)]
    R_adv = case["R_adv"].detach().cpu().numpy()
    is_misclassified = case["is_misclassified"].detach().cpu().numpy()

    correct, misclassified = R_adv[~is_misclassified], R_adv[is_misclassified]
    bin_edges = np.histogram_bin_edges(R_adv, bins=bins)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist([correct, misclassified], bins=bin_edges, stacked=True,
            color=["green", "red"],
            label=[f"not misclassified (n={len(correct)})", f"misclassified (n={len(misclassified)})"])
    ax.set_xlabel(f"R_adv = ||f(x) - f(x_adv)||_2 / {metric_name.lower()}_distance(x, x_adv)")
    ax.set_ylabel("number of images")
    ax.set_title(f"R_adv distribution by attack outcome (epsilon={epsilon:g}, {method}, "
                 f"{metric_name} distance)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_bound_closeness_vs_width(combined_df, metric_name="Euclidean", save_path=None):
    """Plot the closeness of the achieved adversarial sensitivity (max_R_adv) to the tight bound
    (L_full_estimated) and the loose bound (product_bound) vs. CNN width, one panel per bound.
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
    """Plot the most and least sensitive examples from `run_experiment.find_examples_by_criteria` 
    -- the two rows of the figure are the most and least sensitive examples, each with three 
    columns: clean image, adversarial image, and absolute difference.
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


def plot_example_pair(example, metric_name="Euclidean", save_path=None):
    """Plot a single example pair (clean and adversarial) with titles showing 
    the true label, predicted labels, R_adv, epsilon, method, and pixel distance.
    """
    x = example["x"].detach().cpu().numpy().reshape(28, 28)
    x_adv = example["x_adv"].detach().cpu().numpy().reshape(28, 28)

    fig, axes = plt.subplots(1, 2, figsize=(5, 3))
    axes[0].imshow(x, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"clean\ntrue={example['y_true']} pred={example['pred_clean']}", fontsize=9)
    axes[1].imshow(x_adv, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"adversarial\npred={example['pred_adv']}", fontsize=9)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"R_adv={example['R_adv']:.3f}  eps={example['epsilon']:g} {example['method']}  "
                 f"{metric_name.lower()}_distance={example['pixel_distance']:.3f}", fontsize=9)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_euclidean_vs_mahalanobis_R_adv(euclidean_summary_df, mahalanobis_summary_df,
                                         stat="mean_R_adv", save_path=None):
    """Plot the mean or max R_adv (or any other statistic) vs. epsilon, one panel per method (FGSM/PGD),
    with two bars per epsilon (Euclidean vs. Mahalanobis). 
    """
    methods = ("FGSM", "PGD")
    fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 5))

    for ax, method in zip(axes, methods):
        eucl = euclidean_summary_df[euclidean_summary_df["method"] == method].sort_values("epsilon")
        maha = mahalanobis_summary_df[mahalanobis_summary_df["method"] == method].sort_values("epsilon")
        epsilons = eucl["epsilon"].tolist()
        x = range(len(epsilons))
        width = 0.35

        ax.bar([xi - width / 2 for xi in x], eucl[stat], width, label="Euclidean", color="blue")
        ax.bar([xi + width / 2 for xi in x], maha[stat], width, label="Mahalanobis", color="orange")
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
    """Plot the closeness of the achieved adversarial sensitivity (max_R_adv) to the tight bound
    (L_full_estimated) and the loose bound (product_bound) vs. CNN width, one panel per bound,
    with two lines per panel (Euclidean vs. Mahalanobis). 
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


# ---------------------------------------------------------------------------
# Multi-seed confirmation sweep (seed_sweep.py)
# ---------------------------------------------------------------------------

WIDTH_COLORS = {16: "blue", 32: "orange", 64: "green"}


def plot_misclassification_vs_width_with_spread(per_config_df, metric_name="Euclidean", save_path=None):
    """ Plot the mean misclassification rate vs. width, one panel per method (FGSM/PGD), with error bars
    showing the standard deviation across seeds. Each epsilon is a separate line within each panel. 
    """
    sub = per_config_df[per_config_df["metric"] == metric_name]
    methods = [m for m in ("FGSM", "PGD") if m in sub["method"].unique()]
    epsilons = sorted(sub["epsilon"].unique())
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 5), squeeze=False)
    axes = axes[0]

    for ax, method in zip(axes, methods):
        method_sub = sub[sub["method"] == method]
        for i, epsilon in enumerate(epsilons):
            eps_sub = method_sub[method_sub["epsilon"] == epsilon].sort_values("width")
            color = cmap(i / max(len(epsilons) - 1, 1))
            ax.errorbar(eps_sub["width"], eps_sub["misclassification_rate_mean"],
                        yerr=eps_sub["misclassification_rate_std"], marker="o", capsize=3,
                        color=color, label=f"epsilon={epsilon:g}")
        ax.set_xlabel("CNN width")
        ax.set_ylabel("misclassification_rate (mean +/- std across seeds)")
        ax.set_title(method)
        ax.legend(fontsize=8)

    fig.suptitle(f"Misclassification rate vs. width, across seeds ({metric_name} distance)")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_L_full_vs_misclassification(df, metric_name="Euclidean", save_path=None):
    """ Scatter of the achieved misclassification rate vs. the full-logit-vector Lipschitz constant
    (L_full_estimated), one point per (seed, width), colored by width.
    """
    sub = df[df["metric"] == metric_name]
    fig, ax = plt.subplots(figsize=(7, 5.5))

    for width in sorted(sub["width"].unique()):
        width_sub = sub[sub["width"] == width]
        ax.scatter(width_sub["misclassification_rate"], width_sub["L_full_estimated"],
                   color=WIDTH_COLORS.get(width, "gray"), alpha=0.6, label=f"width={width}")

    ax.set_xlabel("misclassification_rate")
    ax.set_ylabel("L_full_estimated")
    ax.set_title(f"L_full_estimated vs. achieved misclassification rate, all seeds ({metric_name} distance)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_margin_vs_full_lipschitz(df, metric_name="Euclidean", save_path=None):
    """ Scatter of the margin-functional Lipschitz constant (L_margin_estimated) vs. the
    full-logit-vector Lipschitz constant (L_full_estimated), one point per (seed, width),
    colored by width. The diagonal line is the equality line (L_margin = L_full).
    """
    sub = df[df["metric"] == metric_name].drop_duplicates(subset=["seed", "width"])
    fig, ax = plt.subplots(figsize=(6.5, 6))

    lims = [
        min(sub["L_margin_estimated"].min(), sub["L_full_estimated"].min()),
        max(sub["L_margin_estimated"].max(), sub["L_full_estimated"].max()),
    ]
    ax.plot(lims, lims, color="black", linestyle="--", linewidth=1, label="L_margin = L_full")

    for width in sorted(sub["width"].unique()):
        width_sub = sub[sub["width"] == width]
        ax.scatter(width_sub["L_full_estimated"], width_sub["L_margin_estimated"],
                   color=WIDTH_COLORS.get(width, "gray"), s=60, label=f"width={width}")

    ax.set_xlabel("L_full_estimated (full logit vector)")
    ax.set_ylabel("L_margin_estimated (scalar margin functional)")
    ax.set_title(f"Margin-functional vs. full-logit Lipschitz constant, per (seed, width) "
                 f"({metric_name} distance)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_bound_closeness_vs_width_with_spread(per_config_df, epsilon=None, metric_name="Euclidean",
                                                save_path=None):
    """ Plot the closeness of the achieved adversarial sensitivity (max_R_adv) to the tight bound
    (L_full_estimated) and the loose bound (product_bound) vs. CNN width (one panel per bound), 
    with error bars showing the standard deviation across seeds.
    """
    sub = per_config_df[per_config_df["metric"] == metric_name]
    if epsilon is None:
        epsilon = sub["epsilon"].max()
    sub = sub[sub["epsilon"] == epsilon]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.axhline(1.0, color="black", linestyle="--", label="L_full_estimated (ratio=1)")
    for method, marker in (("FGSM", "o"), ("PGD", "s")):
        method_sub = sub[sub["method"] == method].sort_values("width")
        ax1.errorbar(method_sub["width"], method_sub["ratio_to_L_full_mean"],
                     yerr=method_sub["ratio_to_L_full_std"], marker=marker, capsize=3,
                     color=METHOD_COLORS[method], label=method)
        ax2.errorbar(method_sub["width"], method_sub["ratio_to_product_bound_mean"],
                     yerr=method_sub["ratio_to_product_bound_std"], marker=marker, capsize=3,
                     color=METHOD_COLORS[method], label=method)

    ax1.set_xlabel("CNN width")
    ax1.set_ylabel("max_R_adv / L_full_estimated (mean +/- std across seeds)")
    ax1.set_title("Closeness to the tight bound (L_full) vs. width")
    ax1.legend(fontsize=8)

    ax2.set_xlabel("CNN width")
    ax2.set_ylabel("max_R_adv / product_bound (mean +/- std across seeds)")
    ax2.set_title("Closeness to the loose bound (product) vs. width")
    ax2.legend(fontsize=8)

    fig.suptitle(f"Adversarial achieved sensitivity vs. capacity, across seeds "
                 f"(epsilon={epsilon:g}, {metric_name} distance)")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# StrongCNN multi-seed sweep (strong_cnn_seed_sweep.py)
# ---------------------------------------------------------------------------

SEED_COLORS = {0: "blue", 1: "orange", 2: "green", 3: "red", 4: "purple"}


def plot_strong_cnn_seed_sweep_r_adv_outcome_grid(df, save_path=None):
    """ Plot the distribution of R_adv for each (seed, epsilon, metric) combination, split by whether
    the attack succeeded in misclassifying the image or not (red = misclassified, green = not misclassified)
    """
    seeds = sorted(df["seed"].unique())
    epsilons = sorted(df["epsilon"].unique())
    metrics = (("R_adv_euclidean", "Euclidean"), ("R_adv_mahalanobis", "Mahalanobis"))
    cols = [(eps, col, name) for eps in epsilons for col, name in metrics]

    fig, axes = plt.subplots(len(seeds), len(cols), figsize=(4 * len(cols), 3 * len(seeds)),
                              squeeze=False)

    for row, seed in enumerate(seeds):
        for c, (epsilon, col, metric_name) in enumerate(cols):
            ax = axes[row][c]
            case = df[(df["seed"] == seed) & (df["epsilon"] == epsilon)]
            R = case[col].to_numpy()
            is_misclassified = case["is_misclassified"].to_numpy()
            correct, misclassified = R[~is_misclassified], R[is_misclassified]
            bin_edges = np.histogram_bin_edges(R, bins=20)
            ax.hist([correct, misclassified], bins=bin_edges, stacked=True,
                    color=["green", "red"])
            if row == 0:
                ax.set_title(f"epsilon={epsilon:g}\n{metric_name}", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"seed={seed}\ncount", fontsize=8)

    fig.suptitle("StrongCNN seed sweep: R_adv distribution by attack outcome "
                 "(green=correct, red=misclassified)")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_strong_cnn_seed_sweep_cross_seed_overlay(df, save_path=None):
    """ Plot the distribution of R_adv for each (epsilon, metric) combination, overlaid across seeds
    (one line per seed, colored by seed), to show the variability of R_adv across seeds for the same epsilon and metric.
    """
    epsilons = sorted(df["epsilon"].unique())
    metrics = (("R_adv_euclidean", "Euclidean"), ("R_adv_mahalanobis", "Mahalanobis"))
    seeds = sorted(df["seed"].unique())

    fig, axes = plt.subplots(len(epsilons), len(metrics),
                              figsize=(6 * len(metrics), 4.5 * len(epsilons)), squeeze=False)

    for row, epsilon in enumerate(epsilons):
        for col_idx, (col, metric_name) in enumerate(metrics):
            ax = axes[row][col_idx]
            eps_df = df[df["epsilon"] == epsilon]
            bin_edges = np.histogram_bin_edges(eps_df[col].to_numpy(), bins=30)
            for seed in seeds:
                values = eps_df[eps_df["seed"] == seed][col].to_numpy()
                ax.hist(values, bins=bin_edges, histtype="step", density=True, linewidth=1.5,
                        color=SEED_COLORS.get(seed, "gray"), label=f"seed={seed}")
            ax.set_xlabel(f"R_adv ({metric_name.lower()} distance)")
            ax.set_ylabel("density")
            ax.set_title(f"epsilon={epsilon:g}, {metric_name}")
            ax.legend(fontsize=7)

    fig.suptitle("StrongCNN seed sweep: unconditional R_adv across seeds (identical common pool)")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_strong_cnn_seed_sweep_euclidean_vs_mahalanobis(primary_stats_df, bounds_df, stat="p99",
                                                          save_path=None):
    """ Plot the mean or max R_adv (or any other statistic) vs. epsilon, one panel per epsilon, 
    with two bars per seed (Euclidean vs. Mahalanobis), and a horizontal line for the tight 
    bound (L_full_estimated) for each seed and metric.
    """
    epsilons = sorted(primary_stats_df["epsilon"].unique())
    seeds = sorted(primary_stats_df["seed"].unique())
    width = 0.35

    fig, axes = plt.subplots(1, len(epsilons), figsize=(6 * len(epsilons), 5), squeeze=False)
    axes = axes[0]

    for ax, epsilon in zip(axes, epsilons):
        eps_df = primary_stats_df[primary_stats_df["epsilon"] == epsilon]
        x = np.arange(len(seeds))
        for offset, metric_name, color in ((-width / 2, "Euclidean", "blue"),
                                            (width / 2, "Mahalanobis", "orange")):
            metric_df = eps_df[eps_df["metric"] == metric_name].set_index("seed").loc[seeds]
            ax.bar(x + offset, metric_df[stat], width, label=f"{metric_name} ({stat})", color=color, alpha=0.7)

            bounds_metric = bounds_df[bounds_df["metric"] == metric_name].set_index("seed").loc[seeds]
            ax.scatter(x + offset, bounds_metric["L_full_estimated"], marker="_", s=400,
                       color="black", linewidths=2,
                       label=f"L_full_estimated ({metric_name})" if ax is axes[0] else None)

        ax.set_xticks(x)
        ax.set_xticklabels([f"seed={s}" for s in seeds])
        ax.set_ylabel(f"{stat}(R_adv)")
        ax.set_title(f"epsilon={epsilon:g}")
        ax.legend(fontsize=7)

    fig.suptitle(f"StrongCNN seed sweep: {stat}(R_adv) vs. L_full_estimated, "
                 f"Euclidean vs. Mahalanobis, per seed")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_strong_cnn_seed_sweep_summary_table(summary_df, save_path=None):
    """ Plot a summary table of the key statistics from the StrongCNN seed sweep
    """
    fig, ax = plt.subplots(figsize=(1.6 * len(summary_df.columns), 0.6 * (len(summary_df) + 1)))
    ax.axis("off")
    display_df = summary_df.round(4)
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns,
                      cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    fig.suptitle("StrongCNN seed sweep: summary (Project 1 punchline)")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Transferability check (strong_cnn_seed_sweep.run_transfer_attack/summarize_transfer_attack)
# ---------------------------------------------------------------------------

def plot_transfer_attack(summary_df, source_seed, save_path=None):
    """ Plot the transferability of adversarial examples crafted against 
    one seed's model and evaluated on every seed's own model.
    """
    sub_all = summary_df[summary_df["source_seed"] == source_seed]
    epsilons = sorted(sub_all["epsilon"].unique())

    fig, axes = plt.subplots(2, len(epsilons), figsize=(6 * len(epsilons), 9), squeeze=False)

    for col, epsilon in enumerate(epsilons):
        sub = sub_all[sub_all["epsilon"] == epsilon].sort_values("eval_seed")
        eval_seeds = sub["eval_seed"].tolist()

        ax_acc = axes[0][col]
        colors = [SEED_COLORS.get(s, "gray") for s in eval_seeds]
        bars = ax_acc.bar([str(s) for s in eval_seeds], sub["transfer_accuracy"], color=colors)
        for bar, seed in zip(bars, eval_seeds):
            if seed == source_seed:
                bar.set_edgecolor("black")
                bar.set_linewidth(2.5)
        ax_acc.set_xlabel("eval_seed")
        ax_acc.set_ylabel("transfer_accuracy")
        ax_acc.set_title(f"epsilon={epsilon:g}")

        ax_sens = axes[1][col]
        width = 0.35
        x = list(range(len(sub)))
        ax_sens.bar([xi - width / 2 for xi in x], sub["p99_R_adv_euclidean"], width,
                    label="Euclidean", color="blue")
        ax_sens.bar([xi + width / 2 for xi in x], sub["p99_R_adv_mahalanobis"], width,
                    label="Mahalanobis", color="orange")
        ax_sens.set_xticks(x)
        ax_sens.set_xticklabels([str(s) for s in eval_seeds])
        ax_sens.set_xlabel("eval_seed")
        ax_sens.set_ylabel("p99(R_adv)")
        ax_sens.legend(fontsize=8)

    fig.suptitle(f"Transfer attack: adversarial examples crafted against seed={source_seed}, "
                 f"evaluated on every seed's own model (black outline = source seed itself)")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig
