"""All plotting functions for the MNIST Lipschitz experiment. 
Every function returns the created `matplotlib.figure.Figure` and optionally saves it to `save_path`. 
No plotting logic should live anywhere else -- matches toy_lipschitz/plots.py's convention.
"""

import matplotlib.pyplot as plt
import numpy as np

MODEL_ORDER = ("logistic_regression", "mlp", "cnn")
MODEL_LABELS = {"logistic_regression": "Logistic\nRegression", "mlp": "MLP", "cnn": "CNN"}

def _maybe_save(fig, save_path):
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_euclidean_vs_mahalanobis(euclidean_results, mahalanobis_results, save_path=None):
    """Three models x three sub-methods x two metrics (Euclidean, Mahalanobis)"""

    submethods = [("pairwise", "Pairwise"), ("local_max", "Local-perturbation (max)"), ("grad_max", "Gradient-norm (max)")]
    models = [m for m in MODEL_ORDER if m in euclidean_results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = range(len(models))
    width = 0.35

    for ax, (key, title) in zip(axes, submethods):
        euclidean_vals = [euclidean_results[m][key] for m in models]
        mahalanobis_vals = [mahalanobis_results[m][key] for m in models]

        ax.bar([xi - width / 2 for xi in x], euclidean_vals, width, label="Euclidean", color="tab:blue")
        ax.bar([xi + width / 2 for xi in x], mahalanobis_vals, width, label="Mahalanobis", color="tab:orange")

        ax.set_xticks(list(x))
        ax.set_xticklabels([MODEL_LABELS[m] for m in models])
        ax.set_title(title)
        ax.set_ylabel("L_hat")
        ax.legend(fontsize=8)

    fig.suptitle("Lipschitz estimate: Euclidean vs. Mahalanobis distance, by model and sub-method")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_epsilon_sweep(epsilon_values, cond_numbers, cv_values, selected_epsilon=None, save_path=None):
    """Condition number and subsample instability (coefficient of variation) both plotted against epsilon (log scale)"""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epsilon_values, cond_numbers, marker="o", color="tab:blue", label="cond(Sigma + eps*I)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("epsilon (log scale)")
    ax.set_ylabel("condition number (log scale)", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax.twinx()
    ax2.plot(epsilon_values, cv_values, marker="s", color="tab:red", label="coefficient of variation")
    ax2.set_ylabel("subsample instability (std/mean)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    if selected_epsilon is not None:
        ax.axvline(selected_epsilon, color="gray", linestyle="--", label=f"selected epsilon={selected_epsilon:g}")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax.set_title("Epsilon selection: conditioning vs. subsample stability")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_submethod_agreement(results, metric_name, save_path=None):
    """For each model, show the three sub-method estimates side by side on a log y-axis (the three live on very different natural scales).
    The validity check made visible, given there's no L* to plot as a reference line here."""

    models = [m for m in MODEL_ORDER if m in results]
    submethod_keys = ["pairwise", "local_max", "grad_max"]
    submethod_labels = ["pairwise", "local-pert.", "grad-norm"]

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        vals = [results[model][k] for k in submethod_keys]
        bars = ax.bar(submethod_labels, vals, color=["tab:blue", "tab:red", "tab:green"])
        ax.set_yscale("log")
        ax.set_title(MODEL_LABELS[model].replace("\n", " "))
        ax.set_ylabel("L_hat (log scale)")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Sub-method agreement per model ({metric_name} distance)")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_covariance_eigenvalues(eigenvalues, epsilon=None, save_path=None):
    """Eigenvalue spectrum of the pixel covariance matrix, optionally with epsilon-regularized version overlaid.
    If `epsilon` is given, also plots the ridge-regularized spectrum `eigenvalues + epsilon` (same eigenvectors 
    as Sigma, just a uniform shift) so it's directly visible how much epsilon lifts the near-zero tail while 
    barely touching the large eigenvalues.
    """
    eigenvalues = eigenvalues.detach().cpu().numpy() if hasattr(eigenvalues, "detach") else np.asarray(eigenvalues)
    eigenvalues_plot = np.clip(eigenvalues, a_min=1e-12, a_max=None)  # avoid log(0)/log(negative) from fp noise
    index = np.arange(1, len(eigenvalues) + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(index, eigenvalues_plot, color="tab:blue", label="eigenvalues of Sigma")

    if epsilon is not None:
        ax.plot(index, eigenvalues_plot + epsilon, color="tab:orange", linestyle="--",
                label=f"eigenvalues of Sigma + {epsilon:g}*I")
        ax.axhline(epsilon, color="gray", linestyle=":", label=f"epsilon={epsilon:g}")

    ax.set_yscale("log")
    ax.set_xlabel("eigenvalue rank (descending)")
    ax.set_ylabel("eigenvalue magnitude (log scale)")
    ax.set_title("Eigenvalue spectrum of the pixel covariance matrix")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_ratio_distribution(ratio_dist_results, metric_name="Euclidean", save_path=None):
    """For each model, show the distribution of pairwise ratios for all pairs vs. nearest-neighbor pairs, side by side."""
    models = [m for m in MODEL_ORDER if m in ratio_dist_results]
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    bins = 50
    for ax, model in zip(axes, models):
        all_r = np.asarray(ratio_dist_results[model]["all_pairs_ratio"])
        near_r = np.asarray(ratio_dist_results[model]["near_neighbor_ratio"])
        ax.hist(all_r, bins=bins, density=True, alpha=0.6, color="tab:blue",
                label=f"all pairs (n={len(all_r)})")
        ax.hist(near_r, bins=bins, density=True, alpha=0.6, color="tab:orange",
                label=f"nearest-neighbor pairs (n={len(near_r)})")
        ax.set_xlabel("ratio: |margin_i - margin_j| / distance(x_i, x_j)")
        ax.set_title(MODEL_LABELS[model].replace("\n", " "))
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")

    fig.suptitle(f"Pairwise ratio distribution: all pairs vs. nearest-neighbor pairs ({metric_name} distance)")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_ratio_distribution_euclidean_vs_mahalanobis(euclidean_ratio_results, mahalanobis_ratio_results, save_path=None):
    """Grouped bar chart comparing the three summary statistics of the pairwise ratio distribution (mean of all pairs, mean of nearest-neighbor pairs, 
    max of nearest-neighbor pairs) for each model and each of the two distance metrics (Euclidean, Mahalanobis)."""
    stats = [("all_pairs_mean", "All-pairs mean ratio"),
             ("near_neighbor_mean", "Near-neighbor mean ratio"),
             ("near_neighbor_max", "Near-neighbor max ratio")]
    models = [m for m in MODEL_ORDER if m in euclidean_ratio_results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = range(len(models))
    width = 0.35

    for ax, (key, title) in zip(axes, stats):
        euclidean_vals = [euclidean_ratio_results[m]["summary"][key] for m in models]
        mahalanobis_vals = [mahalanobis_ratio_results[m]["summary"][key] for m in models]

        ax.bar([xi - width / 2 for xi in x], euclidean_vals, width, label="Euclidean", color="tab:blue")
        ax.bar([xi + width / 2 for xi in x], mahalanobis_vals, width, label="Mahalanobis", color="tab:orange")

        ax.set_xticks(list(x))
        ax.set_xticklabels([MODEL_LABELS[m] for m in models])
        ax.set_title(title)
        ax.set_ylabel("ratio")
        ax.legend(fontsize=8)

    fig.suptitle("Ratio-distribution summary: Euclidean vs. Mahalanobis distance, by model")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_image_pairs(pairs, save_path=None):
    """Given a list of image pairs and their associated true/predicted labels and pairwise Lipschitz ratio, plot the first 6 pairs in a grid."""
    n = min(len(pairs), 6)
    # figsize/dpi chosen for a compact inline notebook rendering (a 6-pair
    # gallery at the old figsize=(5, 2.5*n) with no explicit dpi ran to
    # ~15in tall on screen) -- savefig below still writes a crisp dpi=150
    # PNG regardless of this figure's own (lower) on-screen dpi.
    fig, axes = plt.subplots(n, 2, figsize=(4, 1.8 * n), dpi=80)
    axes = np.atleast_2d(axes)

    for row, (img1, img2, true1, pred1, true2, pred2, ratio, *_rest) in enumerate(pairs[:n]):
        for col, (img, true_l, pred_l) in enumerate([(img1, true1, pred1), (img2, true2, pred2)]):
            ax = axes[row, col]
            ax.imshow(np.asarray(img).reshape(28, 28), cmap="gray")
            ax.set_title(f"true={true_l} pred={pred_l}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(f"ratio={ratio:.3g}", fontsize=9)

    fig.suptitle("Image pairs by pairwise Lipschitz ratio")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_pair_diagnostic_gallery(pairs, title=None, save_path=None):
    """Diagnostic gallery for a hand-picked list of image pairs: extends
    `plot_image_pairs`'s image + true/pred + ratio format with a third
    panel per row showing the pixel-level absolute difference between the
    pair, on its own color scale so a small difference is still visible
    (not squashed to near-black against the [0,1] image range) -- built to
    visually rule out a data/preprocessing artefact (near-duplicate
    images, a normalization bug, an indexing bug pairing a point with
    itself or a corrupted copy) before treating a high-ratio pair as a
    genuine model-confusion finding.

    `pairs`: list of dicts, each with `img1`/`img2` (flat 784 or (28,28)
    arrays), `true1`/`pred1`/`true2`/`pred2`, `ratio`/`dist`/`margin_diff`,
    and optionally `confidence1`/`confidence2` (softmax probability of the
    predicted class) -- included in the label when present; omitted
    (rather than shown as a fabricated number) when not, e.g. when no
    trained-model checkpoint was available to compute them.
    """
    n = len(pairs)
    # figsize/dpi chosen for a compact inline notebook rendering (a 6-pair
    # gallery at the old figsize=(8, 2.6*n) with no explicit dpi ran to
    # ~15.6in tall on screen) -- savefig below still writes a crisp dpi=150
    # PNG regardless of this figure's own (lower) on-screen dpi.
    fig, axes = plt.subplots(n, 3, figsize=(6, 1.8 * n), dpi=80)
    axes = np.atleast_2d(axes)

    for row, p in enumerate(pairs):
        img1 = np.asarray(p["img1"]).reshape(28, 28)
        img2 = np.asarray(p["img2"]).reshape(28, 28)
        diff = np.abs(img1 - img2)

        conf1, conf2 = p.get("confidence1"), p.get("confidence2")
        label1 = f"true={p['true1']} pred={p['pred1']}" + (f"\nconf={conf1:.1%}" if conf1 is not None else "")
        label2 = f"true={p['true2']} pred={p['pred2']}" + (f"\nconf={conf2:.1%}" if conf2 is not None else "")

        axes[row, 0].imshow(img1, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(label1, fontsize=8)
        axes[row, 1].imshow(img2, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(label2, fontsize=8)
        axes[row, 2].imshow(diff, cmap="hot", vmin=0, vmax=max(diff.max(), 1e-6))
        axes[row, 2].set_title(f"|diff| (max={diff.max():.2f}, mean={diff.mean():.3f})", fontsize=8)

        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(
            f"ratio={p['ratio']:.3g}\ndist={p['dist']:.3g}\n|Δmargin|={p['margin_diff']:.3g}",
            fontsize=8, rotation=0, ha="right", va="center", labelpad=45)

    fig.suptitle(title or "Pair diagnostic gallery: images, predictions, and pixel-level difference")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_layer_decomposition_sweep(df, save_path=None):
    """`L_head` (exact + estimated), `L_extractor`, `product`
    (`L_extractor_estimated * L_head_exact`), and `L_full` as a function of
    CNN width (one line each, log-y since they can span more than an order
    of magnitude across widths), plus `looseness_ratio` vs. width in a
    second panel with a reference line at 1.0 -- the theoretical floor the
    submultiplicative bound (Szegedy et al. 2014) guarantees `product`
    should never fall below, the same reference-line convention
    `plot_epsilon_sweep`'s `selected_epsilon` line and `plot_sweep`'s `L*`
    line use elsewhere in this project.

    The second panel also plots `looseness_ratio_estimated` (using
    `L_head_estimated` in the product instead of the exact spectral norm)
    as a dashed line -- it is *not* guaranteed to stay above the 1.0
    reference line the way `looseness_ratio` is, since `L_head_estimated`
    is itself only ever a lower bound on the true `L_head` (see
    `layer_decomposition_experiment`'s docstring), so it's shown for
    comparison rather than as a second checkpoint.

    `df`: the DataFrame `layer_decomposition.run_cnn_width_sweep` returns
    (or anything with the same `width`/`L_head_exact`/`L_head_estimated`/
    `L_extractor_estimated`/`L_full_estimated`/`product`/`looseness_ratio`/
    `looseness_ratio_estimated` columns).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(df["width"], df["L_head_exact"], marker="o", color="tab:blue", label="L_head (exact)")
    ax1.plot(df["width"], df["L_head_estimated"], marker="o", color="tab:blue",
              linestyle="--", alpha=0.6, label="L_head (estimated)")
    ax1.plot(df["width"], df["L_extractor_estimated"], marker="s", color="tab:orange", label="L_extractor")
    ax1.plot(df["width"], df["product"], marker="^", color="tab:green", label="product (L_extractor * L_head_exact)")
    ax1.plot(df["width"], df["L_full_estimated"], marker="D", color="tab:red", label="L_full")
    ax1.set_yscale("log")
    ax1.set_xlabel("CNN width")
    ax1.set_ylabel("Lipschitz estimate (log scale)")
    ax1.set_title("Layer-decomposed Lipschitz estimates vs. width")
    ax1.legend(fontsize=8)

    ax2.axhline(1.0, color="black", linestyle="--", label="theoretical floor (looseness_ratio=1)")
    ax2.plot(df["width"], df["looseness_ratio"], marker="o", color="tab:purple", label="looseness_ratio")
    ax2.plot(df["width"], df["looseness_ratio_estimated"], marker="o", color="tab:purple",
              linestyle="--", alpha=0.6, label="looseness_ratio_estimated (all-estimated)")
    ax2.set_xlabel("CNN width")
    ax2.set_ylabel("looseness_ratio = product / L_full")
    ax2.set_title("Bound looseness vs. width")
    ax2.legend(fontsize=8)

    fig.suptitle("Layer decomposition: L_head, L_extractor, and the per-layer Lipschitz bound")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_embedding_degree_sweep(degree_results, save_path=None):
    """For embeddings.py::elementwise_embedding at each degree in `degree_results` (as built by
    run_experiment.py::run_embedding_degree_sweep): the selected epsilon's condition number
    (left) and the ratio-distribution all-pairs/near-neighbor means (right), both against
    embedding degree -- the two-panel view that makes the sweep's two headline findings directly
    visible: condition number climbs with degree (embedding into a higher-dimensional space), while
    both ratio summary stats shrink with degree and the near-neighbor mean stays below the
    all-pairs mean at every degree tested (the reversal finding first established at degree=1
    under plain Mahalanobis distance -- see README).

    `degree_results`: dict keyed by degree (int), each value having `selected_epsilon`,
    `cond_number_at_selected_epsilon`, and `ratio_summary` (with `all_pairs_mean`/
    `near_neighbor_mean` keys) -- exactly `run_embedding_degree_sweep`'s `degree_results` return.
    """
    degrees = sorted(degree_results.keys())
    cond_numbers = [degree_results[d]["cond_number_at_selected_epsilon"] for d in degrees]
    selected_epsilons = [degree_results[d]["selected_epsilon"] for d in degrees]
    all_pairs_means = [degree_results[d]["ratio_summary"]["all_pairs_mean"] for d in degrees]
    near_neighbor_means = [degree_results[d]["ratio_summary"]["near_neighbor_mean"] for d in degrees]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(degrees, cond_numbers, marker="o", color="tab:blue")
    for d, cond, eps in zip(degrees, cond_numbers, selected_epsilons):
        ax1.annotate(f"eps={eps:g}", (d, cond), textcoords="offset points", xytext=(0, 8),
                      fontsize=8, ha="center")
    ax1.set_xlabel("embedding degree")
    ax1.set_ylabel("cond(Sigma + eps*I) at selected epsilon")
    ax1.set_xticks(degrees)
    ax1.set_title("Condition number at selected epsilon vs. degree")

    ax2.plot(degrees, all_pairs_means, marker="o", color="tab:blue", label="all-pairs mean")
    ax2.plot(degrees, near_neighbor_means, marker="s", color="tab:orange", label="near-neighbor mean")
    ax2.set_xlabel("embedding degree")
    ax2.set_ylabel("ratio: |margin_i - margin_j| / distance(x_i, x_j)")
    ax2.set_xticks(degrees)
    ax2.set_title("Ratio-distribution summary vs. degree")
    ax2.legend(fontsize=8)

    fig.suptitle("Embedding degree sweep: elementwise_embedding, logistic regression, Mahalanobis distance")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig
