"""All plotting functions for the MNIST Lipschitz experiment. 
Every function returns the created `matplotlib.figure.Figure` and optionally saves it to `save_path`. 
"""

import matplotlib.pyplot as plt
import numpy as np

MODEL_ORDER = ("logistic_regression", "mlp", "cnn")
MODEL_LABELS = {"logistic_regression": "Logistic\nRegression", "mlp": "MLP", "cnn": "CNN"}

def _maybe_save(fig, save_path):
    """Save the figure to `save_path` if given"""
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_euclidean_vs_mahalanobis(euclidean_results, mahalanobis_results, save_path=None):
    """Bar chart of the three sub-method Lipschitz estimates, one panel per model, on a log scale."""
    submethods = [("pairwise", "Pairwise"), ("local_max", "Local-perturbation (max)"), ("grad_max", "Gradient-norm (max)")]
    models = [m for m in MODEL_ORDER if m in euclidean_results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = range(len(models))
    width = 0.35

    for ax, (key, title) in zip(axes, submethods):
        euclidean_vals = [euclidean_results[m][key] for m in models]
        mahalanobis_vals = [mahalanobis_results[m][key] for m in models]

        ax.bar([xi - width / 2 for xi in x], euclidean_vals, width, label="Euclidean", color="blue")
        ax.bar([xi + width / 2 for xi in x], mahalanobis_vals, width, label="Mahalanobis", color="orange")

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
    """Condition number and subsample coefficient of variation against epsilon, on a log-log scale, with the selected epsilon marked."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epsilon_values, cond_numbers, marker="o", color="blue", label="cond(Sigma + eps*I)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("epsilon (log scale)")
    ax.set_ylabel("condition number (log scale)", color="blue")
    ax.tick_params(axis="y", labelcolor="blue")

    ax2 = ax.twinx()
    ax2.plot(epsilon_values, cv_values, marker="s", color="red", label="coefficient of variation")
    ax2.set_ylabel("subsample instability (std/mean)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

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
    """Bar chart of the three sub-method Lipschitz estimates, one panel per model, on a log scale."""
    models = [m for m in MODEL_ORDER if m in results]
    submethod_keys = ["pairwise", "local_max", "grad_max"]
    submethod_labels = ["pairwise", "local-pert.", "grad-norm"]

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        vals = [results[model][k] for k in submethod_keys]
        bars = ax.bar(submethod_labels, vals, color=["blue", "red", "green"])
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
    """Eigenvalue spectrum of the pixel covariance matrix, log scale, optionally showing the epsilon-regularised spectrum alongside."""
    eigenvalues = eigenvalues.detach().cpu().numpy() if hasattr(eigenvalues, "detach") else np.asarray(eigenvalues)
    eigenvalues_plot = np.clip(eigenvalues, a_min=1e-12, a_max=None)  # avoid log(0)/log(negative) from fp noise
    index = np.arange(1, len(eigenvalues) + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(index, eigenvalues_plot, color="blue", label="eigenvalues of Sigma")

    if epsilon is not None:
        ax.plot(index, eigenvalues_plot + epsilon, color="orange", linestyle="--",
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
    """Histogram of the pairwise ratio distribution, all pairs vs nearest-neighbour pairs, one panel per model."""
    models = [m for m in MODEL_ORDER if m in ratio_dist_results]
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    bins = 50
    for ax, model in zip(axes, models):
        all_r = np.asarray(ratio_dist_results[model]["all_pairs_ratio"])
        near_r = np.asarray(ratio_dist_results[model]["near_neighbor_ratio"])
        ax.hist(all_r, bins=bins, density=True, alpha=0.6, color="blue",
                label=f"all pairs (n={len(all_r)})")
        ax.hist(near_r, bins=bins, density=True, alpha=0.6, color="orange",
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
    """Grouped bar chart comparing all-pairs mean, near-neighbour mean, and near-neighbour max ratio,
    under Euclidean vs Mahalanobis distance, per model."""
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

        ax.bar([xi - width / 2 for xi in x], euclidean_vals, width, label="Euclidean", color="blue")
        ax.bar([xi + width / 2 for xi in x], mahalanobis_vals, width, label="Mahalanobis", color="orange")

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
    """Grid of up to 6 image pairs with true/predicted labels and their pairwise Lipschitz ratio."""
    n = min(len(pairs), 6)
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
    """For each pair, the two images side by side with labels and confidence, plus a pixel-difference
    heatmap and the ratio/distance/margin values."""
    n = len(pairs)

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


def plot_umap_embedding_scatter(embedded_2d, labels, title=None, save_path=None):
    """Scatter plot of a 2D embedding (e.g., UMAP) colored by true digit labels. 
    `embedded_2d`: 2D array/tensor of shape (n_samples, 2) representing the 2D coordinates of the embedding.
    `labels`: 1D array/tensor of shape (n_samples,) representing the true digit labels (0-9) for each sample.
    """
    embedded_2d = embedded_2d.detach().cpu().numpy() if hasattr(embedded_2d, "detach") else np.asarray(embedded_2d)
    labels = labels.detach().cpu().numpy() if hasattr(labels, "detach") else np.asarray(labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")
    for digit in range(10):
        mask = labels == digit
        ax.scatter(embedded_2d[mask, 0], embedded_2d[mask, 1], s=8, alpha=0.7,
                   color=cmap(digit), label=str(digit))

    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.set_title(title or "UMAP embedding, colored by true digit label")
    ax.legend(title="digit", fontsize=8, markerscale=2, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_layer_decomposition_sweep(df, save_path=None):
    """L_head, L_extractor, product, and L_full against CNN width (left), and the resulting
    looseness ratio against width (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(df["width"], df["L_head_exact"], marker="o", color="blue", label="L_head (exact)")
    ax1.plot(df["width"], df["L_head_estimated"], marker="o", color="blue",
              linestyle="--", alpha=0.6, label="L_head (estimated)")
    ax1.plot(df["width"], df["L_extractor_estimated"], marker="s", color="orange", label="L_extractor")
    ax1.plot(df["width"], df["product"], marker="^", color="green", label="product (L_extractor * L_head_exact)")
    ax1.plot(df["width"], df["L_full_estimated"], marker="D", color="red", label="L_full")
    ax1.set_yscale("log")
    ax1.set_xlabel("CNN width")
    ax1.set_ylabel("Lipschitz estimate (log scale)")
    ax1.set_title("Layer-decomposed Lipschitz estimates vs. width")
    ax1.legend(fontsize=8)

    ax2.axhline(1.0, color="black", linestyle="--", label="theoretical floor (looseness_ratio=1)")
    ax2.plot(df["width"], df["looseness_ratio"], marker="o", color="purple", label="looseness_ratio")
    ax2.plot(df["width"], df["looseness_ratio_estimated"], marker="o", color="purple",
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
    """Condition number at the selected epsilon (left), and all-pairs vs near-neighbour mean ratio
    (right), against embedding degree."""
    degrees = sorted(degree_results.keys())
    cond_numbers = [degree_results[d]["cond_number_at_selected_epsilon"] for d in degrees]
    selected_epsilons = [degree_results[d]["selected_epsilon"] for d in degrees]
    all_pairs_means = [degree_results[d]["ratio_summary"]["all_pairs_mean"] for d in degrees]
    near_neighbor_means = [degree_results[d]["ratio_summary"]["near_neighbor_mean"] for d in degrees]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(degrees, cond_numbers, marker="o", color="blue")
    for d, cond, eps in zip(degrees, cond_numbers, selected_epsilons):
        ax1.annotate(f"eps={eps:g}", (d, cond), textcoords="offset points", xytext=(0, 8),
                      fontsize=8, ha="center")
    ax1.set_xlabel("embedding degree")
    ax1.set_ylabel("cond(Sigma + eps*I) at selected epsilon")
    ax1.set_xticks(degrees)
    ax1.set_title("Condition number at selected epsilon vs. degree")

    ax2.plot(degrees, all_pairs_means, marker="o", color="blue", label="all-pairs mean")
    ax2.plot(degrees, near_neighbor_means, marker="s", color="orange", label="near-neighbor mean")
    ax2.set_xlabel("embedding degree")
    ax2.set_ylabel("ratio: |margin_i - margin_j| / distance(x_i, x_j)")
    ax2.set_xticks(degrees)
    ax2.set_title("Ratio-distribution summary vs. degree")
    ax2.legend(fontsize=8)

    fig.suptitle("Embedding degree sweep: elementwise_embedding, logistic regression, Mahalanobis distance")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_umap_mindist_sweep(sweep_rows, save_path=None):
    """Validation quality (left) and near/all ratio elevation (right) against UMAP's min_dist."""
    min_dists = [r["min_dist"] for r in sweep_rows]
    purities = [r["knn_label_purity"] for r in sweep_rows]
    near_over_all = [r["near_over_all"] for r in sweep_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(min_dists, purities, marker="o", color="blue")
    ax1.axhline(0.10, color="gray", linestyle=":", label="10-class chance baseline")
    ax1.set_xlabel("min_dist")
    ax1.set_ylabel("knn_label_purity (k=5)")
    ax1.set_title("Validation quality vs. min_dist")
    ax1.legend(fontsize=8)

    ax2.plot(min_dists, near_over_all, marker="s", color="orange", label="UMAP near/all")
    ax2.axhline(1.13, color="gray", linestyle="--", label="raw-pixel Euclidean near/all (README)")
    ax2.set_xlabel("min_dist")
    ax2.set_ylabel("near-neighbor mean / all-pairs mean")
    ax2.set_title("Ratio-distribution near/all elevation vs. min_dist")
    ax2.legend(fontsize=8)

    fig.suptitle("UMAP min_dist sweep: does relaxing compression shrink the near/all elevation?")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_variance_explained_curve(curve_all_pairs, curve_near_neighbor, save_path=None):
    """Cumulative variance explained against eigenvector rank, for all-pairs vs near-neighbour pixel differences."""
    curve_all_pairs = curve_all_pairs.detach().cpu().numpy() if hasattr(curve_all_pairs, "detach") else np.asarray(curve_all_pairs)
    curve_near_neighbor = curve_near_neighbor.detach().cpu().numpy() if hasattr(curve_near_neighbor, "detach") else np.asarray(curve_near_neighbor)
    ranks = np.arange(1, len(curve_all_pairs) + 1)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(ranks, curve_all_pairs, color="blue", label="all-pairs")
    ax.plot(ranks, curve_near_neighbor, color="orange", label="near-neighbor")
    ax.set_xlabel("eigenvector rank (descending eigenvalue -- 1 = highest variance)")
    ax.set_ylabel("mean cumulative fraction of squared pixel-difference norm")
    ax.set_title("Where pairwise pixel differences live in the covariance eigenbasis")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_umap_ncomponents_sweep(sweep_rows, save_path=None):
    """Validation quality (left) and near/all ratio elevation (right) against UMAP's n_components."""
    n_components_vals = [r["n_components"] for r in sweep_rows]
    purities = [r["knn_label_purity"] for r in sweep_rows]
    near_over_all = [r["near_over_all"] for r in sweep_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(n_components_vals, purities, marker="o", color="blue")
    ax1.axhline(0.10, color="gray", linestyle=":", label="10-class chance baseline")
    ax1.set_xlabel("n_components")
    ax1.set_ylabel("knn_label_purity (k=5)")
    ax1.set_title("Validation quality vs. n_components")
    ax1.legend(fontsize=8)

    ax2.plot(n_components_vals, near_over_all, marker="s", color="orange", label="UMAP near/all")
    ax2.axhline(1.13, color="gray", linestyle="--", label="raw-pixel Euclidean near/all (README)")
    ax2.set_xlabel("n_components")
    ax2.set_ylabel("near-neighbor mean / all-pairs mean")
    ax2.set_title("Ratio-distribution near/all elevation vs. n_components")
    ax2.legend(fontsize=8)

    fig.suptitle("UMAP n_components sweep: is the near/all elevation specific to low output dimension?")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_umap_nneighbors_sweep(sweep_rows, save_path=None):
    """Validation quality (left) and near/all ratio elevation (right) against UMAP's n_neighbors."""
    n_neighbors_vals = [r["n_neighbors"] for r in sweep_rows]
    purities = [r["knn_label_purity"] for r in sweep_rows]
    near_over_all = [r["near_over_all"] for r in sweep_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(n_neighbors_vals, purities, marker="o", color="blue")
    ax1.axhline(0.10, color="gray", linestyle=":", label="10-class chance baseline")
    ax1.set_xlabel("n_neighbors")
    ax1.set_ylabel("knn_label_purity (k=5)")
    ax1.set_title("Validation quality vs. n_neighbors")
    ax1.legend(fontsize=8)

    ax2.plot(n_neighbors_vals, near_over_all, marker="s", color="orange", label="UMAP near/all")
    ax2.axhline(1.13, color="gray", linestyle="--", label="raw-pixel Euclidean near/all (README)")
    ax2.set_xlabel("n_neighbors")
    ax2.set_ylabel("near-neighbor mean / all-pairs mean")
    ax2.set_title("Ratio-distribution near/all elevation vs. n_neighbors")
    ax2.legend(fontsize=8)

    fig.suptitle("UMAP n_neighbors sweep: does a larger local-neighborhood size shrink the near/all elevation?")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_umap_seed_sweep(sweep_rows, save_path=None):
    """Validation quality (left) and near/all ratio elevation (right) across UMAP random seeds, as
    bar charts since seed has no natural ordering."""
    seeds = [r["seed"] for r in sweep_rows]
    purities = [r["knn_label_purity"] for r in sweep_rows]
    near_over_all = [r["near_over_all"] for r in sweep_rows]
    x = np.arange(len(seeds))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(x, purities, color="blue")
    ax1.axhline(0.10, color="gray", linestyle=":", label="10-class chance baseline")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in seeds])
    ax1.set_xlabel("seed")
    ax1.set_ylabel("knn_label_purity (k=5)")
    ax1.set_title("Validation quality across seeds")
    ax1.legend(fontsize=8)

    ax2.bar(x, near_over_all, color="orange", label="UMAP near/all")
    ax2.axhline(1.13, color="gray", linestyle="--", label="raw-pixel Euclidean near/all (README)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in seeds])
    ax2.set_xlabel("seed")
    ax2.set_ylabel("near-neighbor mean / all-pairs mean")
    ax2.set_title("Ratio-distribution near/all elevation across seeds")
    ax2.legend(fontsize=8)

    fig.suptitle("UMAP seed sweep: is the near/all elevation stable across random restarts?")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_smoothing_gallery(samples, sigma, save_path=None):
    """Original vs Gaussian-blurred versions of a set of sample digits, side by side, at a given sigma."""
    n = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(4, 1.8 * n), dpi=80)
    axes = np.atleast_2d(axes)

    for row, s in enumerate(samples):
        original = np.asarray(s["original"]).reshape(28, 28)
        blurred = np.asarray(s["blurred"]).reshape(28, 28)

        axes[row, 0].imshow(original, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title("original", fontsize=8)
        axes[row, 1].imshow(blurred, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("blurred", fontsize=8)

        for col in range(2):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(f"digit={s['digit']}", fontsize=9, rotation=0, ha="right",
                                 va="center", labelpad=25)

    fig.suptitle(f"Smoothing gallery: sigma={sigma}")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_smoothing_stability_sweep(sweep_rows, max_cv=0.05, save_path=None):
    """Best epsilon-selection coefficient of variation against Gaussian blur strength (sigma)."""
    sigmas = [r["sigma"] for r in sweep_rows]
    min_cvs = [r["min_cv"] for r in sweep_rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sigmas, min_cvs, marker="o", color="blue")
    ax.axhline(max_cv, color="gray", linestyle="--", label=f"stability bound (cv<={max_cv})")
    ax.set_xlabel("sigma (Gaussian blur strength)")
    ax.set_ylabel("best (minimum) coefficient of variation across epsilon candidates")
    ax.set_title("Epsilon-selection stability vs. smoothing strength")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_smoothing_ratio_sweep(sweep_rows, save_path=None):
    """Validation quality (left) and near/all ratio elevation (right) against blur strength."""
    sigmas = [r["sigma"] for r in sweep_rows]
    purities = [r["knn_label_purity"] for r in sweep_rows]
    euclidean_near_over_all = [r["euclidean_near_over_all"] for r in sweep_rows]
    mahalanobis_sigmas = [r["sigma"] for r in sweep_rows if r["mahalanobis_near_over_all"] is not None]
    mahalanobis_near_over_all = [r["mahalanobis_near_over_all"] for r in sweep_rows
                                  if r["mahalanobis_near_over_all"] is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(sigmas, purities, marker="o", color="blue")
    ax1.axhline(0.10, color="gray", linestyle=":", label="10-class chance baseline")
    ax1.set_xlabel("sigma")
    ax1.set_ylabel("knn_label_purity (k=5)")
    ax1.set_title("Validation quality vs. smoothing strength")
    ax1.legend(fontsize=8)

    ax2.plot(sigmas, euclidean_near_over_all, marker="s", color="orange",
              label="smoothed cross-terms + Euclidean")
    if mahalanobis_sigmas:
        ax2.plot(mahalanobis_sigmas, mahalanobis_near_over_all, marker="^", color="green",
                  label="smoothed cross-terms + Mahalanobis")
    ax2.axhline(1.13, color="gray", linestyle="--", label="raw-pixel Euclidean near/all (README)")
    ax2.set_xlabel("sigma")
    ax2.set_ylabel("near-neighbor mean / all-pairs mean")
    ax2.set_title("Ratio-distribution near/all elevation vs. smoothing strength")
    ax2.legend(fontsize=8)

    fig.suptitle("Smoothing-strength sweep: does blurring before cross-terms fix the Mahalanobis instability?")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_truncated_mahalanobis_stability_sweep(feature_space_results, max_cv=0.05, save_path=None):
    """Coefficient of variation against the number of retained top-variance dimensions (k), one line
    per feature space."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, k_results in feature_space_results.items():
        k_values = sorted(k_results.keys())
        cvs = [k_results[k]["cv"] for k in k_values]
        ax.plot(k_values, cvs, marker="o", label=name)
    ax.axhline(max_cv, color="gray", linestyle="--", label=f"stability bound (cv<={max_cv})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("k (retained top-variance dimensions)")
    ax.set_ylabel("coefficient of variation")
    ax.set_title("Truncated-eigenvalue Mahalanobis: epsilon-selection-check stability vs. k")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_truncated_mahalanobis_ratio_sweep(feature_space_results, save_path=None):
    """Validation quality (left) and near/all ratio elevation (right, at stable k values only)
    against k, one line per feature space."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for name, k_results in feature_space_results.items():
        k_values = sorted(k_results.keys())
        purities = [k_results[k]["knn_label_purity"] for k in k_values]
        ax1.plot(k_values, purities, marker="o", label=name)

        passing_k = [k for k in k_values if k_results[k]["ratio_summary"] is not None]
        if passing_k:
            near_over_all = [k_results[k]["ratio_summary"]["near_neighbor_mean"] / k_results[k]["ratio_summary"]["all_pairs_mean"]
                              for k in passing_k]
            ax2.plot(passing_k, near_over_all, marker="s", label=name)

    ax1.axhline(0.10, color="gray", linestyle=":", label="10-class chance baseline")
    ax1.set_xscale("log")
    ax1.set_xlabel("k")
    ax1.set_ylabel("knn_label_purity (k=5, whitened truncated-Mahalanobis coordinates)")
    ax1.set_title("Validation quality vs. k")
    ax1.legend(fontsize=8)

    ax2.axhline(1.13, color="gray", linestyle="--", label="raw-pixel Euclidean near/all (README)")
    ax2.set_xscale("log")
    ax2.set_xlabel("k")
    ax2.set_ylabel("near-neighbor mean / all-pairs mean")
    ax2.set_title("Ratio-distribution near/all elevation vs. k (stable k only)")
    ax2.legend(fontsize=8)

    fig.suptitle("Truncated-eigenvalue Mahalanobis: does discarding low-variance directions fix the instability?")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_radius_multiplier_stability_sweep(sweep_rows, max_cv=0.05, save_path=None):
    """Best epsilon-selection coefficient of variation against the blur kernel's radius_multiplier,
    at fixed sigma=1."""
    multipliers = [r["radius_multiplier"] for r in sweep_rows]
    min_cvs = [r["min_cv"] for r in sweep_rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(multipliers, min_cvs, marker="o", color="blue")
    ax.axhline(max_cv, color="gray", linestyle="--", label=f"stability bound (cv<={max_cv})")
    ax.set_xlabel("radius_multiplier (kernel radius = round(radius_multiplier * sigma))")
    ax.set_ylabel("best (minimum) coefficient of variation across epsilon candidates")
    ax.set_title("Epsilon-selection stability vs. radius_multiplier (sigma=1 fixed)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_radius_multiplier_ratio_sweep(sweep_rows, save_path=None):
    """Validation quality (left) and near/all ratio elevation (right) against radius_multiplier, at fixed sigma=1."""
    multipliers = [r["radius_multiplier"] for r in sweep_rows]
    purities = [r["knn_label_purity"] for r in sweep_rows]
    euclidean_near_over_all = [r["euclidean_near_over_all"] for r in sweep_rows]
    mahalanobis_multipliers = [r["radius_multiplier"] for r in sweep_rows if r["mahalanobis_near_over_all"] is not None]
    mahalanobis_near_over_all = [r["mahalanobis_near_over_all"] for r in sweep_rows
                                  if r["mahalanobis_near_over_all"] is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(multipliers, purities, marker="o", color="blue")
    ax1.axhline(0.10, color="gray", linestyle=":", label="10-class chance baseline")
    ax1.set_xlabel("radius_multiplier")
    ax1.set_ylabel("knn_label_purity (k=5)")
    ax1.set_title("Validation quality vs. radius_multiplier")
    ax1.legend(fontsize=8)

    ax2.plot(multipliers, euclidean_near_over_all, marker="s", color="orange",
              label="smoothed cross-terms + Euclidean")
    if mahalanobis_multipliers:
        ax2.plot(mahalanobis_multipliers, mahalanobis_near_over_all, marker="^", color="green",
                  label="smoothed cross-terms + Mahalanobis")
    ax2.axhline(1.13, color="gray", linestyle="--", label="raw-pixel Euclidean near/all (README)")
    ax2.set_xlabel("radius_multiplier")
    ax2.set_ylabel("near-neighbor mean / all-pairs mean")
    ax2.set_title("Ratio-distribution near/all elevation vs. radius_multiplier")
    ax2.legend(fontsize=8)

    fig.suptitle("radius_multiplier sweep (sigma=1 fixed): was the hardcoded default of 3 actually best?")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig
