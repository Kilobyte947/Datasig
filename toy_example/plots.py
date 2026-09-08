"""This file contains all plotting functions for the Toy Lipschitz experiment."""

import matplotlib.pyplot as plt

def _maybe_save(fig, save_path):
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_gap_vs_uniform(x_grid, f_star_vals, dataset_results, L_star, save_path=None):
    """f*(x), f_hat(x), and the pointwise local Lipschitz estimate
    (finite-difference) overlaid, with training points as a rug plot, for
    both the gap and uniform datasets side by side.

    x_grid: (M,) 1D grid spanning the domain.
    f_star_vals: (M,) true function values on the grid.
    dataset_results: dict with keys "gap" and "uniform", each a dict with
        "x_train": (N,) training x locations,
        "f_hat_vals": (M,) trained model predictions on x_grid,
        "local_lipschitz_vals": (M,) finite-difference local Lipschitz
            estimate on x_grid.
    L_star: true global Lipschitz constant (reference line).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key in zip(axes, ["gap", "uniform"]):
        r = dataset_results[key]
        ax.plot(x_grid, f_star_vals, label="f*(x)", color="black", linewidth=2)
        ax.plot(x_grid, r["f_hat_vals"], label="f_hat(x)", color="red")

        ax2 = ax.twinx()
        ax2.plot(x_grid, r["local_lipschitz_vals"], label="local Lipschitz est. (finite-diff)",
                 color="orange", alpha=0.7)
        ax2.axhline(L_star, color="black", linestyle="--", label="L* (true)")
        ax2.set_ylabel("local Lipschitz estimate")

        y_lo, _ = ax.get_ylim()
        x_train = r["x_train"]
        ax.plot(x_train, [y_lo] * len(x_train), "|", color="gray", markersize=10,
                 markeredgewidth=1.2, label="train points")

        ax.set_title(f"{key} dataset")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    fig.suptitle("f*, f_hat, and local Lipschitz estimate: gap vs. uniform sampling")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_sweep(x_values, L_star, L_hat_data_values, L_hat_model_values, xlabel, title,
                model_label="L_hat_model (held-out grid)", log_x=True, save_path=None):
    """L* (reference line), L_hat_data(x), L_hat_model(x) vs. a swept
    quantity (N or model capacity)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(L_star, color="black", linestyle="--", label="L* (true)")
    ax.plot(x_values, L_hat_data_values, marker="o", color="blue", label="L_hat_data")
    ax.plot(x_values, L_hat_model_values, marker="s", color="red", label=model_label)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Lipschitz estimate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_seed_averaged_sweep(seed_results, L_star, xlabel="N (training samples)", title=None,
                              spread="std", show_individual_seeds=True, log_x=True, save_path=None):
    """Seed-averaged N-sweep plot: mean L_hat_model vs the swept quantity with a shaded spread band,
    mean L_hat_data for comparison, the L* reference line, and optionally faint individual per-seed
    L_hat_model trajectories.
    
    seed_results: dict as returned by sweep_over_N_seed_averaged (N_values, seeds, L_hat_data/
    model_mean/std/min/max, L_hat_data/model_per_seed). spread: "std" for a +-1 std band, "minmax"
    for a min/max band."""
    x_values = seed_results["N_values"]
    mean_model = seed_results["L_hat_model_mean"]
    mean_data = seed_results["L_hat_data_mean"]

    if spread == "std":
        lo = mean_model - seed_results["L_hat_model_std"]
        hi = mean_model + seed_results["L_hat_model_std"]
        band_label = "L_hat_model mean +/- 1 std"
    elif spread == "minmax":
        lo = seed_results["L_hat_model_min"]
        hi = seed_results["L_hat_model_max"]
        band_label = "L_hat_model mean, min-max range"
    else:
        raise ValueError(f"unknown spread: {spread}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axhline(L_star, color="black", linestyle="--", label="L* (true)")

    if show_individual_seeds:
        for i, seed in enumerate(seed_results["seeds"]):
            ax.plot(x_values, seed_results["L_hat_model_per_seed"][i], color="orange",
                     alpha=0.15, linewidth=1, label="individual seeds" if i == 0 else None)

    ax.fill_between(x_values, lo, hi, color="orange", alpha=0.25, label=band_label)
    ax.plot(x_values, mean_model, marker="s", color="orange", label="L_hat_model (mean)")
    ax.plot(x_values, mean_data, marker="o", color="blue", label="L_hat_data (mean)")

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Lipschitz estimate")
    ax.set_title(title or f"Seed-averaged sweep ({len(seed_results['seeds'])} seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_2d_heatmaps(xx, yy, true_grad_norm_grid, model_grad_norm_grid, local_lipschitz_grid,
                      train_points, save_path=None):
    """Heatmaps of true ||grad f*||, model ||grad f_hat|| (autograd), and
    the finite-difference local Lipschitz estimate over [-5,5]^2, with
    training points overlaid as scatter."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    titles = [
        "True ||grad f*(x)||",
        "Model ||grad f_hat(x)|| (autograd)",
        "Finite-diff local Lipschitz estimate",
    ]
    grids = [true_grad_norm_grid, model_grad_norm_grid, local_lipschitz_grid]
    vmax = max(g.max() for g in grids)

    for ax, title, grid in zip(axes, titles, grids):
        im = ax.pcolormesh(xx, yy, grid, shading="auto", vmin=0, vmax=vmax, cmap="viridis")
        ax.scatter(train_points[:, 0], train_points[:, 1], s=8, c="white",
                   edgecolors="black", linewidths=0.4, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_local_vs_global_lipschitz(x_grid, dataset_results, L_star, save_path=None):
    """Global scalar estimates (one number for the whole domain) vs.
    location-resolved local estimates, each shown under both plain
    Euclidean and Mahalanobis-in-embedding distance. Side by side for gap
    and uniform sampling.

    dataset_results: dict with keys "gap"/"uniform", each holding
        "L_hat_data_plain", "L_hat_data_maha": global scalars,
        "local_plain_vals", "local_maha_vals": (M,) arrays over x_grid.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key in zip(axes, ["gap", "uniform"]):
        r = dataset_results[key]
        ax.axhline(L_star, color="black", linestyle="--", label="L* (true)")
        ax.axhline(r["L_hat_data_plain"], color="blue", linestyle=":", label="L_hat_data (plain, global)")
        ax.axhline(r["L_hat_data_maha"], color="green", linestyle=":", label="L_hat_data (Mahalanobis, global)")
        ax.plot(x_grid, r["local_plain_vals"], color="blue", alpha=0.8, label="local L(x) (plain)")
        ax.plot(x_grid, r["local_maha_vals"], color="green", alpha=0.8, label="local L(x) (Mahalanobis)")
        ax.set_title(f"{key} dataset")
        ax.set_xlabel("x")
        ax.set_ylabel("Lipschitz estimate")
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Global -> local, plain -> Mahalanobis: Lipschitz estimate progression")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_degree_sweep(degrees, errors, cond_numbers, save_path=None):
    """Relative error of the global L_hat_mahalanobis vs. L*, and numerical
    conditioning of the fitted embedding covariance, both vs. polynomial
    embedding degree - used to pick the lowest degree that is both accurate
    and well-conditioned."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(degrees, errors, marker="o", color="blue", label="rel. error vs. L*")
    ax.set_xlabel("polynomial embedding degree")
    ax.set_ylabel("relative error", color="blue")
    ax.tick_params(axis="y", labelcolor="blue")

    ax2 = ax.twinx()
    ax2.plot(degrees, cond_numbers, marker="s", color="red", label="cond(cov)")
    ax2.set_yscale("log")
    ax2.set_ylabel("condition number of covariance (log scale)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax.set_title("Polynomial embedding degree: accuracy vs. conditioning")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_presentation_gap_effect(x_grid, dataset_results, L_star, save_path=None):
    """Presentation figure: the trained model's local Lipschitz estimate vs x, gap-sampled vs 
    uniformly-sampled, with L* marked and the two peak values labeled directly.
    
    x_grid: (M,) 1D grid. dataset_results: dict with keys "gap"/"uniform", each holding 
    "local_lipschitz_vals" (M,). L_star: true global Lipschitz constant."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(L_star, color="black", linestyle="--", linewidth=2)
    ax.annotate(f"L* = {L_star:.1f}", xy=(float(x_grid[-1]), L_star), xytext=(-4, 6),
                textcoords="offset points", ha="right", fontsize=12, color="black")

    colors = {"uniform": "blue", "gap": "red"}
    labels = {"uniform": "uniformly-sampled model", "gap": "gap-sampled model"}
    for key in ["uniform", "gap"]:
        vals = dataset_results[key]["local_lipschitz_vals"]
        ax.plot(x_grid, vals, color=colors[key], linewidth=2.5, label=labels[key])
        peak_idx = vals.argmax()
        peak_x, peak_y = float(x_grid[peak_idx]), float(vals[peak_idx])
        ax.annotate(f"{peak_y:.1f}", xy=(peak_x, peak_y), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=13,
                    fontweight="bold", color="black")

    ax.set_xlabel("x", fontsize=13)
    ax.set_ylabel("local Lipschitz estimate", fontsize=13)
    ax.set_title("A sampling gap makes the model underestimate its own sensitivity", fontsize=15)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=12, loc="lower left", frameon=False)
    ax.grid(axis="y", color="0.9", linewidth=1)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_presentation_metric_comparison(L_star, L_hat_plain, L_hat_maha, save_path=None):
    """Presentation figure: plain-Euclidean vs. Mahalanobis error against
    L*, as two bars with the relative error labeled directly on each -
    the headline distance-metric result on one simple axis."""
    err_plain = abs(L_hat_plain - L_star) / L_star * 100
    err_maha = abs(L_hat_maha - L_star) / L_star * 100

    fig, ax = plt.subplots(figsize=(7, 5.5))
    bars = ax.bar(["Plain\nEuclidean", "Mahalanobis\n(degree-3 embedding)"],
                   [err_plain, err_maha], color=["blue", "green"], width=0.55)
    for bar, err in zip(bars, [err_plain, err_maha]):
        label = f"{err:.1f}%" if err >= 1 else "<1%"
        ax.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 8), textcoords="offset points", ha="center",
                    fontsize=15, fontweight="bold", color="black")

    ax.set_ylabel("relative error vs. L*", fontsize=13)
    ax.set_title("Switching the distance metric alone drops the error\nfrom ~19% to under 1%", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.set_ylim(0, max(err_plain, err_maha) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=1)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_presentation_degree_tradeoff(degrees, errors, cond_numbers, save_path=None):
    """Presentation figure: the embedding-degree tradeoff as two small
    multiples sharing one x-axis (degree), each with its own single
    y-axis, instead of one dual-axis plot - accuracy and conditioning are
    different quantities on different scales, so splitting them avoids the
    misleading visual overlap a shared plot area would create. Degree 3
    (the chosen value) is highlighted in both panels."""
    best_idx = 2  # degree 3, by construction of the default sweep (degrees=1..6)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    ax1.plot(degrees, [e * 100 for e in errors], color="blue", linewidth=2.5, marker="o", markersize=8)
    ax1.scatter([degrees[best_idx]], [errors[best_idx] * 100], color="blue", s=160,
                zorder=3, edgecolors="black", linewidths=1.5)
    ax1.set_yscale("log")
    ax1.set_xlabel("polynomial embedding degree", fontsize=12)
    ax1.set_ylabel("relative error vs. L* (%)", fontsize=12)
    ax1.set_title("Accuracy", fontsize=14)
    ax1.tick_params(labelsize=11)
    ax1.grid(axis="y", color="0.9", linewidth=1)
    ax1.set_axisbelow(True)

    ax2.plot(degrees, cond_numbers, color="red", linewidth=2.5, marker="o", markersize=8)
    ax2.scatter([degrees[best_idx]], [cond_numbers[best_idx]], color="red", s=160,
                zorder=3, edgecolors="black", linewidths=1.5)
    ax2.set_yscale("log")
    ax2.set_xlabel("polynomial embedding degree", fontsize=12)
    ax2.set_ylabel("condition number of covariance", fontsize=12)
    ax2.set_title("Numerical stability", fontsize=14)
    ax2.tick_params(labelsize=11)
    ax2.grid(axis="y", color="0.9", linewidth=1)
    ax2.set_axisbelow(True)

    fig.suptitle(f"Degree {degrees[best_idx]}: the lowest degree that is both accurate and well-conditioned", fontsize=15)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_coverage_heatmap(xx, yy, density_grid, train_points, save_path=None):
    """Coverage diagnostic: local training-point density per grid cell,
    plotted separately from the Lipschitz/gradient-norm heatmaps in
    plot_2d_heatmaps so "smooth" and "unsampled" stay visually
    distinguishable.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.pcolormesh(xx, yy, density_grid, shading="auto", cmap="magma")
    ax.scatter(train_points[:, 0], train_points[:, 1], s=8, c="white",
               edgecolors="black", linewidths=0.4, alpha=0.8)
    ax.set_title("Local sample density (coverage diagnostic)")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig
