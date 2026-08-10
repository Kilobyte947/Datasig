"""All plotting functions for the toy Lipschitz experiment.

Every function returns the created `matplotlib.figure.Figure` and optionally
saves it to `save_path`. No plotting logic should live anywhere else
(driver scripts/notebooks only call into this module).
"""

import matplotlib.pyplot as plt


def _maybe_save(fig, save_path):
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_gap_vs_uniform(x_grid, f_star_vals, dataset_results, L_star, save_path=None):
    """Step 6.6: f*(x), f_hat(x), and the pointwise local Lipschitz estimate
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
        ax.plot(x_grid, r["f_hat_vals"], label="f_hat(x)", color="tab:blue")

        ax2 = ax.twinx()
        ax2.plot(x_grid, r["local_lipschitz_vals"], label="local Lipschitz est. (finite-diff)",
                 color="tab:red", alpha=0.7)
        ax2.axhline(L_star, color="gray", linestyle="--", label="L* (true)")
        ax2.set_ylabel("local Lipschitz estimate")

        y_lo, _ = ax.get_ylim()
        x_train = r["x_train"]
        ax.plot(x_train, [y_lo] * len(x_train), "|", color="green", markersize=10,
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
    """Step 7: L* (reference line), L_hat_data(x), L_hat_model(x) vs. a swept
    quantity (N or model capacity)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(L_star, color="black", linestyle="--", label="L* (true)")
    ax.plot(x_values, L_hat_data_values, marker="o", label="L_hat_data")
    ax.plot(x_values, L_hat_model_values, marker="s", label=model_label)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Lipschitz estimate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_2d_heatmaps(xx, yy, true_grad_norm_grid, model_grad_norm_grid, local_lipschitz_grid,
                      train_points, save_path=None):
    """Step 8: heatmaps of true ||grad f*||, model ||grad f_hat|| (autograd),
    and the finite-difference local Lipschitz estimate over [-5,5]^2, with
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


def plot_coverage_heatmap(xx, yy, density_grid, train_points, save_path=None):
    """Coverage diagnostic (point 5): local training-point density per
    grid cell, plotted separately from the Lipschitz/gradient-norm
    heatmaps in plot_2d_heatmaps so "tested and found smooth" and "never
    really tested" stay visually distinguishable rather than conflated.
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
