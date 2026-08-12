"""All plotting functions for the MNIST Lipschitz experiment.

Every function returns the created `matplotlib.figure.Figure` and optionally
saves it to `save_path`. No plotting logic should live anywhere else
(driver scripts/notebooks only call into this module) -- matches
toy_lipschitz/plots.py's convention.
"""

import matplotlib.pyplot as plt

MODEL_ORDER = ("logistic_regression", "mlp", "cnn")
MODEL_LABELS = {"logistic_regression": "Logistic\nRegression", "mlp": "MLP", "cnn": "CNN"}


def _maybe_save(fig, save_path):
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_euclidean_vs_mahalanobis(euclidean_results, mahalanobis_results, save_path=None):
    """Three models x three sub-methods x two metrics (Euclidean,
    Mahalanobis) -- the MNIST analogue of Experiment 1's headline
    plain-vs-Mahalanobis comparison. One panel per sub-method (pairwise,
    local-perturbation max, gradient-norm max), since the three sub-methods
    live on very different natural scales.
    """
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
    """Condition number and subsample instability (coefficient of
    variation) both plotted against epsilon (log scale) -- the MNIST
    analogue of toy_lipschitz's plot_degree_sweep."""
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
    """For each model, show the three sub-method estimates (pairwise,
    local-perturbation max, gradient-norm max) side by side on a log y-axis
    (the three live on very different natural scales) -- the validity check
    made visible, given there's no L* to plot as a reference line here."""
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
