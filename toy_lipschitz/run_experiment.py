"""Driver: Tier A sanity check, Tier B main experiment (gap vs. uniform),
N/capacity sweeps, and the 2D extension. Saves csv/npz results and figures
to results/. All reusable logic lives in the sibling modules — this file
only wires them together.
"""

from pathlib import Path

import numpy as np
import torch

from toy_lipschitz.toy_functions import tier_a_f, tier_a_true_L, tier_b_f, tier_b_grad, tier_b_true_L
from toy_lipschitz.data import sample_uniform, sample_with_gap, make_dataset
from toy_lipschitz.models import SingleTanhUnit, TinyMLP, train_regressor
from toy_lipschitz.estimators import (
    pairwise_lipschitz,
    local_perturbation_lipschitz,
    gradient_norm_estimate,
    gradient_norm_estimate_grid,
    local_perturbation_lipschitz_grid,
)
from toy_lipschitz import plots

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DOMAIN = (-5.0, 5.0)
SEED = 0

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# Step 5: Tier A sanity run
# ---------------------------------------------------------------------------

def run_tier_a_sanity(tol=0.10, verbose=True):
    torch.manual_seed(SEED)
    w = torch.tensor([4.0])
    b = 0.5
    A = 1.5
    d = 1

    L_star = tier_a_true_L(w, A, norm="l2")
    f_star = lambda x: tier_a_f(x, w, b, A)

    x = sample_uniform(500, DOMAIN, d, seed=SEED)
    x_train, y_train = make_dataset(f_star, x, noiseless=True)

    L_hat_data, _, _ = pairwise_lipschitz(x_train, y_train, norm="l2")

    model = SingleTanhUnit(input_dim=d)
    model, history = train_regressor(model, x_train, y_train, epochs=2000, lr=0.05, seed=SEED)

    x0_star = torch.tensor([-b / w[0].item()])  # analytic hyperplane point
    y_pred = model(x_train).detach()
    L_hat_model_pairwise, _, _ = pairwise_lipschitz(x_train, y_pred, norm="l2")
    L_hat_model_local, _ = local_perturbation_lipschitz(model, x0_star, radius=0.3, n_samples=500, norm="l2", seed=SEED + 1)
    L_hat_model_grad = gradient_norm_estimate(model, x0_star, norm="l2")

    results = {
        "L_star": L_star,
        "L_hat_data": L_hat_data,
        "L_hat_model_pairwise": L_hat_model_pairwise,
        "L_hat_model_local": L_hat_model_local,
        "L_hat_model_grad": L_hat_model_grad,
        "final_train_mse": history[-1],
    }

    if verbose:
        print("=== Tier A sanity run ===")
        for k, v in results.items():
            print(f"  {k}: {v}")

    for key in ["L_hat_data", "L_hat_model_pairwise", "L_hat_model_local", "L_hat_model_grad"]:
        rel_err = abs(results[key] - L_star) / L_star
        assert rel_err < tol, f"Tier A sanity check failed for {key}: rel_err={rel_err:.3f} >= {tol}"
        if verbose:
            print(f"  {key} rel_err={rel_err:.4f} (tol={tol})")

    return results


# ---------------------------------------------------------------------------
# Tier A gap demo (single ridge, closed-form L*, gap vs. uniform sampling)
# ---------------------------------------------------------------------------

def run_tier_a_gap_demo(N=400, w=(4.0,), b=0.5, A=1.5, gap_radius=0.5, gap_fraction=0.02,
                         local_radius=0.2, local_n_samples=200, hidden_sizes=(64, 64),
                         epochs=3000, lr=1e-2, grid_size=1000, verbose=True):
    """Tier A analogue of run_main_experiment: same gap-vs-uniform sampling
    comparison, but on the single-ridge f* (closed-form L* = A*||w||) instead
    of the Tier B sum of ridges.

    Uses TinyMLP, not SingleTanhUnit: SingleTanhUnit matches f*'s functional
    form exactly and would recover L* from almost any data, gap or not, so
    it can't demonstrate undershoot. TinyMLP has to learn the shape from
    data, so a sampling gap at the true argmax x* can actually starve it of
    signal there.
    """
    w_t = torch.tensor(w)
    L_star = tier_a_true_L(w_t, A, norm="l2")
    x_star = torch.tensor([-b / w_t[0].item()])  # analytic: where w^Tx+b=0
    f_star = lambda x: tier_a_f(x, w_t, b, A)
    if verbose:
        print(f"\n=== Tier A gap demo: L*={L_star:.4f}, x*={x_star.tolist()} ===")

    x_gap = sample_with_gap(N, DOMAIN, 1, gap_center=x_star, gap_radius=gap_radius, gap_fraction=gap_fraction, seed=SEED)
    x_uniform = sample_uniform(N, DOMAIN, 1, seed=SEED)
    datasets = {}
    for key, x in [("gap", x_gap), ("uniform", x_uniform)]:
        x_train, y_train = make_dataset(f_star, x, noiseless=True)
        datasets[key] = {"x_train": x_train, "y_train": y_train}

    x_grid = torch.linspace(DOMAIN[0], DOMAIN[1], grid_size).unsqueeze(-1)
    f_star_vals = f_star(x_grid).detach()

    dataset_results = {}
    report = {}
    for key, ds in datasets.items():
        x_train, y_train = ds["x_train"], ds["y_train"]
        model, history = train_tiny_mlp(x_train, y_train, hidden_sizes, epochs=epochs, lr=lr)

        L_hat_data, _, _ = pairwise_lipschitz(x_train, y_train, norm="l2")

        y_pred_train = model(x_train).detach()
        L_hat_model_pairwise_train, _, _ = pairwise_lipschitz(x_train, y_pred_train, norm="l2")
        y_pred_grid = model(x_grid).detach()
        L_hat_model_pairwise_grid, _, _ = pairwise_lipschitz(x_grid, y_pred_grid, norm="l2")

        local_lipschitz_vals = local_perturbation_lipschitz_grid(
            model, x_grid, radius=local_radius, n_samples=local_n_samples, norm="l2", seed=SEED)
        grad_norm_vals = gradient_norm_estimate_grid(model, x_grid, norm="l2")

        dataset_results[key] = {
            "x_train": x_train.squeeze(-1),
            "f_hat_vals": y_pred_grid,
            "local_lipschitz_vals": local_lipschitz_vals,
            "grad_norm_vals": grad_norm_vals,
        }

        in_gap = (x_grid.squeeze(-1) - x_star.item()).abs() < gap_radius
        max_in_gap = local_lipschitz_vals[in_gap].max().item() if in_gap.any() else float("nan")
        max_outside_gap = local_lipschitz_vals[~in_gap].max().item()

        report[key] = {
            "L_hat_data": L_hat_data,
            "L_hat_model_pairwise_train": L_hat_model_pairwise_train,
            "L_hat_model_pairwise_grid": L_hat_model_pairwise_grid,
            "final_train_mse": history[-1],
            "max_local_lipschitz_in_gap": max_in_gap,
            "max_local_lipschitz_outside_gap": max_outside_gap,
        }

        if verbose:
            print(f"\n--- {key} dataset ---")
            for k, v in report[key].items():
                print(f"  {k}: {v}")

    RESULTS_DIR.mkdir(exist_ok=True)
    fig = plots.plot_gap_vs_uniform(x_grid.squeeze(-1).numpy(), f_star_vals.numpy(), dataset_results, L_star,
                                     save_path=RESULTS_DIR / "tier_a_gap_vs_uniform.png")

    return {"L_star": L_star, "x_star": x_star, "report": report, "dataset_results": dataset_results, "figure": fig}


# ---------------------------------------------------------------------------
# Tier B setup
# ---------------------------------------------------------------------------

def build_tier_b_1d():
    components = [
        {"w": torch.tensor([8.0]), "b": 3.0, "A": 1.0},   # steep, right-of-center
        {"w": torch.tensor([2.0]), "b": -3.0, "A": 1.0},  # shallow, left-of-center
        {"w": torch.tensor([5.0]), "b": 0.0, "A": 1.0},   # moderate, centered
    ]
    L_star, x_star = tier_b_true_L(components, DOMAIN, norm="l2", grid_points=200000, n_restarts=50, seed=SEED)
    return components, L_star, x_star


def build_tier_b_2d():
    components = [
        {"w": torch.tensor([8.0, 0.0]), "b": 3.0, "A": 1.0},    # steep ridge along x1
        {"w": torch.tensor([0.0, 2.0]), "b": -3.0, "A": 1.0},   # shallow ridge along x2
        {"w": torch.tensor([3.0, 3.0]), "b": 0.0, "A": 1.0},    # moderate, diagonal
    ]
    L_star, x_star = tier_b_true_L(components, DOMAIN, norm="l2", grid_points=200000, n_restarts=50, seed=SEED)
    return components, L_star, x_star


# ---------------------------------------------------------------------------
# Step 6: main experiment (gap vs. uniform, d=1)
# ---------------------------------------------------------------------------

def build_gap_and_uniform_datasets(components, x_star, N, gap_radius=0.5, gap_fraction=0.02, seed=SEED):
    f_star = lambda x: tier_b_f(x, components)
    x_gap = sample_with_gap(N, DOMAIN, 1, gap_center=x_star, gap_radius=gap_radius, gap_fraction=gap_fraction, seed=seed)
    x_uniform = sample_uniform(N, DOMAIN, 1, seed=seed)
    datasets = {}
    for key, x in [("gap", x_gap), ("uniform", x_uniform)]:
        x_train, y_train = make_dataset(f_star, x, noiseless=True)
        datasets[key] = {"x_train": x_train, "y_train": y_train}
    return datasets


def train_tiny_mlp(x_train, y_train, hidden_sizes=(64, 64), activation="tanh", epochs=3000, lr=1e-2, seed=SEED):
    model = TinyMLP(input_dim=x_train.shape[1], hidden_sizes=hidden_sizes, activation=activation)
    model, history = train_regressor(model, x_train, y_train, epochs=epochs, lr=lr, seed=seed)
    return model, history


def run_main_experiment(N=400, gap_radius=0.5, gap_fraction=0.02, local_radius=0.2, local_n_samples=200,
                         hidden_sizes=(64, 64), epochs=3000, lr=1e-2, grid_size=1000, verbose=True):
    components, L_star, x_star = build_tier_b_1d()
    f_star = lambda x: tier_b_f(x, components)
    if verbose:
        print(f"\n=== Tier B main experiment: L*={L_star:.4f}, x*={x_star.tolist()} ===")

    datasets = build_gap_and_uniform_datasets(components, x_star, N, gap_radius, gap_fraction)

    x_grid = torch.linspace(DOMAIN[0], DOMAIN[1], grid_size).unsqueeze(-1)
    f_star_vals = f_star(x_grid).detach()

    dataset_results = {}
    report = {}
    for key, ds in datasets.items():
        x_train, y_train = ds["x_train"], ds["y_train"]
        model, history = train_tiny_mlp(x_train, y_train, hidden_sizes, epochs=epochs, lr=lr)

        L_hat_data, _, _ = pairwise_lipschitz(x_train, y_train, norm="l2")

        y_pred_train = model(x_train).detach()
        L_hat_model_pairwise_train, _, _ = pairwise_lipschitz(x_train, y_pred_train, norm="l2")
        y_pred_grid = model(x_grid).detach()
        L_hat_model_pairwise_grid, _, _ = pairwise_lipschitz(x_grid, y_pred_grid, norm="l2")

        local_lipschitz_vals = local_perturbation_lipschitz_grid(
            model, x_grid, radius=local_radius, n_samples=local_n_samples, norm="l2", seed=SEED)
        grad_norm_vals = gradient_norm_estimate_grid(model, x_grid, norm="l2")

        dataset_results[key] = {
            "x_train": x_train.squeeze(-1),
            "f_hat_vals": y_pred_grid,
            "local_lipschitz_vals": local_lipschitz_vals,
            "grad_norm_vals": grad_norm_vals,
        }

        in_gap = (x_grid.squeeze(-1) - x_star.item()).abs() < gap_radius
        max_in_gap = local_lipschitz_vals[in_gap].max().item() if in_gap.any() else float("nan")
        max_outside_gap = local_lipschitz_vals[~in_gap].max().item()

        report[key] = {
            "L_hat_data": L_hat_data,
            "L_hat_model_pairwise_train": L_hat_model_pairwise_train,
            "L_hat_model_pairwise_grid": L_hat_model_pairwise_grid,
            "final_train_mse": history[-1],
            "max_local_lipschitz_in_gap": max_in_gap,
            "max_local_lipschitz_outside_gap": max_outside_gap,
        }

        if verbose:
            print(f"\n--- {key} dataset ---")
            for k, v in report[key].items():
                print(f"  {k}: {v}")

    RESULTS_DIR.mkdir(exist_ok=True)
    fig = plots.plot_gap_vs_uniform(x_grid.squeeze(-1).numpy(), f_star_vals.numpy(), dataset_results, L_star,
                                     save_path=RESULTS_DIR / "step6_gap_vs_uniform.png")

    return {"L_star": L_star, "x_star": x_star, "report": report, "dataset_results": dataset_results, "figure": fig}


# ---------------------------------------------------------------------------
# Step 7: sweeps
# ---------------------------------------------------------------------------

def _build_dataset(dataset_type, components, x_star, N, gap_radius, gap_fraction, seed):
    f_star = lambda x: tier_b_f(x, components)
    if dataset_type == "gap":
        x = sample_with_gap(N, DOMAIN, 1, gap_center=x_star, gap_radius=gap_radius, gap_fraction=gap_fraction, seed=seed)
    elif dataset_type == "uniform":
        x = sample_uniform(N, DOMAIN, 1, seed=seed)
    else:
        raise ValueError(dataset_type)
    return make_dataset(f_star, x, noiseless=True)


def sweep_over_N(dataset_type, components, L_star, x_star, held_out_grid, N_values=(50, 100, 200, 500, 1000, 2000, 5000),
                  hidden_sizes=(64, 64), epochs=2000, lr=1e-2, gap_radius=0.5, gap_fraction=0.02, verbose=True):
    L_hat_data_vals, L_hat_model_vals = [], []
    for N in N_values:
        x_train, y_train = _build_dataset(dataset_type, components, x_star, N, gap_radius, gap_fraction, seed=SEED)
        L_hat_data, _, _ = pairwise_lipschitz(x_train, y_train, norm="l2")

        model, _ = train_tiny_mlp(x_train, y_train, hidden_sizes, epochs=epochs, lr=lr)
        y_pred_grid = model(held_out_grid).detach()
        L_hat_model, _, _ = pairwise_lipschitz(held_out_grid, y_pred_grid, norm="l2")

        L_hat_data_vals.append(L_hat_data)
        L_hat_model_vals.append(L_hat_model)
        if verbose:
            print(f"  [{dataset_type}] N={N:5d}  L_hat_data={L_hat_data:.3f}  L_hat_model={L_hat_model:.3f}")

    return np.array(L_hat_data_vals), np.array(L_hat_model_vals)


def sweep_over_capacity(dataset_type, components, L_star, x_star, held_out_grid, widths=(4, 8, 16, 32, 64, 128),
                         N=500, epochs=2000, lr=1e-2, gap_radius=0.5, gap_fraction=0.02, verbose=True):
    x_train, y_train = _build_dataset(dataset_type, components, x_star, N, gap_radius, gap_fraction, seed=SEED)
    L_hat_data, _, _ = pairwise_lipschitz(x_train, y_train, norm="l2")

    L_hat_data_vals, L_hat_model_vals = [], []
    for width in widths:
        model, _ = train_tiny_mlp(x_train, y_train, hidden_sizes=(width, width), epochs=epochs, lr=lr)
        y_pred_grid = model(held_out_grid).detach()
        L_hat_model, _, _ = pairwise_lipschitz(held_out_grid, y_pred_grid, norm="l2")

        L_hat_data_vals.append(L_hat_data)  # constant across widths (fixed N)
        L_hat_model_vals.append(L_hat_model)
        if verbose:
            print(f"  [{dataset_type}] width={width:4d}  L_hat_data={L_hat_data:.3f}  L_hat_model={L_hat_model:.3f}")

    return np.array(L_hat_data_vals), np.array(L_hat_model_vals)


def run_sweeps(N_values=(50, 100, 200, 500, 1000, 2000, 5000), widths=(4, 8, 16, 32, 64, 128),
               capacity_N=500, held_out_grid_size=2000, verbose=True):
    components, L_star, x_star = build_tier_b_1d()
    held_out_grid = torch.linspace(DOMAIN[0], DOMAIN[1], held_out_grid_size).unsqueeze(-1)

    if verbose:
        print(f"=== Step 7 sweeps: ground truth L* = {L_star:.4f} (x* = {x_star.tolist()}) ===")

    RESULTS_DIR.mkdir(exist_ok=True)
    all_results = {}

    for dataset_type in ["uniform", "gap"]:
        if verbose:
            print(f"\n=== N-sweep ({dataset_type}) ===")
        L_hat_data_N, L_hat_model_N = sweep_over_N(dataset_type, components, L_star, x_star, held_out_grid,
                                                     N_values=N_values, verbose=verbose)
        fig_N = plots.plot_sweep(N_values, L_star, L_hat_data_N, L_hat_model_N, xlabel="N (training samples)",
                                  title=f"L* vs. N ({dataset_type} sampling)",
                                  save_path=RESULTS_DIR / f"step7_N_sweep_{dataset_type}.png")

        if verbose:
            print(f"\n=== capacity-sweep ({dataset_type}) ===")
        L_hat_data_w, L_hat_model_w = sweep_over_capacity(dataset_type, components, L_star, x_star, held_out_grid,
                                                            widths=widths, N=capacity_N, verbose=verbose)
        fig_w = plots.plot_sweep(widths, L_star, L_hat_data_w, L_hat_model_w, xlabel="hidden width",
                                  title=f"L* vs. model capacity ({dataset_type} sampling, N={capacity_N})",
                                  save_path=RESULTS_DIR / f"step7_capacity_sweep_{dataset_type}.png")

        all_results[dataset_type] = {
            "N_values": np.array(N_values), "L_hat_data_N": L_hat_data_N, "L_hat_model_N": L_hat_model_N,
            "widths": np.array(widths), "L_hat_data_w": L_hat_data_w, "L_hat_model_w": L_hat_model_w,
        }

        np.savez(RESULTS_DIR / f"step7_sweeps_{dataset_type}.npz", L_star=L_star, **all_results[dataset_type])

    return {"L_star": L_star, "results": all_results}


# ---------------------------------------------------------------------------
# Step 8: 2D extension
# ---------------------------------------------------------------------------

def run_2d_extension(N=800, gap_radius=0.7, gap_fraction=0.03, heatmap_grid_side=40, local_radius=0.3,
                      local_n_samples=30, hidden_sizes=(64, 64), epochs=3000, lr=1e-2, verbose=True):
    components, L_star, x_star = build_tier_b_2d()
    f_star = lambda x: tier_b_f(x, components)
    if verbose:
        print(f"\n=== 2D extension: L*={L_star:.4f}, x*={x_star.tolist()} ===")

    x = sample_with_gap(N, DOMAIN, 2, gap_center=x_star, gap_radius=gap_radius, gap_fraction=gap_fraction, seed=SEED)
    x_train, y_train = make_dataset(f_star, x, noiseless=True)
    model, history = train_tiny_mlp(x_train, y_train, hidden_sizes, epochs=epochs, lr=lr)
    if verbose:
        print(f"  final train MSE: {history[-1]:.6f}")

    lo, hi = DOMAIN
    g1 = torch.linspace(lo, hi, heatmap_grid_side)
    g2 = torch.linspace(lo, hi, heatmap_grid_side)
    gg1, gg2 = torch.meshgrid(g1, g2, indexing="ij")
    grid_x = torch.stack([gg1.reshape(-1), gg2.reshape(-1)], dim=-1)

    true_grad_norm = tier_b_grad(grid_x, components).norm(p=2, dim=-1).detach()
    model_grad_norm = gradient_norm_estimate_grid(model, grid_x, norm="l2")
    local_lipschitz = local_perturbation_lipschitz_grid(model, grid_x, radius=local_radius,
                                                          n_samples=local_n_samples, norm="l2", seed=SEED)

    shape = (heatmap_grid_side, heatmap_grid_side)
    RESULTS_DIR.mkdir(exist_ok=True)
    fig = plots.plot_2d_heatmaps(
        gg1.numpy(), gg2.numpy(),
        true_grad_norm.reshape(shape).numpy(),
        model_grad_norm.reshape(shape).numpy(),
        local_lipschitz.reshape(shape).numpy(),
        x_train.numpy(),
        save_path=RESULTS_DIR / "step8_2d_heatmaps.png",
    )

    np.savez(RESULTS_DIR / "step8_2d_results.npz", L_star=L_star, x_star=x_star.numpy(),
             true_grad_norm=true_grad_norm.reshape(shape).numpy(),
             model_grad_norm=model_grad_norm.reshape(shape).numpy(),
             local_lipschitz=local_lipschitz.reshape(shape).numpy(),
             x_train=x_train.numpy())

    return {"L_star": L_star, "x_star": x_star, "figure": fig}


# ---------------------------------------------------------------------------

def main():
    run_tier_a_sanity()
    run_main_experiment()
    run_sweeps()
    run_2d_extension()


if __name__ == "__main__":
    main()
