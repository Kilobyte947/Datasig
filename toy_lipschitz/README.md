inc# toy_lipschitz

Experiment 1 of the Lipschitz-diagnostics project: a toy 1D/2D regression
testbed where the true Lipschitz constant `L*` of the ground-truth function
is known essentially exactly, used to check how well `L*` can be recovered
from data alone versus from a trained model.

Three quantities are compared across sample size `N` and model capacity:

- **`L*`** — the true Lipschitz constant (closed-form for a single ridge,
  grid-search + gradient-ascent-refined for a sum of ridges).
- **`L_hat_data`** — a pairwise empirical estimate computed directly from
  sampled `(x_i, f*(x_i))` pairs, no model involved.
- **`L_hat_model`** — estimates computed from a trained model `f_hat`, via
  three distinct sub-methods: pairwise, local-perturbation (finite
  difference), and gradient-norm (autograd/infinitesimal). These three are
  kept as separate, separately-labeled quantities throughout — never
  merged into a single "local estimate."

The headline result: training on a dataset with a deliberate sampling gap
near the steepest region of `f*` causes the trained model to *flatten*
(undershoot `L*`) there, and this undershoot persists even as model
capacity grows — see [Key results](#key-results) below.

## Layout

| File | Contents |
|---|---|
| `toy_functions.py` | Ground-truth `f*`. **Tier A**: single ridge `A*tanh(w^Tx+b)` with closed-form `L* = A*\|\|w\|\|`. **Tier B**: sum of 2-3 ridges; `tier_b_true_L` estimates `L*` via a dense grid search refined by `torch.optim.LBFGS` gradient ascent from multiple random restarts, returning `(L_star, x_star)`. |
| `data.py` | Sampling schemes: `sample_uniform`, `sample_with_gap` (rejection sampling — most points drawn outside an L2 ball around a "gap center," a small fraction drawn inside it), `make_dataset`. |
| `estimators.py` | The three core estimators — `pairwise_lipschitz`, `local_perturbation_lipschitz`, `gradient_norm_estimate` — plus grid-evaluating variants (`gradient_norm_estimate_grid`, `local_perturbation_lipschitz_grid`) that return the full array of per-point estimates, needed for the sweep and heatmap plots. This module is meant to become the first draft of a shared `lipschitz_diagnostics.py` for Experiments 2-4. |
| `models.py` | `TinyMLP` (configurable depth/width, tanh or relu), `SingleTanhUnit` (matches the Tier A functional form exactly, used only for the sanity check), `train_regressor` (plain MSE/Adam training loop). |
| `plots.py` | All plotting logic: `plot_gap_vs_uniform` (f*, f_hat, and local-Lipschitz-vs-x with a training-point rug plot, gap vs. uniform side by side), `plot_sweep` (3-curve L* / L_hat_data / L_hat_model vs. N or width), `plot_2d_heatmaps` (true/model/finite-diff gradient-norm heatmaps with training points overlaid). |
| `run_experiment.py` | Driver wiring everything together: `run_tier_a_sanity`, `run_main_experiment` (Tier B, gap vs. uniform), `run_sweeps` (N-sweep and capacity-sweep, both dataset types), `run_2d_extension` (Step 8). Saves figures and `.npz`/`.csv`-equivalent results to `results/`. |
| `tests/test_tier_a_closed_form.py` | Checks the hand-derived analytic gradient against `torch.autograd.grad` — must pass before anything else is trusted. |
| `tests/test_estimators.py` | Checks `pairwise_lipschitz` converges toward `tier_a_true_L` as N grows, and `gradient_norm_estimate` matches it almost exactly at the true argmax. |
| `notebook_toy_lipschitz.ipynb` | Thin driver notebook — imports from this package and displays the figures produced by `run_experiment.py`. Contains no reusable logic of its own. |
| `results/` | Generated outputs (git-ignored except `.gitkeep`): step6/7/8 plots and `.npz` result arrays. |

## Design decisions

- **float64 everywhere** — every module calls `torch.set_default_dtype(torch.float64)` at import time, so the true-vs-estimate comparisons aren't contaminated by float32 noise near the true maximum.
- **`domain` is a single `(low, high)` tuple** applied isotropically to every dimension (e.g. `(-5.0, 5.0)` for both d=1 and d=2), not a per-axis list.
- **No `scipy` dependency** — Tier B's gradient-ascent refinement of `L*` uses `torch.optim.LBFGS` instead of `scipy.optimize.minimize`, since the ground-truth gradient (`tier_b_grad`) is itself a plain differentiable torch expression and doesn't need a second autograd trick.
- **`tier_a_true_L(norm='l1')` is a documented simplification** — it returns `A * ||w||_1`, not the true L1-distance dual norm `A * ||w||_inf`. All correctness checkpoints use `norm='l2'`, where the dual-norm identity holds exactly, so this doesn't affect any test.
- **Gap sampling is rejection-based**, not analytic — cheap and exact enough at d=1/d=2.

## How to run it

```bash
# from the repo root
.venv/bin/python -m pytest toy_lipschitz/tests/ -v

.venv/bin/python -c "from toy_lipschitz.run_experiment import main; main()"

# or execute the notebook end-to-end
.venv/bin/jupyter nbconvert --to notebook --execute --inplace toy_lipschitz/notebook_toy_lipschitz.ipynb
```

`run_experiment.main()` runs, in order: `run_tier_a_sanity()` (asserts
agreement within tolerance, halts if not), `run_main_experiment()`,
`run_sweeps()`, `run_2d_extension()`.

## Key results

- **Tier A sanity check**: `L*`, `L_hat_data`, and all three `L_hat_model`
  sub-methods agree to within ~1% (tolerance is 10%).
- **Tier B (1D)**: `L* ≈ 8.46`, attained at `x* ≈ -0.37`.
- **Gap vs. uniform** (`results/step6_gap_vs_uniform.png`): the
  finite-difference local-Lipschitz curve peaks noticeably lower near `x*`
  for the gap-sampled model than for the uniformly-sampled model — direct
  visual evidence of flattening/undershoot from sparse coverage.
- **Capacity sweep on the gap dataset** (`results/step7_capacity_sweep_gap.png`):
  `L_hat_model` stays persistently below both `L_hat_data` and `L*` across
  every hidden width tested (4 to 128) — increasing model capacity alone
  does not compensate for a data-coverage gap.
- **2D extension** (`results/step8_2d_heatmaps.png`): true, model
  (autograd), and finite-difference gradient-norm heatmaps over an
  anisotropic 3-ridge function are visually consistent with each other,
  correctly locating the ridge-intersection hotspot.

## Status

This is Experiment 1 of a larger project studying Lipschitz-constant
estimability (true vs. data vs. trained-model), which motivates later
experiments on MNIST/Fashion-MNIST. `estimators.py` is written generally
enough (arbitrary `f`, arbitrary `d`) to be imported directly by those
experiments rather than reimplemented.
