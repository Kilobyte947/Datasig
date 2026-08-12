# toy_lipschitz

Experiment 1 of the Lipschitz-diagnostics project: a toy 1D/2D regression
testbed where the true Lipschitz constant `L*` of the ground-truth function
is known essentially exactly, used to check how well `L*` can be recovered
from data alone versus from a trained model — and, since Step 9, how much
the choice of *distance metric* used inside that recovery matters.

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

Two headline results:

- Training on a dataset with a deliberate sampling gap near the steepest
  region of `f*` causes the trained model to *flatten* (undershoot `L*`)
  there, and this undershoot persists even as model capacity grows (Steps
  6-7).
- All of the above implicitly assumes plain Euclidean distance is the
  right way to measure "closeness" between input points. Step 9 tests
  that assumption directly: switching to a Mahalanobis distance derived
  from a polynomial embedding of `x`, on the *same* raw data, cuts a
  19%-off data-only estimate down to <1% error.

See [Key results](#key-results) below for the numbers.

## Layout

| File | Contents |
|---|---|
| `toy_functions.py` | Ground-truth `f*`. **Tier A**: single ridge `A*tanh(w^Tx+b)` with closed-form `L* = A*\|\|w\|\|`. **Tier B**: sum of 2-3 ridges; `tier_b_true_L` estimates `L*` via a dense grid search refined by `torch.optim.LBFGS` gradient ascent from multiple random restarts, returning `(L_star, x_star)`. Also defines a **piecewise-linear** ground truth (`piecewise_ramp_f`/`piecewise_sum_f`) whose Lipschitz constant is exactly the largest ramp slope, no numerical search needed — used to test whether matching the model's activation to `f*`'s functional form (smooth tanh vs. piecewise-linear ReLU) affects recovery (`run_cross_architecture_check`; not currently called from the notebook, see [Status](#status)). |
| `data.py` | Sampling schemes: `sample_uniform`, `sample_with_gap` (rejection sampling — most points drawn outside an L2 ball around a "gap center," a small fraction drawn inside it), `make_dataset` (always noiseless — `y = f_star(x)` exactly; if noise is ever needed it belongs in the definition of `f_star` itself, not as a stochastic wrapper). |
| `estimators.py` | The three core estimators — `pairwise_lipschitz`, `local_perturbation_lipschitz`, `gradient_norm_estimate` — plus grid-evaluating variants (`gradient_norm_estimate_grid`, `local_perturbation_lipschitz_grid`) that return the full array of per-point estimates, needed for the sweep and heatmap plots. `pairwise_lipschitz` and `local_perturbation_lipschitz` (+ their grid variants) also accept an optional `embed_fn`/`precision` pair (Step 9): supply both to measure distance as Mahalanobis distance in an embedded feature space instead of plain Euclidean; leave both unset and behavior is unchanged. `local_sample_density` counts nearby training points per query location (a coverage diagnostic, not a Lipschitz estimate — meant to be plotted *alongside* a Lipschitz heatmap, not merged into it; not yet wired into the notebook). This module is meant to become the first draft of a shared `lipschitz_diagnostics.py` for Experiments 2-4. |
| `embeddings.py` | **New in Step 9.** `polynomial_embedding` (`x -> (x, x^2, ..., x^degree)`, 1D only), `augmented_embedding` (the same, with `f(x)` appended as an extra feature — implemented but deliberately not used anywhere, see [Design decisions](#design-decisions)), `empirical_covariance`, `precision_from_covariance` (inverts the covariance to get the quadratic-form matrix for Mahalanobis distance). |
| `models.py` | `TinyMLP` (configurable depth/width, tanh or relu), `SingleTanhUnit` (matches the Tier A functional form exactly, used only for the sanity check), `train_regressor` (plain MSE/Adam training loop). |
| `plots.py` | All plotting logic. Original: `plot_gap_vs_uniform` (f*, f_hat, and local-Lipschitz-vs-x with a training-point rug plot, gap vs. uniform side by side), `plot_sweep` (3-curve L* / L_hat_data / L_hat_model vs. N or width), `plot_2d_heatmaps` (true/model/finite-diff gradient-norm heatmaps with training points overlaid). Added for Step 9: `plot_local_vs_global_lipschitz` (global scalar estimates as reference lines, local curves overlaid, plain vs. Mahalanobis, gap vs. uniform side by side), `plot_degree_sweep` (relative error vs. `L*` and covariance condition number, both vs. polynomial embedding degree). Also present but not yet called by any driver function: `plot_coverage_heatmap` (local training-point density, the visualization counterpart of `local_sample_density`). |
| `run_experiment.py` | Driver wiring everything together. Original: `run_tier_a_sanity`, `run_main_experiment` (Tier B, gap vs. uniform), `run_sweeps` (N-sweep and capacity-sweep, both dataset types), `run_2d_extension` (Step 8). Added since: `run_tier_a_gap_demo` (Tier A analogue of the gap-vs-uniform comparison, using `TinyMLP` instead of `SingleTanhUnit` so undershoot is actually observable — also computes the Mahalanobis-vs-plain local/global comparison used in Step 9's plot), `run_metric_embedding_check` and `sweep_polynomial_degree` (Step 9's global metric comparison and degree selection), `build_gap_dataset_and_embedding` (shared setup for both), `run_cross_architecture_check` (defined and included in `main()`, but not called from the notebook — see [Status](#status)). Saves figures and `.npz` results to `results/`. |
| `new_distance_measure.md` | **New in Step 9.** Standalone write-up of the Mahalanobis-in-polynomial-embedding extension: the idea, the headline result table, the degree-selection table, and why `f(x)` was deliberately left out of the embedding. |
| `tests/test_tier_a_closed_form.py` | Checks the hand-derived analytic gradient against `torch.autograd.grad` — must pass before anything else is trusted. |
| `tests/test_estimators.py` | Checks `pairwise_lipschitz` converges toward `tier_a_true_L` as N grows, and `gradient_norm_estimate` matches it almost exactly at the true argmax. Also checks the Mahalanobis path (Step 9): `pairwise_lipschitz` under a degree-1 polynomial embedding reduces to a closed-form scaled-Euclidean identity, and `local_perturbation_lipschitz` under an embedding converges to its analytic pointwise pullback-metric value as the perturbation radius shrinks. |
| `notebook_toy_lipschitz.ipynb` | Thin driver notebook — imports from this package and displays the figures produced by `run_experiment.py`. Contains no reusable logic of its own. Covers Steps 5 through 9 (see [Notebook contents](#notebook-contents)). |
| `results/` | Generated outputs (git-ignored except `.gitkeep`): step6/7/8 plots and `.npz` result arrays, plus Step 9's `tier_a_gap_vs_uniform.png`, `tier_a_local_vs_global_lipschitz.png`, and `degree_sweep.png`. |

## Notebook contents

`notebook_toy_lipschitz.ipynb` runs, in order:

1. **Step 5** — Tier A sanity check (`run_tier_a_sanity`): the whole
   pipeline validated against a closed-form answer.
2. **Tier A gap demo** (`run_tier_a_gap_demo`): the flattening effect
   shown on the simplest possible ground truth (single ridge), using
   `TinyMLP` so the undershoot is actually observable; also computes the
   plain-vs-Mahalanobis comparison used later in Step 9.
3. **Step 6** — main experiment (`run_main_experiment`): the same
   gap-vs-uniform comparison on the harder Tier B (multi-ridge) function.
4. **Step 7** — N-sweep and capacity-sweep (`run_sweeps`), both dataset
   types, `L*` printed explicitly alongside the curves.
5. **Step 8** — 2D extension (`run_2d_extension`): heatmaps over
   `[-5,5]^2`.
6. **Step 9** — does the distance metric matter? (`run_metric_embedding_check`,
   `sweep_polynomial_degree`, and the local-vs-global figure computed
   earlier in the Tier A gap demo). See [Key results](#key-results) below.
7. **Summary and status** cell.

## Design decisions

- **float64 everywhere** — every module calls `torch.set_default_dtype(torch.float64)` at import time, so the true-vs-estimate comparisons aren't contaminated by float32 noise near the true maximum.
- **`domain` is a single `(low, high)` tuple** applied isotropically to every dimension (e.g. `(-5.0, 5.0)` for both d=1 and d=2), not a per-axis list.
- **No `scipy` dependency** — Tier B's gradient-ascent refinement of `L*` uses `torch.optim.LBFGS` instead of `scipy.optimize.minimize`, since the ground-truth gradient (`tier_b_grad`) is itself a plain differentiable torch expression and doesn't need a second autograd trick.
- **`tier_a_true_L(norm='l1')` is a documented simplification** — it returns `A * ||w||_1`, not the true L1-distance dual norm `A * ||w||_inf`. All correctness checkpoints use `norm='l2'`, where the dual-norm identity holds exactly, so this doesn't affect any test.
- **Gap sampling is rejection-based**, not analytic — cheap and exact enough at d=1/d=2.
- **`make_dataset` is always noiseless** (`y = f_star(x)` exactly) — noise, if ever needed, belongs in the definition of `f_star` itself (a different, still-fixed function), not as a stochastic perturbation layered on top of a fixed function's output.
- **The polynomial embedding never includes `f(x)`** — `augmented_embedding` supports appending `f(x)` as a feature, but using it to measure the Lipschitz behavior of the *same* function it's built from turns out to be self-cancelling (checked directly, not assumed): it collapses the gap-vs-uniform local-Lipschitz contrast from ~6x down to ~1x, erasing exactly the effect this whole project exists to detect. Full explanation in `new_distance_measure.md`.
- **Embedding degree 3 was chosen by measuring, not guessing** — `sweep_polynomial_degree` checks both accuracy against `L*` and the condition number of the fitted covariance, since a higher degree can look more accurate on one dataset while being numerically fragile (collinear polynomial powers on a bounded domain). Degree 3 is the lowest degree that is both accurate (~0.08% error) and well-conditioned (~1.5e3); degrees 5-6 look tempting on raw feature count but blow past 1e6 condition number and lose accuracy along with it.
- **The Mahalanobis extension is 1D only** — `polynomial_embedding` doesn't support `d=2`; extending it to the Tier B 2D dataset would need a 2D polynomial feature map.

## How to run it

```bash
# from the repo root
.venv/bin/python -m pytest toy_lipschitz/tests/ -v

.venv/bin/python -c "from toy_lipschitz.run_experiment import main; main()"

# or execute the notebook end-to-end
.venv/bin/jupyter nbconvert --to notebook --execute --inplace toy_lipschitz/notebook_toy_lipschitz.ipynb
```

`run_experiment.main()` runs, in order: `run_tier_a_sanity()` (asserts
agreement within tolerance, halts if not), `run_tier_a_gap_demo()`,
`run_main_experiment()`, `run_sweeps()`, `run_2d_extension()`,
`run_cross_architecture_check()`, `run_metric_embedding_check()`,
`sweep_polynomial_degree()`.

## Key results

- **Tier A sanity check**: `L*`, `L_hat_data`, and all three `L_hat_model`
  sub-methods agree to within ~1% (tolerance is 10%).
- **Tier A gap demo** (`results/tier_a_gap_vs_uniform.png`, `L*=15`): the
  gap-sampled model's local-Lipschitz peak near `x*` tops out around
  `8.2`, barely half of `L*`, while the uniformly-sampled model reaches
  `15.0` almost exactly — the same undershoot as Step 6, sharper and
  easier to see on this simpler ground truth.
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
- **Step 9 — the distance metric matters** (`run_metric_embedding_check`,
  `L*=6.0`): on identical raw data, `L_hat_data` computed with plain
  Euclidean distance is `~4.87` (19% under `L*`); the same data under a
  Mahalanobis distance in a degree-3 polynomial embedding gives `~6.01`
  (<1% off). `results/tier_a_local_vs_global_lipschitz.png` shows this
  holds at the local (per-point), not just global, level.
- **Step 9 — degree selection isn't free** (`sweep_polynomial_degree`,
  `results/degree_sweep.png`): degrees 1-2 are too simple to capture the
  effect at all (>130% error); degree 3 is the sweet spot (`0.08%` error,
  `cond(cov) ≈ 1.5e3`); degrees 5-6 have more raw features but the
  covariance's condition number blows past `1e6` and accuracy collapses
  with it — more parameters here buys overfitting to sampling noise, not
  a better metric.

## Status

This is Experiment 1 of a larger project studying Lipschitz-constant
estimability (true vs. data vs. trained-model), which motivates later
experiments on MNIST/Fashion-MNIST. `estimators.py` is written generally
enough (arbitrary `f`, arbitrary `d`) to be imported directly by those
experiments rather than reimplemented.

Two pieces of the codebase exist and are exercised by `main()` /
`tests/`, but are **not yet wired into the notebook**: `run_cross_architecture_check`
(a 2x2 check of whether matching the model's activation to `f*`'s
functional form — smooth tanh vs. piecewise-linear ReLU — affects
recovery), and the `local_sample_density` / `plot_coverage_heatmap` pair
(a training-point-density diagnostic meant to sit alongside the 2D
heatmaps, distinguishing "tested and found smooth" from "never really
tested"). Both are candidates for a future Step 10 write-up rather than
gaps in the current one.
