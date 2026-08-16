# toy_lipschitz

A toy 1D/2D regression testbed for studying how well the Lipschitz constant
of a function can be recovered — from raw data alone, and from a model
trained on that data — when the true value is known essentially exactly.
It's Experiment 1 in a broader Lipschitz-diagnostics project; `estimators.py`
is written generally enough (arbitrary `f`, arbitrary input dimension `d`)
to be reused directly by later experiments rather than reimplemented (see
the sibling `mnist_lipschitz` folder).

## What's being compared

For a ground-truth function `f*` with true Lipschitz constant `L*`, three
quantities are computed and compared across sample size `N` and model
capacity:

- **`L*`** — the true Lipschitz constant: closed-form for a single ridge,
  grid-search + gradient-ascent-refined for a sum of ridges.
- **`L_hat_data`** — a pairwise empirical estimate computed directly from
  sampled `(x_i, f*(x_i))` pairs, no model involved.
- **`L_hat_model`** — estimates computed from a trained model `f_hat`, via
  three distinct sub-methods: pairwise, local-perturbation (finite
  difference), and gradient-norm (autograd/infinitesimal). These are kept
  as separate, separately-labeled quantities throughout — never merged
  into a single "local estimate," since they measure related but distinct
  things.

A second, independent question is whether the *distance metric* used
inside these estimates matters: all of them default to plain Euclidean
distance between input points, but that's a modeling choice, not a law.
Both `pairwise_lipschitz` and `local_perturbation_lipschitz` accept an
optional Mahalanobis distance instead, derived from a polynomial embedding
of the data (see [Distance metric: Euclidean vs. Mahalanobis](#distance-metric-euclidean-vs-mahalanobis)).

## How it's designed

**Ground truth (`toy_functions.py`).** Two tiers of `f*`, both with an
independently-verifiable Lipschitz constant so `L*` isn't itself an
estimate:

- **Tier A** — a single ridge, `f*(x) = A * tanh(w^T x + b)`. The steepest
  point of an S-curve has an exactly known slope, so `L* = A * ||w||`
  closed-form.
- **Tier B** — a sum of 2-3 such ridges (no closed form, more like a real
  decision boundary). `tier_b_true_L` locates the steepest point via a
  dense grid search refined by gradient ascent (`torch.optim.LBFGS`) from
  multiple random restarts, returning `(L_star, x_star)`.
- A **piecewise-linear** ground truth (`piecewise_ramp_f`/`piecewise_sum_f`)
  is also defined, whose Lipschitz constant is exactly the largest ramp
  slope — no numerical search needed. It exists to test whether matching
  a model's activation function to `f*`'s functional form (smooth tanh vs.
  piecewise-linear ReLU) affects how well `L*` is recovered
  (`run_cross_architecture_check`).

**Sampling (`data.py`).** Two schemes generate the `x` values that `f*` is
evaluated at:

- `sample_uniform` — i.i.d. uniform over the domain.
- `sample_with_gap` — the same, except a deliberate sampling gap is carved
  out: most points are drawn outside an L2 ball around a "gap center"
  (typically `f*`'s steepest point, since that's where starving data
  should hurt most), and only a small fraction are drawn inside it, via
  rejection sampling.

`make_dataset` is always noiseless (`y = f_star(x)` exactly) — if noise is
ever needed, it belongs in the definition of `f_star` itself, not as a
stochastic wrapper on top of a fixed function.

**Models (`models.py`).** `TinyMLP` (configurable depth/width, tanh or
relu) is the model used almost everywhere. `SingleTanhUnit` matches Tier
A's functional form exactly (`A * tanh(w^Tx+b)` with `A`/`w`/`b` as
learnable parameters) and is used only for the closed-form sanity check,
where an architecturally-matched model is the point.

**Estimators (`estimators.py`).** `pairwise_lipschitz` takes the biggest
ratio `|y_i - y_j| / d(x_i, x_j)` over all (or a random subsample of)
point pairs. `local_perturbation_lipschitz` samples points within a
radius of a query point and takes the biggest finite-difference ratio
there. `gradient_norm_estimate` is the exact autograd gradient norm at a
point — infinitesimal rather than finite-difference. Grid-evaluating
variants (`gradient_norm_estimate_grid`, `local_perturbation_lipschitz_grid`)
return the full array of per-point estimates rather than just the max,
needed for sweep and heatmap plots.

**Coverage diagnostic (`data.py::local_sample_density`).** Not a
Lipschitz quantity — a count of how many training points fall within a
given radius of each query location. A low local-Lipschitz estimate can
mean "genuinely smooth here" or just "barely sampled here," and this is
what tells the two apart. Used alongside the 2D Lipschitz heatmaps,
plotted as its own figure rather than folded into them.

### Distance metric: Euclidean vs. Mahalanobis

Every estimator above defaults to plain Euclidean (or L1) distance
between raw input points. `embeddings.py` provides an alternative:

1. **Embed.** Map `x -> (x, x^2, ..., x^degree)` (`polynomial_embedding`,
   1D only). A linear function of this embedding is a degree-`degree`
   polynomial in `x`, so distance measured in the embedded space can
   reflect curvature that raw Euclidean distance can't.
2. **Derive a metric from the data.** Compute the empirical covariance of
   the embedded training points (`empirical_covariance`), invert it to
   get a precision matrix (`precision_from_covariance`), and use that as
   the quadratic form for a Mahalanobis-style distance,
   `d(a,b)^2 = (a-b)^T Sigma^-1 (a-b)` — which reweights each direction
   in inverse proportion to how much the data naturally varies along it.
3. **Plug it in.** `pairwise_lipschitz` and `local_perturbation_lipschitz`
   (and their grid variants) accept optional `embed_fn`/`precision`
   arguments. Leave both unset and behavior is exactly plain Euclidean;
   supply both and distance is computed in the embedded space instead.

`augmented_embedding` additionally supports appending `f(x)` itself as an
embedding feature, but no driver currently uses it: measuring a
function's Lipschitz behavior with a metric built from that same
function's own output turns out to be self-cancelling (checked directly —
see [Design decisions](#design-decisions)). It's kept available for a
setting where the embedded function differs from the one being measured.

The embedding is currently 1D only (`polynomial_embedding` doesn't
support `d=2`); extending it to the Tier B 2D dataset would need a 2D
polynomial feature map.

## File reference

| File | Contents |
|---|---|
| `toy_functions.py` | Ground-truth `f*` for both tiers, their true Lipschitz constants, and the piecewise-linear ground truth used for the activation-matching check. |
| `data.py` | Sampling schemes (`sample_uniform`, `sample_with_gap`), `make_dataset`, and the coverage diagnostic `local_sample_density`. |
| `estimators.py` | The three Lipschitz estimators (`pairwise_lipschitz`, `local_perturbation_lipschitz`, `gradient_norm_estimate`) plus their grid-evaluating variants. `pairwise_lipschitz` and `local_perturbation_lipschitz` accept the optional `embed_fn`/`precision` pair for Mahalanobis distance. |
| `embeddings.py` | `polynomial_embedding`, `augmented_embedding`, `empirical_covariance`, `precision_from_covariance`, and `_mahalanobis_dist` — the full pipeline for deriving and applying a Mahalanobis distance from an embedded feature space. |
| `models.py` | `TinyMLP`, `SingleTanhUnit`, and `train_regressor` (plain MSE/Adam training loop). |
| `plots.py` | All plotting logic: `plot_gap_vs_uniform`, `plot_sweep`, `plot_2d_heatmaps`, `plot_local_vs_global_lipschitz`, `plot_degree_sweep`, `plot_coverage_heatmap`, `plot_seed_averaged_sweep`. |
| `run_experiment.py` | Driver wiring everything together — see [How to run it](#how-to-run-it) for the full list of entry points. Saves figures and `.npz` result arrays to `results/`. |
| `new_distance_measure.md` | Standalone write-up of the Mahalanobis-in-polynomial-embedding extension and the 2D coverage-density finding, with the full result tables. |
| `tests/test_tier_a_closed_form.py` | Checks the hand-derived analytic gradient against `torch.autograd.grad`. |
| `tests/test_estimators.py` | Checks `pairwise_lipschitz` converges toward `tier_a_true_L` as `N` grows, `gradient_norm_estimate` matches it at the true argmax, and the Mahalanobis path is correct (reduces to a closed-form scaled-Euclidean identity under a degree-1 embedding; `local_perturbation_lipschitz` under an embedding converges to its analytic pointwise value as the perturbation radius shrinks). |
| `tests/test_seed_averaged_sweep.py` | Checks `sweep_over_N_seed_averaged` returns correctly-shaped per-seed and aggregated arrays, and that `n_seeds=1` reproduces `sweep_over_N`'s own single-seed output exactly. |
| `notebook_toy_lipschitz.ipynb` | Thin driver notebook — imports from this package and displays the figures produced by `run_experiment.py`. Contains no reusable logic of its own; see [Notebook contents](#notebook-contents). |
| `results/` | Generated outputs (git-ignored except `.gitkeep`): plots and `.npz` result arrays from every experiment below. |

## Notebook contents

`notebook_toy_lipschitz.ipynb` runs, in order:

1. **Tier A sanity check** (`run_tier_a_sanity`) — the whole pipeline
   validated against a closed-form answer.
2. **Tier A gap demo** (`run_tier_a_gap_demo`) — the flattening effect
   shown on the simplest possible ground truth (single ridge), using
   `TinyMLP` so the undershoot is actually observable; also computes the
   plain-vs-Mahalanobis comparison used later.
3. **Main experiment** (`run_main_experiment`) — the same gap-vs-uniform
   comparison on the harder Tier B (multi-ridge) function.
4. **N-sweep and capacity-sweep** (`run_sweeps`), both dataset types, `L*`
   printed explicitly alongside the curves.
5. **Seed-averaged gap N-sweep** (`run_gap_N_sweep_seed_averaged`) —
   repeats the gap dataset's N-sweep across 5 seeds and plots the mean,
   spread, and individual seed trajectories, to check whether a
   single-seed sweep's apparent non-monotonicity is real or noise.
6. **2D extension** (`run_2d_extension`) — the three Lipschitz heatmaps
   over `[-5,5]^2`, plus a coverage-density heatmap showing how many
   training points actually landed near each grid point.
7. **Distance-metric comparison** (`run_metric_embedding_check`,
   `sweep_polynomial_degree`, and the local-vs-global figure computed
   earlier in the Tier A gap demo) — does switching to a Mahalanobis
   distance change the recovered `L_hat`, and what embedding degree is
   the right choice?
8. **Summary cell.**

`run_cross_architecture_check` (the activation-matching check) and
`sweep_over_N_seed_averaged`'s underlying single-seed sweep are exercised
by `tests/` and `run_experiment.main()`, but the former isn't called from
the notebook.

## Design decisions

- **float64 everywhere** — every module calls `torch.set_default_dtype(torch.float64)` at import time, so true-vs-estimate comparisons aren't contaminated by float32 noise near the true maximum.
- **`domain` is a single `(low, high)` tuple** applied isotropically to every dimension (e.g. `(-5.0, 5.0)` for both `d=1` and `d=2`), not a per-axis list.
- **No `scipy` dependency** — Tier B's gradient-ascent refinement of `L*` uses `torch.optim.LBFGS` instead of `scipy.optimize.minimize`, since the ground-truth gradient (`tier_b_grad`) is itself a plain differentiable torch expression.
- **`tier_a_true_L(norm='l1')` is a documented simplification** — it returns `A * ||w||_1`, not the true L1-distance dual norm `A * ||w||_inf`. All correctness checkpoints use `norm='l2'`, where the dual-norm identity holds exactly, so this doesn't affect any test.
- **Gap sampling is rejection-based**, not analytic — cheap and exact enough at `d=1`/`d=2`.
- **Seeding covers weight initialization, not just training.** `train_regressor`'s `seed` argument only controls randomness inside that function (there is none — full-batch gradient descent is deterministic). Every call site that trains a model seeds `torch.manual_seed(seed)` immediately *before* constructing the model, so a fixed seed actually controls initialization too.
- **`make_dataset` is always noiseless** (`y = f_star(x)` exactly) — noise, if needed, belongs in the definition of `f_star` itself, not as a stochastic perturbation on a fixed function's output.
- **The polynomial embedding never includes `f(x)`** — `augmented_embedding` supports it, but using it to measure the Lipschitz behavior of the same function it's built from is self-cancelling: checked directly, it collapses the gap-vs-uniform local-Lipschitz contrast from roughly 6x down to roughly 1x, erasing the effect this whole project exists to detect. Full explanation in `new_distance_measure.md`.
- **Embedding degree 3 was chosen by measuring, not guessing** — `sweep_polynomial_degree` checks both accuracy against `L*` and the condition number of the fitted covariance, since a higher degree can look more accurate on one dataset while being numerically fragile. Degree 3 is the lowest degree that is both accurate (~0.08% error) and well-conditioned (`cond ≈ 1.5e3`); degrees 5-6 have more raw features but blow past `1e6` condition number and lose accuracy along with it.
- **`sweep_over_N` takes an optional `seed`, defaulting to the module-level `SEED`** — purely additive: neither the uniform nor the gap N-sweep inside `run_sweeps()` passes `seed=`, so both are unaffected. This is what lets `sweep_over_N_seed_averaged` repeat the exact same procedure at different seeds by calling `sweep_over_N` directly rather than duplicating its logic.

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

The seed-averaged gap N-sweep is opt-in and not part of `main()` — it's
several times slower than a single-seed sweep — so run it directly:

```bash
.venv/bin/python -c "from toy_lipschitz.run_experiment import run_gap_N_sweep_seed_averaged; run_gap_N_sweep_seed_averaged(n_seeds=5)"
```

## Results

- **Tier A sanity check**: `L*`, `L_hat_data`, and all three
  `L_hat_model` sub-methods agree to within ~1% (tolerance is 10%).
- **Tier A gap demo** (`results/tier_a_gap_vs_uniform.png`, `L*=15`): the
  gap-sampled model's local-Lipschitz peak near `x*` tops out around
  `8.2`, barely half of `L*`, while the uniformly-sampled model reaches
  `15.0` almost exactly.
- **Tier B (1D) ground truth**: `L* ≈ 8.46`, attained at `x* ≈ -0.37`.
- **Gap vs. uniform, Tier B** (`results/step6_gap_vs_uniform.png`): the
  finite-difference local-Lipschitz curve peaks noticeably lower near
  `x*` for the gap-sampled model than for the uniformly-sampled model —
  direct visual evidence of flattening/undershoot from sparse coverage.
- **Seed-averaged N-sweep on the gap dataset**
  (`results/step7_N_sweep_gap_seed_averaged.png`): at a single seed, the
  gap dataset's `L_hat_model` vs. `N` curve is non-monotonic — it bounces
  between `~4.6` and `~8.4` as `N` grows from 50 to 5000, unlike the
  uniform dataset's clean convergence. Repeating the sweep across 5 seeds
  and averaging shows the bounce is mostly single-seed noise (individual
  curves scatter widely at every `N`), but reveals a different, real
  effect underneath: the mean `L_hat_model` stays essentially **flat** at
  `~6.3-7.2` across the entire `N` range, never climbing toward
  `L*=8.46` the way `L_hat_data` (data-only, no model) does over the same
  range. At `N=5000`: `L_hat_model = 6.89 ± 0.56` (mean ± std across 5
  seeds) vs. `L*=8.46`. More gap-sampled data does not resolve the
  undershoot — a persistent-plateau effect, not a converges-with-noise one.
- **Capacity sweep on the gap dataset**
  (`results/step7_capacity_sweep_gap.png`): `L_hat_model` stays
  persistently below both `L_hat_data` and `L*` across every hidden width
  tested (4 to 128) — more model capacity alone does not compensate for a
  data-coverage gap.
- **2D extension** (`results/step8_2d_heatmaps.png`): true, model
  (autograd), and finite-difference gradient-norm heatmaps over an
  anisotropic 3-ridge function are visually consistent with each other,
  correctly locating the ridge-intersection hotspot.
- **2D coverage check** (`results/step8_coverage_heatmap.png`):
  `local_sample_density` counts training points within `local_radius` of
  each grid point. At the default `run_2d_extension` settings
  (`gap_radius=0.7`, `gap_fraction=0.03`), the hotspot is **not**
  actually undersampled — its density (3, at `x*`) is above the
  grid-wide average (~2.15), because `gap_fraction=0.03` is nearly 2x the
  ~0.0154 density a plain uniform sample would put in a ball that size by
  area alone. Correcting to `gap_fraction≈0.0077` (half that
  naive-uniform baseline) produces genuine local undersampling (density 0
  vs. grid-wide mean ~2.15). Averaged over 5 seeds, the hotspot's
  local-Lipschitz estimate then undershoots the true gradient norm by
  ~9% on average (mean ratio 0.91, stdev 0.07) — real, but far weaker
  than the 20-50%+ undershoot seen in 1D. Why the 2D effect is weaker
  isn't established here; one untested hypothesis is that a 2D gap can be
  approached from more directions than a 1D one, giving a smooth model
  more surrounding signal to interpolate the peak from.
- **Distance metric matters** (`run_metric_embedding_check`, `L*=6.0`):
  on identical raw data, `L_hat_data` computed with plain Euclidean
  distance is `~4.87` (19% under `L*`); the same data under a Mahalanobis
  distance in a degree-3 polynomial embedding gives `~6.01` (<1% off).
  `results/tier_a_local_vs_global_lipschitz.png` shows this holds at the
  local (per-point), not just global, level.
- **Degree selection isn't free** (`sweep_polynomial_degree`,
  `results/degree_sweep.png`): degrees 1-2 are too simple to capture the
  effect at all (>130% error); degree 3 is the sweet spot (`0.08%` error,
  `cond(cov) ≈ 1.5e3`); degrees 5-6 have more raw features but the
  covariance's condition number blows past `1e6` and accuracy collapses
  with it — more parameters here buys overfitting to sampling noise, not
  a better metric.

See `new_distance_measure.md` for the full write-up and result tables
behind the last two points.
