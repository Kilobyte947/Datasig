# mnist_lipschitz

Scales the three Lipschitz estimators built for a toy regression problem
(`toy_lipschitz`, sibling folder) up to real classifiers trained on MNIST,
and asks the same underlying question: how much can a model's output
change for a given change in input?

The toy setting had a known, closed-form ground-truth Lipschitz constant
`L*` to check every estimate against. **There is no such ground truth
here** — a trained neural network's exact Lipschitz constant is not
analytically knowable in general. Validity therefore comes from two
weaker checks instead:

1. **Do the three independently implemented sub-methods (pairwise,
   local-perturbation, gradient-norm) roughly agree with each other?**
   Disagreement doesn't necessarily mean a bug, but it tells you how much
   to trust any single number.
2. **Is an estimate stable under resampling?** An estimate that swings
   wildly depending on which random subset of the data it was computed
   from is fitting sampling noise, not a real property of the model.

Both are checked directly, not assumed, and both turn out to be more
interesting — and less reassuring — than in the toy setting. See
[Results](#results).

**Out of scope** (may return in a later experiment): adversarial
examples, FGSM/PGD attacks, robustness mitigations (weight decay,
spectral norm, gradient penalty, adversarial training, input smoothing),
Fashion-MNIST.

## What's being measured

A classifier doesn't have a single scalar output the way the toy
regression problem did — it outputs 10 logits. The estimators here are
instead applied to the **margin function**,
`margin(x) = logit[y_true] - max(logit[j] for j != y_true)`: the natural
classifier analogue of a scalar output, and the quantity robustness
actually depends on, since the margin crossing zero is exactly where the
predicted class flips.

Three classifiers are trained (logistic regression, a small MLP, a small
CNN), and for each, all three Lipschitz sub-methods are computed twice:
once under plain Euclidean distance, once under a ridge-regularized
Mahalanobis distance fit from MNIST's own pixel covariance.

## How it's designed

**Data (`data.py`).** `load_mnist` loads MNIST via torchvision and hands
back three aligned views of the same points: `x_flat` (N, 784) — what the
estimators actually operate on — `x_image` (N, 1, 28, 28) — needed for
the CNN's convolutions — and integer labels `y`. Pixel values are kept in
`[0, 1]` (no ImageNet-style normalization), so raw pixel differences stay
directly interpretable for the covariance/Mahalanobis work in
`distance.py`. `get_dev_subset` gives a small seeded subset for fast
iteration; `stratified_subset_idx` draws a class-balanced subset (with
optional exclusion of already-used indices), used to build the
ratio-distribution query sets below.

**Models (`models.py`).** `LogisticRegressionModel` (784 → 10, no hidden
layers) is the one model whose margin is exactly linear in `x`, giving it
a closed-form Euclidean Lipschitz constant
(`||w_true - w_runner_up||_2` for a fixed class pair) — this is what the
whole estimator pipeline is validated against before being trusted on the
other two models. `SmallMLP` (one or two hidden layers, ReLU by default)
and `SmallCNN` (two conv+pool blocks plus a linear head) are the two
models with no closed form. `FlattenedInputWrapper` wraps the CNN so it
accepts the same flat `(N, 784)` input as the other two models, letting
all three share one estimator code path. `train_classifier` is a plain
cross-entropy/Adam training loop; `margin_fn` is described above.

**Estimators (`estimators.py`).** The same three sub-methods as the toy
experiment, generalized to take `margin_fn`/`y_batch` and a pluggable
`distance_fn` (or `precision` matrix, for the gradient dual norm):
`pairwise_lipschitz`, `local_perturbation_lipschitz`,
`gradient_norm_estimate`. `euclidean_distance_fn` is the default metric.
Three more functions support the ratio-distribution analysis:
`pairwise_lipschitz_all` (the full per-pair ratio array instead of just
the max), and `ratio_and_components_for_pairs`/`ratio_for_pairs` (the
ratio, and its raw distance/margin-difference components, for an
arbitrary set of pairs — not just all `i < j` — so a caller with a
different pairing scheme, like nearest neighbors, can reuse the same core
computation).

**Distance metric: Euclidean vs. Mahalanobis (`distance.py`).**
`euclidean_distance_fn` is plain `||x-y||_2`. The alternative starts from
`pixel_covariance` — the empirical covariance of centered pixel vectors —
which is exactly singular in practice (many border pixels are 0 in every
MNIST image, so their variance and covariance with everything else is
exactly 0) and therefore can't be inverted directly. `ridge_precision`
fixes this by inverting `Sigma + epsilon*I` instead; `mahalanobis_distance`
then computes `sqrt((x-y)^T P (x-y))` for a given precision matrix `P`,
and `make_mahalanobis_distance_fn` wraps a fixed `P` into a `distance_fn`
closure for direct use with `estimators.py`. `covariance_eigenvalues`
exposes the sorted eigenvalue spectrum of `Sigma`, making the
rank-deficiency directly visible rather than only inferred from a large
condition number. `sweep_epsilon` — the pure linear-algebra half of
epsilon selection — reports `cond(Sigma + epsilon*I)` for each candidate
epsilon.

**Choosing epsilon (`run_experiment.py`).** How much regularization to
use is itself selected by a data-driven sweep, mirroring the toy
experiment's polynomial-embedding-degree selection.
`epsilon_stability_check` is the other half: for each candidate epsilon,
it draws independent random subsamples of the training set, fits a
Mahalanobis precision matrix on each, and computes the mean
gradient-norm estimate (over a fixed reference model — logistic
regression's `margin_fn`, chosen only for a cheap, consistent yardstick,
not because it's the final model). The coefficient of variation
(std/mean) across subsamples measures stability: a stable epsilon
reproduces its estimate across resamples, an unstable one is fitting
noise in that subsample's covariance. `select_epsilon` then picks the
smallest epsilon meeting both a condition-number bound and a stability
bound, falling back to the lowest-cv candidate (with an explicit warning)
if none qualify.

**Ratio-distribution analysis (`run_ratio_distribution_analysis` in
`run_experiment.py`).** Every Lipschitz sub-method above ultimately
reduces to a ratio, `|margin_i - margin_j| / distance(x_i, x_j)`. This
analysis compares that ratio's full distribution across two different
populations of pairs, on a class-stratified subset: **all pairs** versus
**nearest-neighbor pairs** (found in raw pixel space via
`sklearn.neighbors.NearestNeighbors`, regardless of which `distance_fn`
is being evaluated). Nearest neighbors are deliberately always found in
raw pixel space so the comparison isolates one question — do pairs a
human would call visually similar behave differently, ratio-wise, than
the general pair population? — without the answer being confounded by
which metric is also being used to compute the ratio itself. The top
highest-ratio nearest-neighbor pairs (deduplicated, since mutual nearest
neighbors would otherwise appear twice) are kept alongside their images
and predictions for visual inspection.

## File reference

| File | Contents |
|---|---|
| `data.py` | `load_mnist`, `get_dev_subset`, `stratified_subset_idx`, `make_loader`. |
| `models.py` | `LogisticRegressionModel`, `SmallMLP`, `SmallCNN`, `FlattenedInputWrapper`, `train_classifier`, `evaluate_accuracy`, `margin_fn`. |
| `estimators.py` | The three Lipschitz sub-methods (`pairwise_lipschitz`, `local_perturbation_lipschitz`, `gradient_norm_estimate`) plus the ratio-distribution support functions (`pairwise_lipschitz_all`, `ratio_and_components_for_pairs`, `ratio_for_pairs`). Imports `euclidean_distance_fn` from `distance.py` as the default metric. |
| `distance.py` | Both distance functions (`euclidean_distance_fn`, `mahalanobis_distance`) and the pieces that build a Mahalanobis metric from data: `pixel_covariance`, `ridge_precision`, `make_mahalanobis_distance_fn`, `covariance_eigenvalues`, `sweep_epsilon`. Pure linear algebra — no dependency on `models.py` or `estimators.py`. |
| `run_experiment.py` | `epsilon_stability_check` and `select_epsilon` (the model-dependent half of epsilon selection), `run_ratio_distribution_analysis`, and `run_mnist_experiment` — the main driver that trains all three models, runs all three estimators and the ratio-distribution analysis under Euclidean distance, selects an epsilon, repeats both under Mahalanobis distance, and saves everything to `results/`. |
| `plots.py` | `MODEL_ORDER`/`MODEL_LABELS` (the single source of truth for model iteration order and display names). `plot_euclidean_vs_mahalanobis`, `plot_epsilon_sweep`, `plot_submethod_agreement`, `plot_covariance_eigenvalues`, `plot_ratio_distribution`, `plot_ratio_distribution_euclidean_vs_mahalanobis`, `plot_image_pairs`. |
| `tests/test_data.py` | MNIST shapes, pixel value range `[0,1]`, label range `{0..9}`, dev-subset seed reproducibility. |
| `tests/test_models.py` | Each model trains above an accuracy threshold on the full test set (with headroom below what's actually observed, so the test isn't flaky against normal seed variance); `margin_fn` matches a manual per-example computation and is differentiable w.r.t. `x`. |
| `tests/test_estimators.py` | On `LogisticRegressionModel(num_classes=2)`, `margin_fn` reduces to an exactly linear function with a closed-form Euclidean Lipschitz constant `||w_0-w_1||_2` — all three sub-methods are checked against it (10% tolerance) before being trusted on the MLP/CNN. Also checks the Mahalanobis "P vs. P^-1" dual-norm convention in `gradient_norm_estimate` against an independent closed-form identity. |
| `tests/test_distance.py` | As epsilon grows large, `mahalanobis_distance` converges to a constant multiple (`1/sqrt(epsilon)`) of Euclidean distance; real MNIST pixel covariance is confirmed exactly rank-deficient (`cond=inf`) while the ridge-regularized version is well-conditioned. |
| `tests/test_epsilon_selection.py` | Condition number is (deterministically) non-increasing as epsilon grows; subsample instability does not increase going from a near-singular to a well-regularized epsilon; `select_epsilon`'s bound-matching and fallback logic. |
| `notebook_mnist_lipschitz.ipynb` | Thin driver notebook — imports from this package, runs `run_mnist_experiment()`, displays every plot with introductory markdown. No reusable logic of its own. |
| `results/` | Generated outputs (git-ignored except `.gitkeep`): `mnist_experiment_results.json` (scalar summary), `mnist_experiment_arrays.npz` (full per-point local-perturbation/gradient-norm/ratio arrays), and every plot (epsilon sweep, covariance eigenvalues, Euclidean-vs-Mahalanobis comparison, sub-method agreement, ratio distributions, top near-neighbor image pairs). |
| `data/` | Downloaded MNIST files (git-ignored, ~63MB) — recreated automatically by `load_mnist` on first run. |

## Design decisions

- **ReLU for the MLP, not tanh.** `toy_lipschitz` used tanh throughout for continuity with its smooth closed-form ground truth; there's no such ground truth here, and ReLU is the standard choice for MNIST classifiers (faster to train, no vanishing-gradient concern at this depth). `SmallMLP` still supports `activation="tanh"` for anyone who wants to compare.
- **Pairwise sampling keeps `N` modest rather than random-subsampling from a huge pool.** `pairwise_lipschitz` is handed a few hundred points (300 in the main run) and scores *all* pairs among them (~45k pairs, cheap), rather than defaulting to random-pair subsampling from a much larger pool. `max_pairs` is still supported as a safety valve.
- **The `precision` vs. `precision^-1` convention in `gradient_norm_estimate`.** For a Mahalanobis distance with quadratic-form matrix `P` (i.e. `distance_fn` computes `sqrt((x-y)^T P (x-y))`), the correct dual norm of a gradient `g` is `sqrt(g^T P^-1 g)` — **not** `sqrt(g^T P g)`. This is easy to get backwards with nothing to catch it once real MNIST data is involved (no `L*` here), so it's checked directly against an independent closed-form identity (the maximizer `delta* = P^-1 g / sqrt(g^T P^-1 g)` provably attains the claimed value) in `tests/test_estimators.py`, not just derived and trusted. `gradient_norm_estimate` inverts the given `precision` internally to recover `Sigma` — a second inversion (the first happens in `distance.py` to build `precision` in the first place) — but a dense 784x784 inverse measured at ~10ms is negligible next to model training, and keeping one `precision`-matrix convention across all three estimators was judged simpler than threading a Cholesky factor through three different call sites.
- **Ridge regularization, not PCA truncation, for the singular pixel covariance.** Ridge is the more direct generalization of `toy_lipschitz/embeddings.py`'s existing `precision_from_covariance` (which already adds `eps*I`) and needs no extra machinery — choosing a truncation rank, deciding how to handle the discarded subspace in the distance formula. It hasn't been compared head-to-head against PCA truncation (see [Limitations](#limitations-and-open-questions)).
- **`epsilon_stability_check` uses the mean gradient-norm estimate, not `pairwise_lipschitz`'s max.** Using `pairwise_lipschitz` (a max over ~200-300 pairs) as the per-subsample yardstick gives a coefficient of variation that bounces between roughly 0.04 and 0.26 with no clean trend against epsilon — an extreme-value statistic over a modest number of pairs is dominated by which specific pair happens to land near the metric's steepest direction, adding a lot of irreducible sampling noise on top of (and swamping) the genuine metric-shape instability epsilon is meant to control. The *mean* gradient-norm estimate over 100 points gives a clean, repeatable (checked across several seeds) decreasing-then-flat trend with cv in the 0.007-0.045 range instead.
- **The stability check's model is always logistic regression**, regardless of which model the final comparison uses — epsilon selection only needs *a* consistent, cheap-to-evaluate yardstick, not the specific model being analyzed. Unlike the toy experiment's strict "no model involved" `L_hat_data`, there's no model-free scalar function of `x` on MNIST to fall back on.
- **`select_epsilon` never silently returns a value outside its stated criteria** — if no candidate meets both the condition-number and stability bounds, it falls back to the lowest-cv candidate and prints an explicit warning, rather than picking silently or raising.
- **Nearest neighbors for the ratio-distribution analysis are always found in raw pixel space**, regardless of `distance_fn` — this keeps "which pairs look similar to a human" a fixed, metric-independent question, so the comparison across Euclidean and Mahalanobis isolates how the *ratio* changes, not also how the neighbor selection changes.

## How to run it

```bash
# from the repo root
.venv/bin/python -m pytest mnist_lipschitz/tests/ -v

.venv/bin/python -c "from mnist_lipschitz.run_experiment import main; main()"

# or execute the notebook end-to-end (~1 minute on CPU)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace mnist_lipschitz/notebook_mnist_lipschitz.ipynb
```

`run_experiment.main()` calls `run_mnist_experiment()` with its default
configuration (15 epochs for logistic regression and the MLP, 8 for the
CNN; **1000** Lipschitz query points by default; a 1000-point
ratio-distribution subset; a 7-point epsilon sweep from `1e-6` to `100`).
Its pipeline, in order: train all three models; run all three Lipschitz
sub-methods and the ratio-distribution analysis under Euclidean distance;
select an epsilon from the pixel covariance; repeat both under
Mahalanobis distance; save everything to `results/`.

The numbers in [Results](#results) below are from a run with
`n_lipschitz_points=300` passed explicitly (an override, not the
current default of 1000) — reproduce them with
`run_mnist_experiment(n_lipschitz_points=300)`, not with `main()`'s
plain defaults.

## Results

All numbers below are from an actual run (`results/mnist_experiment_results.json`), full MNIST train/test sets, seed 0, `n_lipschitz_points=300` (an explicit override of the function's current default of 1000 — see [How to run it](#how-to-run-it)).

**Model accuracies** (test set):

| Model | Train acc. | Test acc. |
|---|---|---|
| Logistic regression | 92.87% | 92.70% |
| MLP (1 hidden layer, 128 units, ReLU) | 99.31% | 97.82% |
| CNN (2 conv+pool blocks) | 98.89% | 98.65% |

**Epsilon selection**: swept `{1e-6, 1e-4, 1e-2, 1e-1, 1, 10, 100}`. Both
the condition number (`5.1e6 -> 1.05`) and the subsample coefficient of
variation (`0.045 -> 0.007`) decrease monotonically across the whole
sweep. Selected epsilon = **0.01** (`cond=513`, `cv=0.019`) — the
smallest epsilon meeting both bounds (`cond<=1e4`, `cv<=0.05`); the next
smaller candidate, `1e-4`, fails the condition-number bound
(`cond=51170`) even though its stability is already fine.

**Lipschitz estimates, Euclidean distance** (300 held-out query points, `local_radius=1.0`, `n_directions=20`):

| Model | pairwise | local-pert. (max) | grad-norm (max) |
|---|---|---|---|
| Logistic regression | 1.829 | 1.286 | 10.505 |
| MLP | 2.376 | 2.288 | 22.403 |
| CNN | 2.665 | 1.530 | 11.261 |

**Lipschitz estimates, Mahalanobis distance** (epsilon=0.01, same query points):

| Model | pairwise | local-pert. (max) | grad-norm (max) |
|---|---|---|---|
| Logistic regression | 0.771 | 0.167 | 9.420 |
| MLP | 1.014 | 0.280 | 16.036 |
| CNN | 1.122 | 0.188 | 9.703 |

**The metric choice changes the estimates substantially, and unevenly
across sub-methods.** Switching from Euclidean to Mahalanobis distance
(same raw data, same trained models) drops `pairwise` by ~57-58% and
`local-perturbation` by ~87-88% across all three models, but
`gradient-norm` by only ~10-28%. Unlike the toy experiment, where
Mahalanobis distance moved a data-only estimate *toward* a known `L*`,
there's no ground truth to compare against here — the honest statement is
just that the metric matters a great deal, and matters differently
depending on which sub-method is asked, not that one number is "more
correct" than another.

**Sub-method agreement is weak, and gets weaker under Mahalanobis.** This
is the more important and more surprising finding. In the toy setting,
all three sub-methods landed within about 10% of each other and of `L*`.
Here, gradient-norm is **4-9x larger** than pairwise under Euclidean
distance (5.7x for logistic regression, 9.4x for the MLP, 4.2x for the
CNN), and **9-16x larger** under Mahalanobis distance (12.2x, 15.8x, 8.7x
respectively). Pairwise and local-perturbation stay roughly comparable to
each other throughout; gradient-norm is the consistent outlier. A
plausible (not rigorously confirmed) explanation: gradient-norm captures
the *exact* steepest direction at each query point via autograd, while
pairwise (bounded to the ~300 sampled query points, mean pairwise
distance ~9.3 in raw pixel space) and local-perturbation (a fixed
Euclidean radius of 1.0, only 20 sampled directions) are much more likely
to miss a network's sharp, narrow high-sensitivity directions than to hit
one.

**Ratio-distribution analysis** (1000-point class-stratified subset, 5
nearest neighbors per point, all-pairs vs. near-neighbor-pairs ratio
`|margin_i-margin_j|/distance(x_i,x_j)`): visually-similar pairs (nearest
neighbors in raw pixel space) don't behave like a scaled-down version of
the general pair population — they behave *differently* depending on the
metric used to compute the ratio, in opposite directions:

| Model | Euclidean: all-pairs mean | Euclidean: near-neighbor mean | Mahalanobis: all-pairs mean | Mahalanobis: near-neighbor mean |
|---|---|---|---|---|
| Logistic regression | 0.312 | 0.354 | 0.139 | 0.110 |
| MLP | 0.409 | 0.502 | 0.182 | 0.156 |
| CNN | 0.418 | 0.487 | 0.186 | 0.151 |

Under Euclidean distance, near-neighbor pairs have a **higher** mean
ratio than the general pair population (+13-23% across the three
models) — pairs a human would call visually similar are, if anything,
slightly more likely to have a disproportionately large margin swing for
their (small) pixel-space distance. Under Mahalanobis distance this
**flips**: near-neighbor pairs have a **lower** mean ratio than the
general population (−14-21%). Both directions are consistent across all
three models. Why the direction flips with the metric is not established
here — a plausible starting point is that the Mahalanobis metric
downweights the very directions raw-pixel nearest neighbors are most
likely to differ along (the high-variance directions the covariance
already captures), but this hasn't been checked directly.

## Limitations and open questions

- **Sub-method disagreement (4-16x) is larger here than in the toy
  setting**, and this experiment reports it with a candidate explanation
  but doesn't resolve it — e.g. by checking whether increasing
  `n_directions` or `local_radius` in `local_perturbation_lipschitz`
  narrows the gap with gradient-norm (which would support the
  "missing the sharp direction" hypothesis), or whether it doesn't
  (which would point somewhere else).
- **PCA-truncation regularization hasn't been tried** as an alternative
  to ridge regularization for the singular pixel covariance.
- **The Mahalanobis extension reuses one epsilon across all three
  models.** Epsilon is selected using only the logistic-regression model
  and 100 points via `epsilon_stability_check`; whether that choice is
  also good specifically for the MLP or CNN's estimates hasn't been
  checked — the experiment assumes one epsilon suffices for all three
  rather than testing that assumption.
- **`local_radius=1.0` and `n_directions=20` were chosen by inspecting
  typical MNIST pixel-space distances** (image L2 norms ~9.3, so radius
  1.0 is a genuinely "local" ~10% perturbation) rather than by a
  dedicated sensitivity sweep the way epsilon was.
- **Why the ratio-distribution flip (see above) reverses direction
  between Euclidean and Mahalanobis distance is not established.**
