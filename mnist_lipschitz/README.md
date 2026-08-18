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
`euclidean_distance_fn` is plain `||x-y||_2`. The alternative is a
ridge-regularized Mahalanobis distance: `svd_ridge_precision` builds the
precision matrix `(Sigma + epsilon*I)^-1`, where `Sigma` is the empirical
covariance of centered pixel vectors, directly from the SVD of the
(centered) pixel matrix — without ever forming `Sigma` as a `(784, 784)`
matrix. This matters because MNIST's raw pixel covariance is exactly
singular in practice (many border pixels are 0 in every image, so their
variance and covariance with everything else is exactly 0), and forming
`X^T @ X` to get `Sigma` squares the condition number of the data before
any regularization is even applied — working from the pixel matrix's own
singular values instead avoids that amplification. `mahalanobis_distance`
then computes `sqrt((x-y)^T P (x-y))` for a given precision matrix `P`,
and `make_mahalanobis_distance_fn` wraps a fixed `P` into a `distance_fn`
closure for direct use with `estimators.py`. `covariance_eigenvalues`
exposes the sorted eigenvalue spectrum of `Sigma` the same way — from
singular values, not a formed matrix — making the rank-deficiency
directly visible rather than only inferred from a large condition
number. `sweep_epsilon` — the pure linear-algebra half of epsilon
selection — reports `cond(Sigma + epsilon*I)` for each candidate epsilon,
also computed from those singular values.

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
| `distance.py` | Both distance functions (`euclidean_distance_fn`, `mahalanobis_distance`) and the pieces that build a Mahalanobis metric from data: `svd_ridge_precision`, `make_mahalanobis_distance_fn`, `covariance_eigenvalues`, `sweep_epsilon`. All built from the SVD of the raw pixel matrix directly — nothing in this file ever forms the `(784, 784)` covariance matrix explicitly. Pure linear algebra — no dependency on `models.py` or `estimators.py`. |
| `embeddings.py` | `elementwise_embedding(x_flat, degree)` — maps each pixel through `[x, x**2, ..., x**degree]` independently (no cross-pixel terms), generalizing `toy_lipschitz/embeddings.py`'s polynomial-embedding convention to the 784-pixel setting. `degree=1` is the identity. `local_patch_cross_terms(x_image)` — maps each pixel to its raw value plus one cross-term product with each of its immediate 3x3-window spatial neighbors (zero-padded at the border), operating on the actual 28x28 image layout. |
| `run_experiment.py` | `epsilon_stability_check` and `select_epsilon` (the model-dependent half of epsilon selection), `run_ratio_distribution_analysis`, `run_mnist_experiment` — the main driver that trains all three models, runs all three estimators and the ratio-distribution analysis under Euclidean distance, selects an epsilon, repeats both under Mahalanobis distance, and saves everything to `results/` — and `run_embedding_degree_sweep`, an opt-in (not part of `main()`) driver that repeats epsilon selection and the ratio-distribution analysis once per `elementwise_embedding` degree. |
| `plots.py` | `MODEL_ORDER`/`MODEL_LABELS` (the single source of truth for model iteration order and display names). `plot_euclidean_vs_mahalanobis`, `plot_epsilon_sweep`, `plot_submethod_agreement`, `plot_covariance_eigenvalues`, `plot_ratio_distribution`, `plot_ratio_distribution_euclidean_vs_mahalanobis`, `plot_image_pairs`, `plot_embedding_degree_sweep`. |
| `tests/test_data.py` | MNIST shapes, pixel value range `[0,1]`, label range `{0..9}`, dev-subset seed reproducibility. |
| `tests/test_models.py` | Each model trains above an accuracy threshold on the full test set (with headroom below what's actually observed, so the test isn't flaky against normal seed variance); `margin_fn` matches a manual per-example computation and is differentiable w.r.t. `x`. |
| `tests/test_estimators.py` | On `LogisticRegressionModel(num_classes=2)`, `margin_fn` reduces to an exactly linear function with a closed-form Euclidean Lipschitz constant `||w_0-w_1||_2` — all three sub-methods are checked against it (10% tolerance) before being trusted on the MLP/CNN. Also checks the Mahalanobis "P vs. P^-1" dual-norm convention in `gradient_norm_estimate` against an independent closed-form identity, and `gradient_norm_estimate`'s `embed_fn`-aware pullback-metric path (identity, linear, and `elementwise_embedding` cases, each against an independently computed closed form). |
| `tests/test_embeddings.py` | `elementwise_embedding`'s identity case, shape, and exact block layout on a hand-checkable example; `make_mahalanobis_distance_fn(..., embed_fn=...)` matches the raw-pixel path exactly at `degree=1`, and leaves existing behavior unchanged when `embed_fn` is left `None`. `local_patch_cross_terms`'s exact neighbor products and border zero-padding on a hand-checkable 3x3 example, output shape, and batch-vs-single-example consistency (relevant since `gradient_norm_estimate` calls `embed_fn` on single examples via `jacrev`/`vmap`). |
| `tests/test_distance.py` | As epsilon grows large, `mahalanobis_distance` converges to a constant multiple (`1/sqrt(epsilon)`) of Euclidean distance; real MNIST pixel covariance is confirmed exactly rank-deficient (`cond=inf`) while the ridge-regularized version is well-conditioned. |
| `tests/test_epsilon_selection.py` | Condition number is (deterministically) non-increasing as epsilon grows; subsample instability does not increase going from a near-singular to a well-regularized epsilon; `select_epsilon`'s bound-matching and fallback logic; `run_embedding_degree_sweep`'s `degree=1` result matches the pre-existing raw-pixel pipeline exactly on an identical pool/model/seed. |
| `notebook_mnist_lipschitz.ipynb` | Thin driver notebook — imports from this package, runs `run_mnist_experiment()` and (further down) the opt-in `run_embedding_degree_sweep()`, displays every plot and table with introductory markdown. No reusable logic of its own. |
| `results/` | Generated outputs (git-ignored except `.gitkeep`): `mnist_experiment_results.json` (scalar summary), `mnist_experiment_arrays.npz` (full per-point local-perturbation/gradient-norm/ratio arrays), `embedding_degree_sweep_results.json`/`embedding_degree_sweep_arrays.npz` (the degree-sweep equivalents), and every plot (epsilon sweep, covariance eigenvalues, Euclidean-vs-Mahalanobis comparison, sub-method agreement, ratio distributions, top near-neighbor image pairs, embedding degree sweep). |
| `data/` | Downloaded MNIST files (git-ignored, ~63MB) — recreated automatically by `load_mnist` on first run. |

## Design decisions

- **ReLU for the MLP, not tanh.** `toy_lipschitz` used tanh throughout for continuity with its smooth closed-form ground truth; there's no such ground truth here, and ReLU is the standard choice for MNIST classifiers (faster to train, no vanishing-gradient concern at this depth). `SmallMLP` still supports `activation="tanh"` for anyone who wants to compare.
- **Pairwise sampling keeps `N` modest rather than random-subsampling from a huge pool.** `pairwise_lipschitz` is handed a few hundred to a thousand points (1000 in the main run) and scores *all* pairs among them (~500k pairs, still cheap), rather than defaulting to random-pair subsampling from a much larger pool. `max_pairs` is still supported as a safety valve.
- **The `precision` vs. `precision^-1` convention in `gradient_norm_estimate`.** For a Mahalanobis distance with quadratic-form matrix `P` (i.e. `distance_fn` computes `sqrt((x-y)^T P (x-y))`), the correct dual norm of a gradient `g` is `sqrt(g^T P^-1 g)` — **not** `sqrt(g^T P g)`. This is easy to get backwards with nothing to catch it once real MNIST data is involved (no `L*` here), so it's checked directly against an independent closed-form identity (the maximizer `delta* = P^-1 g / sqrt(g^T P^-1 g)` provably attains the claimed value) in `tests/test_estimators.py`, not just derived and trusted. `gradient_norm_estimate` inverts the given `precision` internally to recover `Sigma` — a second inversion (the first happens in `distance.py` to build `precision` in the first place) — but a dense 784x784 inverse measured at ~10ms is negligible next to model training, and keeping one `precision`-matrix convention across all three estimators was judged simpler than threading a Cholesky factor through three different call sites.
- **Ridge regularization, not PCA truncation, for the singular pixel covariance.** Ridge is the more direct generalization of `toy_lipschitz/embeddings.py`'s existing `precision_from_covariance` (which already adds `eps*I`) and needs no extra machinery — choosing a truncation rank, deciding how to handle the discarded subspace in the distance formula. It hasn't been compared head-to-head against PCA truncation (see [Limitations](#limitations-and-open-questions)).
- **The ridge precision matrix is built from the pixel matrix's SVD, not from a formed covariance matrix.** `svd_ridge_precision` computes `(Sigma + epsilon*I)^-1` directly from the singular values of the (centered) pixel matrix, rather than first forming `Sigma = X^T @ X / (N-1)` and inverting it. Forming `X^T @ X` squares the condition number of the data (`cond(X^T @ X) = cond(X)^2`) before regularization is even applied — concretely, `torch.linalg.cond` on the formed 784x784 `Sigma` reports it as **exactly singular** (`cond=inf`), while the same quantity derived from the pixel matrix's own singular values is merely astronomically ill-conditioned but finite (`~8.3e33`). Downstream, this made no practical difference at MNIST's scale — the selected epsilon, the epsilon-stability coefficients of variation, and the ratio-distribution statistics all agreed with the previous covariance-forming route to 5+ significant figures before the switch — but working from the SVD directly is the more numerically defensible route on principle, and is now the only route in this file.
- **`epsilon_stability_check` uses the mean gradient-norm estimate, not `pairwise_lipschitz`'s max.** Using `pairwise_lipschitz` (a max over ~200-300 pairs) as the per-subsample yardstick gives a coefficient of variation that bounces between roughly 0.04 and 0.26 with no clean trend against epsilon — an extreme-value statistic over a modest number of pairs is dominated by which specific pair happens to land near the metric's steepest direction, adding a lot of irreducible sampling noise on top of (and swamping) the genuine metric-shape instability epsilon is meant to control. The *mean* gradient-norm estimate over 100 points gives a clean, repeatable (checked across several seeds) decreasing-then-flat trend with cv in the 0.007-0.045 range instead.
- **The stability check's model is always logistic regression**, regardless of which model the final comparison uses — epsilon selection only needs *a* consistent, cheap-to-evaluate yardstick, not the specific model being analyzed. Unlike the toy experiment's strict "no model involved" `L_hat_data`, there's no model-free scalar function of `x` on MNIST to fall back on.
- **`select_epsilon` never silently returns a value outside its stated criteria** — if no candidate meets both the condition-number and stability bounds, it falls back to the lowest-cv candidate and prints an explicit warning, rather than picking silently or raising.
- **Nearest neighbors for the ratio-distribution analysis are always found in raw pixel space**, regardless of `distance_fn` — this keeps "which pairs look similar to a human" a fixed, metric-independent question, so the comparison across Euclidean and Mahalanobis isolates how the *ratio* changes, not also how the neighbor selection changes.
- **The Mahalanobis metric can be fit over an `embed_fn`-mapped feature space, not just raw pixels.** `make_mahalanobis_distance_fn`, `epsilon_stability_check`, and `gradient_norm_estimate` all take an optional `embed_fn` (e.g. `embeddings.py::elementwise_embedding`); left `None`, every one of them leaves pre-existing raw-pixel behavior exactly unchanged. `gradient_norm_estimate`'s embedded path is the subtle piece: the correct dual norm isn't the raw-pixel Mahalanobis formula reapplied to the embedded gradient, but the *pullback* of the embedded metric through `embed_fn`'s (generally per-point, for nonlinear `embed_fn`) Jacobian, computed via `torch.func.jacrev`/`vmap` rather than a formula hand-derived for one specific embedding — checked against closed-form/manual-Jacobian formulas for the identity, a generic linear map, and `elementwise_embedding` itself in `tests/test_estimators.py`, since there's no MNIST-scale ground truth to check it against otherwise.
- **`run_embedding_degree_sweep` is self-contained and opt-in, like `toy_lipschitz`'s seed-averaged N-sweep.** It trains its own reference logistic-regression model and loads MNIST fresh unless already-built ones are passed in, and is deliberately not called from `main()` — fitting a precision matrix on the full 60k-point set at `degree=3` (a `(60000, 2352)` SVD) makes it markedly slower than `run_mnist_experiment()` alone.

## How to run it

```bash
# from the repo root
.venv/bin/python -m pytest mnist_lipschitz/tests/ -v

.venv/bin/python -c "from mnist_lipschitz.run_experiment import main; main()"

# opt-in embedding-degree sweep (not part of main() -- markedly slower, see above)
.venv/bin/python -c "from mnist_lipschitz.run_experiment import run_embedding_degree_sweep; run_embedding_degree_sweep(degrees=(1, 2, 3))"

# or execute the notebook end-to-end (~1 minute on CPU for run_mnist_experiment();
# meaningfully longer with the embedding-degree sweep further down included)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace mnist_lipschitz/notebook_mnist_lipschitz.ipynb
```

As of the StrongCNN addition (`models.py`'s `StrongCNN`/`augmentation.py`),
`mnist_lipschitz/tests/` has **71 tests**. A figure of 58 has circulated in
discussion of this suite before; it doesn't match any commit in this repo's
history, so it was likely just stale. Treat the exact count as something that
drifts as tests are added rather than a fixed number to remember -- run the
command above for the current figure.

`run_experiment.main()` calls `run_mnist_experiment()` with its default
configuration (15 epochs for logistic regression and the MLP, 8 for the
CNN; 1000 Lipschitz query points; a 1000-point ratio-distribution subset;
a 7-point epsilon sweep from `1e-6` to `100`). Its pipeline, in order:
train all three models; run all three Lipschitz sub-methods and the
ratio-distribution analysis under Euclidean distance; select an epsilon
from the pixel covariance; repeat both under Mahalanobis distance; save
everything to `results/`. The numbers in [Results](#results) below are
from exactly this default configuration.

## Results

All numbers below are from an actual run (`results/mnist_experiment_results.json`), full MNIST train/test sets, seed 0, `run_mnist_experiment()`'s plain default configuration (1000 Lipschitz query points).

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

**Lipschitz estimates, Euclidean distance** (1000 held-out query points, `local_radius=1.0`, `n_directions=20`):

| Model | pairwise | local-pert. (max) | grad-norm (max) |
|---|---|---|---|
| Logistic regression | 2.615 | 1.382 | 10.590 |
| MLP | 2.905 | 2.998 | 23.977 |
| CNN | 2.841 | 1.508 | 11.261 |

**Lipschitz estimates, Mahalanobis distance** (epsilon=0.01, same query points):

| Model | pairwise | local-pert. (max) | grad-norm (max) |
|---|---|---|---|
| Logistic regression | 0.998 | 0.170 | 9.420 |
| MLP | 1.176 | 0.367 | 16.036 |
| CNN | 1.122 | 0.190 | 9.703 |

**The metric choice changes the estimates substantially, and unevenly
across sub-methods.** Switching from Euclidean to Mahalanobis distance
(same raw data, same trained models) drops `pairwise` by ~60-62% and
`local-perturbation` by ~87-88% across all three models, but
`gradient-norm` by only ~11-33%. Unlike the toy experiment, where
Mahalanobis distance moved a data-only estimate *toward* a known `L*`,
there's no ground truth to compare against here — the honest statement is
just that the metric matters a great deal, and matters differently
depending on which sub-method is asked, not that one number is "more
correct" than another.

**Sub-method agreement is weak, and gets weaker under Mahalanobis.** This
is the more important and more surprising finding. In the toy setting,
all three sub-methods landed within about 10% of each other and of `L*`.
Here, gradient-norm is **4-8x larger** than pairwise under Euclidean
distance (4.0x for logistic regression, 8.3x for the MLP, 4.0x for the
CNN), and **9-14x larger** under Mahalanobis distance (9.4x, 13.6x, 8.6x
respectively). Pairwise and local-perturbation stay roughly comparable to
each other throughout; gradient-norm is the consistent outlier. A
plausible (not rigorously confirmed) explanation: gradient-norm captures
the *exact* steepest direction at each query point via autograd, while
pairwise (bounded to the ~1000 sampled query points, mean pairwise
distance ~10.2 in raw pixel space) and local-perturbation (a fixed
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
| Logistic regression | 0.310 | 0.349 | 0.139 | 0.108 |
| MLP | 0.419 | 0.497 | 0.187 | 0.154 |
| CNN | 0.445 | 0.488 | 0.198 | 0.151 |

Under Euclidean distance, near-neighbor pairs have a **higher** mean
ratio than the general pair population (+10-19% across the three
models) — pairs a human would call visually similar are, if anything,
slightly more likely to have a disproportionately large margin swing for
their (small) pixel-space distance. Under Mahalanobis distance this
**flips**: near-neighbor pairs have a **lower** mean ratio than the
general population (−18-24%). Both directions are consistent across all
three models. Why the direction flips with the metric is not established
here — a plausible starting point is that the Mahalanobis metric
downweights the very directions raw-pixel nearest neighbors are most
likely to differ along (the high-variance directions the covariance
already captures), but this hasn't been checked directly.

**Validity check: are the flagged near-neighbor pairs just mislabeled examples?** A natural
objection to treating a high-ratio near-neighbor pair as evidence of real model sensitivity is
that it might instead just be one image with a wrong label — a large margin swing for a small
pixel distance is exactly what a mislabeled example would also produce. This was checked directly
against [cleanlab/label-errors](https://github.com/cleanlab/label-errors)'s published,
human-verified MNIST test-set label corrections (15 confirmed errors out of 10,000 test images) —
not by retraining or modifying the dataset, purely a cross-reference of already-computed
near-neighbor pair indices against that published list, across all three original models' top-6
Euclidean near-neighbor pairs plus the higher-capacity CNN's own top pair and its high-ratio
same-digit "1/1" pairs (22 pairs checked in total). **21 of the 22 are clean** — including logistic
regression's top-6, which turn out to be almost entirely the same digit ("1" vs. "1") rather than
cross-digit confusions, and every one of them checks out clean too. That supports genuine model
behavior as the general pattern.

**The one exception is itself a positive result, not a caveat.** The original CNN's actual rank-3
highest-ratio near-neighbor pair (Euclidean) is 6 vs. 2 (test indices 9679 and 2200) — and index
**9679 is itself one of the 15 confirmed known label errors** (original label "6", rejected by a
majority of mTurk workers). This is direct, independent evidence that the near-neighbor diagnostic
can surface genuine *labeling* problems in MNIST's own test set, not just model confusions — the
same kind of real, interpretable failure this checkpoint's whole design is meant to catch, just
from an unexpected source. See `results/label_error_crossref.md` for the full pair-by-pair table,
including a resolved data-provenance issue this cross-reference surfaced along the way: an earlier
version of this table checked a different set of "documented" pairs for the original CNN and
logistic regression that turned out to be stale, generated under a since-changed default of
`run_mnist_experiment`'s `n_lipschitz_points` parameter (300, changed to 1000 the day after that
documentation was written) — not file corruption or non-determinism, just documentation that never
got updated after a later, unrelated commit changed an upstream default. `results/label_error_crossref.md`
has the full diagnosis.

**Visual sanity check on logistic regression's same-digit pairs**: the same check already run on
the higher-capacity CNN's "1/1" pairs, now run on logistic regression's top-6 (`plot_pair_diagnostic_gallery`,
`results/pair_diagnostic_lr_top6.png`, images + predictions + a pixel-level difference panel per
pair). Structurally, a data indexing bug pairing a point with itself or a corrupted duplicate
already couldn't have produced these ratios at all — `estimators.py`'s ratio computation zeroes any
pair with `dist<=1e-12` — but that's checked visually here too, not just assumed. **Genuine, not an
artifact**: every pair is two visibly distinct handwriting samples (different slant, stroke width,
and in one recurring image a stray extra mark), every pairwise distance is comfortably nonzero
(6.3-7.9 in raw pixel space), and every difference panel shows real, spatially-concentrated
structure along the actual strokes, not a near-blank frame. The *mechanism* turns out to be
different from the CNN's case, though, and more classical: in every one of the 6 pairs, one member
is **misclassified** by logistic regression (a "1" predicted "5", or a "3" predicted "8") while its
near-neighbor of the same true digit is correctly classified — five of the six pairs share one
specific image (test index 5642, a "1" with a small stray mark) as their misclassified member,
consistently predicted "5" everywhere it appears. This is a point sitting at or past a genuine LR
decision boundary, with a same-digit neighbor sitting clearly on the correct side — the large
margin swing is close to definitionally expected there, not a surprising finding on its own. What
*is* still a genuine confirmation is that the near-neighbor search reliably finds these
boundary-straddling pairs at all, rather than something degenerate — a second, independent
confirmation (alongside the CNN's own "1/1" result, which involved two *correctly and confidently*
classified points instead) that this diagnostic surfaces real within-class decision-boundary
sensitivity, via two different underlying mechanisms, not a coincidence specific to one model.

**Embedding degree sweep** (`run_embedding_degree_sweep(degrees=(1, 2, 3))`, logistic regression,
`elementwise_embedding`, epsilon selected on a fixed 3000-point pool, final precision matrix fit
on the full 60k training set, 1000-point ratio-distribution subset — same setup as the rest of
this section, repeated once per degree):

| Degree | Selected epsilon | Cond. number at selected epsilon | All-pairs mean | Near-neighbor mean |
|---|---|---|---|---|
| 1 (identity) | 0.01 | 508 | 0.146 | 0.111 |
| 2 | 0.01 | 929 | 0.121 | 0.090 |
| 3 | 0.01 | 1309 | 0.110 | 0.080 |

Raising the embedding degree **shrinks both ratio-distribution summary statistics
monotonically** (all-pairs mean and near-neighbor mean both decrease at every step from degree 1
to 3), while the condition number at the selected epsilon grows roughly in proportion to the
embedded dimension (`784*degree`) instead — a higher-dimensional embedded space is intrinsically
harder to condition well, regardless of what the ratio statistics do. **This looks like a pure
scale effect, not a structural or locality one**: the near-vs-all-pairs reversal already
established above under plain (degree=1) Mahalanobis distance — near-neighbor pairs have a
*lower* mean ratio than the general population — **holds at every degree tested**, with a
comparable relative gap each time (near-neighbor mean is roughly 24-27% below the all-pairs mean
at all three degrees). Raising the embedding degree changes the ratio's overall scale without
changing *which* pairs (visually-similar vs. general population) are more or less sensitive
relative to each other.

**A structurally different embedding: local spatial cross-terms (negative result).**
`embeddings.py::local_patch_cross_terms(x_image)` was built as a second, deliberately different
embedding to compare against `elementwise_embedding`: instead of same-pixel powers, it maps each
pixel to its raw value plus one cross-term product (`x_i * x_j`) with each of its immediate
spatial neighbors in a 3x3 window (using the actual 28x28 image layout, zero-padded at the
border), giving a 3920-dimensional embedded space (`784` raw `+ 4*784` cross-terms — each
unordered adjacent-pixel pair counted once, not twice). It's exercised through the exact same
`epsilon_stability_check(embed_fn=...)` path as the degree sweep above, with the same 3000-point
epsilon-selection pool and full-60k final precision matrix.

**Epsilon selection fails categorically for this embedding.** All 7 candidates in the standard
sweep (`1e-6` through `100`) produced coefficients of variation of **0.91-1.45** — every single
one an order of magnitude past the `cv<=0.05` stability bound — with mean gradient-norm estimates
in the billions to trillions. This is not a conditioning failure: at `epsilon=0.01`,
`cond=1821` alone would have passed the `cond<=1e4` bound; it's specifically the resampling
*stability* of `gradient_norm_estimate`'s per-point dual norm that never settles, at any
regularization level tried, including the most heavily regularized candidate (`epsilon=100`,
`cv=1.25`). `select_epsilon` falls back, with its explicit warning, to `epsilon=1` (lowest cv
among a set of uniformly bad candidates). This is qualitatively different from
`elementwise_embedding`, which degrades *gracefully* as degree increases (condition number climbs,
but the resampling coefficient of variation stays in the same 0.01-0.03 range at every degree
tested — see the table above).

A working hypothesis, not yet independently verified: `gradient_norm_estimate`'s embedded path
pulls the metric back through `embed_fn`'s per-point Jacobian (see its docstring). For
`local_patch_cross_terms`, a cross-term feature's Jacobian entries are literally the *value* of
the neighboring pixel it's paired with — on MNIST, where most pixels are exactly 0 (background),
most of a typical query point's cross-term Jacobian rows are near-zero. The aggregate,
whole-dataset precision matrix doesn't reveal this (it's built from covariance across many points,
where every pixel takes nonzero values somewhere), but at any *individual* query point the local
pullback metric can be erratic in a way the aggregate never surfaces — a genuine structural
property of pairing raw-pixel-value-weighted cross-terms with a mostly-sparse image, not a bug
(`local_patch_cross_terms` itself is fully unit-tested, including an exact hand-checked
boundary/zero-padding example, in `tests/test_embeddings.py`).

**The fallback-epsilon ratio-distribution numbers below are not a clean comparison against the
table above — included for completeness, not as a headline result:**

| | All-pairs mean | Near-neighbor mean |
|---|---|---|
| `local_patch_cross_terms` (epsilon=1 fallback, n=300) | 0.288 | 0.246 |

Two caveats apply simultaneously: (1) `epsilon=1` is **100x** the regularization used for every
other row in this README (`epsilon=0.01`) — per `distance.py`'s own documented property
(`mahalanobis_distance` converges to scaled Euclidean distance as epsilon grows), this metric
behaves considerably more like Euclidean distance than a genuine data-fit Mahalanobis metric, and
(2) the ratio-distribution subset here is 300 points (~44,850 pairs), not the 1000 points
(~499,500 pairs) used everywhere else in this file — cut after the original 1000-point run
exhausted this machine's available memory and swap partway through (gathering `embed_fn` over
hundreds of thousands of pairs at 3920 dimensions has a much larger memory footprint than the
same step at `elementwise_embedding`'s largest dimension tested, 2352 — a real scaling gap in how
this embedding was wired into the existing pairwise machinery, left unfixed since this was a
one-off exploratory comparison, not promoted into the permanent pipeline).

With both caveats in mind: the near-vs-all-pairs reversal still holds direction (near-neighbor
mean below all-pairs mean, `-14.6%`), consistent with every other Mahalanobis-metric result in
this file, but the gap is shallower than any of them (`-22%` to `-27%` elsewhere) — plausibly
because the heavy `epsilon=1` regularization is already pulling this specific comparison partway
back toward Euclidean-distance territory, where the reversal direction is known to flip entirely
(Euclidean logistic regression: near-neighbor mean *above* all-pairs mean, see the Euclidean vs.
Mahalanobis table earlier in this section). Neither `local_patch_cross_terms` nor this comparison
is wired into `run_experiment.py`, the notebook, or `main()` — it's a completed one-off exploratory
result, not a permanent part of this experiment's pipeline.

**Follow-up: does plain Euclidean distance on this embedding show a locality signal on its own,
without any covariance-based reweighting?** Yes — closing the open question of whether the
epsilon-selection instability above was a property of the embedding itself or specifically of the
covariance reweighting. Applying `euclidean_distance_fn` directly to `local_patch_cross_terms`-embedded
features (no covariance, precision, or epsilon anywhere) on the same logistic-regression setup,
near-neighbor pairs still sit above the all-pairs mean (all-pairs mean 0.165, near-neighbor mean
0.171, `+3.9%`) — the same direction as raw-pixel Euclidean distance's own locality signal — so the
embedding is locally informative on its own, and the earlier instability was specifically a
Mahalanobis/covariance-reweighting problem, not a defect of the embedding. That said, the signal is
markedly weaker than raw pixels' own Euclidean result (`+13%` for logistic regression, see the
Euclidean-vs-Mahalanobis ratio-distribution table earlier in this section), so this embedding
doesn't add value over raw pixels under Euclidean distance either. And even the heavily-damped
`epsilon=1` Mahalanobis fallback above — already noted as behaving considerably more like Euclidean
distance than a genuine data-fit metric — still reverses the near/all sign relative to true
Euclidean distance on the same features (`-14.6%` vs. `+3.9%`), which generalizes this project's
"self-referential covariance reweighting shrinks/reverses the signal" finding beyond just
full-strength or numerically unstable Mahalanobis: even a weak, near-Euclidean one flips it. Run as
a one-off scratch script, same exploratory status as the rest of this section — see
`results/local_patch_cross_terms_euclidean_followup.md` for the full setup, comparison table, and a
reproducible snippet.

## Limitations and open questions

- **Sub-method disagreement (4-14x) is larger here than in the toy
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
- **The embedding degree sweep only covers logistic regression, only Mahalanobis distance, and
  only degrees 1-3.** Whether the same monotonic-shrink-plus-stable-reversal pattern holds for
  the MLP/CNN, under Euclidean distance in the embedded space, or at higher degrees (where
  condition-number growth may eventually force a much larger selected epsilon, or none of the
  swept candidates may qualify) hasn't been checked.
