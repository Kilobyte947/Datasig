# mnist_lipschitz

Experiment 2 of the Lipschitz-diagnostics project: scales the three
Lipschitz estimators built and validated in Experiment 1 (`toy_lipschitz`,
sibling folder) up to real classifiers trained on MNIST.

Experiment 1 had a known, closed-form ground-truth Lipschitz constant `L*`
to check estimates against. **There is no such ground truth here** -- a
trained neural network's exact Lipschitz constant is not analytically
knowable in general. Validity in this experiment therefore comes from two
weaker checks instead: **(a) agreement between the three independently
implemented sub-methods** (pairwise, local-perturbation, gradient-norm),
and **(b) stability of an estimate across independent resamples of the
data**. Both are checked directly (not assumed) throughout, and both turn
out to be more interesting -- and less reassuring -- than in the toy
setting. See [Key results](#key-results).

**Out of scope for this experiment** (may return in a later one):
adversarial examples, FGSM/PGD attacks, robustness mitigations (weight
decay, spectral norm, gradient penalty, adversarial training, input
smoothing), Fashion-MNIST.

## Layout

| File | Contents |
|---|---|
| `data.py` | `load_mnist` (via torchvision, pixel values kept in `[0,1]` -- no ImageNet-style normalization, so raw pixel differences stay directly interpretable for the covariance/Mahalanobis work in `distance.py`; returns both a flat `(N,784)` and image `(N,1,28,28)` view of the same pixels), `get_dev_subset` (small seeded subset for fast iteration), `make_loader` (wraps tensors in a `DataLoader` for training). |
| `models.py` | `LogisticRegressionModel` (784 -> 10, the one model with a closed-form Lipschitz constant, used for Checkpoint 3's test), `SmallMLP` (configurable hidden layers, ReLU by default), `SmallCNN` (two conv+pool blocks + FC head), `FlattenedInputWrapper` (wraps the CNN so it accepts flat `(N,784)` input like the other two models -- lets all three models share the same estimator code path), `train_classifier` (cross-entropy + Adam), `margin_fn` (`logit[y_true] - max(logit[j] for j != y_true)`, the function the estimators are actually applied to). |
| `estimators.py` | `pairwise_lipschitz`, `local_perturbation_lipschitz`, `gradient_norm_estimate` -- generalized from `toy_lipschitz/estimators.py` to take `margin_fn`/`y_batch` and a pluggable `distance_fn` (or `precision` matrix, for the gradient dual norm). `euclidean_distance_fn` is the default; `distance.py` supplies the Mahalanobis alternative. |
| `distance.py` | `pixel_covariance` (empirical covariance of centered pixel vectors), `ridge_precision` (dense `(Sigma + epsilon*I)^-1`), `mahalanobis_distance`, `make_mahalanobis_distance_fn` (wraps a fixed precision matrix into a `distance_fn` closure for direct use with `estimators.py`). |
| `run_experiment.py` | `sweep_epsilon`, `epsilon_stability_check`, `select_epsilon` (Checkpoint 5's epsilon selection); `run_mnist_experiment` (the main driver -- trains all three models, runs all three estimators under Euclidean distance, selects an epsilon, re-runs under Mahalanobis distance, saves everything to `results/`). |
| `plots.py` | `plot_euclidean_vs_mahalanobis` (grouped bars, one panel per sub-method), `plot_epsilon_sweep` (condition number + subsample instability vs. epsilon, log scale), `plot_submethod_agreement` (the three sub-methods side by side per model, log scale -- the validity check made visible, since there's no `L*` to plot as a reference line). |
| `tests/test_data.py` | MNIST shapes, pixel value range `[0,1]`, label range `{0..9}`, dev-subset seed reproducibility. |
| `tests/test_models.py` | Each model trains above an accuracy threshold on the full test set (with headroom below what's actually observed, so the test isn't flaky against normal seed variance); `margin_fn` matches a manual per-example computation and is differentiable w.r.t. `x`. |
| `tests/test_estimators.py` | **Critical checkpoint.** On `LogisticRegressionModel(num_classes=2)`, `margin_fn` reduces to an exactly linear function with a closed-form Euclidean Lipschitz constant `\|\|w_0-w_1\|\|_2` -- all three sub-methods are checked against it (10% tolerance) before being trusted on the MLP/CNN. Also directly checks the Mahalanobis "P vs. P^-1" dual-norm convention in `gradient_norm_estimate` against an independent closed-form identity. |
| `tests/test_distance.py` | As epsilon grows large, `mahalanobis_distance` converges to a constant multiple (`1/sqrt(epsilon)`) of Euclidean distance; real MNIST pixel covariance is confirmed exactly rank-deficient (`cond=inf`) while the ridge-regularized version is well-conditioned. |
| `tests/test_epsilon_selection.py` | Condition number is (deterministically) non-increasing as epsilon grows; subsample instability does not increase going from a near-singular to a well-regularized epsilon; `select_epsilon`'s bound-matching and fallback logic. |
| `notebook_mnist_lipschitz.ipynb` | Thin driver notebook -- imports from this package, runs `run_mnist_experiment()`, displays all three plots with introductory markdown. No reusable logic of its own. |
| `results/` | Generated outputs (git-ignored except `.gitkeep`): `mnist_experiment_results.json` (scalar summary), `mnist_experiment_arrays.npz` (full per-point local-perturbation/gradient-norm arrays), and the three plots. |
| `data/` | Downloaded MNIST files (git-ignored, ~63MB) -- recreated automatically by `load_mnist` on first run. |

## Design decisions

- **ReLU for the MLP, not tanh.** `toy_lipschitz` used tanh throughout for continuity with its smooth closed-form ground truth; there's no such ground truth here, and ReLU is the standard choice for MNIST classifiers (faster to train, no vanishing-gradient concern at this depth). `SmallMLP` still supports `activation="tanh"` for anyone who wants to compare.
- **Pairwise sampling: keep N modest, not random-subsample from a huge pool.** `pairwise_lipschitz` is handed a few hundred points (300 in the main run) and scores *all* pairs among them (~45k pairs, cheap), rather than defaulting to random-pair subsampling from a much larger pool. `max_pairs` is still supported as a safety valve.
- **The `precision` vs. `precision^-1` convention in `gradient_norm_estimate`.** For a Mahalanobis distance with quadratic-form matrix `P` (i.e. `distance_fn` computes `sqrt((x-y)^T P (x-y))`), the correct dual norm of a gradient `g` is `sqrt(g^T P^-1 g)` -- **not** `sqrt(g^T P g)`. This is easy to get backwards with nothing to catch it once real MNIST data is involved (no `L*` there), so it was checked directly against an independent closed-form identity (the maximizer `delta* = P^-1 g / sqrt(g^T P^-1 g)` provably attains the claimed value) in `tests/test_estimators.py`, not just derived and trusted. `gradient_norm_estimate` inverts the given `precision` internally to recover `Sigma`; this is a second inversion (the first happens in `distance.py` to build `precision` in the first place), but a dense 784x784 inverse measured at ~10ms is negligible next to model training, and keeping one `precision`-matrix convention across all three estimators (matching `toy_lipschitz/embeddings.py`'s dense-matrix convention) was judged simpler than threading a Cholesky factor through three different call sites.
- **Ridge regularization, not PCA truncation, for the singular pixel covariance.** Not compared head-to-head here (see [Status](#status)) -- ridge was the more direct generalization of `toy_lipschitz`'s existing `precision_from_covariance` (which already adds `eps*I`), and needed no extra machinery (choosing a truncation rank, deciding how to handle the discarded subspace in the distance formula) to get a first result.
- **`epsilon_stability_check` uses the mean gradient-norm estimate, not `pairwise_lipschitz`'s max.** This was changed after the first implementation: using `pairwise_lipschitz` (a max over ~200-300 pairs) as the per-subsample yardstick gave a coefficient-of-variation that bounced between roughly 0.04 and 0.26 with no clean trend against epsilon -- an extreme-value statistic over a modest number of pairs is dominated by which specific pair happens to be sampled near the metric's steepest direction, which adds a lot of irreducible sampling noise on top of (and swamping) the genuine metric-shape instability epsilon is meant to control. Switching to the *mean* gradient-norm estimate over 100 points gave a clean, repeatable (checked across several seeds) decreasing-then-flat trend with cv in the 0.007-0.045 range instead.
- **The stability check's model is always logistic regression**, regardless of which model the final comparison uses -- epsilon selection only needs *a* consistent, cheap-to-evaluate yardstick, not the specific model being analyzed. Unlike Experiment 1's strict "no model involved" `L_hat_data`, there's no model-free scalar function of `x` on MNIST to fall back on.
- **`select_epsilon` never silently returns a value outside its stated criteria** -- if no candidate epsilon meets both the condition-number and stability bounds, it falls back to the lowest-cv candidate and prints an explicit warning, rather than picking silently or raising.

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
CNN; 300 Lipschitz query points; a 7-point epsilon sweep from `1e-6` to
`100`), matching what produced the numbers below.

## Key results

All numbers below are from an actual run (`results/mnist_experiment_results.json`), full MNIST train/test sets, seed 0.

**Model accuracies** (test set):

| Model | Train acc. | Test acc. |
|---|---|---|
| Logistic regression | 92.87% | 92.70% |
| MLP (1 hidden layer, 128 units, ReLU) | 99.31% | 97.82% |
| CNN (2 conv+pool blocks) | 98.89% | 98.65% |

**Epsilon selection**: swept `{1e-6, 1e-4, 1e-2, 1e-1, 1, 10, 100}`. Both
the condition number (`5.1e6 -> 1.05`) and the subsample coefficient of
variation (`0.045 -> 0.007`) decrease monotonically across the whole
sweep. Selected epsilon = **0.01** (`cond=513`, `cv=0.019`) -- the
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

**Does the metric choice change the estimates? Yes, substantially, and
unevenly across sub-methods.** Switching from Euclidean to Mahalanobis
distance (same raw data, same trained models) drops `pairwise` by
~57-58% and `local-perturbation` by ~87-88% across all three models, but
`gradient-norm` by only ~10-28%. This is the opposite direction from
Experiment 1's headline result, where Mahalanobis distance *increased* a
data-only estimate that had been undershooting `L*`; here, with no `L*`
to compare against, the honest statement is just that the metric matters
a great deal, and matters differently depending on which sub-method is
asked -- not that one number is "more correct" than another.

**Sub-method agreement is weak, and gets weaker under Mahalanobis.** This
is the more important and more surprising finding. In Experiment 1's toy
setting, all three sub-methods landed within about 10% of each other and
of `L*`. Here, gradient-norm is **4-9x larger** than pairwise under
Euclidean distance (5.7x for logistic regression, 9.4x for the MLP, 4.2x
for the CNN), and **9-16x larger** under Mahalanobis distance (12.2x,
15.8x, 8.7x respectively). Pairwise and local-perturbation stay roughly
comparable to each other throughout; gradient-norm is the consistent
outlier. A plausible (not rigorously confirmed) explanation: gradient-norm
captures the *exact* steepest direction at each query point via autograd,
while pairwise (bounded to the ~300 sampled query points, mean pairwise
distance ~9.3 in raw pixel space) and local-perturbation (a fixed
Euclidean radius of 1.0, only 20 sampled directions) are much more likely
to miss a network's sharp, narrow high-sensitivity directions than to hit
one. This was not investigated further within this experiment -- see
[Status](#status).

## Status

**Confirmed working, per the checkpoint-gating rule (no checkpoint was
skipped):** MNIST loading, all three models train above threshold, the
closed-form logistic-regression check (Euclidean and Mahalanobis) passes,
the ridge-regularization sanity limits pass, epsilon selection produces a
monotonic condition-number trend (guaranteed) and a monotonic-then-flat
stability trend (empirically confirmed, not guaranteed, across several
seeds during development), and the full experiment runs end-to-end in
about a minute on CPU.

**Explicitly out of scope, not attempted:** adversarial examples,
FGSM/PGD attacks, any robustness mitigation (weight decay, spectral norm,
gradient penalty, adversarial training, input smoothing), Fashion-MNIST.

**Worth revisiting:**
- **Sub-method disagreement (4-16x) is larger here than expected** going
  in, and larger than Experiment 1's toy setting ever showed. This
  experiment reports the disagreement and a candidate explanation (see
  above) but does not resolve it -- e.g. by checking whether increasing
  `n_directions` or `local_radius` in `local_perturbation_lipschitz`
  narrows the gap with gradient-norm, which would support the "missing
  the sharp direction" hypothesis, or whether it doesn't, which would
  point somewhere else entirely.
- **PCA-truncation regularization was not tried** as an alternative to
  ridge regularization for the singular pixel covariance; ridge was used
  because it was the more direct extension of Experiment 1's existing
  code, not because it was compared and found better.
- **The Mahalanobis extension only reaches the same 300 query points used
  for Euclidean distance** -- it was not checked whether epsilon selected
  via `epsilon_stability_check` (which uses the logistic-regression model
  and 100 different points) is also a good choice specifically for the
  MLP or CNN's estimates; the experiment assumes one epsilon suffices for
  all three models rather than testing that assumption.
- **`local_radius=1.0` and `n_directions=20` were chosen by inspecting
  typical MNIST pixel-space distances (image L2 norms ~9.3, so radius 1.0
  is a genuinely "local" ~10% perturbation) rather than by a dedicated
  sensitivity sweep** the way epsilon was; Experiment 1's toy setting
  didn't need this kind of judgment call since its estimator-correctness
  tests operated at a scale small enough to make convergence properties
  analytically clear.
