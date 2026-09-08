# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a research codebase studying **Lipschitz-constant estimability**: how well the
Lipschitz constant of a function can be recovered from data alone vs. from a trained model,
and how much the choice of distance metric (Euclidean vs. Mahalanobis) affects that recovery.
It's organized as a sequence of experiments, each in its own top-level package:

- **`toy_example/`** — Experiment 1. A 1D/2D regression testbed with a closed-form or
  numerically-refined ground-truth Lipschitz constant `L*`, used to validate the estimation
  methodology itself before applying it to real models.
- **`mnist_example/`** — Experiment 2. Scales the same three estimators to real classifiers
  (logistic regression, MLP, CNN) trained on MNIST. There is no `L*` here — validity comes from
  (a) agreement between the three sub-methods and (b) stability across resampling.

Each experiment package has its own detailed `README.md` (`toy_example/README.md`,
`mnist_example/README.md`) — **read the relevant one before making non-trivial changes**; they
document the exact rationale behind non-obvious design choices (numerical conventions, why a
particular regularization/algorithm was chosen over an alternative, what's still open/unresolved).
Do not duplicate that content here; this file only covers what's common across both packages.

## Environment and commands

No `pyproject.toml`/`requirements.txt` — dependencies live in the committed `.venv/` (torch,
torchvision, numpy, scikit-learn, matplotlib, pandas, pytest, jupyter). Always invoke tools
through it rather than a bare `python`/`pytest`:

```bash
# run all tests for one experiment
.venv/bin/python -m pytest toy_example/tests/ -v
.venv/bin/python -m pytest mnist_example/tests/ -v

# run a single test file / test
.venv/bin/python -m pytest mnist_example/tests/test_estimators.py -v
.venv/bin/python -m pytest mnist_example/tests/test_estimators.py::test_name -v

# run the full experiment driver
.venv/bin/python -c "from toy_example.run_experiment import main; main()"
.venv/bin/python -c "from mnist_example.run_experiment import main; main()"

# execute a notebook end-to-end (regenerates results/ and plots in place)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace mnist_example/notebook_distance_measures.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace mnist_example/notebook_adversarial.ipynb
```

`mnist_example`'s full experiment run takes about a minute on CPU. Both packages' `results/`
directories are git-ignored (generated `.png`/`.json`/`.npz` outputs); notebooks are thin drivers
that import from the package and display saved figures — no reusable logic belongs in a notebook.

## Shared architecture across experiments

Both packages follow the same module split, and `mnist_example` is a direct generalization of
`toy_example`'s estimator code (same function names, extended to take an `output_fn`/`y_batch`
and a pluggable `distance_fn`):

- **`estimators.py`** — the actual Lipschitz-estimation logic, independent per package but
  structurally parallel: `pairwise_lipschitz`, `local_perturbation_lipschitz`,
  `gradient_norm_estimate`. Treated as the most safety-critical code in the repo — see below.
  `mnist_example`'s versions take `output_fn` rather than a fixed margin function — usually
  `models.margin_fn` (scalar per example), but `layer_decomposition.py` also hands them
  vector-valued functions (`model.extractor`, raw logits), which `_diff_norm`/the Jacobian path
  in `gradient_norm_estimate` handle as a strict generalization, not a behavior change for
  scalar callers.
- **`data.py`** — dataset construction/loading.
- **`models.py`** — model definitions and training loops.
- **`distance.py`** *(`mnist_example` only)* — `euclidean_distance_fn`, and the Mahalanobis
  machinery (`svd_ridge_precision`, `mahalanobis_distance`, `make_mahalanobis_distance_fn`,
  `covariance_eigenvalues`, `sweep_epsilon`), all built from the SVD of the raw pixel matrix so
  nothing here ever forms the `(784, 784)` covariance explicitly (forming `X^T @ X` squares the
  condition number before regularization is applied). `toy_example` keeps the equivalent
  Mahalanobis logic inside `embeddings.py` instead of a separate file.
- **`embeddings.py`** — feature-space embeddings (`elementwise_embedding`, and in
  `mnist_example`, also `local_patch_cross_terms`) used to fit the Mahalanobis metric over a
  transformed space rather than raw pixels; optional everywhere it's threaded through
  (`embed_fn=None` leaves existing behavior unchanged).
- **`plots.py`** — all plotting logic, kept separate from computation.
- **`run_experiment.py`** — driver functions that wire data/model/estimator/plots together and
  save to `results/`; `main()` runs the full sequence for that experiment.
- **`tests/`** — see "Test methodology" below; mirrors the module names above.

`mnist_example` also has several self-contained sub-experiments, layered on top of the shared
modules above rather than replacing them, documented across its own `README.md`,
`distance_measures.md`, and `adversarial.md` rather than exhaustively here — read those before
changing one. Two notebooks cover the whole package: `notebook_distance_measures.ipynb` (the
flagship comparison, embedding-degree sweep, UMAP, smoothing/`radius_multiplier`/truncated-
Mahalanobis) and `notebook_adversarial.ipynb` (layer decomposition, SmallCNN and StrongCNN
adversarial-vs-bound comparisons, the StrongCNN multi-seed sweep), each as sequential sections
rather than one notebook per sub-experiment.

- **`mnist_example/layer_decomposition.py`** — splits `SmallCNN` into `extractor`/`head` to check
  the submultiplicative bound `L_extractor * L_head >= L_full` (`tests/test_layer_decomposition.py`,
  Section 1 of `notebook_adversarial.ipynb`). Documented in `distance_measures.md`, not
  `mnist_example/README.md` — read the module's own docstring and
  `run_cnn_width_sweep`/`layer_decomposition_experiment` before changing it.
- **The adversarial sub-experiment** (`adversarial_run_experiment.py`, `adversarial_plots.py`,
  `attacks.py`, flattened into the top level of `mnist_example/`, not a separate subpackage) —
  built on top of `layer_decomposition.py`: generates real FGSM/PGD adversarial examples against
  trained CNNs and checks whether their achieved sensitivity approaches the tight
  (`L_full_estimated`) vs. loose (`product_bound`) Lipschitz bounds, under both Euclidean and
  Mahalanobis distance. `adversarial_strong_cnn.py` repeats the same comparison for `StrongCNN` via
  an externally-built extractor/head split, since `StrongCNN` itself has no `.extractor`/`.head`
  attributes; `adversarial_strong_cnn_seed_sweep.py` extends that to independently trained
  `StrongCNN` seeds (5 by design, though `notebook_adversarial.ipynb`'s own default was reduced to
  2 for time — see `adversarial.md`'s "StrongCNN multi-seed sweep" section), checking whether the
  Euclidean-vs-Mahalanobis gap-narrowing effect reproduces and how much adversarial sensitivity
  varies given matched clean accuracy. Read `mnist_example/adversarial.md` before changing anything
  here — it documents why the Mahalanobis repeat reuses the same checkpoints/adversarial examples
  rather than re-deriving them, and why `StrongCNN`'s functions raise on `model.training` instead
  of silently calling `.eval()`.
- **`mnist_example/umap_embedding.py`** — tries a *learned* UMAP embedding as a distance metric
  instead of a hand-built Mahalanobis one (`tests/test_umap_embedding.py`, the UMAP section of
  `notebook_distance_measures.ipynb`). Verdict, documented at length in `distance_measures.md`: the
  elevated near-neighbor ratio it produces is a compression artifact of UMAP's local-structure
  objective, not evidence of real model sensitivity — don't treat it as a validated distance metric
  for other work in this repo.
- **`mnist_example/augmentation.py`** (`random_affine_augment`) and `models.py`'s `StrongCNN` — a
  higher-capacity, near-state-of-the-art CNN baseline (BatchNorm + Dropout, trained with light
  rotation/translation augmentation) built ahead of a later data-cleaning experiment.
  `StrongCNN` deliberately has no `extractor`/`head` split like `SmallCNN` — see
  `adversarial.md`'s "StrongCNN sub-experiment" section for how the adversarial comparison works
  around that. Its BatchNorm/Dropout layers are also why eval-mode discipline (checked explicitly,
  not assumed) first became a real concern in this codebase.
- **`mnist_example` has one canonical trained `SmallCNN`/`StrongCNN` checkpoint**
  (`mnist_example/checkpoints/`, `models.py::train_or_load_small_cnn`/`train_or_load_strong_cnn`)
  that every experiment inside this package reuses rather than independently retraining a copy —
  see `mnist_example/README.md`'s "Shared model checkpoint" section. `signature_distance` is
  intended to load this same checkpoint too, for direct cross-package comparability; not yet wired
  up as of this writing, tracked as a separate follow-up.

## Test methodology — checkpoint-gating

Both experiments follow a strict rule: **an estimator or numerical convention is not used
downstream until it has been checked against an independent closed-form or analytic identity**.
Concretely:

- `toy_example/tests/test_tier_a_closed_form.py` checks the hand-derived analytic gradient
  against `torch.autograd.grad` before anything else is trusted.
- `mnist_example/tests/test_estimators.py` is the critical checkpoint for that package: on a
  2-class logistic regression model, `margin_fn` reduces to an exactly linear function with a
  closed-form Lipschitz constant, and all three sub-methods are checked against it (10%
  tolerance) *before* being trusted on the MLP/CNN. The Mahalanobis "precision vs. precision⁻¹"
  dual-norm convention in `gradient_norm_estimate`, and its `embed_fn`-aware pullback-metric path
  (identity/linear/`elementwise_embedding` cases), are similarly checked against independent
  closed-form identities, not just derived and trusted — these are easy to get backwards with
  nothing to catch it once real data (no `L*`) is involved.

When adding a new estimator or changing a numerical convention (e.g. a distance metric, a
dual-norm), follow the same pattern: add a closed-form or independently-derived check in
`tests/`, and don't wire the new logic into `run_experiment.py`/notebooks until that check passes.
This isn't just belt-and-suspenders: `mnist_example`'s `local_patch_cross_terms` embedding
(`embeddings.py`) passed unit tests but made `epsilon_stability_check`'s resampling stability fail
categorically at every regularization level (`cv` of 0.9-1.45 against a `<=0.05` bound) — the
result is documented as a negative finding in the README rather than silently wired into
`run_experiment.py`/`main()`. Don't promote a new embedding/estimator into the default pipeline
just because it runs without error; check that it actually passes the stability/agreement bounds
first.

## Conventions worth knowing before editing

- **`toy_example` uses `torch.float64` everywhere** (`torch.set_default_dtype(torch.float64)` at
  import time in every module) so true-vs-estimate comparisons aren't contaminated by float32
  noise. `mnist_example` does not follow this (no closed-form ground truth to protect at that
  precision).
- **Distance metrics are threaded through as a `distance_fn` (or `precision` matrix) parameter**,
  not hardcoded — `euclidean_distance_fn` is the default; Mahalanobis variants are built via
  `make_mahalanobis_distance_fn` (mnist: `distance.py`; toy: `embeddings.py`). Both packages'
  Mahalanobis path also accepts an optional `embed_fn` to fit the metric over an embedded feature
  space instead of raw input — leaving it `None` must leave existing raw-space behavior exactly
  unchanged (checked directly in tests, not just asserted). In `gradient_norm_estimate`, a
  nonlinear `embed_fn` makes the correct dual norm a per-point *pullback* through `embed_fn`'s
  Jacobian, not the fixed-`precision` formula reapplied to the embedded gradient — computed via
  `torch.func.jacrev`/`vmap`, checked against closed-form Jacobians for the identity/linear/
  `elementwise_embedding` cases.
- **`mnist_example/distance.py`'s Mahalanobis precision matrix is built from the pixel matrix's
  SVD, never from a formed `(784, 784)` covariance matrix** — `X^T @ X` squares the condition
  number before ridge regularization is applied, and MNIST's raw covariance is exactly singular
  in practice (constant-zero border pixels). Any new covariance/precision-related function in that
  file should follow the same SVD-based route rather than forming `Sigma` directly.
- **The three sub-methods (pairwise, local-perturbation, gradient-norm) are never merged into a
  single "local estimate."** They're kept separate and separately labeled throughout, including
  in plots — their disagreement is itself a finding, not noise to average away.
- Datasets in `toy_example` are always noiseless (`y = f_star(x)` exactly); if noise is ever
  needed it belongs in the definition of `f_star`, not as a stochastic wrapper.
- New driver functions added to `run_experiment.py` are not automatically added to `main()` —
  several (e.g. `run_gap_N_sweep_seed_averaged` in `toy_example`) are deliberately kept opt-in
  because they're multiples slower than the standard run. Check each README's "How to run it" /
  "Status" section before assuming a function is part of the default pipeline.
