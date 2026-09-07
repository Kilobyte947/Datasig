# signature_distance

Path-signature-based image distance metrics for MNIST, evaluated as
denominators in a Lipschitz-ratio pipeline (`|margin_i - margin_j| /
distance(x_i, x_j)`) against plain pixel-space Euclidean distance. Three
independent, fixed (non-learned) constructions turn an image into one or
more paths, each producing its own truncated path signature:

- **Method A -- patch singular-value stream.** A fixed set of 64 interior
  pixel locations; at each, the largest singular value of the surrounding
  3x3 patch. One stream per image, shape `(64, 2)`.
- **Method B -- reference-line stream.** A fixed set of straight lines
  through the image, each sampled at 32 points via bilinear interpolation.
  Geometry and truncation depth are hyperparameters (see below).
- **Method C -- Hilbert-curve stream.** A single order-5 space-filling
  curve through the image (1024 cells, resampled to 512 points, cut into
  16 segments of 32 points -- the same point budget as Method B).

All three constructions are fixed and non-learned, independent of any
trained classifier's own features. Method B and Method C are evaluated
**per-path**: each line/segment produces its own local Lipschitz ratio,
examined as a collection rather than merged into a single score.

## Shared infrastructure

- **Models.** `SmallCNN`/`StrongCNN` are the same trained checkpoints used
  in the `mnist_example` experiment (loaded, not retrained), so results
  here are directly comparable to that experiment's numbers. SmallCNN:
  98.60% test accuracy; StrongCNN (BatchNorm + dropout + augmentation):
  99.63%.
- **Attacks.** FGSM (single-step) and PGD (10-step, L-infinity projected
  gradient descent), both compared against a magnitude-matched random-noise
  control (same pixel-space L2 norm as the attack, not gradient-directed).
- **Metric.** For a genuinely flipped (misclassified) image pair, the
  fold-ratio is the mean adversarial Lipschitz ratio divided by the mean
  control-perturbation ratio on the same images -- how much more strongly a
  distance separates a targeted attack from equally-sized random noise.
- **Evaluation pool.** A fixed, class-balanced, deterministic subset of the
  MNIST test set (typically 200 images, 20/class).
- **Signature computation.** Truncated path signatures via `roughpy_jax`,
  checked against two closed-form identities (a straight line's exact
  tensor exponential, an L-shaped path's hand-computed area term) before
  use. JAX computes signatures in float32 regardless of this package's
  float64 default elsewhere (JAX's x64 mode is a separate, unused opt-in)
  -- a fixed precision ceiling on every signature-based distance.

## Within/cross-digit sanity check

Mean pairwise distance for same-digit vs. different-digit pairs (300
images, 30/class), cross/within ratio as the summary statistic:

| method | within-digit mean | cross-digit mean | ratio |
|---|---|---|---|
| Pixel-Euclidean | -- | -- | 1.15 |
| Method A | 14.60 | 17.18 | 1.18 |
| Method B (winner geometry) | -- | -- | 1.32 |
| Method C | -- | -- | 1.17 |

Method B's optimized geometry is the only signature distance showing a
real improvement over pixel-Euclidean on this check; Method A and Method C
sit close to the baseline. This within/cross-digit check is a weaker
signal than the adversarial evaluation below and predates Method B's
geometry optimization for Method A/C (shown at non-optimized defaults).

**Level decomposition.** Masking out the truncated signature's level-1
term and keeping only levels 2..depth reproduces the full ratio almost
exactly for both Method A and Method B (A: 1.175 vs. 1.176; B: 1.160 vs.
1.160) -- the higher-order signature terms carry the class-separation
signal, not the net-displacement (level-1) term. This is structural for
Method B specifically: every reference line's endpoints sit on the image
border, where MNIST intensity is ~0 for nearly every image, so 99.96% of
per-line net displacements are exactly zero by construction.

## Method B: geometry and depth selection

Method B's original geometry (8 horizontal + 8 vertical lines, truncation
depth 4) was optimized in two stages:

- **Stage A** -- a cheap screen (no model, no attack): for each of 32
  geometry x points-per-line x interpolation combinations, at 5 truncation
  depths, same/different-digit AUC on clean images. Depth is the dominant
  lever (lower is better; the original depth-4 default was not optimal),
  and shifting weight toward horizontal lines helps. Top single
  configuration: **12 horizontal + 4 vertical lines, depth 2** (AUC 0.6811
  vs. the original configuration's 0.6469).
- **Stage B** -- full adversarial fold-ratio validation (FGSM, three
  epsilons, both models) on the finalists from Stage A. **16 horizontal +
  0 vertical lines, depth 2 wins** (mean fold-ratio 13.45x vs. the original
  configuration's 7.03x, 2/84 line-level exceptions), beating 12h+4v
  (9.40x, 3/72 exceptions) on both models.

**Stage A and Stage B pick different winners**, and this is not a
contradiction: Stage A measures same/different-digit separation on clean
images; Stage B measures adversarial-vs-random-noise separation on a
trained classifier's actual predictions. There is no requirement that the
same geometry optimizes both. Stage B's winner (16h+0v, depth 2) is the
one used everywhere below, since the adversarial fold-ratio is this
project's actual target metric.

Two structurally border-adjacent lines (16h+0v's first and last horizontal
line, running along image rows 0 and 27) carry no same/different-digit
signal and are excluded from the informative-line convention used below
(14 of 16 lines).

## Method B vs. Method C vs. pixel-Euclidean

Same 200-image sample, same two models, FGSM and PGD, mean fold-ratio on
genuinely flipped pairs:

| | FGSM | PGD |
|---|---|---|
| Pixel-Euclidean | 8.31x | -- |
| Method B (14 informative lines) | 13.45x | 11.59x |
| Method B (all 16 lines, diagnostic only) | 12.6x | 13.4x |
| Method C (Hilbert, depth 3, all 16 segments) | 6.23x | 8.95x |

Method B beats both pixel-Euclidean and Method C under both attacks. The
all-16-line Method B figure is diagnostic only: it includes the two
border-adjacent lines, which can produce a near-zero baseline signature
distance for some images and destabilize a naive mean -- the
informative-line figure is the one to use for Method B.

FGSM produces a slightly higher fold-ratio than PGD for Method B here
(13.45x vs. 11.59x), despite PGD flipping more images at every epsilon.
This traces to StrongCNN's behavior at low epsilon, where only 5-9 of 200
images flip under either attack -- a small enough sample that individual
ratios swing the aggregate substantially. Fold-ratio ordering between
attacks is not a stable ranking at this sample size; flip rate and
fold-ratio answer different questions.

Method C is a coherent, correctly-verified alternative construction --
exception-free adversarial/control separation across every model, epsilon,
and segment -- but trails Method B by a consistent, non-trivial margin
under both attacks.

## Headline result: local Lipschitz sensitivity vs. model accuracy

Clean accuracy, FGSM adversarial accuracy (eps=0.03), and the 90th
percentile of the local Lipschitz-ratio distribution (`ratio_control` for
"clean", `ratio_adv` for "adversarial"), computed over every image x every
line/segment:

| Model | Clean acc | Adv acc | Method B P90 clean | Method B P90 adv | Method C P90 clean | Method C P90 adv |
|---|---|---|---|---|---|---|
| SmallCNN | 98.60% | 95.50% | 7.54 | 44.68 | 2.37 | 10.29 |
| StrongCNN | 99.63% | 95.50% | 59.49 | 96.86 | 18.42 | 41.00 |

SmallCNN and StrongCNN's accuracies sit close together, but StrongCNN's
P90 local Lipschitz ratio is roughly 8x SmallCNN's on clean (random-noise)
perturbations and 2-4x higher under adversarial perturbations, consistent
across both independently-constructed distance methods. **The
higher-capacity, higher-accuracy model has a substantially more sensitive
high-quantile tail, not a less sensitive one** -- accuracy does not predict
local extension behaviour.

Bootstrap resampling (1000 resamples over images, 90% CI) confirms this is
not sampling noise: every SmallCNN/StrongCNN interval pair is
non-overlapping, for both methods and both conditions.

| Method | Condition | SmallCNN 90% CI | StrongCNN 90% CI |
|---|---|---|---|
| B | clean | [7.06, 8.07] | [56.55, 64.16] |
| B | adversarial | [41.35, 48.76] | [89.60, 101.99] |
| C | clean | [2.18, 2.51] | [17.28, 19.45] |
| C | adversarial | [10.03, 10.58] | [39.36, 43.10] |

## Method A

Method A separates adversarial from random-noise-control perturbations
clearly (near-bimodal ratio distributions) but its adv/control ratio sits
consistently a little behind plain pixel-Euclidean's, at every epsilon and
on both models:

| model | eps | pixel adv/ctrl | Method A adv/ctrl |
|---|---|---|---|
| SmallCNN | 0.02 | 9.12 | 6.83 |
| SmallCNN | 0.03 | 8.44 | 6.35 |
| SmallCNN | 0.05 | 7.59 | 5.91 |
| StrongCNN | 0.02 | 3.77 | 3.60 |
| StrongCNN | 0.03 | 2.82 | 2.42 |
| StrongCNN | 0.05 | 1.78 | 1.43 |

Directly against Method B's merged (pre-per-path) distance at the same
epsilons, Method A and Method B trade off narrowly (neither consistently
ahead). At larger epsilons (0.05-0.3, beyond the imperceptible regime),
adv/control separation collapses monotonically for both distances as the
random control itself starts flipping predictions -- expected saturation,
not a change in which distance is stronger.

## Files

| File | Contents |
|---|---|
| `data_pool.py` | Fixed, deterministic MNIST test-set pool. |
| `models.py` | `SmallCNN`/`StrongCNN` (imported from `mnist_example.models`), training utilities, and load-only accessors for the shared trained checkpoint. |
| `attacks.py` | `fgsm_attack`, `pgd_attack`, `random_noise_perturbation`. |
| `streams.py` | Stream construction for all three methods. |
| `signatures.py` | `signature_of_stream` -- truncated signature via `roughpy_jax`, shared by all three methods. |
| `distances.py` | Rescaling, feature-vector construction, per-line/per-path distances, within/cross-digit checks, level decomposition, Method B's winner-geometry constants. |
| `plots.py` | All plotting. |
| `adversarial_eval.py` | Every adversarial/Lipschitz-ratio evaluation driver, sectioned by method and attack. |
| `method_b_sweep.py` | Method B's geometry/depth/interpolation hyperparameter sweep (Stage A screen, Stage B validation). |
| `headline_bootstrap.py` | The P90 local-Lipschitz-estimate analysis and its bootstrap confidence intervals. |
| `signatures_formation.ipynb` | Stream/signature/distance construction for all three methods, plus Method B's Stage A geometry screen. |
| `adversarial_eval.ipynb` | Full adversarial evaluation for all three methods, the Stage B validation, the headline P90 plot, and bootstrap confidence intervals. |
| `tests/` | Unit tests, mirroring the module list above. |
