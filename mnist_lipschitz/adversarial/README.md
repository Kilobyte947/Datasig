# adversarial

Sub-experiment within `mnist_lipschitz` (Experiment 2), built on top of `layer_decomposition.py`
(sibling module, one level up). Asks a practical question the layer-decomposition sub-experiment's
looseness-ratio number alone can't answer: **does the looseness of the per-layer Lipschitz bound
actually matter?** It generates real FGSM/PGD adversarial examples against trained CNN
checkpoints and checks whether their achieved sensitivity sits close to the tight bound, close to
the loose bound, or well below both — then repeats the entire comparison under Mahalanobis
distance instead of Euclidean, to see whether the metric choice changes the answer.

## Background: the two bounds

`layer_decomposition.py` splits a trained `SmallCNN` into `f = head o extractor` and computes two
theoretical upper bounds on how much the network's full 10-d logit vector can move per unit of
input change:

- **`L_full_estimated`** — the TIGHT bound: the network's own empirical Lipschitz constant,
  measured directly on `f`.
- **`product_bound`** (`L_extractor_estimated * L_head_exact`) — the LOOSE bound: the classic
  submultiplicative per-layer bound (Szegedy et al., 2014, *"Intriguing Properties of Neural
  Networks"*), which that sub-experiment already finds to be loose by roughly 2x-6x depending on
  model width (see below).

## What this experiment does

1. **Generates adversarial examples.** `attacks.py` implements FGSM (Goodfellow et al., 2015) and
   PGD (Madry et al., 2018) from scratch, operating only on cross-entropy loss within an L_inf
   pixel-space ball — neither has any notion of Euclidean vs. Mahalanobis distance.
2. **Attacks only correctly-classified points.** `filter_correctly_classified` restricts the
   evaluation pool before any sampling or attacking — attacking an already-wrong prediction isn't
   a meaningful question here.
3. **Measures the achieved ratio** `R_adv = ||f(x) - f(x_adv)||_2 / distance(x, x_adv)`, per
   example — the numerator is always the full 10-d logit vector (matching `L_full_estimated`'s
   convention, **not** the scalar margin used elsewhere in this project), the denominator is
   pluggable (`achieved_ratio`'s `distance_fn`).
4. **Compares against both bounds**, across an epsilon sweep (`{0.05, 0.1, 0.15, 0.2, 0.25}`,
   Goodfellow et al. 2015's MNIST range) and a CNN-width sweep (`{4, 8, 16, 32, 64}`, matching
   `layer_decomposition.run_cnn_width_sweep`'s own widths), reusing
   `layer_decomposition_experiment` directly rather than recomputing its numbers independently.
5. **Repeats the entire comparison under Mahalanobis distance** (`run_experiment.py`'s
   `*_with_distance_fn` functions), fit from MNIST's own pixel covariance the same way
   `mnist_lipschitz`'s main experiment does, then compares the two metrics directly.

**`max_R_adv` is a LOWER BOUND on the network's true worst-case sensitivity, not an upper bound or
exact value** — FGSM/PGD maximize cross-entropy loss, not `R_adv` directly. Results throughout are
"achieved under this attack," never "the" worst case.

## File reference

| File | Contents |
|---|---|
| `attacks.py` | `fgsm_attack`, `pgd_attack` (with L_inf projection, random restarts, `[0,1]` clipping). |
| `run_experiment.py` | `filter_correctly_classified`, `achieved_ratio`, `run_epsilon_sweep`, `summarize_epsilon_sweep`, `most_and_least_sensitive_examples`, `head_layer_bound_check`, `run_bound_comparison`, `run_cnn_adversarial_width_sweep`, `main` — the Euclidean pipeline — plus `compute_bounds_with_distance_fn`, `run_bound_comparison_with_distance_fn`, `run_cnn_adversarial_width_sweep_with_distance_fn`, `main_with_distance_fn`, `build_pixel_mahalanobis_distance_fn` — pluggable-metric generalizations used for the Mahalanobis repeat. |
| `plots.py` | `plot_R_adv_distribution`, `plot_bound_closeness_vs_width`, `plot_extreme_examples` (all accept a `metric_name` for the title/axis wording), plus `plot_euclidean_vs_mahalanobis_R_adv` and `plot_euclidean_vs_mahalanobis_bounds_vs_width` for the direct comparison. |
| `notebook_adversarial_lipschitz.ipynb` | Thin driver notebook — Euclidean baseline, Euclidean width sweep + per-width most/least-sensitive examples, then the identical sequence again under Mahalanobis distance, then the Euclidean-vs-Mahalanobis comparison. No reusable logic of its own. |
| `tests/test_attacks.py` | FGSM/PGD stay within the epsilon ball and `[0,1]`; `epsilon=0` is a no-op; PGD's single-step/single-restart case reduces exactly to FGSM. |
| `tests/test_run_experiment.py` | Correctly-classified-only filtering; `achieved_ratio`'s logit-space convention; PGD achieves at least as much sensitivity as FGSM; the `max_R_adv > L_full_estimated` sanity check warns without dropping the row; `head_layer_bound_check` never exceeds its own Cauchy-Schwarz bound. |
| `tests/test_mahalanobis.py` | **Central correctness checkpoint**: `compute_bounds_with_distance_fn(distance_fn=euclidean_distance_fn)` reduces to EXACTLY `layer_decomposition_experiment`'s own numbers. Also: Mahalanobis `distance_fn` correctness, retraining determinism (the `*_with_distance_fn` width sweep reproduces bit-identical checkpoints to the original), and end-to-end Mahalanobis runs. |
| `strong_cnn_experiment.py` | StrongCNN variant of this comparison (see "StrongCNN sub-experiment" below): `strong_cnn_extractor_fn`, `strong_cnn_head_module`, `fit_strong_cnn_feature_normalizer`, `compute_strong_cnn_bounds`, `strong_cnn_head_layer_bound_check`, `strong_cnn_bound_comparison`, `main` — reuses `attacks.py`/`plots.py`/`run_experiment.py`'s architecture-agnostic functions unchanged. |
| `notebook_adversarial_strongcnn.ipynb` | Thin driver notebook for the StrongCNN baseline (Euclidean + Mahalanobis, no width/seed sweep). No reusable logic of its own. |
| `tests/test_strong_cnn_experiment.py` | Composition identity (`head(extractor(x)) == model(x)` exactly); eval-mode discipline (BatchNorm train-mode batch-dependence confirmed directly, every StrongCNN-specific function raises if `model.training`); self-consistency parity for `compute_strong_cnn_bounds`; `strong_cnn_head_layer_bound_check` never exceeds its own bound; feature-normalizer floor; end-to-end smoke test. |
| `results/` | Generated outputs (git-ignored except `.gitkeep`): per-`(epsilon, method)` CSVs, width-sweep CSVs, and every plot, for both metrics (SmallCNN `adversarial_*` and StrongCNN `strong_cnn_*` prefixes). |

## Design decisions

- **The Mahalanobis repeat reuses the SAME trained checkpoints and the SAME adversarial examples
  as the Euclidean sections — not independently retrained/reattacked ones.** Training is
  deterministic given the same seed (checked directly in `tests/test_mahalanobis.py`), and
  `fgsm_attack`/`pgd_attack` have no dependency on distance metric at all, so calling the
  `*_with_distance_fn` functions with the same seed reproduces bit-identical models and `x_adv`
  values. Only how sensitivity is *measured* differs between the two halves of the notebook.
- **`L_head_exact` is provably identical between the two metrics.** The head layer maps extracted
  features (always Euclidean) to logits (always Euclidean) — only the input-side pixel distance
  is pluggable. `compute_bounds_with_distance_fn` reuses `layer_decomposition.py`'s own
  feature-standardization helpers (including its underscore-prefixed "private" ones, deliberately,
  to guarantee no drift from that module's own Euclidean computation) rather than re-deriving the
  same logic, and this identity is checked directly in tests, not just asserted.
- **The Mahalanobis precision matrix reuses `mnist_lipschitz`'s own established
  `epsilon=0.01`** (`run_experiment.MAHALANOBIS_EPSILON`) rather than re-running epsilon selection
  from scratch — MNIST's pixel covariance structure doesn't depend on which sub-experiment is
  consuming it, and that choice is already justified in `mnist_lipschitz/README.md`.
- **Width-sweep bound/`R_adv` columns are evaluated at the largest swept epsilon** (`0.25`) — the
  strongest, most informative attack condition for the width-vs-bound-closeness comparison.
- **The CNN-width sweep retrains fresh models per width** rather than reusing
  `layer_decomposition.run_cnn_width_sweep`'s checkpoints, which aren't saved to disk.

## How to run it

```bash
# from the repo root
.venv/bin/python -m pytest mnist_lipschitz/adversarial/tests/ -v

# single-checkpoint baseline (Euclidean)
.venv/bin/python -c "from mnist_lipschitz.adversarial.run_experiment import main; main()"

# CNN-width sweep (Euclidean)
.venv/bin/python -c "from mnist_lipschitz.adversarial.run_experiment import run_cnn_adversarial_width_sweep; run_cnn_adversarial_width_sweep()"

# or execute the notebook end-to-end (~10 minutes on CPU: two baseline trainings,
# two 5-width sweeps, and a 60000x784 SVD for the Mahalanobis precision matrix)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace mnist_lipschitz/adversarial/notebook_adversarial_lipschitz.ipynb
```

## Results

All numbers below are from an actual run (`results/`), the default `SmallCNN`
(`conv_channels=(16, 32)`, matching every other CNN trained elsewhere in this project) trained on
full MNIST (`train_acc=0.9889`, `test_acc=0.9860`), seed 0, `n_points=500`, PGD with 20 steps and
5 random restarts.

### Baseline checkpoint

`L_head_exact=2.870` (identical under both metrics, by construction — see Design decisions).

| Metric | `L_extractor_est` | `L_full_estimated` | `product_bound` | looseness_ratio |
|---|---|---|---|---|
| Euclidean | 31.765 | 5.564 | 91.154 | 16.38x |
| Mahalanobis | 10.466 | 2.854 | 30.033 | 10.52x |

| Metric | Method | mean_R_adv @ eps=0.05 | mean_R_adv @ eps=0.25 | max_R_adv @ eps=0.25 | ratio_to_L_full @ eps=0.25 |
|---|---|---|---|---|---|
| Euclidean | FGSM | 3.065 | 2.907 | 4.010 | 0.721 |
| Euclidean | PGD | 3.756 | 4.449 | 5.660 | **1.017** |
| Mahalanobis | FGSM | 0.471 | 0.446 | 0.641 | 0.225 |
| Mahalanobis | PGD | 0.597 | 0.699 | 0.964 | 0.338 |

**Under Euclidean distance, PGD's `max_R_adv` slightly exceeds `L_full_estimated` at
`epsilon=0.2` and `0.25`** (ratio_to_L_full 1.004 and 1.017) — `summarize_epsilon_sweep`'s sanity
check fires two warnings, as designed: `L_full_estimated` is itself only a `pairwise` estimate
over 200 query points, so a 500-point, 5-restart PGD search occasionally finds a locally sharper
direction than those particular sampled pairs happened to hit. **Under Mahalanobis distance, this
never happens in this run** — the tight bound is never grazed, let alone exceeded, at any epsilon.

**Mahalanobis `mean_R_adv` is roughly 6-7x smaller than Euclidean's** for the same underlying
attacks — pixel-space Mahalanobis distance (ridge-regularized against MNIST's own covariance) is
numerically much larger than Euclidean distance for a typical perturbation, so the same
logit-space movement divided by a larger denominator gives a smaller ratio.

### CNN-width sweep

`L_head_exact` matches exactly between the two metrics at every width (confirming the
metric-independence claim directly, not just in theory):

| width | L_head_exact | L_extractor_est (Eucl.) | L_extractor_est (Maha.) | looseness_ratio (Eucl.) | looseness_ratio (Maha.) |
|---|---|---|---|---|---|
| 4  | 1.590 | 4.478  | 1.668 | 2.33x | 1.79x |
| 8  | 1.336 | 6.600  | 2.447 | 2.70x | 1.91x |
| 16 | 1.090 | 9.100  | 3.432 | 3.10x | 2.17x |
| 32 | 0.992 | 13.013 | 4.900 | 3.43x | 2.55x |
| 64 | 1.005 | 26.681 | 8.642 | 6.13x | 3.89x |

`ratio_to_L_full`/`ratio_to_product_bound` at the largest epsilon (`0.25`), both attacks:

| width | ratio_to_L_full (Eucl., PGD) | ratio_to_L_full (Maha., PGD) | ratio_to_product_bound (Eucl., PGD) | ratio_to_product_bound (Maha., PGD) |
|---|---|---|---|---|
| 4  | 1.223 | 0.446 | 0.524 | 0.250 |
| 8  | 1.202 | 0.401 | 0.445 | 0.210 |
| 16 | 1.209 | 0.394 | 0.389 | 0.181 |
| 32 | 1.018 | 0.358 | 0.297 | 0.140 |
| 64 | 0.920 | 0.329 | 0.150 | 0.085 |

**The Euclidean under-sampling warning fires 17 times across the width sweep's 50
`(width, epsilon, method)` combinations** (concentrated at the smaller widths); **the Mahalanobis
sweep triggers it zero times.** Consistent with the baseline finding: achieved sensitivity sits
comfortably under the Mahalanobis tight bound throughout this run, but occasionally grazes/exceeds
the Euclidean one.

**`ratio_to_L_full` and `ratio_to_product_bound` both decrease with width under both metrics** —
real attacks fall further behind both bounds as the network gets wider — but **the Mahalanobis
curve sits consistently below the Euclidean one at every single width**, for both bounds and both
attack methods (see `results/adversarial_euclidean_vs_mahalanobis_bounds_vs_width.png`). The
practical gap between achieved sensitivity and the theoretical bounds is larger under Mahalanobis
distance than under Euclidean, not smaller, and that ordering never flips across the whole width
range tested.

### Most/least sensitive attacked example, per width

The single largest-`R_adv` example is a PGD hit at the largest swept epsilon (`0.25`) for every
width under Euclidean, and for 4 of 5 widths under Mahalanobis (width=4's Mahalanobis maximum
comes from `epsilon=0.2` instead); the smallest is always an FGSM hit under both metrics, mostly
(4 of 5 widths) at the smallest epsilon (`0.05`), with the two largest widths' minimum shifting to
a larger epsilon under both metrics identically. **The specific example selected sometimes matches
between metrics and sometimes doesn't** — the smallest-`R_adv` example happens to be the exact
same image at 4 of 5 widths, but the largest-`R_adv` example matches at only 2 of 5 (`width=8` and
`width=64`) — Mahalanobis distance reweights which pixel directions count as "close," so which
attacked point ends up most/least sensitive is not fixed across metrics.

The largest-`R_adv` example's fraction of the head layer's own exact Cauchy-Schwarz bound
(`actual_logit_distance / head_bound`) shrinks monotonically with width under Euclidean
(70.6% at width=4 down to 46.3% at width=64) — mirroring the aggregate looseness-vs-capacity
finding at the level of one specific image pair. This quantity is unaffected by which metric
selected the example (it only ever depends on the extracted features and the head weights, both
always Euclidean) — see `results/adversarial_extreme_examples_width*.png` for the actual
clean/adversarial image pairs.

## Status

**Confirmed working:** all 28 tests in `tests/` pass, including the central
`compute_bounds_with_distance_fn`-matches-`layer_decomposition_experiment` parity check and the
retraining-determinism check the Mahalanobis width sweep depends on. The full notebook (Euclidean
baseline + width sweep, Mahalanobis baseline + width sweep, comparison) executes end to end with
zero errors.

**Explicitly out of scope, not attempted:** Carlini-Wagner or other attack methods (FGSM/PGD
only); adversarial training or any other mitigation (this sub-experiment measures existing model
vulnerability only); a Mahalanobis metric over the extractor's feature space (this project's
Mahalanobis distance is only ever defined over raw pixel input or an explicit embedding of it, see
`compute_bounds_with_distance_fn`'s docstring).

**Worth revisiting:**
- **Why Mahalanobis distance widens the practical gap to both bounds, rather than narrowing it,
  is not established.** A plausible starting point: FGSM/PGD select perturbation directions by
  maximizing cross-entropy loss in raw pixel space, with no awareness of the Mahalanobis metric at
  all — so there's no reason to expect they'd land on directions that are also "efficient" moves
  under a metric they never optimized against. This hasn't been checked directly (e.g. by
  designing a Mahalanobis-aware attack objective and seeing whether the gap closes).
- **The specific extreme examples Mahalanobis distance selects were not cross-referenced against
  Euclidean's** beyond noting they can differ — which specific pixel directions the Mahalanobis
  metric down/up-weights for these particular examples hasn't been inspected.
- Only the default CNN width range (`4`-`64`) and epsilon range (`0.05`-`0.25`) were tested; larger
  or non-uniform widths/epsilons might behave differently.

## StrongCNN sub-experiment

`strong_cnn_experiment.py` + `notebook_adversarial_strongcnn.ipynb` repeat this sub-experiment's
baseline comparison (single checkpoint, Euclidean + Mahalanobis, FGSM/PGD, WITH the tight/loose
bound comparison) for `StrongCNN` (`models.py`) — the project's higher-capacity,
near-state-of-the-art baseline — instead of `SmallCNN`. **Deliberately out of scope**: any
CNN-capacity/width sweep (`StrongCNN`'s conv channels are hardcoded, not a constructor parameter,
unlike `SmallCNN`'s) and the multi-seed confirmation sweep (`seed_sweep.py`'s pattern, also
`SmallCNN`-specific).

**Why this needed a separate module rather than reusing `run_experiment.py`/`layer_decomposition.py`
directly**: `StrongCNN` has no `.extractor`/`.head` submodule split — `layer_decomposition.py`'s
own docstring explicitly excludes it ("this model isn't part of" that sub-experiment). But
`StrongCNN.classifier[4]` (the final layer of its classifier) IS a plain `nn.Linear` with nothing
nonlinear after it, so the same tight/loose bound methodology still applies via an
EXTERNALLY-constructed extractor/head split (`strong_cnn_extractor_fn` = `model.features` +
`model.classifier[:4]`, `strong_cnn_head_module` = `model.classifier[4]`) — built in a new module
rather than by editing `models.py` or `layer_decomposition.py`, following the same "reuse, never
edit `layer_decomposition.py`" precedent `compute_bounds_with_distance_fn` already established for
the Mahalanobis generalization. `attacks.py`, `plots.py`, and every architecture-agnostic function
in `run_experiment.py` (`run_epsilon_sweep`, `summarize_epsilon_sweep`,
`most_and_least_sensitive_examples`, ...) are reused UNCHANGED.

**Eval-mode discipline (the one genuinely new correctness concern vs. `SmallCNN`)**: `SmallCNN`
has no `BatchNorm`/`Dropout`, so nothing in this sub-experiment's original machinery ever needed
to care about train/eval mode. `StrongCNN` has `BatchNorm1d`/`Dropout2d`/`Dropout` — every
`strong_cnn_experiment.py` function that forwards data through the model RAISES `ValueError` if
`model.training` is `True`, rather than silently calling `.eval()` itself: a caller who forgot to
set eval mode has a real bug (BatchNorm in train mode uses per-batch statistics, so a
bound/attack computation would silently depend on which other points happen to share a batch)
worth surfacing loudly, not hiding.

**Weaker checkpoint-gating than the `SmallCNN` machinery, stated explicitly**: no independently-
existing `layer_decomposition_experiment`-equivalent exists for `StrongCNN` to check
`compute_strong_cnn_bounds` against (unlike `compute_bounds_with_distance_fn`'s check against
`layer_decomposition_experiment` itself). `tests/test_strong_cnn_experiment.py`'s parity test is
therefore a SELF-consistency check (does `compute_strong_cnn_bounds` reduce to a from-scratch
manual `pairwise_lipschitz` computation using the same closures), not a check against an
independently-validated reference.

### Results (seed=0 run)

Trained via `STRONG_CNN_CONFIG`'s exact recipe (full 60k MNIST, 25 epochs, light
rotation/translation augmentation, cosine-annealed LR — same recipe
`mnist_lipschitz.run_experiment.run_stronger_cnn_raw_mnist_experiment` uses elsewhere in this
project): `train_acc=0.9981`, `test_acc=0.9966`. Total standalone wall-clock time: ~19.7 minutes
on CPU (~18 min training, ~100s for the bound comparison + full FGSM/PGD sweep).

`L_head_exact=2.103` (identical between metrics, by construction — confirmed directly).

| Metric | `L_extractor_est` | `L_full_estimated` | `product_bound` | `looseness_ratio` |
|---|---|---|---|---|
| Euclidean | 4.295 | 6.148 | 9.034 | 1.47x |
| Mahalanobis | 1.901 | 2.948 | 3.998 | 1.36x |

Far tighter than `SmallCNN`'s baseline looseness (16.4x under Euclidean) — `StrongCNN`'s much
deeper extractor (6 conv/BatchNorm layers + a 256-d FC layer, vs. `SmallCNN`'s 2 conv layers)
apparently makes the per-layer submultiplicative bound a much closer approximation to the
network's actual end-to-end sensitivity, at least at this query-sample size.

**A much more severe tight-bound gap than `SmallCNN` ever showed, under Euclidean distance**:
`max_R_adv` exceeds `L_full_estimated` at 9 of the 10 epsilon/method combinations tested
(`ratio_to_L_full` from 0.92x up to 3.73x; only `epsilon=0.25`/FGSM stays under the bound), not
the occasional ~1.00-1.02x graze `SmallCNN`'s baseline showed. Still consistent with
`L_full_estimated` being a valid lower-bound *estimate* from a finite, randomly-sampled
1000-point query set (not a violation of any mathematical guarantee) — but the magnitude
suggests `StrongCNN`'s higher capacity/depth has sharper, more localized sensitivity directions
that only PGD's targeted, iterative search finds, which random natural-image query pairs are
unlikely to land near. Raising `n_query_points` from 200 to 1000 only modestly tightened the
bound (`L_full_estimated`: `5.716 -> 6.148` Euclidean, `2.508 -> 2.948` Mahalanobis), so the gap
looks like an intrinsic property of random-pair sampling here, not a simple undersampling
artifact that more query points would fix.

**Under Mahalanobis distance, the same gap is dramatically smaller** (`ratio_to_L_full` ranges
`0.29x-1.38x`, vs. Euclidean's `0.92x-3.73x` at the same epsilons) — the OPPOSITE of the pattern
found for `SmallCNN`'s width sweep, where Mahalanobis distance WIDENED the gap rather than
narrowing it. Why this effect flips direction between the two architectures is not established.

Full per-epsilon tables, the most/least-sensitive example breakdown, and the discussion above are
in `notebook_adversarial_strongcnn.ipynb`'s own "Findings and observations" section.

### Status

**Confirmed working:** all `tests/test_strong_cnn_experiment.py` tests pass (composition
identity, eval-mode-matters, self-consistency parity, head-bound-check, feature-normalizer floor,
end-to-end smoke). The notebook executes end to end with zero errors.

**Explicitly out of scope, not attempted:** any CNN-capacity/width sweep; a multi-seed
confirmation sweep; Carlini-Wagner or other attack methods; adversarial training or any other
mitigation.

**Worth revisiting:**
- Why the Euclidean-vs-Mahalanobis gap-width effect flips direction between `StrongCNN` and
  `SmallCNN` — the single most interesting open question from this run.
- `n_query_points` was raised from 200 to 1000 for the Lipschitz-bound estimate and only
  modestly tightened `L_full_estimated`, while the achieved-vs-bound gap stayed large —
  consistent with the gap being intrinsic to random-pair sampling missing PGD's sharp
  sensitivity directions, not simple undersampling; not pushed further (e.g. 5000+ points).
- This is a single-seed run — stability across seeds is unknown.
