# Distance-aware Lipschitz estimation: Mahalanobis distance in a polynomial embedding

Extension to Experiment 1, following up on a question raised after the
original README/notebook work: every estimator so far (`pairwise_lipschitz`,
`local_perturbation_lipschitz`, `gradient_norm_estimate`) measures how
"close" two input points are using plain Euclidean distance in raw `x`.
That's a modeling choice, not a law — a fixed-size step in `x` doesn't
have to mean the same thing everywhere in the domain, and the choice of
distance can change how much of the true Lipschitz constant an estimate
actually recovers. This note describes what was added to test that idea,
and what it found.

## The idea

Instead of measuring distance directly in `x`, embed `x` into a richer
feature space first, and measure distance there:

1. **Embed.** Map `x -> (x, x^2, ..., x^degree)`
   (`embeddings.py::polynomial_embedding`). A linear function of this
   embedding is a degree-`degree` polynomial in `x`, so distance measured
   in the embedded space can reflect structure that raw distance in `x`
   can't.
2. **Derive a metric from the data.** Compute the empirical covariance of
   the embedded training points (`empirical_covariance`), invert it to get
   a precision matrix (`precision_from_covariance`), and use that as the
   quadratic form for a Mahalanobis-style distance:
   `d(a,b)^2 = (a-b)^T * Sigma^-1 * (a-b)`.
3. **Plug it into the existing estimators.** `pairwise_lipschitz` and
   `local_perturbation_lipschitz` (`estimators.py`) both accept optional
   `embed_fn`/`precision` arguments. With neither supplied, they behave
   exactly as before (plain Euclidean); with both supplied, distance is
   computed in the embedded space instead. Every pre-existing call site is
   unaffected.

## Result: the metric choice measurably matters

On the Tier A gap-sampled dataset (single ridge, `L* = 6.0` exactly, a
closed-form ground truth), using the same raw data and the same trained
model throughout, only the distance metric changes:

| | global `L_hat_data` | error vs `L*` |
|---|---|---|
| plain Euclidean | 4.87 | 19% |
| Mahalanobis (degree-3 polynomial embedding) | 6.01 | <1% |

That's the headline result (`run_experiment.run_metric_embedding_check`):
switching only the distance metric cuts the error from 19% to under 1%.

This was also extended from a single global number into a **local**
estimate — the same comparison computed as a curve over `x`, so it shows
*where* the plain-Euclidean estimate falls short rather than only that it
does, and lets the global scalar and the local curve be compared side by
side (`run_experiment.run_tier_a_gap_demo`,
`plots.plot_local_vs_global_lipschitz`,
`results/tier_a_local_vs_global_lipschitz.png`).

## Choosing the embedding degree

The polynomial degree is a free parameter that matters more than it looks:
too low, and the embedding has too little structure to capture the effect;
too high, and the embedded covariance becomes ill-conditioned (polynomial
powers are highly collinear on a bounded domain), making the metric
numerically unreliable even though it still inverts.
`run_experiment.sweep_polynomial_degree` checks both accuracy and
conditioning together:

| degree | rel. error vs `L*` | cond(covariance) |
|---|---|---|
| 1 | 137% | 1 |
| 2 | 133% | 6 |
| 3 | **0.08%** | 1.5e3 |
| 4 | 1.2% | 2.0e4 |
| 5 | 44% | 2.0e6 |
| 6 | 45% | 3.4e7 |

Degree 3 is the clear choice — lowest error, and still well within a
reasonable condition number. Degrees 1-2 are too simple to capture the
effect at all; degrees 5-6 are numerically fragile (`results/degree_sweep.png`).

## Why the embedding doesn't include `f(x)`

A natural next idea is to make the metric aware of the function's own
shape by embedding `f(x)` itself alongside the polynomial features —
`embeddings.py::augmented_embedding` implements exactly this. The
intuition is reasonable: a metric that "knows" the function is flat in
some region should treat a disturbance there as more significant than the
same-size disturbance somewhere the function is already changing quickly.

In practice, though, using this embedding to measure the Lipschitz
behavior of the *same* function it was built from is self-cancelling: the
resulting distance ends up scaled by almost exactly the quantity being
measured, so the ratio collapses toward a near-constant value everywhere,
and the effect it was meant to reveal disappears rather than becoming
clearer. This was checked directly rather than assumed — with `f(x)`
included in the embedding, the gap-vs-uniform local-Lipschitz contrast
that plain Euclidean distance shows clearly (roughly 6x higher inside the
undersampled region) dropped to roughly 1x, i.e. the flattening effect
this whole experiment exists to detect became invisible.

`augmented_embedding` is still available in `embeddings.py` for a future
use where the function being embedded isn't the same one being measured,
but no current driver uses it.

## Where this lives

- `embeddings.py` — `polynomial_embedding`, `augmented_embedding`,
  `empirical_covariance`, `precision_from_covariance`.
- `estimators.py` — `pairwise_lipschitz` and `local_perturbation_lipschitz`
  (+ `_grid`) accept the optional `embed_fn`/`precision` pair.
- `run_experiment.py` — `run_metric_embedding_check` (global comparison),
  `run_tier_a_gap_demo` (global + local comparison together, one plot),
  `sweep_polynomial_degree` (degree selection).
- `tests/test_estimators.py` — the Mahalanobis distance is checked against
  a closed-form identity (reduces to scaled Euclidean distance under a
  simple 1D embedding), and the finite-radius local estimator is checked
  to converge to its analytic/pointwise limit as the radius shrinks.

Currently 1D only (`polynomial_embedding` doesn't support `d=2`);
extending this to the Tier B 2D dataset would need a 2D polynomial feature
map.
