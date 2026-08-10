# New distance measure: Mahalanobis-in-polynomial-embedding

This documents the second round of changes to `toy_lipschitz/`, made in
response to Terry's feedback (meeting 2) and a separate discussion about
what that feedback actually implied for the implementation. It's additive
to the original Experiment 1 work described in `README.md` — nothing
there was changed or invalidated, this only adds a new metric option on
top of it.

## Motivation (Terry's feedback, points 1–6)

Every estimator up to this point — `pairwise_lipschitz`,
`local_perturbation_lipschitz`, `gradient_norm_estimate` — measures
distance between points as plain Euclidean distance in raw `x`. Terry's
objection: raw Euclidean distance treats "close in `x`" as the only
notion of closeness that matters, but a clump of points near one region
of `x` doesn't necessarily behave like a clump of the same size elsewhere
(points 1–2). If you instead embed `x` into a richer feature space — e.g.
`(x, x^2, ..., x^degree)` — and measure distance there, a linear function
of the embedding is a degree-`degree` polynomial in `x`, and "distance"
adapts to how the function actually varies rather than assuming it's
uniform. Concretely: derive a Mahalanobis-style distance from the
empirical covariance of the embedded training points — this is the
"L2 norm in the dual space, pulled back" framing (points 3–4). Point 6
went further: the embedding should also include `f(x)` itself, since
that's already computed and available. Point 5, separately, was about a
missing diagnostic: a low local-Lipschitz estimate in some region only
means "no stretch found here," not "no stretch exists" — it could just
mean that region was barely sampled, and nothing in the codebase
distinguished those two cases.

## What was added

### 1. New file: `embeddings.py`

Four functions:

- **`polynomial_embedding(x, degree)`** — maps `x` (shape `(N,)` or
  `(N,1)`) to `(x, x^2, ..., x^degree)`, shape `(N, degree)`. 1D input
  only. This is the direct answer to points 1–2.
- **`augmented_embedding(x, degree, f_vals)`** — `polynomial_embedding`
  with a precomputed `f(x)` column appended. This is the direct answer to
  point 6. **Not wired into any driver** — see [Status: item 4](#status-item-4-augmented_embedding)
  below.
- **`empirical_covariance(z, eps=1e-8)`** — covariance of the embedded
  points, with a small `eps * I` ridge so it's always invertible even
  when `N` is small relative to the embedding dimension.
- **`precision_from_covariance(cov)`** — `Sigma^{-1}` via
  `torch.linalg.inv`, the matrix that defines the quadratic form
  `d(a,b)^2 = (a-b)^T Sigma^{-1} (a-b)`. This is points 3–4.

### 2. Changes to `estimators.py`

- Added `_mahalanobis_dist(diff, precision)` — computes
  `sqrt(diff^T precision diff)` via `torch.einsum`, clamped at 0 for
  numerical safety.
- `pairwise_lipschitz` gained two new optional keyword arguments,
  `embed_fn` and `precision`, both defaulting to `None`. When neither is
  supplied, behavior is byte-for-byte identical to before (plain
  Euclidean/L1 in raw `x`) — every existing call site (Tier A/B sanity
  checks, sweeps, 2D extension) is unaffected. When both are supplied,
  the function embeds `x` via `embed_fn`, then measures pairwise distance
  in the embedded space via the Mahalanobis form; the `norm` argument is
  ignored in that case.
- Added `local_sample_density(x_query, x_train, radius, norm="l2")` at
  the end of the file — for each query point, counts how many training
  points fall within `radius`. This is the answer to point 5: a coverage
  diagnostic, deliberately *not* a Lipschitz quantity itself, meant to be
  plotted alongside (never merged into) the existing Lipschitz heatmaps
  so "tested and found smooth" stays visually distinct from "never really
  tested."

### 3. Changes to `plots.py`

Added `plot_coverage_heatmap(xx, yy, density_grid, train_points,
save_path=None)`, placed right after `plot_2d_heatmaps`. Same visual
grammar (pcolormesh + white-outlined training-point scatter) but its own
figure, its own colormap (`magma` instead of `viridis`), and its own
title — reinforcing that this is a different kind of quantity, not a
fourth panel bolted onto the existing 2D heatmap trio.

### 4. New driver in `run_experiment.py`

Added `run_metric_embedding_check(N=400, w=(4.0,), b=0.5, A=1.5, degree=3,
gap_radius=0.5, gap_fraction=0.02, seed=SEED, verbose=True)`:

- Reuses the exact same Tier A gap-sampled dataset construction as
  `run_tier_a_gap_demo` (single ridge, closed-form `L*`).
- Computes `L_hat_data` twice on that dataset: once with plain Euclidean
  `pairwise_lipschitz(x_train, y_train, norm="l2")` (unchanged call
  signature), once with `embed_fn=lambda xx: polynomial_embedding(xx,
  degree=degree)` and `precision` built from the empirical covariance of
  the embedded training points.
- Reports `L_star`, `L_hat_plain` + its argmax pair, `L_hat_mahalanobis`
  + its argmax pair.

Wired into `main()` as the last call, after `run_cross_architecture_check()`.

## Result and interpretation

Running `run_metric_embedding_check()` with the defaults:

```
L_star:                 6.0
L_hat_plain:             4.87
argmax_pair_plain:      (0.1009, -0.3236)
L_hat_mahalanobis:       6.01
argmax_pair_mahalanobis: (0.1009, -0.3236)
```

Two things worth noting:

- **The Mahalanobis estimate is much closer to `L*`.** Plain Euclidean
  distance on this gap-sampled dataset underestimates the true constant
  by about 19% (4.87 vs. 6.0); the polynomial-embedding Mahalanobis
  distance recovers it almost exactly (6.01 vs. 6.0, <1% error). That's
  a real, checkable signal that the choice of metric isn't cosmetic — it
  materially changes how well `L_hat_data` tracks `L*` on the same
  underlying data.
- **The argmax pair happened to be identical in this run** — both
  metrics flag the same two training points as the steepest pair. This
  wasn't the outcome originally expected (the working assumption was that
  changing the metric would also change *which* pair gets flagged, which
  would have been an even more visually direct way to show the effect to
  Terry). It didn't happen here because, at Tier A with a single ridge,
  the steepest region of `f*` is narrow and well-defined, so both metrics
  end up agreeing about where it is even though they disagree about *how
  steep* it is there. The magnitude gap (4.87 vs. 6.01) is still the
  concrete, presentable evidence — just via `L_hat` value rather than
  argmax location. Worth trying `gap_fraction`/`gap_radius` variations or
  the Tier B (multi-ridge) dataset if a differing-argmax example is
  wanted for the write-up.

## Status: item 4 (`augmented_embedding`)

`augmented_embedding` (points 1–2 embedding with `f(x)` appended, per
point 6) is implemented and unit-testable in isolation but **not wired
into `run_metric_embedding_check` or any other driver**. This was
intentional — appending the model's own output to the embedding used to
*measure* that model's Lipschitz constant introduces a subtlety (does
`f(x)` mean the trained model's predictions, the ground-truth `f*`, or
something recomputed per-candidate-model during a sweep?) that wasn't
resolved by the description of point 6 alone. This is the one item
flagged as worth confirming with Terry before building out further, per
the earlier discussion.

## What's still not done

- `augmented_embedding` is not wired into any driver (see above).
- `local_sample_density` / `plot_coverage_heatmap` are implemented but
  not yet called from `run_2d_extension` or any other driver — the
  natural next step is computing a density grid over the same `[-5,5]^2`
  domain as the existing 2D heatmaps and calling
  `plot_coverage_heatmap` alongside `plot_2d_heatmaps` in
  `run_2d_extension`, using the same `x_train` already computed there.
- No unit tests were added for `embeddings.py` or the new
  `local_sample_density` / Mahalanobis path in `estimators.py` — the
  existing `tests/test_estimators.py` and
  `tests/test_tier_a_closed_form.py` are unaffected and still pass, but
  nothing new is covered by the test suite yet.
- 2D polynomial embedding: `polynomial_embedding` is 1D-only by design
  (`x = _as_tensor(x).reshape(-1)`); extending the Mahalanobis check to
  the Tier B 2D dataset would need a 2D-embedding variant (e.g. degree-2
  polynomial features in two variables) that doesn't exist yet.
