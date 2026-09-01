# Method C: Hilbert-Curve Signature Distance

## Overview

Method C replaces Method B's 16 straight reference lines with a single
space-filling curve through the image: an order-5 Hilbert curve (a
32x32-cell grid, 1024 cells), scaled into the 28x28 image domain and
sampled at 512 points evenly spaced along its arc length, then cut into
16 contiguous segments of 32 points each. This matches Method B's current
default total point budget exactly (16 lines x 32 points), so a
comparison between the two methods isn't confounded by a differing sample
size. Each segment gets its own signature and its own local Lipschitz
ratio, following the same per-path (never-merged) evaluation approach
validated for Method B: the 16 segments are examined as a collection,
never combined into a single vector before comparison.

The question this method answers is direct: does a single continuous,
space-filling traversal of the image do better, worse, or about the same
as 16 independent straight scans, once both are evaluated the same way?

## Construction

**Curve generation.** The standard index-to-coordinate Hilbert-curve
algorithm produces, for order 5, a sequence of 1024 grid cells covering
every cell of a 32x32 grid exactly once. This was checked directly before
anything was built on top of it: every cell is visited exactly once, every
coordinate stays in bounds, and — the property that matters most for what
follows — every consecutive pair of cells is exactly one grid unit apart,
always axis-aligned (never diagonal).

**Scaling and resampling.** Grid coordinates are scaled by `28/32` into
the image domain, then the full 1024-point curve is resampled to exactly
512 points evenly spaced along its arc length (cumulative-length
parameterization plus linear interpolation between the original vertices).
This is a genuine arc-length resampling, not a shortcut: an earlier
assumption that it would reduce to simple index subsampling (since
1024/512 = 2 and every original step is the same length) turned out to be
wrong, and a test caught it — arc-length-even sampling of a *bending*
path is not the same as evenly-spaced-by-index sampling once the path
changes direction between two consecutive output points, even when every
underlying step is the same length. The resampling logic was pulled out
and re-verified in isolation on a simple, hand-computable L-shaped path
before trusting it on the actual curve.

One consequence of the scaling worth stating plainly rather than glossing
over: the curve's farthest point lands at coordinate 27.125, just past the
last valid pixel index (27), not strictly within it as a tidier-sounding
description might suggest. This was caught by a test that checked the
actual bound rather than an assumed one. In practice this is harmless —
the same bilinear sampling technique Method B's lines use reads intensity
with border-clamping, so this one slightly-out-of-range point just reads
the border pixel rather than erroring or extrapolating.

**Segments.** The 512-point sequence is cut into 16 contiguous blocks of
32 points, each with its own `[t, intensity]` time coordinate (t running
0 to 1 within that segment, the same convention used throughout this
project). The curve is entirely fixed and deterministic — unlike Method
B's lines, there isn't even a reserved-but-unused seed parameter, since
nothing about this construction is random. No cross-segment
concatenation: each segment's signature is computed independently, and
segments are combined only by keeping their 16 ratios separate, never by
merging into one vector.

Everything downstream of stream construction reuses Method B's existing,
unmodified infrastructure: the same signature computation, the same
level-rescaling, the same per-line/per-segment distance function, the
same adversarial evaluation machinery (model training, FGSM, the
magnitude-matched random control), and the same same/different-digit AUC
diagnostic — applied to Hilbert segments instead of reference lines.

## Stage A: depth mini-sweep

Method B's sweep found depth 2 clearly and monotonically best, by a wide
margin, for straight reference lines. There was no reason to assume that
transfers to a contiguous space-filling traversal, which has different
local statistics, so depth was swept separately for Method C rather than
carried over.

| depth | mean AUC | best-segment AUC |
|---|---|---|
| 2 | 0.5654 | 0.6485 |
| **3** | **0.5655** | **0.6537** |
| 4 | 0.5633 | 0.6475 |

Depth 3 wins, but narrowly — about 0.008 higher best-segment AUC than
depth 2, essentially tied on mean AUC, and depth 4 trails both by a
similarly small margin. This is a genuinely different pattern from Method
B's sweep: there, depth 2 beat depth 6 by roughly 0.026 AUC in a clean,
monotonic trend; here, the three depths tested are close together with no
strong trend in either direction. Depth 3 was carried forward as Method
C's setting for the full validation, but the margin over depth 2 is small
enough that it shouldn't be read as a strong, confidently-established
optimum the way Method B's depth-2 result was.

A secondary observation from this screen: all 16 segments carried
above-chance signal at every depth tested (none at the exact AUC-0.5000
chance level the way 4 of Method B's 16 lines did). This makes structural
sense — a space-filling curve doesn't have an equivalent to a straight
line placed exactly on the image border; every segment covers a
genuinely different, non-degenerate arc of the image regardless of where
it falls in the traversal order.

## Stage B: full validation and head-to-head comparison

Same evaluation framework as Method B's per-path result: two freshly
trained classifiers (SmallCNN, ~98.2% test accuracy; StrongCNN, ~99.4%),
FGSM at three epsilons, a magnitude-matched random-noise control, 200
images, mean ratio on genuinely flipped pairs vs. the same images' control
ratio, at depth 3.

| | mean fold-ratio (all 16) | exception-free? |
|---|---|---|
| Method B winner (12h+4v, depth 2) | **13.53x** | 95/96 (1 exception) |
| **Method C (Hilbert, depth 3)** | 9.88x | **96/96 (0 exceptions)** |

Both numbers above use all 16 lines/segments on each side (not a
subset), on the same 200-image sample, same models, same epsilons — a
direct, fair comparison.

**Method C does not beat Method B on the primary magnitude measure.**
Method B's winning configuration separates genuinely adversarial pairs
from random-noise controls by roughly 13.5x on average; Method C reaches
roughly 9.9x — meaningfully lower, not a close call. Per epsilon and
model, Method C's fold-ratios (13.4x, 11.5x, 11.9x for SmallCNN; 12.7x,
5.2x, 4.6x for StrongCNN) follow the same general shape as Method B's
own numbers (larger on the smaller model, smaller at higher epsilon on
the larger model) but sit lower throughout — this is a consistent gap
across every model/epsilon combination, not a mixed result that favors
Method C anywhere.

**Where Method C does edge ahead**: it was completely exception-free
across all 96 line/segment x model x epsilon combinations, where Method
B's winner had one (on StrongCNN's smallest-sample condition, already
understood as sample-size noise rather than a real reversal). This is a
minor point in Method C's favor, not a substantial one — one exception
out of 72 wasn't treated as concerning for Method B either.

**Robustness check** (same approach as Method B's): the two segments with
the smallest baseline signature distance were identified and excluded;
the mean fold-ratio moved by less than 1x in every case (e.g. 13.42x to
12.89x, 5.21x to 5.09x) and the pattern held throughout — Method C's
result isn't resting on a small number of unusually sensitive segments
either.

## Verdict

Method C works — it produces a coherent, deterministic, per-path
Lipschitz-ratio evaluation with a completely exception-free
adversarial-vs-control separation, using the same infrastructure and
methodology validated for Method B. But on the metric that actually
matters for this project (how strongly the distance separates genuinely
adversarial perturbations from equally-sized random ones), it trails
Method B's current winner by a consistent, non-trivial margin across
every model and epsilon tested, not just on average. The honest
conclusion is that Method C should be kept as a documented alternative —
the construction is sound, correctly verified, and a legitimate
comparison point — rather than adopted as a replacement for Method B's
reference-line approach. Method B's straight, independently-placed lines
currently do a better job at the actual task than one continuous curve
does, at the same total sampling budget.

## Files

- `hilbert_stream.py` — curve generation, arc-length resampling, segment
  construction, Stage A depth screen, Stage B evaluation driver, and the
  robustness check, all in one module.
- `tests/test_hilbert_stream.py` — 17 tests, including the corrected
  arc-length resampling check (isolated on a hand-computable synthetic
  path) and the same depth-prefix-shortcut correctness check used for
  Method B's sweep.
- `artifacts/hilbert_curve_seed0.npy` — the fixed 512-point curve.
