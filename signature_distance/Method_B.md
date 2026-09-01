# Method B: Reference-Line Signature Distance

## Overview

Method B represents an MNIST image as a fixed set of straight reference
lines through the 28x28 grid. Each line is sampled at a fixed number of
points via bilinear interpolation into a `[t, intensity]` stream, and each
stream gets its own truncated path signature. The signatures are used to
build a distance between two images, which is tested as the denominator in
the project's Lipschitz-ratio pipeline (`|margin_i - margin_j| / distance`),
compared against plain pixel-space Euclidean distance under adversarial
perturbation.

The design went through one major pivot, driven by evidence rather than
assumption at every step: the distance was first built by merging all
lines' signatures into a single vector before comparing images, then
rebuilt to keep each line's comparison separate and examine the resulting
collection of per-line distances instead. The per-path version produced
the strongest, most reproducible result in the project so far, and a
subsequent hyperparameter sweep improved on it further. This document
covers the full arc: design, what was built, what was measured, and where
it landed.

## Design

**Stream construction** (`streams.py`, `make_reference_lines` +
`line_stream`). Lines are placed by `angles_deg`/`counts` pairs — angle 0
gives horizontal lines (rows evenly spaced across image height, each
sampled left to right), angle 90 gives vertical lines (columns evenly
spaced, sampled top to bottom). The original default was 8 horizontal + 8
vertical lines, 32 points per line; both the orientation split and the
point count are parameters of one function, not separate code paths, so
any split (all-horizontal, all-vertical, uneven) is just a different call.
No clipping is needed in either orientation — every sample point is
in-bounds by construction, since lines run edge to edge along the axis
they're aligned with. Intensity is read via batched bilinear interpolation
(`grid_sample`), not a Python loop over images.

**Signature computation** (`signatures.py`, shared with the project's other
stream-based method). Each line's stream is treated as a piecewise-linear
path — straight-line interpolation between consecutive sampled points —
and its truncated signature is computed via low-level tensor/Lie-algebra
primitives (`Lie`, `cbh`, `to_signature`) rather than higher-level stream
wrapper objects, batched with `jax.vmap`. Correctness was checked against
two closed-form identities before being used downstream: a straight-line
path's signature matches the exact analytic tensor exponential at every
truncation level, and an L-shaped (two-segment, non-collinear) path
matches a hand-computed nonzero area term — the second check specifically
exercises whether multi-segment combination is correct, not just the
trivial straight-line case.

**No cross-line merging before the signature step.** Each line's signature
is computed independently over its own points only; if lines are combined
at all, it happens after signatures are computed; the raw streams
themselves are never concatenated.

## Distance function and sanity check

The first distance built from these signatures rescaled each line's
signature level-`n` by `r**n` (raw coefficients decay roughly `1/n!` with
depth, so without rescaling a Euclidean distance mostly measures the
depth-1 terms), then concatenated all 16 rescaled per-line signatures into
one 496-dimensional feature vector per image, and took a single Euclidean
distance between two such vectors. `r` was derived empirically per
sample (geometric mean of the level-to-level magnitude ratio, inverted, so
that `r**n` roughly flattens the decay), landing at r ~ 2.86.

A within-digit-vs-cross-digit check on a 300-image sample (30 per class)
found same-label pairs closer than different-label pairs on average
(cross/within ratio ~1.16 after rescaling), so the distance was behaving
sensibly rather than randomly — but the effect size was modest, in the
same range as plain raw-pixel Euclidean distance's own weak ratio
(~1.13) on the same kind of check elsewhere in the project. Rescaling
barely moved the ratio (1.19 to 1.16), so rescaling itself wasn't the
bottleneck. This result was ambiguous enough that it would ordinarily
warrant pausing before building further on it, but work proceeded to the
actual target application (adversarial evaluation) on the reasoning that
hyperparameter tuning was always going to follow regardless of how this
first, unweighted attempt looked.

## Adversarial evaluation, round 1: merged distance

The merged 496-dimensional distance was tested directly as a Lipschitz-
ratio denominator: FGSM perturbations at three epsilons (0.02, 0.03, 0.05)
against two freshly trained classifiers (a small CNN, ~98.2% test
accuracy, and a stronger CNN with batch normalization and dropout, ~99.4%
test accuracy), on a 200-image sample, compared against plain pixel-space
Euclidean distance and against a magnitude-matched random-noise control
(same L2 pixel-space norm as the FGSM perturbation, but not
gradient-directed).

Both distances separated genuinely adversarial perturbations from the
random-noise control clearly (near-bimodal ratio distributions, minimal
overlap) — the Method B distance was not simply noise. But its
adversarial-vs-control separation ratio was consistently a little behind
plain pixel-Euclidean's, on both models, at every epsilon (roughly
7.0-7.4x vs. pixel distance's 8.5-9.1x on the smaller model; 3.3-4.6x vs.
3.6-5.0x on the larger one). Both distances also registered the expected
capacity difference between the two models. The highest-ratio pairs under
the merged distance were mostly cases where the prediction hadn't actually
flipped (large margin swing, still correctly classified) rather than
misclassifications.

Two independent checks — the sanity check above and this evaluation — now
pointed the same direction: the merged distance behaved sensibly but
hadn't yet beaten the pixel-space baseline it was meant to improve on.

## Diagnostic: is concatenation diluting a real per-line signal?

The concept for this method had originally been described in terms of
comparing images line by line — asking whether two images are close *on
each individual path* — rather than merging all paths into one score
first. The pipeline as built did the opposite. A same/different-digit AUC
diagnostic tested this directly: computed per individual line and
compared against the merged distance, on the same 300-image sample, all
44,850 unique pairs.

One line narrowly beat the merged distance (AUC 0.6394 vs. 0.6315) — a
real but small margin, and 13 of the 16 individual lines scored *below*
the merged distance, so this wasn't broad evidence that concatenation was
drowning out most lines' signal. It was, however, clear evidence of one
specific inefficiency: 4 of the 16 lines — the ones running exactly along
the image border (row or column 0 or 27) — sat at exactly AUC 0.5000,
chance level, carrying zero signal. MNIST digits essentially never reach
the image edge, so these lines run through background regardless of the
image and contribute pure noise to the merged vector. Horizontal lines
dominated the top of the ranking generally.

## The pivot: per-path evaluation

Rather than merging the 16 lines into any single score — not the 496-dim
concatenation, not a max or weighted sum — each line was instead treated
as an independent coordinate, producing its own local Lipschitz ratio
(`margin_difference / line_distance`), examined as a collection of 16
numbers per image pair. This mirrors how the project's earlier,
pixel-space Lipschitz work treats individual pixels: as separate
coordinates, never reduced to one number before being examined.

Primary reporting used the 12 informative lines (excluding the 4
chance-level border lines identified above). Same 200-image sample, same
two models, same three epsilons and random-noise control as the merged
evaluation.

**Result 1.** For every one of the 12 informative lines, in every one of
the 6 model x epsilon combinations (72 comparisons total, no exceptions),
the mean ratio on genuinely adversarial (prediction-flipping) pairs
exceeded the mean ratio on magnitude-matched random-control pairs from the
same images — fold-ratios of roughly 4-11x on average across lines, up to
~20x for the single best line. This is a substantially clearer separation
than the merged distance's own 7.0-7.4x / 3.3-4.6x. The number of
genuinely flipped pairs was small at these conservative epsilons (4-14 out
of 200), so exact magnitudes should be read as indicative rather than
precise — but the direction, with zero exceptions across all 72
comparisons, was the trustworthy part.

**Result 2.** Looking at which line shows the largest ratio for each pair,
the distribution of "which line spikes" was more concentrated (lower
entropy) under adversarial perturbation than under the random control, in
all 6 of 6 model/epsilon combinations — modest (roughly 0.2-0.3 bits out
of a 3.585-bit maximum) but completely consistent. This is evidence,
though not dramatic, that an adversarial perturbation concentrates on
fewer paths than an equally large random shift, even though it touches
every pixel.

**A confound, checked rather than assumed away.** Two lines dominated the
raw "which line spikes" counts under *both* adversarial and control
conditions, because they had systematically smaller baseline signature
distances than the other ten lines, regardless of perturbation type — a
smaller denominator mechanically inflates a ratio independent of whether
anything adversarially meaningful happened. This meant raw spike counts
were not a clean read on their own; the entropy comparison above, being a
within-condition comparison, was the more trustworthy summary.

**Visual check on two example pairs** (original image, perturbed image,
and the single highest-ratio line highlighted) gave a mixed picture on
whether the spiking line visually crosses the digit's stroke: one example
did, one mostly ran through background near the stroke. Two examples were
not enough to establish a clean pattern either way.

## Robustness check on the per-path finding

Because two lines had already been flagged as a possible scale confound,
the fold-ratio finding above (adversarial ratio exceeding control ratio on
every line, every model, every epsilon) was checked directly against
their exclusion, using the same already-computed evaluation data. Removing
the two lines left 60 of the original 72 comparisons; all 60 still showed
the same pattern, with zero exceptions. The mean fold-ratio moved by at
most about 0.5x on the 4-11x scale, in no consistent direction — the two
flagged lines were not doing meaningful work for the aggregate result. A
Pearson correlation between each remaining line's own baseline distance
and its fold-ratio was small and inconsistent in sign across models and
epsilons (|r| <= 0.32), unlike the two excluded lines' much cleaner
pattern (roughly half the distance of the rest, dominating spike counts
under both conditions) — the scale confound looked fairly well isolated to
those two specific lines, not a systemic property of the whole set.

## Hyperparameter sweep

With the per-path approach established as the stronger design, its own
hyperparameters were swept: line geometry (orientation split), points per
line, truncation depth, and interpolation method (straight-line joins
between sample points vs. a cubic spline fit through them).

A cheap screen (no model training, using the same per-path same/
different-digit AUC diagnostic) covered the joint grid: 4 geometry
variants x 4 point counts x 5 depths x 2 interpolation methods, 160 scored
configurations. One efficiency shortcut was verified directly before being
relied on: a depth-D truncated signature's lower-level coefficients are an
exact, bit-identical prefix of a higher-depth computation, so each
(geometry, points, interpolation) stream only needed the expensive
signature step computed once, at the maximum depth swept, with every
lower depth sliced from that single result.

**Depth turned out to be the dominant lever, and the original default (4)
was not the best choice** — mean best-line AUC decreased monotonically
from depth 2 (0.6585) to depth 6 (0.6320), a consistent trend rather than
noise. **Geometry mattered too**: shifting weight toward horizontal lines
helped (12 horizontal + 4 vertical: 0.6592 mean, vs. the original 8+8
split's 0.6492), and an all-vertical split was clearly worst (0.6219).
**Points per line and interpolation method barely mattered**, plateauing
by 32 points and showing no meaningful difference between straight-line
and cubic-spline joins. The single best configuration found — 12
horizontal + 4 vertical lines at depth 2 — reached AUC 0.6811 against the
original configuration's 0.6469, and this wasn't a single fortunate row:
the entire top cluster of the ranking used this geometry-and-depth
combination across different point counts and interpolation choices.

The top candidates from this screen were then validated with the full
per-path adversarial evaluation (both models trained once and reused
across every candidate, since training doesn't depend on Method B's
configuration). The winning configuration (12 horizontal + 4 vertical
lines, depth 2) reached a mean fold-ratio of roughly 14.8-15.0x against
the original configuration's 8.4x — a substantial improvement, and one
that agreed with the independent cheap screen rather than contradicting
it. It was not perfectly exception-free at the individual-line level (one
of 72 line/model/epsilon combinations showed the reverse direction,
isolated to the larger model's smallest-sample condition, consistent with
sample-size noise rather than a new pattern), and a second candidate
(all-horizontal-plus-a-few-vertical variants aside, specifically the
16-horizontal/0-vertical split at depth 2) was competitive rather than a
distant runner-up — it actually outperformed the winner on the larger
model specifically while losing on the smaller one, a genuinely mixed
result on that comparison. A robustness check on the winning configuration
(same method as above) found no lines behaving like the earlier scale
confound; baseline distances across its 12 informative lines spanned a
narrow range (about 1.4x top to bottom, compared to the earlier ~2x gap),
and excluding its two smallest-distance lines moved the mean fold-ratio
from 14.8x to 12.6x while the pattern remained intact.

Points-per-line showed the weakest effect of any axis in this sweep, which
is the main signal it offers for choosing a comparable parameter — a
segment or step length — in any related construction built afterward: no
axis here supports a case for a long segment length, and geometry- and
depth-style choices look like the higher-leverage places to spend tuning
effort by comparison.

## Current status

**12 horizontal + 4 vertical lines at truncation depth 2 is the adopted
Method B default**, replacing the original 8+8 split at depth 4. This was
confirmed by two independent evaluations — a cheap AUC-based screen and
the full per-path adversarial evaluation — and checked for robustness
against depending on a small number of particular lines.

Method B behaves sensibly as a Lipschitz-ratio denominator, is sensitive
to model capacity, and — once compared path by path rather than merged —
shows a clear, reproducible, and now-improved separation between
genuinely adversarial perturbations and equally large random ones. The
overall picture across every stage was one of honest, incremental
evidence rather than a single clean result: the first (merged) design was
close to but behind a plain pixel-distance baseline; identifying and
removing the merging step produced the strongest result in the project;
a subsequent parameter sweep improved on that further while surfacing a
concrete, fixable inefficiency (uninformative border-line placement) and
two genuine open questions (a mixed second-best configuration, and no
firm conclusion yet on whether spikes reliably land on stroke-relevant
image structure).
