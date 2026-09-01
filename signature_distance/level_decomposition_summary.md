# Level-wise decomposition of signature distances

Diagnostic answering one question, additive alongside the existing Phase 3/4
pipeline (no distance/stream/signature behaviour changed): is the within- vs.
cross-digit signal in Method A's and Method B's signature distance carried
entirely by the level-1 terms, or do the higher-order levels (2..depth)
contribute? Same 300-image sample as Phase 4 (30/class, seed 0, depth 4), `r`
derived independently per method exactly as in `run_experiment.sanity_check_demo`
(1.656 for A, 2.860 for B), applied before any level masking. Full method:
`level_decomposition.py::run_level_decomposition`.

## Variant × method table

Ratio is cross-digit mean / within-digit mean; `all` reproduces the
documented Phase 4 table exactly (Gate 2).

### Method A (r ≈ 1.656)

| variant | levels kept | within-digit mean | cross-digit mean | ratio |
|---|---|---|---|---|
| all | 1,2,3,4 | 14.60 | 17.18 | 1.176 |
| level1_only | 1 | 1.50 | 1.91 | 1.276 |
| level2plus | 2,3,4 | 14.50 | 17.05 | 1.175 |
| level2_only | 2 | 4.54 | 5.58 | 1.228 |
| level3_only | 3 | 7.90 | 9.38 | 1.187 |
| level4_only | 4 | 11.18 | 12.98 | 1.162 |

### Method B (r ≈ 2.860)

| variant | levels kept | within-digit mean | cross-digit mean | ratio |
|---|---|---|---|---|
| all | 1,2,3,4 | 28.60 | 33.17 | 1.160 |
| level1_only | 1 | 0.0229 | 0.0229 | 1.000 |
| level2plus | 2,3,4 | 28.60 | 33.17 | 1.160 |
| level2_only | 2 | 4.66 | 5.68 | 1.219 |
| level3_only | 3 | 13.08 | 15.39 | 1.177 |
| level4_only | 4 | 24.99 | 28.81 | 1.153 |

## Per-level squared-distance contribution

Mean fraction of total squared pairwise distance contributed by each level,
averaged over all 44,850 unique pairs in the sample (levels 1..4 sum to
1.0 — level 0 is the constant `1.0` term, identical for every image, and so
contributes exactly zero to any pairwise distance).

| level | Method A fraction | Method B fraction |
|---|---|---|
| 1 | 1.79% | 0.0031% |
| 2 | 11.35% | 2.99% |
| 3 | 28.85% | 21.60% |
| 4 | 58.01% | 75.41% |

## Verdict: (b) — the higher-order signature terms carry genuine information, not level-1

For both methods, `level2plus`'s ratio (A 1.175, B 1.160) is essentially
identical to `all`'s (A 1.176, B 1.160) — dropping level 1 entirely costs
almost nothing. That alone would already point to (b) over (a). But the
per-level fraction table makes the finding sharper than "higher levels
matter": **level 1 is nearly inert for both methods**, not just
non-dominant. For Method A it carries only 1.79% of total squared distance;
for Method B it carries 0.003% — three orders of magnitude smaller than
level 4. The reason is structural for Method B specifically: every reference
line runs the full width/height of the image, so both of its endpoints sit
on the image border, where MNIST pixel intensity is ~0 for nearly every
image (99.96% of the 4,800 per-line net-displacement values in this sample
are *exactly* 0.0, not just small) — Method B's level-1 term, the path's net
displacement, is measuring a quantity that's degenerate by construction, not
one that merely turned out to be weak. `level1_only`'s ratio for Method B is
exactly 1.000 to 3 decimals: no within/cross separation at all.

A second, non-obvious point the fraction table alone would miss: **ratio and
magnitude rank levels in opposite orders**. `level1_only` has the *highest*
individual-level ratio for Method A (1.276, above even `all`'s 1.176)
despite carrying almost none of the distance magnitude (1.79%) — a small,
noisy-looking quantity that happens to separate classes relatively well per
unit of scale. Conversely `level4_only`, which carries the most magnitude
for both methods (58%/75%), has the *lowest* individual-level ratio of any
level (1.162 A, 1.153 B) — bulk signal that's comparatively unfocused
class-wise. Level 2 alone lands closest to `level1_only`'s per-unit
separation-vs-magnitude tradeoff (ratio 1.228 A / 1.219 B on 11%/3% of the
magnitude) without level 1's degeneracy problem.

Net conclusion: the signature machinery is doing real work. Method A's
Phase 4 ratio of 1.176 is not a relabelled version of the single
`Δσ₁ = sigma1(anchor 63) − sigma1(anchor 0)` scalar that motivated this
check — that scalar (`level1_only`) contributes under 2% of the distance's
magnitude and is not what's driving the `all` result; `level2plus` alone
reproduces `all` almost exactly. Method B's case is even more direct: its
level-1 term is close to degenerate by construction (border-anchored lines
against a near-black background), so essentially all of its Phase 4 signal
already had to come from levels 2-4. This is carried forward as the honest
finding, per this project's convention — it doesn't change anything about
Phase 4's already-documented modest effect size relative to the pixel-space
Euclidean baseline, and doesn't by itself motivate a scope change (e.g. to
line placement) beyond what `per_line_auc_summary.md`'s border-line finding
already flagged independently.
