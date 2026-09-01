# signature_distance

Builds candidate **image distance metrics from path signatures** on MNIST, for
eventual comparison against the Euclidean and Mahalanobis pixel-space distances
already used in `mnist_lipschitz`'s Lipschitz/adversarial work. Every Lipschitz
estimate in that experiment reduces to the same ratio,
`|margin_i - margin_j| / distance(x_i, x_j)` — every denominator tried there
(Euclidean, ridge-regularized Mahalanobis, a UMAP embedding) has been a distance
over the *raw pixel grid*. This package asks whether representing an image as a
handful of low-dimensional paths, then measuring distance between their
**truncated path signatures**, gives a better-behaved denominator.

Two independent, fixed (non-learned) constructions turn an image into one or
more paths — deliberately not derived from a trained classifier's own features,
to stay clear of the self-referential-metric problem (measuring a function's
sensitivity with a metric built from that same function collapses the signal —
documented in `toy_lipschitz/README.md`'s `augmented_embedding` finding):

- **Method A — patch singular-value stream**: visit a fixed, shared set of 64
  interior pixel locations; at each, take the largest singular value of the
  surrounding 3×3 patch. One stream per image, shape `(64, 2)`.
- **Method B — reference-line stream**: 8 horizontal + 8 vertical fixed lines
  through the image, each sampled at 32 points via bilinear interpolation.
  16 streams per image, shape `(16, 32, 2)`.

The full staged plan (Phases 0-9) lives in `PLAN.md`; this README summarizes
what's actually been built and found so far. **Read `PLAN.md` before making
non-trivial changes** — it records design decisions that must not change
without flagging (fixed pixel/line geometry, no cross-line concatenation
before the signature step, SVD-based patch statistic, etc.).

## How it's designed

**Evaluation pool (`data_pool.py`).** `load_eval_pool(n_per_class, seed)`
mirrors `mnist_lipschitz`'s pool-based protocol: a deterministic,
class-balanced subset of the MNIST test set (shuffle indices with `seed`,
take the first `n_per_class` per class). Default pool used throughout is 1000
images (`n_per_class=100`).

**Streams (`streams.py`).**
- `time_channel(n)` — the shared `arange(n)/(n-1)` time coordinate used by
  both methods, so every stream is time-augmented identically regardless of
  which method built it (a signature of a scalar path degenerates to a
  function of the net increment alone without a time coordinate).
- Method A: `make_pixel_order(k=64, seed=0)` samples `k` interior
  `(row, col)` locations without replacement (fixed order, one call, reused
  for every image); `patch_sv_stream` batches all 3×3 patch extraction +
  `torch.linalg.svdvals` (no Python loop over images), returning
  `(N, K, 2)` columns `[t, sigma1]` (`mode="top1"`, default) or `(N, K, 4)`
  `[t, s1, s2, s3]` (`mode="all3"`).
- Method B: `make_reference_lines(angles_deg=(0,90), counts=(8,8), ...)` is
  one parameterized function for both orientations (not separate code paths)
  — only 0°/90° are supported (in-bounds by construction, no clipping logic);
  `line_stream` reads intensity via batched `grid_sample` bilinear
  interpolation, returning `(N, 16, 32, 2)`. Lines are never concatenated
  into one raw stream before the signature step (Method_B.md's "no
  cross-line concatenation" rule) — each stays a separate `(32, 2)` path.
- `row_stream` is a **superseded draft** of Method B (plain row/column
  vectors, no time augmentation) — kept in the file, not part of the active
  plan; `make_reference_lines`/`line_stream` replaced it after a naming
  mix-up between two independently drafted plans was reconciled (see
  `PLAN.md`'s Checkpoint 3 note).

**Signatures (`signatures.py`).** `signature_of_stream(stream, depth)` computes
the truncated signature of a batch of 2D piecewise-linear paths (shared by
both methods — same function, so Method A and B signatures are directly
comparable). Built on `roughpy_jax`'s low-level primitives (`Lie`, `cbh`,
`to_signature`) rather than its `Stream` wrapper classes, specifically because
those primitives are `custom_vjp`-registered JAX functions that `jax.vmap`
batches cleanly. Each piecewise-linear segment's log-signature is exactly its
displacement vector (a straight line has no higher-order/area terms), so `cbh`
(Campbell-Baker-Hausdorff) combining the segments in sequence gives the
log-signature of the whole path, matching Chen's identity. Output dimension
for `width=2` is `1 + 2 + 4 + ... + 2**depth` (31 at the project's default
`depth=4`); index 0 is always the constant term `1.0`, indices `1:3` are
always the path's exact net displacement (`stream[-1] - stream[0]`) regardless
of depth or intermediate path shape.

**Distance and rescaling (`distances.py`).** Signature coefficients decay
~1/n! with depth (verified: level-4 magnitudes ~4-50x smaller than level-1),
so raw Euclidean distance on a signature vector mostly measures the depth-1
term. `rescale_signature(sig, r, depth)` scales each level-`n` block by `r**n`
before any distance computation; `choose_rescale_factor` derives `r` per
batch as the inverse geometric mean of the level-2..depth magnitude ratio.
**`r` is always derived independently per method, never shared** — Method A
and Method B are two separate distance functions throughout this package,
never merged into one metric (mirrors the root project's "the three Lipschitz
sub-methods are never merged" convention). Per-image feature vectors:
`method_a_feature_vector` is the identity (the rescaled signature already
is the feature vector); `method_b_feature_vector` concatenates the 16
per-line rescaled signatures into one 496-dim vector — **the first point the
16 lines combine**, never before. `per_line_distances` stops one step short
of that concatenation, comparing line `i` to line `i` directly ("path by
path") instead of merging first — used by the per-path diagnostics below,
additive alongside the merged pipeline, not a replacement for it.
`within_vs_cross_digit_distance` is the cheap label-based sanity check
(Phase 4): mean pairwise distance for same-digit vs. different-digit pairs.

## File reference

| File | Contents |
|---|---|
| `data_pool.py` | `load_eval_pool` — fixed, deterministic MNIST test-set pool, mirrors `mnist_lipschitz`'s protocol. |
| `streams.py` | Stream construction for both methods: `time_channel`, `make_pixel_order`/`patch_sv_stream` (Method A), `make_reference_lines`/`line_stream` (Method B), and the superseded `row_stream`. |
| `signatures.py` | `signature_of_stream` — truncated signature via `roughpy_jax`, shared by both methods. |
| `distances.py` | `rescale_signature`, `choose_rescale_factor`, `method_a_feature_vector`, `method_b_feature_vector`, `pairwise_euclidean_distance`, `per_line_distances`, `within_vs_cross_digit_distance`. |
| `plots.py` | All plotting: pixel-order/reference-line overlays, stream plots, signature bar chart/heatmap, per-line AUC ranking bar chart. |
| `run_experiment.py` | `stream_construction_demo`/`method_a_demo` (per-digit stream+signature figures for both methods), `sanity_check_demo` (Phase 4, both methods). |
| `method_b_adversarial_eval.py` | Standalone (isolated from `mnist_lipschitz`) FGSM Lipschitz-ratio evaluation: fresh `SmallCNN`/`StrongCNN` reimplementations, `margin`, `fgsm_attack`, `random_noise_perturbation` control, `method_b_signature_distance`, `run_adversarial_evaluation`. |
| `method_a_adversarial_eval.py` | Method A's counterpart to `method_b_adversarial_eval.py` — same evaluation, `method_a_signature_distance` in place of Method B's distance; reuses that module's `SmallCNN`/`StrongCNN`/`train_classifier`/`margin`/`fgsm_attack`/`random_noise_perturbation`/`pixel_euclidean_distance`/`load_mnist_train_test` unmodified. Result-dict fields use `denom_pixel_*`/`denom_sig_*` (not `_a`/`_b`) to avoid clashing with Method B's naming convention. |
| `per_line_diagnostics.py` | `run_per_line_auc_diagnostic` — same/different-digit ROC AUC per individual line vs. the merged distance. Additive diagnostic, doesn't change the merged pipeline. |
| `per_path_adversarial_eval.py` | `run_per_path_adversarial_eval`, `summarize_informative_subset`, `spike_analysis`, `plot_spike_gallery` — 16 separate per-line Lipschitz ratios, never merged, reusing `method_b_adversarial_eval.py`'s models/attack unmodified. Defines `BORDER_LINE_INDICES`/`INFORMATIVE_LINE_INDICES`/`BEST_LINE_INDEX`. |
| `per_path_ratio_robustness_check.py` | `fold_ratio_robustness`, `run_and_report` — read-only check of whether the per-path fold-ratio finding survives excluding lines 9 and 14 (flagged elsewhere as having a distance-scale confound). |
| `level_decomposition.py` | `level_slices`, `mask_signature_levels`, `run_level_decomposition` — read-only diagnostic: is Phase 4's within/cross-digit signal carried by the level-1 signature terms or the higher-order ones (2..depth)? Reuses `distances.py`/`streams.py`/`signatures.py` unmodified. |
| `PLAN.md` | The full staged plan (Phases 0-9), Method A/B naming reconciliation, and the running status/scope-boundary notes for each checkpoint. Read before changing stream construction or the signature/distance conventions. |
| `Method_B.md` | Method B's own design doc (line geometry, sampling, artifact convention) — written before Checkpoint 3 was implemented. |
| `PHASE1_SUMMARY.md` | Stream-construction shapes and CPU wall-clock timing on the 1000-image default pool, written at the Phase 1 gate. Test count in this file (33) reflects that snapshot only — see [How to run it](#how-to-run-it) for the current total. |
| `adversarial_eval_summary.md` | Full write-up of `method_b_adversarial_eval.py`'s results (see [Results](#results)). |
| `method_a_adversarial_eval_summary.md` | Full write-up of `method_a_adversarial_eval.py`'s results. |
| `per_line_auc_summary.md` | Full write-up of `per_line_diagnostics.py`'s results. |
| `per_path_adversarial_summary.md` | Full write-up of `per_path_adversarial_eval.py`'s results. |
| `per_path_ratio_robustness_summary.md` | Full write-up of `per_path_ratio_robustness_check.py`'s results. |
| `level_decomposition_summary.md` | Full write-up of `level_decomposition.py`'s results. |
| `notebook_method_a.ipynb` | Method A only: streams + signatures for one sample image per digit. |
| `notebook_method_b.ipynb` | Method B: streams + per-line signatures per digit, plus Phase 3/4 (distance + sanity check) for both methods. |
| `notebook_method_b_adversarial_eval.ipynb` | Trains both models, runs `method_b_adversarial_eval.run_adversarial_evaluation`, shows ratio tables/distributions/galleries. |
| `notebook_method_a_adversarial_eval.ipynb` | Method A's counterpart: trains both models, runs `method_a_adversarial_eval.run_adversarial_evaluation` at the imperceptible epsilons (0.02/0.03/0.05), shows ratio tables/distributions/galleries, then a second section reruns it at a larger sweep (0.05-0.3). |
| `notebook_per_path_adversarial_eval.ipynb` | Per-path (unmerged) evaluation — Findings 1/2, spike galleries. |
| `tests/` | Mirrors the module list above — see [How to run it](#how-to-run-it). |
| `artifacts/` | `pixel_order_seed0.npy`, `reference_lines_seed0.npy` — the fixed, shared geometry each method uses for every image, saved once at the default seed/config. |
| `results/` | Generated figures (git-ignored except `.gitkeep`). |

## Design decisions

- **Two methods, always kept separate.** Method A and Method B are two
  distinct distance functions throughout — never averaged, never given a
  shared rescale factor `r` (derived independently per method via
  `choose_rescale_factor`, since their raw signature scales differ).
- **No signature library import** — `roughpy_jax`'s low-level primitives are
  used (hand-driven `Lie`/`cbh`/`to_signature` calls), not `iisignature`,
  `signatory`, or `sigkernel`, and not `roughpy_jax`'s own `Stream` wrapper
  classes — chosen specifically because the low-level primitives are
  `custom_vjp`-registered and `vmap`-batchable, and because their behavior is
  pinned down by two closed-form checks (a straight line's exact tensor
  exponential across every level, and an L-shaped path's hand-computed area
  term) rather than inferred from sparse docs on the `Stream` classes.
- **Checkpoint-gating followed strictly for the signature step**, per the
  root project's convention: `signature_of_stream` is checked against the
  straight-line and L-shape closed forms (`tests/test_signatures_method_b.py`)
  before being trusted on real Method A/B streams.
- **`method_b_adversarial_eval.py` is deliberately isolated from
  `mnist_lipschitz`** — no imports of its `SmallCNN`/`StrongCNN`/
  `fgsm_attack`/`margin_fn`; everything is a fresh, individually-flagged
  reimplementation, trained with a different seed/init. This is a departure
  from `PLAN.md`'s Phase 5 text (which originally said to reuse
  `mnist_lipschitz/adversarial/attacks.py`) — the isolation was a deliberate
  choice, not an oversight, so any numeric comparison against Experiment 2's
  documented results (e.g. its CNN confusable pairs 6/5, 8/2, 8/0) is
  architecture-level only, not an exact reproduction. `per_path_adversarial_eval.py`,
  `per_path_ratio_robustness_check.py`, and `method_a_adversarial_eval.py` in
  turn reuse *this* module's models/attack code unmodified, rather than
  re-deriving it again — `method_a_adversarial_eval.py` swaps in
  `method_a_signature_distance` as the new denominator but retrains fresh
  with the same seed/recipe, so its `SmallCNN`/`StrongCNN` end up
  statistically indistinguishable from this module's own trained models
  (verified: test_acc within 0.02pp, matching flip fractions) even though
  they're not the literal same objects.
- **Method A and Method B are still not evaluated symmetrically, though less
  so than before.** Both went through streams -> signature -> Phase 4 sanity
  check (`run_experiment.sanity_check_demo` computes both), and both now have
  an adversarial/Lipschitz-ratio evaluation (`method_a_adversarial_eval.py` /
  `method_b_adversarial_eval.py`, same protocol, same reused
  model/training/attack code). Method B still additionally has a per-line AUC
  diagnostic and a per-path evaluation — those don't have a Method A
  equivalent, since they're specific to Method B's 16-separate-lines
  structure (Method A's single 64-anchor stream has no analogous "per-path"
  split to examine unmerged).
- **Method A's pixel visiting order is a random sample, not a spatially
  coherent walk.** `make_pixel_order` uses `torch.randperm`; measured mean
  step between consecutive anchors is 14.69px (range 1.41-30.41px) on a 26x26
  interior grid. Since a signature is order-sensitive by construction, this
  is flagged as a likely contributor to Method A's weak Phase 4 signal, not
  yet fixed.
- **4 of Method B's 16 lines carry zero same/different-digit signal by
  construction.** The border lines (indices 0, 7, 8, 15 — row/col exactly 0
  or 27) sit at exactly AUC=0.500 (chance) in `per_line_diagnostics.py`,
  since MNIST digits essentially never touch the image edge — a concrete,
  unfixed line-placement inefficiency (a quarter of the 496-dim merged vector
  contributes pure noise), independent of whichever aggregation method is
  used downstream.
- **Confounds are checked directly, not assumed away.** Two lines (9, 14)
  dominate the per-path "which line spikes" popularity count under *both*
  adversarial and control perturbations, because they have systematically
  smaller baseline distances regardless of perturbation type — flagged in
  `per_path_adversarial_eval.py` and directly re-checked in
  `per_path_ratio_robustness_check.py`, which confirms the headline
  adv>control fold-ratio finding does not depend on those two lines.

## How to run it

```bash
# from the repo root
.venv/bin/python -m pytest signature_distance/tests/ -v   # 83 tests

.venv/bin/python -c "from signature_distance.run_experiment import sanity_check_demo; print(sanity_check_demo())"

.venv/bin/python -c "from signature_distance.method_b_adversarial_eval import run_adversarial_evaluation; run_adversarial_evaluation()"

.venv/bin/python -c "from signature_distance.method_a_adversarial_eval import run_adversarial_evaluation; run_adversarial_evaluation()"

.venv/bin/python -m signature_distance.per_path_ratio_robustness_check   # ~5 min, retrains both CNNs

# or execute a notebook end-to-end
.venv/bin/jupyter nbconvert --to notebook --execute --inplace signature_distance/notebook_method_b.ipynb
```

There is no single `main()`/driver that runs every phase in sequence (unlike
`toy_lipschitz`/`mnist_lipschitz`'s `run_experiment.main()`) — each
notebook/module above is its own entry point, reflecting the staged,
gated-checkpoint way this package was built.

## Results

**Phase 1 — stream construction shapes and timing** (1000-image default
pool, `n_per_class=100, seed=0`, Apple Silicon CPU, no GPU): pool loading
~0.007s; Method A stream construction ~0.018s; Method B stream construction
(via batched `grid_sample`) ~0.001s — both effectively instantaneous. Full
detail: `PHASE1_SUMMARY.md`.

**Phase 2/3 — signature + distance (checkpoint-gated before use).**
`signature_of_stream` matches the exact closed-form tensor exponential for a
straight-line path (every level, not just level 1) and a hand-computed area
term for an L-shaped path, before being trusted on real Method A/B streams.
Raw signature magnitudes decay ~4-50x from level 1 to level 4; `r` (derived
independently per method) is `r_A ≈ 1.656`, `r_B ≈ 2.860` at the Phase 4
sample settings.

**Phase 4 — within- vs. cross-digit sanity check** (300 images, 30/class,
seed 0, depth 4):

| method | r | within-digit mean | cross-digit mean | cross/within ratio |
|---|---|---|---|---|
| Method A | 1.656 | 14.60 | 17.18 | 1.176 |
| Method B | 2.860 | 28.60 | 33.17 | 1.160 |

Both show the right direction (within < cross) with or without rescaling —
rescaling barely moved either ratio. But the effect size is comparable to
plain raw-pixel Euclidean's own weak ratio in `mnist_lipschitz` (~1.13):
**neither method yet shows a stronger class-separation signal than the
pixel-space baseline it's meant to improve on.** Per the checkpoint-gating
rule, this didn't block moving forward, but it's an honest, still-open
finding (full detail and a spatial-coherence caveat on Method A's ordering:
`PLAN.md`'s Phase 4 section).

**Method B adversarial/Lipschitz-ratio evaluation** (FGSM, eps ∈
{0.02, 0.03, 0.05}, 200 images/20 per class, fresh `SmallCNN`
test_acc 98.24% / `StrongCNN` test_acc 99.36%, magnitude-matched random-noise
control): both pixel-Euclidean and Method B distances separate adversarial
from control shifts clearly (near-bimodal ratio histograms), so Method B is
not just noise. But **pixel-Euclidean's adv/control separation ratio is
larger than Method B's in every one of the 6 model×epsilon combinations**
(SmallCNN 8.5-9.1x vs. 7.0-7.4x; StrongCNN 3.6-5.0x vs. 3.3-4.6x) — the same
"sensible but not yet better than the baseline" pattern as Phase 4, now a
second independent data point. Both distances register the capacity contrast
between the two models. Full table and the top-10-ratio-pairs check:
`adversarial_eval_summary.md`.

**Method A adversarial/Lipschitz-ratio evaluation** (same protocol,
retrained fresh with the same seed/recipe — trained models statistically
indistinguishable from Method B's run: `SmallCNN` test_acc 98.26% /
`StrongCNN` test_acc 99.34%, matching flip fractions almost exactly): same
result as Method B — both distances clearly separate adversarial from
control shifts, but **pixel-Euclidean's adv/control ratio beats Method A's
in every one of the 6 combinations too** (SmallCNN 8.4-9.1x vs. 7.0-7.8x;
StrongCNN 3.6-5.0x vs. 2.9-4.2x) — a third independent test landing in the
same place. Compared directly against Method B (valid here since the
trained models/numerator match): Method A's adv/control ratio edges out
Method B's on 4 of 6 model×epsilon combinations, but narrowly and
inconsistently — neither method is a clearly better denominator than the
other. Full table and the top-10-ratio-pairs check:
`method_a_adversarial_eval_summary.md`.

### Sub-experiment: Method A larger-epsilon sweep (verdict: pixel's edge erodes and briefly flips as epsilon grows, tracking control saturation, not a real capability edge)

Same `method_a_adversarial_eval.py` protocol, retrained fresh again with the
same seed/recipe, now run at epsilons {0.05, 0.1, 0.15, 0.2, 0.25, 0.3} —
well past the imperceptible regime the other evaluations above use (at
eps=0.3 a pixel can shift by up to 30% of the intensity range, visibly so in
`notebook_method_a_adversarial_eval.ipynb`'s example gallery). Flip fraction
saturates fast (SmallCNN 6.5%→99.5%, StrongCNN 4.5%→85.0% across the sweep),
and every adv/control ratio — pixel and Method A, both models — falls
monotonically as epsilon grows (e.g. SmallCNN pixel 8.63→4.17): once a
same-sized *random* shift starts flipping predictions on its own, the
control stops being a clean negative and the adversarial-vs-random signal
both distances measure erodes. The pixel-vs-Method-A gap narrows in lockstep
and, at the largest epsilon, briefly flips: StrongCNN's Method A ratio
(1.48) narrowly exceeds pixel's (1.47) at eps=0.3 — the first time anywhere
in this project a signature distance has matched or beaten the
pixel-Euclidean baseline on this comparison. Flagged, not overclaimed: a
0.01 gap from 200 images at the one condition (85% flip rate) where the
comparison is least informative is well within sampling noise, and it
coincides exactly with both ratios being compressed toward each other by
the same saturation effect, not obvious evidence of a real capability edge.
Full tables: `method_a_adversarial_eval_summary.md`.

### Sub-experiment: per-line AUC diagnostic (verdict: one line beats merged narrowly; 4 lines are dead weight)

Tests the "path by path" framing directly: is any individual line a better
same/different-digit classifier than the merged 496-dim distance? On the
same 300-image Phase 4 sample, `line_6` (horizontal, row≈23) reaches
AUC 0.6394 vs. merged's 0.6315 — real but narrow, and only 1 of 16 lines
beats merged (13 score below it), so this isn't broad dilution. The 4 border
lines (0, 7, 8, 15) sit at exactly AUC=0.500 — zero signal by construction —
contributing pure noise to the merged vector. Every measure here, merged and
individual, is a weak discriminator in absolute terms (FPR > 0.8 at 90% TPR
everywhere). Full ranking: `per_line_auc_summary.md`.

### Sub-experiment: per-path (unmerged) Lipschitz ratios (verdict: stronger separation than merged distance, with two honest caveats)

Never merges the 16 lines — each produces its own local ratio
`margin_change / line_i_distance`, examined as a collection (mirrors how
pixel-space Lipschitz work treats individual pixels). **Finding 1**: across
all 12 informative lines (excludes the 4 border lines) and all 6
model×epsilon combinations (72/72, no exceptions), mean ratio on genuinely
flipped adversarial pairs exceeds the matched random-control ratio — fold
ratios 4-11x (up to ~20x for `line_6`), noticeably stronger than the merged
distance's own 3.3-7.4x adv/control separation. Caveat: `n_flipped` is small
(4-14/200) at these conservative epsilons, so exact magnitudes are noisy —
the *direction* is what's trustworthy. **Finding 2**: the informative line
carrying the largest ratio is more concentrated (lower entropy) under
adversarial perturbation than under control, in all 6/6 combinations
(~0.2-0.3 bits of a 3.585-bit max) — modest but zero-exception evidence FGSM
concentrates on fewer paths than an equally-large random shift. A confound
(two lines with systematically smaller baseline distances dominate raw
"which line spikes" counts under both conditions) is flagged and separated
from Finding 1. Full write-up, including the mixed stroke-relevance gallery
read: `per_path_adversarial_summary.md`.

### Sub-experiment: per-path robustness check (verdict: Finding 1 is not resting on the two confounded lines)

Checks whether Finding 1 above survives excluding the two lines (9, 14)
flagged for the baseline-distance-scale confound. **Yes, unconditionally**:
all 60 of the reduced 10-line×6-combination comparisons still show
adversarial > control (mathematically guaranteed once 72/72 held), and the
mean fold-ratio moves by at most ~0.5x on the 4-11x scale (no consistent
direction). A follow-up correlation check across the remaining 10 lines
finds no meaningful relationship between baseline distance and fold-ratio
(`|r| ≤ 0.32`, sign flips across models/epsilons) — the scale confound looks
isolated to lines 9 and 14 specifically, not systemic. Full detail:
`per_path_ratio_robustness_summary.md`.

### Sub-experiment: signature level decomposition (verdict: higher-order levels carry the signal, not level 1 — and Method B's level 1 is near-degenerate)

Tests the concern that motivated it directly: is Phase 4's within/cross-digit
signal carried entirely by the level-1 signature terms (for Method A, a
single scalar patch-SV difference), or do levels 2..depth contribute real
information? On the same 300-image Phase 4 sample, masking out level 1
entirely (`level2plus`) reproduces the full `all` ratio almost exactly for
both methods (A: 1.175 vs. 1.176; B: 1.160 vs. 1.160) — level 1 turns out to
be carrying almost none of the distance's *magnitude* (1.79% of total
squared distance for Method A, 0.003% for Method B) despite `level1_only`
posting the single *highest* per-level ratio for Method A (1.276). Method
B's level-1 near-vanishing has a structural cause, not just a weak signal:
every reference line's endpoints sit on the image border, where MNIST
intensity is ~0 for nearly every image, so 99.96% of its per-line net
displacements are exactly 0.0 by construction. Level 4 carries the most
magnitude for both methods (58%/75%) but the *lowest* per-level ratio
(1.162 A / 1.153 B) — magnitude and per-unit class-separation rank levels in
opposite orders. Full tables and discussion: `level_decomposition_summary.md`.

## Status / what's not done yet

Per `PLAN.md`'s phase numbering: Phases 0-4 are implemented and gated for
both methods; the adversarial evaluation (now both methods) and Method B's
follow-on diagnostics (per-line AUC, per-path ratios, robustness check) go
beyond Phase 4 but don't map onto a single later phase number cleanly. **Not
yet done**, in `PLAN.md`'s terms:

- **Phase 5's PGD leg** — only single-step FGSM has been run, for both
  methods; multi-step PGD is deferred.
- **Phase 6 — integration into the existing `mnist_lipschitz/adversarial`
  pipeline** (`run_ratio_distribution_analysis` as a new denominator option)
  has not been done; `method_a_adversarial_eval.py`/`method_b_adversarial_eval.py`
  are standalone evaluations, not plugged into that harness.
- **Phase 8 — the hyperparameter sweep** (line count, points/line, patch
  count K, truncation depth, rescale factor) has not been run. The two
  diagnostics above (border-line inefficiency, Method A's non-coherent
  pixel order) are concrete candidates for what such a sweep should fix
  first.
- **Method A still has no per-line/per-path-style unmerged evaluation** —
  its single 64-anchor stream has no analogous per-path split the way
  Method B's 16 separate lines do, so `per_line_diagnostics.py`/
  `per_path_adversarial_eval.py`/`per_path_ratio_robustness_check.py` have
  no Method A counterpart (nor an obvious one to build without a comparable
  sub-structure to split on).

None of the ambiguous/negative findings above (Phase 4's modest ratio, both
methods sitting behind the pixel-Euclidean baseline) have been treated as a
stop condition — per this project's "move forward regardless, but carry the
finding" instruction — but none have been promoted into a default pipeline
either, consistent with the root project's checkpoint-gating discipline: a
new metric earns its way into `main()`-style default usage by beating
existing baselines, not just by running without error.
