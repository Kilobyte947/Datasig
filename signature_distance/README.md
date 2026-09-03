# signature_distance

Builds candidate **image distance metrics from path signatures** on MNIST, for
eventual comparison against the Euclidean and Mahalanobis pixel-space distances
already used in `mnist_lipschitz`'s Lipschitz/adversarial work. Every Lipschitz
estimate in that experiment reduces to the same ratio,
`|margin_i - margin_j| / distance(x_i, x_j)` - every denominator tried there
(Euclidean, ridge-regularized Mahalanobis, a UMAP embedding) has been a distance
over the *raw pixel grid*. This package asks whether representing an image as a
handful of low-dimensional paths, then measuring distance between their
**truncated path signatures**, gives a better-behaved denominator.

Two independent, fixed (non-learned) constructions turn an image into one or
more paths - deliberately not derived from a trained classifier's own features,
to stay clear of the self-referential-metric problem (measuring a function's
sensitivity with a metric built from that same function collapses the signal -
documented in `toy_lipschitz/README.md`'s `augmented_embedding` finding):

- **Method A - patch singular-value stream**: visit a fixed, shared set of 64
  interior pixel locations; at each, take the largest singular value of the
  surrounding 3x3 patch. One stream per image, shape `(64, 2)`.
- **Method B - reference-line stream**: originally 8 horizontal + 8 vertical
  fixed lines through the image (16 streams per image, shape `(16, 32, 2)`),
  each sampled at 32 points via bilinear interpolation. A later hyperparameter
  sweep (`method_b_sweep.py`) adopted **12 horizontal + 4 vertical lines at
  truncation depth 2** as the new default, replacing the original 8+8/depth-4
  split - see "Method B: Reference-Line Signature Distance"'s "Hyperparameter
  sweep"/"Current status" sections below. The original 8+8/depth-4
  configuration remains fully available (it's the default `streams.make_reference_lines()`
  still produces) and is what every "Method B" number in this document's
  Phase 1-4/adversarial-eval sections uses, unless stated otherwise - those
  sections predate the sweep and haven't been re-run against the new default.
- **Method C - Hilbert-curve stream**: a single order-5 space-filling curve
  through the image (1024 cells, resampled to 512 points, cut into 16
  segments of 32 points - the same total point budget as Method B's
  original default), evaluated the same per-path way. Full design/results:
  "Method C: Hilbert-Curve Signature Distance" below. Verdict: works sensibly,
  exception-free adversarial/control separation, but trails Method B's swept
  winner by a consistent margin (9.88x vs. 13.53x mean fold-ratio) - kept as a
  documented alternative, not adopted as a replacement.

The full staged plan (originally Phases 0-9) is described in "Implementation
plan and design constraints" below; this document summarizes what's actually
been built and found so far for Method A specifically (the Method B sweep and
Method C are summarized here only briefly - their own sections, "Method B:
Reference-Line Signature Distance"/"Method C: Hilbert-Curve Signature
Distance", are authoritative). **Read "Implementation plan and design
constraints" before making non-trivial changes** - it records design
decisions that must not change without flagging (fixed pixel/line geometry,
no cross-line concatenation before the signature step, SVD-based patch
statistic, etc.).

## Document map

This file merges what used to be seven separate markdown documents in this
package. Rough guide to where things are:

- **How it's designed** - architecture of the evaluation pool, stream
  construction, signature computation, and distance/rescaling code.
- **Code and notebook reference** - what each `.py` module, notebook, and
  data directory contains.
- **Design decisions** - fixed conventions and known inefficiencies/open
  issues (border lines, pixel-order incoherence, etc.).
- **How to run it** - test/driver commands.
- **Implementation plan and design constraints** - the original staged plan
  (formerly `PLAN.md`): why signatures over Euclidean/Mahalanobis, the two
  methods' original specs, and constraints that must not change without
  flagging.
- **Method B: Reference-Line Signature Distance** - Method B's full design
  and experimental history, including the pivot to per-path evaluation and
  the hyperparameter sweep (formerly `Method_B.md`).
- **Method C: Hilbert-Curve Signature Distance** - the Hilbert-curve
  alternative's design and head-to-head comparison against Method B (formerly
  `Method_C.md`).
- **Method A: Adversarial / Lipschitz-Ratio Evaluation Results** - Method A's
  adversarial evaluation, both the imperceptible-epsilon and larger-epsilon
  sweeps (formerly `method_a_adversarial_eval_summary.md`).
- **Level-wise decomposition diagnostic** - whether the level-1 signature
  terms or the higher-order ones carry the within/cross-digit signal
  (formerly `level_decomposition.md` + `level_decomposition_summary.md`).
- **Results** - the chronological walkthrough of every phase and
  sub-experiment, cross-referencing the sections above for full detail.
- **Status / what's not done yet** - open work, in the original plan's phase
  numbering.

## How it's designed

**Evaluation pool (`data_pool.py`).** `load_eval_pool(n_per_class, seed)`
mirrors `mnist_lipschitz`'s pool-based protocol: a deterministic,
class-balanced subset of the MNIST test set (shuffle indices with `seed`,
take the first `n_per_class` per class). Default pool used throughout is 1000
images (`n_per_class=100`).

**Streams (`streams.py`).**
- `time_channel(n)` - the shared `arange(n)/(n-1)` time coordinate used by
  both methods, so every stream is time-augmented identically regardless of
  which method built it (a signature of a scalar path degenerates to a
  function of the net increment alone without a time coordinate).
- Method A: `make_pixel_order(k=64, seed=0)` samples `k` interior
  `(row, col)` locations without replacement (fixed order, one call, reused
  for every image); `patch_sv_stream` batches all 3x3 patch extraction +
  `torch.linalg.svdvals` (no Python loop over images), returning
  `(N, K, 2)` columns `[t, sigma1]` (`mode="top1"`, default) or `(N, K, 4)`
  `[t, s1, s2, s3]` (`mode="all3"`).
- Method B: `make_reference_lines(angles_deg=(0,90), counts=(8,8), ...)` is
  one parameterized function for both orientations (not separate code paths)
  - only 0deg/90deg are supported (in-bounds by construction, no clipping
  logic); `line_stream` reads intensity via batched `grid_sample` bilinear
  interpolation, returning `(N, 16, 32, 2)`. Lines are never concatenated
  into one raw stream before the signature step (see "Method B:
  Reference-Line Signature Distance" below's "no cross-line concatenation"
  rule) - each stays a separate `(32, 2)` path. `counts=(8,8)` is still the
  function's default (the original geometry); the swept 12h+4v default is
  applied by callers passing `counts=(12,4)`, not by a change to this
  function's own default.

**Signatures (`signatures.py`).** `signature_of_stream(stream, depth)` computes
the truncated signature of a batch of 2D piecewise-linear paths (shared by
both methods - same function, so Method A and B signatures are directly
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
**`r` is always derived independently per method, never shared** - Method A
and Method B are two separate distance functions throughout this package,
never merged into one metric (mirrors the root project's "the three Lipschitz
sub-methods are never merged" convention). Per-image feature vectors:
`method_a_feature_vector` is the identity (the rescaled signature already
is the feature vector); `method_b_feature_vector` concatenates the 16
per-line rescaled signatures into one 496-dim vector - **the first point the
16 lines combine**, never before. `per_line_distances` stops one step short
of that concatenation, comparing line `i` to line `i` directly ("path by
path") instead of merging first - used by the per-path diagnostics below,
additive alongside the merged pipeline, not a replacement for it.
`within_vs_cross_digit_distance` is the cheap label-based sanity check
(Phase 4): mean pairwise distance for same-digit vs. different-digit pairs.

## Code and notebook reference

Package documentation now lives entirely in this file (see "Document map"
above); the table below covers code, notebooks, and data only.

| File | Contents |
|---|---|
| `data_pool.py` | `load_eval_pool` - fixed, deterministic MNIST test-set pool, mirrors `mnist_lipschitz`'s protocol. |
| `streams.py` | Stream construction for Methods A and B: `time_channel`, `make_pixel_order`/`patch_sv_stream` (Method A), `make_reference_lines`/`line_stream` (Method B). `row_stream` (a superseded draft) has been removed. |
| `hilbert_stream.py` | Method C: Hilbert-curve generation, arc-length resampling, segment construction, and its own depth-sweep/adversarial-eval/robustness-check driver functions. |
| `signatures.py` | `signature_of_stream` - truncated signature via `roughpy_jax`, shared by all three methods. |
| `distances.py` | `rescale_signature`, `choose_rescale_factor`, `method_a_feature_vector`, `method_b_feature_vector`, `pairwise_euclidean_distance`, `per_line_distances`, `within_vs_cross_digit_distance`, `auc_for_distance` (shared same/different-digit AUC helper, factored out of `per_line_diagnostics.py`/`method_b_sweep.py`). |
| `plots.py` | All plotting: pixel-order/reference-line overlays, stream plots, signature bar chart/heatmap, per-line AUC ranking bar chart. |
| `run_experiment.py` | `stream_construction_demo`/`method_a_demo` (per-digit stream+signature figures for both methods), `sanity_check_demo` (Phase 4, both methods). |
| `method_b_adversarial_eval.py` | Standalone (isolated from `mnist_lipschitz`) FGSM Lipschitz-ratio evaluation: fresh `SmallCNN`/`StrongCNN` reimplementations, `margin`, `fgsm_attack`, `random_noise_perturbation` control, `method_b_signature_distance`, `run_adversarial_evaluation`. Its model/training/attack code is reused unmodified by every other adversarial-eval module in this package (Method A's, per-path, the sweep, Method C). |
| `method_a_adversarial_eval.py` | Method A's counterpart to `method_b_adversarial_eval.py` - same evaluation, `method_a_signature_distance` in place of Method B's distance; reuses that module's `SmallCNN`/`StrongCNN`/`train_classifier`/`margin`/`fgsm_attack`/`random_noise_perturbation`/`pixel_euclidean_distance`/`load_mnist_train_test` unmodified. Result-dict fields use `denom_pixel_*`/`denom_sig_*` (not `_a`/`_b`) to avoid clashing with Method B's naming convention. |
| `method_b_sweep.py` | Stage 8 hyperparameter sweep for Method B (geometry, points/line, depth, interpolation) - Stage A cheap AUC screen (160 configs) and Stage B full per-path adversarial validation on finalists; adopted the 12h+4v/depth=2 winner as the new default. |
| `per_line_diagnostics.py` | `run_per_line_auc_diagnostic` - same/different-digit ROC AUC per individual line vs. the merged distance. Additive diagnostic, doesn't change the merged pipeline. |
| `per_path_adversarial_eval.py` | `run_per_path_adversarial_eval`, `summarize_informative_subset`, `spike_analysis`, `plot_spike_gallery` - 16 separate per-line Lipschitz ratios, never merged, reusing `method_b_adversarial_eval.py`'s models/attack unmodified. Defines `BORDER_LINE_INDICES`/`INFORMATIVE_LINE_INDICES`/`BEST_LINE_INDEX`. Also now includes `fold_ratio_robustness`/`run_robustness_report` (the robustness check, merged in from the former standalone `per_path_ratio_robustness_check.py`). |
| `level_decomposition.py` | `level_slices`, `mask_signature_levels`, `run_level_decomposition` - read-only diagnostic: is Phase 4's within/cross-digit signal carried by the level-1 signature terms or the higher-order ones (2..depth)? Reuses `distances.py`/`streams.py`/`signatures.py` unmodified. |
| `notebook_method_a.ipynb` | Method A only: streams + signatures for one sample image per digit, plus the level-decomposition and pixel-order-robustness sections. |
| `notebook_method_b_streams.ipynb` | Method B stream/signature construction (Phase 1/2) - renamed from `notebook_method_b.ipynb`. |
| `notebook_method_b_further_work.ipynb` | Method B's distance/sanity-check, adversarial evaluation, and hyperparameter-sweep notebook (supersedes the former separate `notebook_method_b_adversarial_eval.ipynb`/`notebook_per_path_adversarial_eval.ipynb`, both removed). |
| `notebook_method_a_adversarial_eval.ipynb` | Method A's counterpart: trains both models, runs `method_a_adversarial_eval.run_adversarial_evaluation` at the imperceptible epsilons (0.02/0.03/0.05), shows ratio tables/distributions/galleries, then a second section reruns it at a larger sweep (0.05-0.3). |
| `notebook_method_c.ipynb` | Method C construction, both sweep stages, and the head-to-head comparison against Method B's winner, executed end to end. |
| `tests/` | Mirrors the module list above - see [How to run it](#how-to-run-it). |
| `artifacts/` | `pixel_order_seed0.npy`, `reference_lines_seed0.npy`, `hilbert_curve_seed0.npy` - the fixed, shared geometry each method uses for every image, saved once at the default seed/config. |
| `results/` | Generated figures (git-ignored except `.gitkeep`). |

## Design decisions

- **Two methods, always kept separate.** Method A and Method B are two
  distinct distance functions throughout - never averaged, never given a
  shared rescale factor `r` (derived independently per method via
  `choose_rescale_factor`, since their raw signature scales differ).
- **No signature library import** - `roughpy_jax`'s low-level primitives are
  used (hand-driven `Lie`/`cbh`/`to_signature` calls), not `iisignature`,
  `signatory`, or `sigkernel`, and not `roughpy_jax`'s own `Stream` wrapper
  classes - chosen specifically because the low-level primitives are
  `custom_vjp`-registered and `vmap`-batchable, and because their behavior is
  pinned down by two closed-form checks (a straight line's exact tensor
  exponential across every level, and an L-shaped path's hand-computed area
  term) rather than inferred from sparse docs on the `Stream` classes.
- **Checkpoint-gating followed strictly for the signature step**, per the
  root project's convention: `signature_of_stream` is checked against the
  straight-line and L-shape closed forms (`tests/test_signatures_method_b.py`)
  before being trusted on real Method A/B streams.
- **`method_b_adversarial_eval.py` is deliberately isolated from
  `mnist_lipschitz`** - no imports of its `SmallCNN`/`StrongCNN`/
  `fgsm_attack`/`margin_fn`; everything is a fresh, individually-flagged
  reimplementation, trained with a different seed/init. This is a departure
  from the original plan's Phase 5 text (which originally said to reuse
  `mnist_lipschitz/adversarial/attacks.py` - see "Implementation plan and
  design constraints" below) - the isolation was a deliberate choice, not an
  oversight, so any numeric comparison against Experiment 2's documented
  results (e.g. its CNN confusable pairs 6/5, 8/2, 8/0) is architecture-level
  only, not an exact reproduction. `per_path_adversarial_eval.py`
  (including its merged-in robustness check), `method_b_sweep.py`,
  `hilbert_stream.py` (Method C), and `method_a_adversarial_eval.py` in
  turn reuse *this* module's models/attack code unmodified, rather than
  re-deriving it again - `method_a_adversarial_eval.py` swaps in
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
  diagnostic and a per-path evaluation - those don't have a Method A
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
  construction.** The border lines (indices 0, 7, 8, 15 - row/col exactly 0
  or 27) sit at exactly AUC=0.500 (chance) in `per_line_diagnostics.py`,
  since MNIST digits essentially never touch the image edge - a concrete,
  unfixed line-placement inefficiency (a quarter of the 496-dim merged vector
  contributes pure noise), independent of whichever aggregation method is
  used downstream.
- **Confounds are checked directly, not assumed away.** Two lines (9, 14)
  dominate the per-path "which line spikes" popularity count under *both*
  adversarial and control perturbations, because they have systematically
  smaller baseline distances regardless of perturbation type - flagged and
  directly re-checked within `per_path_adversarial_eval.py` (via
  `fold_ratio_robustness`, formerly a separate `per_path_ratio_robustness_check.py`
  module, merged in once its history was recorded in "Method B:
  Reference-Line Signature Distance" below), which confirms the headline
  adv>control fold-ratio finding does not depend on those two lines.
- **Method B's hyperparameter sweep changed the recommended default, not
  the underlying functions' own defaults.** `method_b_sweep.py` found 12h+4v
  lines at depth 2 beats the original 8h+8v/depth-4 split by a wide margin
  (mean fold-ratio ~14.8-15x vs. ~8.4x) and adopted it as the new default for
  *new* work - but `make_reference_lines`'s own `counts=(8,8)` default and
  `signature_of_stream`'s depth parameter are unchanged, so every existing
  number in this document computed before the sweep (Phase 1-4, both
  adversarial evaluations, the level-decomposition sub-experiment) remains
  exactly reproducible as documented, using the original configuration -
  it just no longer reflects Method B's best-known setting. See "Method B:
  Reference-Line Signature Distance" below for the full sweep and "Method A:
  Adversarial / Lipschitz-Ratio Evaluation Results" below for what a
  Method-A-vs-Method-B comparison would need to account for if re-run
  against the new default (not done here).
- **Method C (Hilbert curve) exists as a third, documented alternative to
  Method B**, evaluated with the same per-path infrastructure and reusing
  `method_b_adversarial_eval.py`'s models/attack code unmodified (same
  pattern as every other adversarial-eval module here). It trails Method
  B's swept winner by a consistent, non-trivial margin (9.88x vs. 13.53x
  mean fold-ratio, same 200-image/model/epsilon comparison) - kept as a
  documented alternative, not adopted as a replacement. Full detail: see
  "Method C: Hilbert-Curve Signature Distance" below.

## How to run it

```bash
# from the repo root
.venv/bin/python -m pytest signature_distance/tests/ -v   # 104 tests

.venv/bin/python -c "from signature_distance.run_experiment import sanity_check_demo; print(sanity_check_demo())"

.venv/bin/python -c "from signature_distance.method_b_adversarial_eval import run_adversarial_evaluation; run_adversarial_evaluation()"

.venv/bin/python -c "from signature_distance.method_a_adversarial_eval import run_adversarial_evaluation; run_adversarial_evaluation()"

.venv/bin/python -c "from signature_distance.per_path_adversarial_eval import run_robustness_report; run_robustness_report()"   # ~5 min, retrains both CNNs

# or execute a notebook end-to-end
.venv/bin/jupyter nbconvert --to notebook --execute --inplace signature_distance/notebook_method_b_streams.ipynb
```

There is no single `main()`/driver that runs every phase in sequence (unlike
`toy_lipschitz`/`mnist_lipschitz`'s `run_experiment.main()`) - each
notebook/module above is its own entry point, reflecting the staged,
gated-checkpoint way this package was built.

## Implementation plan and design constraints

This package was built from a staged implementation plan (originally
`PLAN.md`, folded into this section) that named the two original
stream-construction methods consistently throughout the project as
**Method A** (the "convolution" method: a local, patch-based statistic at a
fixed set of pixel locations) and **Method B** (the "line reference" method:
intensity sampled along a fixed set of straight lines). Method C (the
Hilbert-curve variant, see below) was added later, layered on top of the same
signature/distance infrastructure once Methods A and B were already built and
evaluated - it isn't part of the original phase plan described here.

**Context.** The experiment builds candidate distance metrics on MNIST images
using path signatures, for eventual comparison against the Euclidean and
Mahalanobis distances already used in `mnist_lipschitz`'s Lipschitz/
adversarial work: represent each image via a small number of low-dimensional
paths/streams (not raw pixels), compute a distance between images in this
path-signature space, and use that distance as the new denominator in the
existing Lipschitz-ratio pipeline, tested against imperceptible adversarial
examples. Both methods use fixed, non-learned constructions only - no
dependence on a trained classifier's own features, deliberately, to stay
clear of the self-referential-metric problem documented elsewhere in the
project (`toy_lipschitz/README.md`'s `augmented_embedding` finding).

**Why no signature-library import, even later.** The plan from the outset
was to use `roughpy-jax` (same semantics as `roughpy`, faster), hand-driven
rather than pulled in as an opaque dependency - not a placeholder decision to
revisit later, but deliberate: it's why the earliest phase of the project
avoided importing `iisignature`, `signatory`, `sigkernel`, `roughpy`, or
`roughpy-jax` at all, even in code paths that wouldn't be exercised yet, so
that no early scaffolding would need to be reworked around a library choice
made afterward.

**Design constraints that must not change without flagging** (fixed at the
outset, still true of the current code):

- Method A's pixel locations are sampled once with a fixed seed and reused
  identically for every image (`make_pixel_order`), from interior rows/
  columns 1..26 only, so every 3x3 patch stays fully inside the 28x28 image
  with no padding. The patch statistic is the largest singular value of the
  3x3 patch (`torch.linalg.svdvals`), structured so a `mode="all3"` (all
  three singular values) extension was trivial without being required
  initially. Default `K=64`, exposed as a parameter.
- Method B's `make_reference_lines(angles_deg, counts, points_per_line,
  image_size, seed)` is one parameterized function, not separate
  per-orientation implementations - any orientation split is a different
  call to the same function, keeping it tunable rather than hardcoded. Only
  horizontal (0deg) and vertical (90deg) angles are supported, by
  construction in-bounds with no clipping logic needed (an earlier
  circle-based geometry draft was dropped for exactly this reason -
  unnecessary complexity next to axis-aligned lines); arbitrary angles would
  need clipping logic, deferred indefinitely, `ValueError` otherwise. Sample
  points generally miss pixel centers, so intensity is read via batched
  bilinear interpolation (`grid_sample`), not nearest-pixel snapping. No
  cross-line concatenation: each line's signature is computed independently
  over its own points; lines are combined only after signatures are
  computed, never by concatenating raw streams first.
- Every stream is time-augmented via the shared `time_channel` helper
  (`arange(n)/(n-1)`), since a signature of a scalar path degenerates to a
  function of the net increment alone without a time coordinate.
- Style constraints held throughout: pure functions operating on
  tensors/arrays, no notebook code in modules; every public function takes
  explicit arguments (no hidden globals) and documents input/output shapes;
  all randomness goes through an explicit `seed` argument; `torch` is used
  throughout for consistency with the rest of the project, kept callable on
  CPU. The original plan's repository-layout sketch (module list, notebook
  names, artifacts folder) matched what the first checkpoints actually
  produced; see "Code and notebook reference" above for the current,
  complete file list, since the package grew well beyond that initial
  sketch.

**How the phases played out.** The plan was staged into checkpoints with
test gates - each checkpoint's tests had to pass before the next began,
mirroring the root project's checkpoint-gating convention. Checkpoints 0-3
(folder setup, the evaluation data pool, Method A streams, Method B streams)
were explicitly scoped as the only phase to implement at first; everything
past stream construction (signature computation, distance, sanity checks,
adversarial evaluation, pipeline integration, sweeps) was recorded up front
only so the stream-construction interfaces would already have the right
shape for what came next, not implemented at that point. All of Checkpoints
0-3 passed their gates - folder structure and empty test collection; pool
shape/dtype/determinism checks; Method A's interior-bound, determinism,
analytic-SVD (constant image -> singular values `(3c, 0, 0)`), time-channel,
and order-sensitivity checks; Method B's shape, in-bounds, directionality,
determinism, interpolation-correctness, time-channel, line-sensitivity, and
invalid-input checks - before signature computation began.

Signature computation (Phase 2) followed the same self-coded
`roughpy-jax`-targeting approach described above, starting at one fixed
truncation level for the basic pass before a full depth sweep was deferred to
later work (ultimately done for Method B in the hyperparameter sweep - see
"Method B: Reference-Line Signature Distance" below). Distance and rescaling
(Phase 3) and the within/cross-digit sanity check (Phase 4) are described in
"How it's designed" above and "Results" below; Phase 4's results were
ambiguous enough that they would ordinarily have warranted pausing, but work
proceeded to the adversarial evaluation anyway, on the reasoning that
hyperparameter tuning was always going to follow regardless of how that
first, unweighted attempt looked. One diagnosis worth preserving from that
stage: roughly 67% of Method A's 64 fixed anchor locations land on
near-zero-intensity patches on average across the pool, a background-sparsity
effect flagged as a likely contributor to its weak Phase 4 signal, alongside
the separately-flagged non-spatially-coherent pixel visiting order (see
"Design decisions" above) - though Method B, which has no equivalent sparsity
issue, showed a comparably weak Phase 4 ratio too, so sparsity isn't the
whole story.

The remaining phases (5 through 9 - imperceptible adversarial examples and
the Lipschitz ratio, integration into the existing `mnist_lipschitz/adversarial`
pipeline, baseline diagnostics, the hyperparameter sweep, and documentation)
were recorded up front as a roadmap, each phase gated by its own tests only
once actually implemented, per the same "don't wire in a new phase just
because it runs without error" discipline used elsewhere in the project.
Phase 5's FGSM leg, Phase 7's diagnostics, and Phase 8's sweep for Method B
are covered in "Results" below and in the Method B/Method A sections below;
see "Status / what's not done yet" below for which of these phases remain
incomplete as of the material folded into this document.

## Method B: Reference-Line Signature Distance

### Overview

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
subsequent hyperparameter sweep improved on it further. This section covers
the full arc: design, what was built, what was measured, and where it
landed.

### Design

**Stream construction** (`streams.py`, `make_reference_lines` +
`line_stream`). Lines are placed by `angles_deg`/`counts` pairs - angle 0
gives horizontal lines (rows evenly spaced across image height, each
sampled left to right), angle 90 gives vertical lines (columns evenly
spaced, sampled top to bottom). The original default was 8 horizontal + 8
vertical lines, 32 points per line; both the orientation split and the
point count are parameters of one function, not separate code paths, so
any split (all-horizontal, all-vertical, uneven) is just a different call.
No clipping is needed in either orientation - every sample point is
in-bounds by construction, since lines run edge to edge along the axis
they're aligned with. Intensity is read via batched bilinear interpolation
(`grid_sample`), not a Python loop over images.

**Signature computation** (`signatures.py`, shared with the project's other
stream-based method). Each line's stream is treated as a piecewise-linear
path - straight-line interpolation between consecutive sampled points -
and its truncated signature is computed via low-level tensor/Lie-algebra
primitives (`Lie`, `cbh`, `to_signature`) rather than higher-level stream
wrapper objects, batched with `jax.vmap`. Correctness was checked against
two closed-form identities before being used downstream: a straight-line
path's signature matches the exact analytic tensor exponential at every
truncation level, and an L-shaped (two-segment, non-collinear) path
matches a hand-computed nonzero area term - the second check specifically
exercises whether multi-segment combination is correct, not just the
trivial straight-line case.

**No cross-line merging before the signature step.** Each line's signature
is computed independently over its own points only; if lines are combined
at all, it happens after signatures are computed; the raw streams
themselves are never concatenated.

### Distance function and sanity check

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
sensibly rather than randomly - but the effect size was modest, in the
same range as plain raw-pixel Euclidean distance's own weak ratio
(~1.13) on the same kind of check elsewhere in the project. Rescaling
barely moved the ratio (1.19 to 1.16), so rescaling itself wasn't the
bottleneck. This result was ambiguous enough that it would ordinarily
warrant pausing before building further on it, but work proceeded to the
actual target application (adversarial evaluation) on the reasoning that
hyperparameter tuning was always going to follow regardless of how this
first, unweighted attempt looked.

### Adversarial evaluation, round 1: merged distance

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
overlap) - the Method B distance was not simply noise. But its
adversarial-vs-control separation ratio was consistently a little behind
plain pixel-Euclidean's, on both models, at every epsilon (roughly
7.0-7.4x vs. pixel distance's 8.5-9.1x on the smaller model; 3.3-4.6x vs.
3.6-5.0x on the larger one). Both distances also registered the expected
capacity difference between the two models. The highest-ratio pairs under
the merged distance were mostly cases where the prediction hadn't actually
flipped (large margin swing, still correctly classified) rather than
misclassifications.

Two independent checks - the sanity check above and this evaluation - now
pointed the same direction: the merged distance behaved sensibly but
hadn't yet beaten the pixel-space baseline it was meant to improve on.

### Diagnostic: is concatenation diluting a real per-line signal?

The concept for this method had originally been described in terms of
comparing images line by line - asking whether two images are close *on
each individual path* - rather than merging all paths into one score
first. The pipeline as built did the opposite. A same/different-digit AUC
diagnostic tested this directly: computed per individual line and
compared against the merged distance, on the same 300-image sample, all
44,850 unique pairs.

One line narrowly beat the merged distance (AUC 0.6394 vs. 0.6315) - a
real but small margin, and 13 of the 16 individual lines scored *below*
the merged distance, so this wasn't broad evidence that concatenation was
drowning out most lines' signal. It was, however, clear evidence of one
specific inefficiency: 4 of the 16 lines - the ones running exactly along
the image border (row or column 0 or 27) - sat at exactly AUC 0.5000,
chance level, carrying zero signal. MNIST digits essentially never reach
the image edge, so these lines run through background regardless of the
image and contribute pure noise to the merged vector. Horizontal lines
dominated the top of the ranking generally.

### The pivot: per-path evaluation

Rather than merging the 16 lines into any single score - not the 496-dim
concatenation, not a max or weighted sum - each line was instead treated
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
same images - fold-ratios of roughly 4-11x on average across lines, up to
~20x for the single best line. This is a substantially clearer separation
than the merged distance's own 7.0-7.4x / 3.3-4.6x. The number of
genuinely flipped pairs was small at these conservative epsilons (4-14 out
of 200), so exact magnitudes should be read as indicative rather than
precise - but the direction, with zero exceptions across all 72
comparisons, was the trustworthy part.

**Result 2.** Looking at which line shows the largest ratio for each pair,
the distribution of "which line spikes" was more concentrated (lower
entropy) under adversarial perturbation than under the random control, in
all 6 of 6 model/epsilon combinations - modest (roughly 0.2-0.3 bits out
of a 3.585-bit maximum) but completely consistent. This is evidence,
though not dramatic, that an adversarial perturbation concentrates on
fewer paths than an equally large random shift, even though it touches
every pixel.

**A confound, checked rather than assumed away.** Two lines dominated the
raw "which line spikes" counts under *both* adversarial and control
conditions, because they had systematically smaller baseline signature
distances than the other ten lines, regardless of perturbation type - a
smaller denominator mechanically inflates a ratio independent of whether
anything adversarially meaningful happened. This meant raw spike counts
were not a clean read on their own; the entropy comparison above, being a
within-condition comparison, was the more trustworthy summary.

**Visual check on two example pairs** (original image, perturbed image,
and the single highest-ratio line highlighted) gave a mixed picture on
whether the spiking line visually crosses the digit's stroke: one example
did, one mostly ran through background near the stroke. Two examples were
not enough to establish a clean pattern either way.

### Robustness check on the per-path finding

Because two lines had already been flagged as a possible scale confound,
the fold-ratio finding above (adversarial ratio exceeding control ratio on
every line, every model, every epsilon) was checked directly against
their exclusion, using the same already-computed evaluation data. Removing
the two lines left 60 of the original 72 comparisons; all 60 still showed
the same pattern, with zero exceptions. The mean fold-ratio moved by at
most about 0.5x on the 4-11x scale, in no consistent direction - the two
flagged lines were not doing meaningful work for the aggregate result. A
Pearson correlation between each remaining line's own baseline distance
and its fold-ratio was small and inconsistent in sign across models and
epsilons (|r| <= 0.32), unlike the two excluded lines' much cleaner
pattern (roughly half the distance of the rest, dominating spike counts
under both conditions) - the scale confound looked fairly well isolated to
those two specific lines, not a systemic property of the whole set.

### Hyperparameter sweep

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
was not the best choice** - mean best-line AUC decreased monotonically
from depth 2 (0.6585) to depth 6 (0.6320), a consistent trend rather than
noise. **Geometry mattered too**: shifting weight toward horizontal lines
helped (12 horizontal + 4 vertical: 0.6592 mean, vs. the original 8+8
split's 0.6492), and an all-vertical split was clearly worst (0.6219).
**Points per line and interpolation method barely mattered**, plateauing
by 32 points and showing no meaningful difference between straight-line
and cubic-spline joins. The single best configuration found - 12
horizontal + 4 vertical lines at depth 2 - reached AUC 0.6811 against the
original configuration's 0.6469, and this wasn't a single fortunate row:
the entire top cluster of the ranking used this geometry-and-depth
combination across different point counts and interpolation choices.

The top candidates from this screen were then validated with the full
per-path adversarial evaluation (both models trained once and reused
across every candidate, since training doesn't depend on Method B's
configuration). The winning configuration (12 horizontal + 4 vertical
lines, depth 2) reached a mean fold-ratio of roughly 14.8-15.0x against
the original configuration's 8.4x - a substantial improvement, and one
that agreed with the independent cheap screen rather than contradicting
it. It was not perfectly exception-free at the individual-line level (one
of 72 line/model/epsilon combinations showed the reverse direction,
isolated to the larger model's smallest-sample condition, consistent with
sample-size noise rather than a new pattern), and a second candidate
(all-horizontal-plus-a-few-vertical variants aside, specifically the
16-horizontal/0-vertical split at depth 2) was competitive rather than a
distant runner-up - it actually outperformed the winner on the larger
model specifically while losing on the smaller one, a genuinely mixed
result on that comparison. A robustness check on the winning configuration
(same method as above) found no lines behaving like the earlier scale
confound; baseline distances across its 12 informative lines spanned a
narrow range (about 1.4x top to bottom, compared to the earlier ~2x gap),
and excluding its two smallest-distance lines moved the mean fold-ratio
from 14.8x to 12.6x while the pattern remained intact.

Points-per-line showed the weakest effect of any axis in this sweep, which
is the main signal it offers for choosing a comparable parameter - a
segment or step length - in any related construction built afterward: no
axis here supports a case for a long segment length, and geometry- and
depth-style choices look like the higher-leverage places to spend tuning
effort by comparison.

### Current status

**12 horizontal + 4 vertical lines at truncation depth 2 is the adopted
Method B default**, replacing the original 8+8 split at depth 4. This was
confirmed by two independent evaluations - a cheap AUC-based screen and
the full per-path adversarial evaluation - and checked for robustness
against depending on a small number of particular lines.

Method B behaves sensibly as a Lipschitz-ratio denominator, is sensitive
to model capacity, and - once compared path by path rather than merged -
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

## Method C: Hilbert-Curve Signature Distance

### Overview

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

### Construction

**Curve generation.** The standard index-to-coordinate Hilbert-curve
algorithm produces, for order 5, a sequence of 1024 grid cells covering
every cell of a 32x32 grid exactly once. This was checked directly before
anything was built on top of it: every cell is visited exactly once, every
coordinate stays in bounds, and - the property that matters most for what
follows - every consecutive pair of cells is exactly one grid unit apart,
always axis-aligned (never diagonal).

**Scaling and resampling.** Grid coordinates are scaled by `28/32` into
the image domain, then the full 1024-point curve is resampled to exactly
512 points evenly spaced along its arc length (cumulative-length
parameterization plus linear interpolation between the original vertices).
This is a genuine arc-length resampling, not a shortcut: an earlier
assumption that it would reduce to simple index subsampling (since
1024/512 = 2 and every original step is the same length) turned out to be
wrong, and a test caught it - arc-length-even sampling of a *bending*
path is not the same as evenly-spaced-by-index sampling once the path
changes direction between two consecutive output points, even when every
underlying step is the same length. The resampling logic was pulled out
and re-verified in isolation on a simple, hand-computable L-shaped path
before trusting it on the actual curve.

One consequence of the scaling worth stating plainly rather than glossing
over: the curve's farthest point lands at coordinate 27.125, just past the
last valid pixel index (27), not strictly within it as a tidier-sounding
description might suggest. This was caught by a test that checked the
actual bound rather than an assumed one. In practice this is harmless -
the same bilinear sampling technique Method B's lines use reads intensity
with border-clamping, so this one slightly-out-of-range point just reads
the border pixel rather than erroring or extrapolating.

**Segments.** The 512-point sequence is cut into 16 contiguous blocks of
32 points, each with its own `[t, intensity]` time coordinate (t running
0 to 1 within that segment, the same convention used throughout this
project). The curve is entirely fixed and deterministic - unlike Method
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
diagnostic - applied to Hilbert segments instead of reference lines.

### Stage A: depth mini-sweep

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

Depth 3 wins, but narrowly - about 0.008 higher best-segment AUC than
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
sense - a space-filling curve doesn't have an equivalent to a straight
line placed exactly on the image border; every segment covers a
genuinely different, non-degenerate arc of the image regardless of where
it falls in the traversal order.

### Stage B: full validation and head-to-head comparison

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
subset), on the same 200-image sample, same models, same epsilons - a
direct, fair comparison.

**Method C does not beat Method B on the primary magnitude measure.**
Method B's winning configuration separates genuinely adversarial pairs
from random-noise controls by roughly 13.5x on average; Method C reaches
roughly 9.9x - meaningfully lower, not a close call. Per epsilon and
model, Method C's fold-ratios (13.4x, 11.5x, 11.9x for SmallCNN; 12.7x,
5.2x, 4.6x for StrongCNN) follow the same general shape as Method B's
own numbers (larger on the smaller model, smaller at higher epsilon on
the larger model) but sit lower throughout - this is a consistent gap
across every model/epsilon combination, not a mixed result that favors
Method C anywhere.

**Where Method C does edge ahead**: it was completely exception-free
across all 96 line/segment x model x epsilon combinations, where Method
B's winner had one (on StrongCNN's smallest-sample condition, already
understood as sample-size noise rather than a real reversal). This is a
minor point in Method C's favor, not a substantial one - one exception
out of 72 wasn't treated as concerning for Method B either.

**Robustness check** (same approach as Method B's): the two segments with
the smallest baseline signature distance were identified and excluded;
the mean fold-ratio moved by less than 1x in every case (e.g. 13.42x to
12.89x, 5.21x to 5.09x) and the pattern held throughout - Method C's
result isn't resting on a small number of unusually sensitive segments
either.

### Verdict

Method C works - it produces a coherent, deterministic, per-path
Lipschitz-ratio evaluation with a completely exception-free
adversarial-vs-control separation, using the same infrastructure and
methodology validated for Method B. But on the metric that actually
matters for this project (how strongly the distance separates genuinely
adversarial perturbations from equally-sized random ones), it trails
Method B's current winner by a consistent, non-trivial margin across
every model and epsilon tested, not just on average. The honest
conclusion is that Method C should be kept as a documented alternative -
the construction is sound, correctly verified, and a legitimate
comparison point - rather than adopted as a replacement for Method B's
reference-line approach. Method B's straight, independently-placed lines
currently do a better job at the actual task than one continuous curve
does, at the same total sampling budget.

### Files

- `hilbert_stream.py` - curve generation, arc-length resampling, segment
  construction, Stage A depth screen, Stage B evaluation driver, and the
  robustness check, all in one module.
- `tests/test_hilbert_stream.py` - 17 tests, including the corrected
  arc-length resampling check (isolated on a hand-computable synthetic
  path) and the same depth-prefix-shortcut correctness check used for
  Method B's sweep.
- `artifacts/hilbert_curve_seed0.npy` - the fixed 512-point curve.

## Method A: Adversarial / Lipschitz-Ratio Evaluation Results

`method_a_adversarial_eval.py` (standalone, no imports from `toy_lipschitz`/
`mnist_lipschitz`; reuses `method_b_adversarial_eval.py`'s model/training/
attack infrastructure unmodified). FGSM, two epsilon ranges tested in
`notebook_method_a_adversarial_eval.ipynb`: the original imperceptible range
{0.02, 0.03, 0.05} (Findings 1-5 below, directly comparable to Method B's
own evaluation), and a much larger sweep {0.05, 0.1, 0.15, 0.2, 0.25, 0.3}
run afterward to see how the picture changes well past the imperceptible
regime (Findings 6-9). 200 test images (20/class) in both cases, two
freshly-trained classifiers per run (SmallCNN test_acc 98.26%, StrongCNN
test_acc 99.34% - both retrained fresh here with the same seed/recipe as
`method_b_adversarial_eval.py`'s run, not reused from it, but statistically
indistinguishable from its documented 98.24%/99.36%: training is independent
of which distance denominator is evaluated downstream; the larger-epsilon
sweep retrains fresh again with the same recipe, so its own reported
accuracies match too). Method A distance uses the fixed r~1.656 from the
Phase 4 sanity check, unmodified in both runs.

Note: every Method B number cited below (Finding 3's comparison table) uses
Method B's original 8h+8v/depth-4 configuration, which predates
`method_b_sweep.py`'s later adoption of a 12h+4v/depth-2 default (see
"Method B: Reference-Line Signature Distance" above) - that default reaches
a substantially higher per-path fold-ratio (~14.8-15x) than the original
configuration's numbers used here. Finding 3's "narrow, inconsistent edge"
for Method A almost certainly would not survive a re-comparison against the
new default; not re-run here.

### Headline numbers

| | eps | flip_frac | pixel ratio (adv) | pixel ratio (ctrl) | pixel adv/ctrl | Method A ratio (adv) | Method A ratio (ctrl) | Method A adv/ctrl |
|---|---|---|---|---|---|---|---|---|
| SmallCNN  | 0.02 | 0.035 | 2.274 | 0.249 | **9.14** | 1.782 | 0.230 | **7.75** |
| SmallCNN  | 0.03 | 0.040 | 2.257 | 0.266 | **8.47** | 1.744 | 0.248 | **7.04** |
| SmallCNN  | 0.05 | 0.065 | 2.294 | 0.273 | **8.41** | 1.775 | 0.239 | **7.42** |
| StrongCNN | 0.02 | 0.020 | 2.759 | 0.551 | **5.00** | 1.996 | 0.471 | **4.24** |
| StrongCNN | 0.03 | 0.035 | 2.808 | 0.654 | **4.29** | 1.995 | 0.526 | **3.80** |
| StrongCNN | 0.05 | 0.045 | 2.953 | 0.812 | **3.64** | 2.114 | 0.726 | **2.91** |

"ctrl" = random (non-gradient-directed) noise, matched in L2 pixel-norm to
the FGSM perturbation for the same image - same convention as Method B's
own adversarial evaluation (see "Method B: Reference-Line Signature
Distance" above, "Adversarial evaluation, round 1" section).

### Finding 1: both distances separate adversarial from random-noise control, clearly

Both the pixel and Method A ratios are bimodal, well-separated distributions
across adversarial vs. control pairs, for both models (see the histogram
cell in `notebook_method_a_adversarial_eval.ipynb` - minimal overlap in
every case). Method A is not just noise: it distinguishes a
gradient-directed perturbation from an equally-sized undirected one, same as
Method B.

### Finding 2: Method A's separation is real but consistently slightly weaker than plain pixel distance

Look at the "adv/ctrl" columns: **pixel-Euclidean's separation ratio is
larger than Method A's in every one of the 6 rows above** (SmallCNN:
8.4-9.1x vs. 7.0-7.8x; StrongCNN: 3.6-5.0x vs. 2.9-4.2x). Same direction,
same consistency, as Method B's finding and the Phase 4 sanity check - a
third independent test now pointing the same way for Method A specifically.

### Finding 3: Method A vs. Method B, directly (not just architecture-level)

Because both modules train `SmallCNN`/`StrongCNN` with the identical
seed/recipe, and training is independent of which distance is evaluated
against the trained model afterward, the two runs' models end up
statistically indistinguishable (test_acc within 0.02pp, flip fractions
matching almost exactly - e.g. SmallCNN eps=0.03: `flip_frac=0.040` in both
runs). That makes a direct, same-numerator comparison meaningful, not just
an architecture-level one:

| model | eps | pixel a/c | Method A a/c | Method B a/c |
|---|---|---|---|---|
| SmallCNN | 0.02 | 9.14 | **7.75** | 7.42 |
| SmallCNN | 0.03 | 8.47 | **7.04** | 7.04 |
| SmallCNN | 0.05 | 8.41 | **7.42** | 7.20 |
| StrongCNN | 0.02 | 5.00 | **4.24** | 4.55 |
| StrongCNN | 0.03 | 4.29 | **3.80** | 3.76 |
| StrongCNN | 0.05 | 3.64 | **2.91** | 3.25 |

Method A's adv/ctrl ratio is higher than Method B's on all 3 SmallCNN
epsilons and 1 of 3 StrongCNN epsilons (4/6 rows) - a narrow, inconsistent
edge, not a clean win for either method. Both sit in the same ballpark,
both consistently below the pixel-Euclidean baseline. Nothing here suggests
one method's signature distance is a categorically better adversarial-shift
denominator than the other's.

### Finding 4: capacity contrast is visible in both distances

StrongCNN is harder to fool at these epsilons (flip_frac 0.020-0.045 vs.
SmallCNN's 0.035-0.065) - expected, the higher-capacity, higher-accuracy
model. Both distances register the capacity difference: StrongCNN's ratios
(pixel and Method A) are consistently higher in absolute terms than
SmallCNN's at matched epsilon. Same pattern documented for Method B.

### Finding 5: top-10 highest-Method-A-ratio pairs - same shape as Method B, different specific pair

At eps=0.03: SmallCNN's top 10 has **one** actual flip (`2 -> 7`); StrongCNN's
top 10 has **zero** - matching Method B's documented shape (1 flip for
SmallCNN, 0 for StrongCNN) exactly, even though the *specific* flipped pair
differs (Method B's was `6 -> 4`) - expected, since the two runs' distance
rankings differ even when the underlying trained models and adversarial
images are nearly identical. Neither this run's flip (`2 -> 7`) nor Method
B's (`6 -> 4`) matches Experiment 2's documented CNN confusable pairs (6/5,
8/2, 8/0) - same honest non-overlap as Method B's run, not glossed over.

### Larger-epsilon sweep (0.05-0.3): beyond the imperceptible regime

Same protocol, same models retrained fresh with the identical seed/recipe,
now run at epsilons {0.05, 0.1, 0.15, 0.2, 0.25, 0.3} - at eps=0.3 a
perturbation can shift any pixel by up to 30% of the full intensity range,
no longer imperceptible (confirmed directly in the notebook's example
gallery at eps=0.3). No Method B comparison here - Method B has only ever
been evaluated at 0.02/0.03/0.05, so there is nothing documented to compare
against at these epsilons.

#### Headline numbers: larger-epsilon sweep

| | eps | flip_frac | pixel ratio (adv) | pixel ratio (ctrl) | pixel adv/ctrl | Method A ratio (adv) | Method A ratio (ctrl) | Method A adv/ctrl |
|---|---|---|---|---|---|---|---|---|
| SmallCNN  | 0.05 | 0.065 | 2.294 | 0.266 | **8.63** | 1.775 | 0.219 | **8.11** |
| SmallCNN  | 0.10 | 0.190 | 2.319 | 0.307 | **7.56** | 1.850 | 0.268 | **6.89** |
| SmallCNN  | 0.15 | 0.495 | 2.349 | 0.362 | **6.48** | 1.954 | 0.311 | **6.29** |
| SmallCNN  | 0.20 | 0.790 | 2.308 | 0.412 | **5.60** | 2.022 | 0.366 | **5.53** |
| SmallCNN  | 0.25 | 0.950 | 2.233 | 0.451 | **4.95** | 2.067 | 0.436 | **4.74** |
| SmallCNN  | 0.30 | 0.995 | 2.147 | 0.515 | **4.17** | 2.104 | 0.511 | **4.12** |
| StrongCNN | 0.05 | 0.045 | 2.953 | 0.795 | **3.71** | 2.114 | 0.699 | **3.02** |
| StrongCNN | 0.10 | 0.170 | 3.033 | 1.107 | **2.74** | 2.313 | 1.002 | **2.31** |
| StrongCNN | 0.15 | 0.355 | 2.718 | 1.277 | **2.13** | 2.192 | 1.165 | **1.88** |
| StrongCNN | 0.20 | 0.585 | 2.383 | 1.273 | **1.87** | 2.039 | 1.162 | **1.75** |
| StrongCNN | 0.25 | 0.740 | 2.090 | 1.312 | **1.59** | 1.901 | 1.271 | **1.50** |
| StrongCNN | 0.30 | 0.850 | 1.838 | 1.254 | **1.47** | 1.781 | 1.202 | **1.48** |

(Note the eps=0.05 row here differs slightly from Finding 1-5's table above
in the `ctrl` columns and adv/ctrl ratio - `flip_fraction` and `ratio(adv)`
match exactly (both depend only on the model/FGSM, unaffected by epsilon
ordering), but `random_noise_perturbation`'s generator is seeded once per
`run_adversarial_evaluation` call and drawn from sequentially across the
epsilon loop, so eps=0.05 being the 3rd epsilon in Finding 1-5's tuple
`(0.02, 0.03, 0.05)` vs. the 1st in this sweep's `(0.05, 0.1, ...)` draws a
different random control each time even with the same seed; the effect on
the adv/ctrl ratio is small, 8.41 vs. 8.63 for SmallCNN pixel.)

#### Finding 6: flip fraction saturates fast

SmallCNN goes from 6.5% flipped at eps=0.05 to 99.5% at eps=0.3; StrongCNN
goes from 4.5% to 85.0%. StrongCNN stays consistently harder to fool at
every epsilon in the sweep - the same capacity contrast as Finding 4, now
visible across a much wider range.

#### Finding 7: adv/control separation collapses monotonically as epsilon grows, for both distances, on both models

Every ratio falls monotonically as epsilon increases: SmallCNN pixel a/c
8.63->7.56->6.48->5.60->4.95->4.17; Method A a/c 8.11->6.89->6.29->5.53->4.74->4.12.
Same monotonic collapse for StrongCNN (pixel 3.71->...->1.47; Method A
3.02->...->1.48). This tracks the flip fraction directly: once the
perturbation is large enough that an equally-sized *random* shift starts
flipping predictions on its own, the control stops being a clean negative,
and the adversarial-vs-random distinction both distances are measuring
erodes.

#### Finding 8: the pixel-vs-Method-A gap narrows sharply as epsilon grows - and briefly flips

At eps=0.05 pixel still leads clearly (SmallCNN +0.52, StrongCNN +0.69). By
eps=0.3 the SmallCNN gap has shrunk to near-zero (4.17 vs. 4.12), and for
StrongCNN specifically **Method A's ratio (1.48) narrowly exceeds pixel's
(1.47)** - the first time in this entire project (Phase 4, Method B's
evaluation, or Method A's own small-epsilon section above) that a signature
distance has matched or beaten the pixel-Euclidean baseline on this
comparison. Flagged plainly, not overclaimed: a 0.01 gap on a ~1.5 scale
from 200 images is well within sampling noise, and it occurs at the one
condition (85% flip rate) where the comparison is least informative to
begin with - both ratios are being compressed toward each other by the same
saturation effect documented in Finding 7, not obviously a genuine
capability edge for Method A.

#### Finding 9: the top-10 highest-ratio pairs flip from mostly-correct to almost-all-flips

At eps=0.3, the top-10 highest-Method-A-ratio pairs are 9/10 genuine flips
for SmallCNN and 9/10 for StrongCNN - a complete reversal from Finding 5's
small-epsilon result (at most 1/10 flips there). Expected given Finding 6's
flip fractions (85-99.5% at this epsilon): the highest-ratio examples are no
longer "large margin swing, still correct" but genuinely misclassified
images.

### Verdict

Method A behaves sensibly as a Lipschitz-ratio denominator - it separates
genuinely adversarial shifts from equally-sized random ones, and it's
sensitive to model capacity - but like Method B, it sits slightly *below*
plain pixel-Euclidean distance at imperceptible epsilons, not above it.
Directly compared against Method B (same trained-model behaviour, same
numerator), Method A is neither clearly better nor clearly worse - a narrow
4/6 edge in adv/ctrl ratio that doesn't survive as a consistent pattern
across both models. Three independent tests now (Phase 4, Method B's
adversarial eval, this one) all land in the same place for both methods at
small epsilons: sensible, capacity-aware, but not yet beating the
pixel-space baseline either is meant to improve on.

The larger-epsilon sweep adds one more honest data point rather than
overturning that picture: pixel-Euclidean's advantage over Method A erodes
steadily and nearly vanishes by eps=0.3, but that convergence tracks the
control condition becoming uninformative as flip fraction saturates
(Finding 7), not clear evidence Method A becomes the *better* denominator at
large epsilon. The one StrongCNN crossover (Finding 8) is noted as an honest
observation, not a result to build on without a larger sample. Per the
project's "move forward regardless, but carry the finding" convention, both
the small- and large-epsilon results are carried forward as-is, not treated
as a stop condition and not smoothed into a single tidier narrative.

## Level-wise decomposition diagnostic

This additive, read-only diagnostic (`level_decomposition.py`) was added to
answer one question directly, without changing any existing distance,
stream, or signature behaviour and without training any model (it's a
label-based check on the same Phase 4 sample): is the within- vs.
cross-digit signal in Method A's (and Method B's) signature distance carried
entirely by the level-1 terms, or do the higher-order levels contribute?

The motivating concern was specific to Method A's stream shape:
`signature_of_stream`'s output for `width=2, depth=4` has 31 dimensions,
where index 0 is always the constant term `1.0` and indices `1:3` are always
the path's exact net displacement (`stream[-1] - stream[0]`), regardless of
depth or intermediate path shape. Method A's stream is `(t, sigma1)`, and its
time channel is `arange(n)/(n-1)`, so `delta_t = 1.0` identically for every
image and contributes exactly zero to any pairwise distance - meaning Method
A's entire level-1 content reduces to a single scalar,
`delta_sigma1 = sigma1(anchor 63) - sigma1(anchor 0)`, the difference of two
3x3 patch singular values out of 64 anchors. If the distance turned out to
be level-1 dominated, Method A's Phase 4 ratio of 1.176 would be produced by
that one scalar alone, with the signature machinery doing no real work -
worth checking before investing in any stream-construction changes.

**Harness.** `level_slices(depth, width=2)` maps each signature level to its
index block (`{0: slice(0,1), 1: slice(1,1+width), ...}`, block sizes
`1, 2, 4, 8, 16` summing to 31 for `width=2, depth=4`).
`mask_signature_levels(sig, levels, depth, width=2)` zeroes out every level
*not* in the requested set - zeroing rather than slicing, so the full 31-dim
(and Method B's 496-dim concatenated) layout is preserved and the existing
distance functions can be reused unmodified, including on Method B's
`(N, 16, 31)` shape before concatenation. Both utilities were checked before
use: `level_slices` was verified to partition `range(0, 31)` exactly with no
gaps or overlap; an orthogonality identity (the sum of squared per-level
distances equals the total squared distance, since the level blocks are
disjoint coordinates - a Pythagorean check) was verified on random signature
vectors; and a closed-form check confirmed that for a straight-line path,
the level-1-masked distance between two signatures equals the exact
Euclidean distance between their net displacements.

`run_level_decomposition(n_per_class=30, seed=0, depth=4)` then reproduced
the existing Phase 4 protocol exactly so the numbers would be comparable:
load the 300-image pool, build streams and signatures for both methods
unmodified, derive `r` independently per method via `choose_rescale_factor`
and apply `rescale_signature` *before* any masking (never re-derived per
level variant - only which levels survive changes across variants), then
build the feature vector via the existing `method_a_feature_vector`/
`method_b_feature_vector` and call `within_vs_cross_digit_distance`, for six
variants: `all` (levels 1..depth), `level1_only`, `level2plus` (2..depth),
`level2_only`, `level3_only`, `level4_only`. The mean fraction of total
squared pairwise distance contributed by each level was also recorded per
method and per level, since a level can carry a meaningful ratio while
contributing almost nothing to the distance magnitude. As a hard gate before
trusting any of this, the `all` variant had to reproduce the already-
documented Phase 4 numbers (Method A: r=1.656, within 14.60, cross 17.18,
ratio 1.176; Method B: r=2.860, within 28.60, cross 33.17, ratio 1.160) to
floating-point tolerance - it did. A second closed-form gate confirmed
Method A's `level1_only` pairwise distance equals `r * |delta_sigma1_i -
delta_sigma1_j|` exactly (the `delta_t` component cancels identically across
every pair), with the time-channel endpoints checked as identical across all
images as part of the same test.

### Variant x method table

Ratio is cross-digit mean / within-digit mean; `all` reproduces the
documented Phase 4 table exactly (the reproduction gate above).

#### Method A (r ~ 1.656)

| variant | levels kept | within-digit mean | cross-digit mean | ratio |
|---|---|---|---|---|
| all | 1,2,3,4 | 14.60 | 17.18 | 1.176 |
| level1_only | 1 | 1.50 | 1.91 | 1.276 |
| level2plus | 2,3,4 | 14.50 | 17.05 | 1.175 |
| level2_only | 2 | 4.54 | 5.58 | 1.228 |
| level3_only | 3 | 7.90 | 9.38 | 1.187 |
| level4_only | 4 | 11.18 | 12.98 | 1.162 |

#### Method B (r ~ 2.860)

Note: Method B's numbers here (r=2.860, depth=4) use its original
8h+8v/depth-4 configuration, which predates `method_b_sweep.py`'s later
adoption of a 12h+4v/depth-2 default (see "Method B: Reference-Line
Signature Distance" above) - still exactly reproducible as documented (the
original config remains `make_reference_lines`'s own default), just not
re-run against the new one.

| variant | levels kept | within-digit mean | cross-digit mean | ratio |
|---|---|---|---|---|
| all | 1,2,3,4 | 28.60 | 33.17 | 1.160 |
| level1_only | 1 | 0.0229 | 0.0229 | 1.000 |
| level2plus | 2,3,4 | 28.60 | 33.17 | 1.160 |
| level2_only | 2 | 4.66 | 5.68 | 1.219 |
| level3_only | 3 | 13.08 | 15.39 | 1.177 |
| level4_only | 4 | 24.99 | 28.81 | 1.153 |

### Per-level squared-distance contribution

Mean fraction of total squared pairwise distance contributed by each level,
averaged over all 44,850 unique pairs in the sample (levels 1..4 sum to
1.0 - level 0 is the constant `1.0` term, identical for every image, and so
contributes exactly zero to any pairwise distance).

| level | Method A fraction | Method B fraction |
|---|---|---|
| 1 | 1.79% | 0.0031% |
| 2 | 11.35% | 2.99% |
| 3 | 28.85% | 21.60% |
| 4 | 58.01% | 75.41% |

### Verdict: (b) - the higher-order signature terms carry genuine information, not level-1

For both methods, `level2plus`'s ratio (A 1.175, B 1.160) is essentially
identical to `all`'s (A 1.176, B 1.160) - dropping level 1 entirely costs
almost nothing. That alone would already point to (b) over (a). But the
per-level fraction table makes the finding sharper than "higher levels
matter": **level 1 is nearly inert for both methods**, not just
non-dominant. For Method A it carries only 1.79% of total squared distance;
for Method B it carries 0.003% - three orders of magnitude smaller than
level 4. The reason is structural for Method B specifically: every reference
line runs the full width/height of the image, so both of its endpoints sit
on the image border, where MNIST pixel intensity is ~0 for nearly every
image (99.96% of the 4,800 per-line net-displacement values in this sample
are *exactly* 0.0, not just small) - Method B's level-1 term, the path's net
displacement, is measuring a quantity that's degenerate by construction, not
one that merely turned out to be weak. `level1_only`'s ratio for Method B is
exactly 1.000 to 3 decimals: no within/cross separation at all.

A second, non-obvious point the fraction table alone would miss: **ratio and
magnitude rank levels in opposite orders**. `level1_only` has the *highest*
individual-level ratio for Method A (1.276, above even `all`'s 1.176)
despite carrying almost none of the distance magnitude (1.79%) - a small,
noisy-looking quantity that happens to separate classes relatively well per
unit of scale. Conversely `level4_only`, which carries the most magnitude
for both methods (58%/75%), has the *lowest* individual-level ratio of any
level (1.162 A, 1.153 B) - bulk signal that's comparatively unfocused
class-wise. Level 2 alone lands closest to `level1_only`'s per-unit
separation-vs-magnitude tradeoff (ratio 1.228 A / 1.219 B on 11%/3% of the
magnitude) without level 1's degeneracy problem.

Net conclusion: the signature machinery is doing real work. Method A's
Phase 4 ratio of 1.176 is not a relabelled version of the single
`delta_sigma1 = sigma1(anchor 63) - sigma1(anchor 0)` scalar that motivated
this check - that scalar (`level1_only`) contributes under 2% of the
distance's magnitude and is not what's driving the `all` result;
`level2plus` alone reproduces `all` almost exactly. Method B's case is even
more direct: its level-1 term is close to degenerate by construction
(border-anchored lines against a near-black background), so essentially all
of its Phase 4 signal already had to come from levels 2-4. This is carried
forward as the honest finding, per this project's convention - it doesn't
change anything about Phase 4's already-documented modest effect size
relative to the pixel-space Euclidean baseline, and doesn't by itself
motivate a scope change (e.g. to line placement) beyond what "Method B:
Reference-Line Signature Distance" above's per-line AUC diagnostic
(border-line finding) already flagged independently.

## Results

Everything in this section through the level-decomposition sub-experiment
uses Method B's **original 8h+8v/depth-4 configuration** (predates the
sweep below) unless stated otherwise - still exactly reproducible, since
that configuration remains `make_reference_lines`'s own default, just no
longer the recommended one for new Method B work. See "Method B
hyperparameter sweep and Method C" further down for the current default and
what replaced it.

**Phase 1 - stream construction shapes and timing** (1000-image default
pool, `n_per_class=100, seed=0`, Apple Silicon CPU, no GPU): pool loading
~0.007s; Method A stream construction ~0.018s; Method B stream construction
(via batched `grid_sample`) ~0.001s - both effectively instantaneous. Full
detail (former `PHASE1_SUMMARY.md`, now folded in): see "Method B:
Reference-Line Signature Distance" above, "Design" section.

**Phase 2/3 - signature + distance (checkpoint-gated before use).**
`signature_of_stream` matches the exact closed-form tensor exponential for a
straight-line path (every level, not just level 1) and a hand-computed area
term for an L-shaped path, before being trusted on real Method A/B streams.
Raw signature magnitudes decay ~4-50x from level 1 to level 4; `r` (derived
independently per method) is `r_A ~ 1.656`, `r_B ~ 2.860` at the Phase 4
sample settings.

**Phase 4 - within- vs. cross-digit sanity check** (300 images, 30/class,
seed 0, depth 4):

| method | r | within-digit mean | cross-digit mean | cross/within ratio |
|---|---|---|---|---|
| Method A | 1.656 | 14.60 | 17.18 | 1.176 |
| Method B | 2.860 | 28.60 | 33.17 | 1.160 |

Both show the right direction (within < cross) with or without rescaling -
rescaling barely moved either ratio. But the effect size is comparable to
plain raw-pixel Euclidean's own weak ratio in `mnist_lipschitz` (~1.13):
**neither method yet shows a stronger class-separation signal than the
pixel-space baseline it's meant to improve on.** Per the checkpoint-gating
rule, this didn't block moving forward, but it's an honest, still-open
finding (full detail and a spatial-coherence caveat on Method A's ordering:
see "Implementation plan and design constraints" above).

**Method B adversarial/Lipschitz-ratio evaluation** (FGSM, eps in
{0.02, 0.03, 0.05}, 200 images/20 per class, fresh `SmallCNN`
test_acc 98.24% / `StrongCNN` test_acc 99.36%, magnitude-matched random-noise
control): both pixel-Euclidean and Method B distances separate adversarial
from control shifts clearly (near-bimodal ratio histograms), so Method B is
not just noise. But **pixel-Euclidean's adv/control separation ratio is
larger than Method B's in every one of the 6 model x epsilon combinations**
(SmallCNN 8.5-9.1x vs. 7.0-7.4x; StrongCNN 3.6-5.0x vs. 3.3-4.6x) - the same
"sensible but not yet better than the baseline" pattern as Phase 4, now a
second independent data point. Both distances register the capacity contrast
between the two models. Full table and the top-10-ratio-pairs check (former
`adversarial_eval_summary.md`, now folded in): see "Method B: Reference-Line
Signature Distance" above, "Adversarial evaluation, round 1" section - this
is the *merged*-distance result, superseded in strength (not correctness) by
the per-path pivot documented in the same section.

**Method A adversarial/Lipschitz-ratio evaluation** (same protocol,
retrained fresh with the same seed/recipe - trained models statistically
indistinguishable from Method B's run: `SmallCNN` test_acc 98.26% /
`StrongCNN` test_acc 99.34%, matching flip fractions almost exactly): same
result as Method B - both distances clearly separate adversarial from
control shifts, but **pixel-Euclidean's adv/control ratio beats Method A's
in every one of the 6 combinations too** (SmallCNN 8.4-9.1x vs. 7.0-7.8x;
StrongCNN 3.6-5.0x vs. 2.9-4.2x) - a third independent test landing in the
same place. Compared directly against Method B (valid here since the
trained models/numerator match, and both still use Method B's original
config): Method A's adv/control ratio edges out Method B's on 4 of 6
model x epsilon combinations, but narrowly and inconsistently - neither
method is a clearly better denominator than the other. This comparison
hasn't been re-run against Method B's new swept default (12h+4v, depth 2,
~14.8-15x mean fold-ratio per-path) - doing so would very likely favor
Method B by a wide margin, since that default already beats Method B's own
original-config numbers substantially; noted here rather than left
implicit. Full table and the top-10-ratio-pairs check: see "Method A:
Adversarial / Lipschitz-Ratio Evaluation Results" above.

### Method B hyperparameter sweep and Method C (not part of this session's work - summarized for context)

`method_b_sweep.py` (Stage 8) found Method B's original defaults were not
optimal: a 160-configuration cheap AUC screen plus full per-path validation
on finalists found depth is the dominant lever (lower is better, contrary
to the original depth=4 default) and 12 horizontal + 4 vertical lines beats
the original 8h+8v split. **12h+4v at depth 2 is now the adopted Method B
default**, reaching a mean per-path fold-ratio of ~14.8-15x against the
original configuration's ~8.4x - a substantial, cross-validated improvement.
Separately, **Method C** (`hilbert_stream.py`) tests a single Hilbert-curve
traversal instead of 16 independent lines, at the same total point budget;
it works sensibly (exception-free adversarial/control separation across all
16 segments) but trails Method B's swept winner by a consistent margin
(9.88x vs. 13.53x mean fold-ratio on the same fair, all-16-paths
comparison) - kept as a documented alternative, not adopted as a
replacement. Full detail for both: see "Method B: Reference-Line Signature
Distance" and "Method C: Hilbert-Curve Signature Distance" above.

### Sub-experiment: Method A larger-epsilon sweep (verdict: pixel's edge erodes and briefly flips as epsilon grows, tracking control saturation, not a real capability edge)

Same `method_a_adversarial_eval.py` protocol, retrained fresh again with the
same seed/recipe, now run at epsilons {0.05, 0.1, 0.15, 0.2, 0.25, 0.3} -
well past the imperceptible regime the other evaluations above use (at
eps=0.3 a pixel can shift by up to 30% of the intensity range, visibly so in
`notebook_method_a_adversarial_eval.ipynb`'s example gallery). Flip fraction
saturates fast (SmallCNN 6.5%->99.5%, StrongCNN 4.5%->85.0% across the sweep),
and every adv/control ratio - pixel and Method A, both models - falls
monotonically as epsilon grows (e.g. SmallCNN pixel 8.63->4.17): once a
same-sized *random* shift starts flipping predictions on its own, the
control stops being a clean negative and the adversarial-vs-random signal
both distances measure erodes. The pixel-vs-Method-A gap narrows in lockstep
and, at the largest epsilon, briefly flips: StrongCNN's Method A ratio
(1.48) narrowly exceeds pixel's (1.47) at eps=0.3 - the first time anywhere
in this project a signature distance has matched or beaten the
pixel-Euclidean baseline on this comparison. Flagged, not overclaimed: a
0.01 gap from 200 images at the one condition (85% flip rate) where the
comparison is least informative is well within sampling noise, and it
coincides exactly with both ratios being compressed toward each other by
the same saturation effect, not obvious evidence of a real capability edge.
Full tables: see "Method A: Adversarial / Lipschitz-Ratio Evaluation
Results" above.

### Sub-experiment: per-line AUC diagnostic (verdict: one line beats merged narrowly; 4 lines are dead weight)

Tests the "path by path" framing directly: is any individual line a better
same/different-digit classifier than the merged 496-dim distance? On the
same 300-image Phase 4 sample, `line_6` (horizontal, row~23) reaches
AUC 0.6394 vs. merged's 0.6315 - real but narrow, and only 1 of 16 lines
beats merged (13 score below it), so this isn't broad dilution. The 4 border
lines (0, 7, 8, 15) sit at exactly AUC=0.500 - zero signal by construction -
contributing pure noise to the merged vector. Every measure here, merged and
individual, is a weak discriminator in absolute terms (FPR > 0.8 at 90% TPR
everywhere). Full ranking (former `per_line_auc_summary.md`, now folded
in): see "Method B: Reference-Line Signature Distance" above, "Diagnostic:
is concatenation diluting a real per-line signal?" section.

### Sub-experiment: per-path (unmerged) Lipschitz ratios (verdict: stronger separation than merged distance, with two honest caveats)

Never merges the 16 lines - each produces its own local ratio
`margin_change / line_i_distance`, examined as a collection (mirrors how
pixel-space Lipschitz work treats individual pixels). **Finding 1**: across
all 12 informative lines (excludes the 4 border lines) and all 6
model x epsilon combinations (72/72, no exceptions), mean ratio on genuinely
flipped adversarial pairs exceeds the matched random-control ratio - fold
ratios 4-11x (up to ~20x for `line_6`), noticeably stronger than the merged
distance's own 3.3-7.4x adv/control separation. Caveat: `n_flipped` is small
(4-14/200) at these conservative epsilons, so exact magnitudes are noisy -
the *direction* is what's trustworthy. **Finding 2**: the informative line
carrying the largest ratio is more concentrated (lower entropy) under
adversarial perturbation than under control, in all 6/6 combinations
(~0.2-0.3 bits of a 3.585-bit max) - modest but zero-exception evidence FGSM
concentrates on fewer paths than an equally-large random shift. A confound
(two lines with systematically smaller baseline distances dominate raw
"which line spikes" counts under both conditions) is flagged and separated
from Finding 1. Full write-up, including the mixed stroke-relevance gallery
read (former `per_path_adversarial_summary.md`, now folded in): see "Method
B: Reference-Line Signature Distance" above, "The pivot: per-path
evaluation" section.

### Sub-experiment: per-path robustness check (verdict: Finding 1 is not resting on the two confounded lines)

Checks whether Finding 1 above survives excluding the two lines (9, 14)
flagged for the baseline-distance-scale confound. **Yes, unconditionally**:
all 60 of the reduced 10-line x 6-combination comparisons still show
adversarial > control (mathematically guaranteed once 72/72 held), and the
mean fold-ratio moves by at most ~0.5x on the 4-11x scale (no consistent
direction). A follow-up correlation check across the remaining 10 lines
finds no meaningful relationship between baseline distance and fold-ratio
(`|r| <= 0.32`, sign flips across models/epsilons) - the scale confound looks
isolated to lines 9 and 14 specifically, not systemic. Full detail (former
`per_path_ratio_robustness_summary.md`, now folded in): see "Method B:
Reference-Line Signature Distance" above, "Robustness check on the per-path
finding" section; the check itself now lives in `per_path_adversarial_eval.py`
(`fold_ratio_robustness`) rather than a separate module.

### Sub-experiment: signature level decomposition (verdict: higher-order levels carry the signal, not level 1 - and Method B's level 1 is near-degenerate)

Tests the concern that motivated it directly: is Phase 4's within/cross-digit
signal carried entirely by the level-1 signature terms (for Method A, a
single scalar patch-SV difference), or do levels 2..depth contribute real
information? On the same 300-image Phase 4 sample, masking out level 1
entirely (`level2plus`) reproduces the full `all` ratio almost exactly for
both methods (A: 1.175 vs. 1.176; B: 1.160 vs. 1.160) - level 1 turns out to
be carrying almost none of the distance's *magnitude* (1.79% of total
squared distance for Method A, 0.003% for Method B) despite `level1_only`
posting the single *highest* per-level ratio for Method A (1.276). Method
B's level-1 near-vanishing has a structural cause, not just a weak signal:
every reference line's endpoints sit on the image border, where MNIST
intensity is ~0 for nearly every image, so 99.96% of its per-line net
displacements are exactly 0.0 by construction. Level 4 carries the most
magnitude for both methods (58%/75%) but the *lowest* per-level ratio
(1.162 A / 1.153 B) - magnitude and per-unit class-separation rank levels in
opposite orders. Uses Method B's original 8h+8v/depth-4 configuration (r_B
~ 2.860), predating the sweep below - not re-run against the new 12h+4v/
depth-2 default. Full tables and discussion: see "Level-wise decomposition
diagnostic" above.

## Status / what's not done yet

Per the original plan's phase numbering (see "Implementation plan and design
constraints" above): Phases 0-4 are implemented and gated for both methods;
the adversarial evaluation (now both methods) and Method B's follow-on
diagnostics (per-line AUC, per-path ratios, robustness check) go beyond
Phase 4 but don't map onto a single later phase number cleanly. **Not yet
done**, in that plan's terms:

- **Phase 5's PGD leg** - only single-step FGSM has been run, for both
  methods; multi-step PGD is deferred.
- **Phase 6 - integration into the existing `mnist_lipschitz/adversarial`
  pipeline** (`run_ratio_distribution_analysis` as a new denominator option)
  has not been done; `method_a_adversarial_eval.py`/`method_b_adversarial_eval.py`
  are standalone evaluations, not plugged into that harness.
- **Phase 8's hyperparameter sweep is done for Method B** (`method_b_sweep.py`
  - line geometry, points/line, truncation depth, interpolation; see "Method
  B hyperparameter sweep and Method C" above) **but not for Method A** -
  Method A's own candidates (patch count K, patch-statistic mode, truncation
  depth, rescale factor, and specifically the non-coherent pixel-order issue
  flagged below) haven't been swept.
- **Method A still has no per-line/per-path-style unmerged evaluation** -
  its single 64-anchor stream has no analogous per-path split the way
  Method B's 16 separate lines (or Method C's 16 Hilbert segments) do, so
  `per_line_diagnostics.py`/`per_path_adversarial_eval.py` have no Method A
  counterpart (nor an obvious one to build without a comparable sub-structure
  to split on).
- **Method A vs. Method B's new default hasn't been compared** - every
  Method-A-vs-Method-B comparison in this document (Phase 4, the adversarial
  evaluation) uses Method B's original, now-superseded configuration; a
  fair re-comparison against the swept 12h+4v/depth-2 default would need
  Method A's own equivalent sweep first to stay apples-to-apples, per the
  same logic used to justify Method C's fair-comparison point budget.

None of the ambiguous/negative findings above (Phase 4's modest ratio, both
methods sitting behind the pixel-Euclidean baseline) have been treated as a
stop condition - per this project's "move forward regardless, but carry the
finding" instruction - but none have been promoted into a default pipeline
either, consistent with the root project's checkpoint-gating discipline: a
new metric earns its way into `main()`-style default usage by beating
existing baselines, not just by running without error.
