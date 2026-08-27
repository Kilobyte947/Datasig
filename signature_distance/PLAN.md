# Implementation Plan: Signature-Based Image Distance

## Instructions for Claude Code

Work through the checkpoints in order. **Each checkpoint has a test gate: do not
proceed to the next checkpoint until all tests for the current one pass.** Report
which checkpoint you are on and the test results at each gate.

Two stream-construction methods are used throughout, named consistently as
**Method A** and **Method B** (this replaces an earlier draft where the two
methods were numbered 1/2 and briefly got swapped between two independently
written plans — the naming below is the reconciled, canonical version):

- **Method A — patch singular-value stream.** The "convolution" method: a local,
  patch-based statistic computed at a fixed set of pixel locations.
- **Method B — reference-line stream.** The "line reference" method: intensity
  sampled along a fixed set of straight lines through the image.

## Scope boundary — read carefully

**Implement Checkpoints 0–3 only (stream construction).** Checkpoints 4 onward
(signature computation, distance, sanity checks, adversarial/Lipschitz-ratio
evaluation, pipeline integration, sweeps, documentation) are recorded below **for
reference only, so the stream-construction interfaces are designed with the right
shape for what comes next** — they must **NOT** be implemented in this phase.
Specifically, in this phase:

- Do NOT compute signatures (truncated or otherwise).
- Do NOT install or import `iisignature`, `signatory`, `sigkernel`, `roughpy`, or
  `roughpy-jax`.
- Do NOT compute any distances or evaluation metrics.
- Stop after Checkpoint 3's test gate passes and produce the summary described there.

Why no signature library import even in later phases: the plan is to use
**`roughpy-jax`** (same semantics as `roughpy`, faster) for truncated-signature
computation, but hand-written/self-coded rather than pulled in as an opaque
dependency — see Phase 2 below. That's a deliberate reason to keep this phase
free of any signature-library code, not just an arbitrary sequencing choice.

## Context

This experiment builds candidate distance metrics on MNIST images using path
signatures, for eventual comparison against the Euclidean and Mahalanobis
distances already used in the Lipschitz/adversarial experiments (Experiment 2,
`mnist_lipschitz/`). The core idea: represent each image via a small number of
low-dimensional paths/streams (not raw pixels), compute a distance between images
in this path-signature space, use that distance as the new denominator in the
existing Lipschitz-ratio pipeline, and test it against imperceptible adversarial
examples.

Both methods use fixed, non-learned constructions only — no dependence on the
trained classifier's own features, deliberately, to stay clear of the
self-referential-metric problem documented elsewhere in the project.

Phase 1 (this document's in-scope part) constructs the *streams* (paths) from
images under both methods. Later phases (documented below, not implemented yet)
compute truncated signatures of these streams and evaluate the induced distance.

### Method A — patch singular-value stream (time-augmented)

For a fixed, shared sequence of K pixel locations, extract the 3×3 patch centred at
each location, compute its largest singular value σ₁, and form the 2-dimensional
stream of points (t_k, σ₁(patch_k)) where t_k = k/(K−1) is the normalised step
index. The time coordinate is essential: without it the signature of a scalar path
degenerates to a function of the net increment alone.

Design decisions (fixed, do not change without flagging):

- **Shared deterministic pixel ordering.** Sample the K pixel locations *once*,
  with a fixed seed, and use the identical ordered sequence for every image. This
  makes the stream a deterministic, comparable feature across images. Save the
  ordering to disk as an artifact.
- **Interior pixels only.** Sample locations from rows and columns 1..26
  (0-indexed) so every 3×3 patch is fully inside the 28×28 image. No padding.
- **Patch statistic.** σ₁ = largest singular value of the 3×3 patch matrix,
  via `torch.linalg.svdvals` (or `numpy.linalg.svd(compute_uv=False)`).
  Implement with a flag `mode="top1"` (default) but structure the code so
  `mode="all3"` (stream in R⁴: time + all three singular values) is a trivial
  extension later. `all3` may be implemented now if it falls out naturally, but
  it is not required and needs no extra tests beyond shape.
- **Default K = 64.** Expose as a parameter.

Stream shape per image: `(K, 2)` float32, columns `[t, sigma1]`.

**Status: implemented** (`make_pixel_order`, `patch_sv_stream` in `streams.py`,
Checkpoint 2, tests passing).

### Method B — reference-line stream (interpolated path)

For a fixed set of horizontal + vertical reference lines through the 28×28 grid
(same lines for every image), sample pixel intensity at a fixed number of points
along each line, joined into a `[t, intensity]` path. Full design detail,
rationale, and test plan: `Method_B.md`.

Design decisions (fixed, do not change without flagging):

- **`make_reference_lines(angles_deg, counts, points_per_line, image_size, seed)`
  — one parameterized function, not separate per-orientation implementations.**
  Default `angles_deg=(0, 90), counts=(8, 8)` (16 lines total: 8 horizontal + 8
  vertical); any other split (15:0, 0:15, 12:3, ...) is a different call to the
  same function, keeping the orientation split tunable for Stage 8 rather than
  hardcoded.
- **Horizontal/vertical only, no clipping.** Angle 0 = rows evenly spaced across
  image height, each sampled left→right; angle 90 = columns evenly spaced across
  image width, each sampled top→bottom. Every point is in `[0, image_size - 1]`
  by construction — no bounding-box or circle clipping needed (an earlier
  circle-based geometry draft was dropped for exactly this reason: unnecessary
  complexity vs. axis-aligned lines). Arbitrary angles would need clipping logic;
  deferred to Stage 8, not currently supported (`ValueError` otherwise).
- **Sampling off the pixel grid.** Line points generally don't land on pixel
  centers; intensity is read via batched bilinear interpolation
  (`torch.nn.functional.grid_sample`), not nearest-pixel snapping.
- **Points per line.** `points_per_line = 32` default, exposed as a parameter,
  real tuning deferred to Stage 8.
- **No cross-line concatenation.** Each line's future signature (Phase 2) is
  computed independently over its own points only; lines are combined by
  concatenating resulting signature vectors, never by concatenating raw streams
  first. Already satisfied by `line_stream`'s output shape (one stream per line,
  never flattened) — no extra code needed for this.
- Shared `time_channel` helper from Checkpoint 2 reused per line (see
  `Method_B.md`), and the artifact convention mirrors
  `artifacts/pixel_order_seed0.npy` → `artifacts/reference_lines_seed0.npy`.

Stream shape per image: `(num_lines, points_per_line, 2)`, columns
`[t, intensity]` per line — `(16, 32, 2)` for the default config, analogous to
Method A's time-augmented stream.

**Status: implemented** (`make_reference_lines`, `line_stream` in `streams.py`,
Checkpoint 3, tests passing). An earlier draft of this plan had a different
Method 2 (row/column vector stream, traversing all 28 rows or columns directly as
28-dimensional vectors) implemented at this checkpoint instead — see `row_stream`
in `streams.py`. That method is superseded by the reference-line method above,
kept in the repo for now, and no longer part of the active plan.

## Repository layout

```
signature_distance/
├── PLAN.md                  # this file
├── PHASE1_SUMMARY.md         # written after Checkpoint 3's gate passes
├── Method_B.md                # Method B's concrete design/implementation plan
├── streams.py                # stream construction (Phase 1 scope, both methods)
├── data_pool.py               # fixed evaluation pool loading (Phase 1 scope, shared)
├── plots.py                    # Method B display plots only (see note below)
├── run_experiment.py           # Method B demo driver only (see note below)
├── notebook_method_b.ipynb     # Method B's own display notebook (Phase 1)
├── tests/
│   ├── test_data_pool.py
│   └── test_streams.py
├── artifacts/
│   ├── pixel_order_seed0.npy       # created by Checkpoint 2
│   └── reference_lines_seed0.npy   # created by Checkpoint 3
└── results/                    # gitignored, Method B demo figures (.gitkeep tracked)

# Nick's Method A analog (plots/driver/notebook) lives in its own files,
# not added to plots.py/run_experiment.py above - kept separate deliberately
# so neither side's Phase-1 presentation work overlaps with the other's.

# Future phases (reference only, not created in Phase 1):
#   signatures.py    — Phase 2, truncated signature computation via roughpy-jax
#   distances.py      — Phase 3, Euclidean distance over concatenated signature vectors
#   notebook_signature_distance.ipynb — Phase 6, combined driver notebook once both
#     methods reach the signature/distance stage (distinct from notebook_method_b.ipynb)
```

Style constraints (consistent with the rest of the project):

- Pure functions operating on tensors/arrays; no notebook code in modules.
- Every public function takes explicit arguments (no hidden globals) and has a
  docstring stating input/output shapes.
- Deterministic: any randomness goes through an explicit `seed` argument.
- Use `torch` throughout for consistency with `lipschitz_diagnostics.py`;
  keep functions callable on CPU.

---

## Checkpoint 0 — Folder setup

Tasks:

1. Create the folder structure above.
2. Save this plan as `signature_distance/PLAN.md`.
3. Create empty `tests/` package with an `__init__.py` if needed for the test
   runner.

Test gate: `pytest signature_distance/tests/` runs (collecting zero tests is fine)
and the folder structure matches the layout above.

**Status: done.**

---

## Checkpoint 1 — Evaluation data pool (`data_pool.py`)

A fixed, reproducible pool of MNIST test images, matching the pool-based protocol
used in Experiment 2.

Function to implement:

```python
def load_eval_pool(n_per_class: int = 100, seed: int = 0,
                   root: str = "./data") -> tuple[torch.Tensor, torch.Tensor]:
    """Return (images, labels).

    images: (10 * n_per_class, 28, 28) float32 in [0, 1]
    labels: (10 * n_per_class,) int64

    Deterministic: for each class, take the first n_per_class images of that
    class in MNIST test-set order after shuffling indices with the given seed.
    Images are sorted by class in the returned tensors (all 0s, then all 1s, ...).
    """
```

Notes:

- Use `torchvision.datasets.MNIST(train=False)`. Convert to float32 in [0, 1];
  no other transforms.
- The pool must be identical across runs with the same seed (this will be tested).

Test gate (`tests/test_data_pool.py`):

1. Shapes and dtypes are exactly as documented.
2. Exactly `n_per_class` images per class; labels sorted ascending by class.
3. Pixel values in [0, 1].
4. Determinism: two calls with the same seed return bitwise-identical tensors;
   two calls with different seeds differ.

**Status: done.**

---

## Checkpoint 2 — Method A streams (`streams.py`)

Functions to implement:

```python
def make_pixel_order(k: int = 64, seed: int = 0,
                     image_size: int = 28) -> torch.Tensor:
    """Return (k, 2) int64 tensor of (row, col) locations, sampled uniformly
    WITHOUT replacement from the interior grid [1, image_size-2]^2, in a fixed
    order determined by seed. Raises ValueError if k exceeds the number of
    interior pixels."""

def patch_sv_stream(images: torch.Tensor, pixel_order: torch.Tensor,
                    mode: str = "top1") -> torch.Tensor:
    """Method A stream construction.

    images: (N, 28, 28) float32
    pixel_order: (K, 2) int64 from make_pixel_order
    returns: (N, K, 2) float32 for mode="top1", columns [t, sigma1],
             with t = arange(K) / (K - 1).
             (N, K, 4) for mode="all3", columns [t, s1, s2, s3], s1 >= s2 >= s3.
    """
```

Implementation notes:

- Extract all patches with indexing/`unfold`; batch the SVD with
  `torch.linalg.svdvals` on a tensor of shape (N, K, 3, 3). Avoid Python loops
  over images.
- After creating the default ordering (`k=64, seed=0`), save it to
  `artifacts/pixel_order_seed0.npy`.

Test gate (`tests/test_streams.py`, Method A section):

1. **Shapes**: output is (N, K, 2) float32 for a small batch.
2. **Interior bound**: every location in `make_pixel_order` output lies in
   [1, 26] for both coordinates; no duplicate locations.
3. **Determinism**: same seed → identical ordering; same image passed twice →
   bitwise-identical streams. Different seeds → different orderings.
4. **Analytic SVD check**: for a constant image with pixel value c, every 3×3
   patch is c·(all-ones matrix), whose singular values are (3c, 0, 0). Assert
   sigma1 == 3c to within 1e-5 at every stream step, for c ∈ {0.0, 0.5, 1.0}.
5. **Time channel**: column 0 equals arange(K)/(K−1) exactly; first entry 0,
   last entry 1.
6. **Order sensitivity sanity check**: for a non-constant image, streams built
   from two different pixel orderings differ (this documents WHY the shared
   fixed ordering matters).

**Status: done.**

---

## Checkpoint 3 — Method B streams (`streams.py`)

Full design detail and rationale: `Method_B.md`. Implemented as:

```python
def make_reference_lines(angles_deg: tuple = (0, 90), counts: tuple = (8, 8),
                          points_per_line: int = 32,
                          image_size: int = 28, seed: int = 0) -> torch.Tensor:
    """Return (sum(counts), points_per_line, 2) float32 tensor of (row, col)
    continuous sample coordinates. angles_deg[i] gets counts[i] evenly-spaced
    parallel lines (0 deg = horizontal, 90 deg = vertical); only 0/90 are
    currently supported (in-bounds by construction, no clipping)."""

def line_stream(images: torch.Tensor, lines: torch.Tensor) -> torch.Tensor:
    """Method B stream construction.

    images: (N, 28, 28) float32
    lines: (num_lines, points_per_line, 2) from make_reference_lines.
    returns: (N, num_lines, points_per_line, 2) float32, columns
             [t, intensity]; intensity via batched grid_sample bilinear
             interpolation, t via the shared time_channel helper.
    """
```

Test gate (`tests/test_streams.py`, Method B section):

1. **Shapes**: `(16, 32, 2)` for the default config; `angles_deg`/`counts`
   combinations (all-horizontal, all-vertical, uneven mixes) produce the
   expected line count via the same function.
2. **In-bounds**: every `(row, col)` in `make_reference_lines` lies in
   `[0, image_size - 1]`, across configurations.
3. **Directionality**: horizontal lines' column coordinate increases along the
   line; vertical lines' row coordinate increases along the line.
4. **Determinism**: same call → identical line geometry; same image passed
   twice → bitwise-identical streams.
5. **Interpolation correctness**: a sample point exactly on a pixel center
   equals the raw pixel value (checked for both a horizontal and a vertical
   line); a constant image gives constant intensity everywhere.
6. **Time channel**: per-line time coordinate equals
   `time_channel(points_per_line)` exactly, for every line.
7. **Line sensitivity sanity check**: streams built from two different
   `angles_deg`/`counts` configs differ for a non-constant image (mirrors
   Method A's ordering-sensitivity check).
8. **Invalid input**: mismatched `angles_deg`/`counts` lengths, or an angle
   other than 0/90, raise `ValueError`.

**Status: done** (`make_reference_lines`, `line_stream` in `streams.py`, 33
tests passing including the above). `row_stream` — the row/column vector
stream that previously occupied this checkpoint — is superseded, kept in the
repo, not part of the active plan (follow-up: decide whether to remove it).

Final task for this phase (after the gate passes): refresh
`signature_distance/PHASE1_SUMMARY.md` to state (a) the exact shapes produced
by Method A and Method B on the default pool (1000 images), (b) wall-clock
time to build all streams on CPU, and (c) confirmation that no signature/
distance code was written. **Done** — see `PHASE1_SUMMARY.md`. Then STOP — do
not proceed to Phase 2 without explicit sign-off.

---

## Future phases (reference only — do NOT implement in Phase 1)

These are recorded so Phase 1's interfaces (stream shapes, artifact conventions)
are designed with the right shape for what comes next, and so the full pipeline
is documented in one place. Each phase's own test-gate discipline (per this
project's checkpoint-gating convention — see the repo's root `CLAUDE.md`) applies
when it's actually implemented: don't wire a new phase into the pipeline just
because it runs without error.

### Phase 2 — Signature computation (Stage 2)

- Truncated signature per stream: per line for Method B, for Method A's single
  stream (or per-patch-location stream, depending on how Method A's shape is
  finalized).
- Computed via a **self-coded implementation targeting `roughpy-jax`** semantics
  (same as `roughpy`, faster) — not an opaque import of `iisignature`,
  `signatory`, or `sigkernel`. This is a deliberate choice (see Scope boundary
  above), not a placeholder for "pick a library later."
- Start at one fixed truncation level (level 2) for the basic pass — a full
  truncation-level sweep is deferred to Phase 8.
- Apply rescaling (level-n terms scaled by rⁿ) before any distance comparison.
- Normalisation/rescaling decisions belong here, not in stream construction
  (Phase 1) — this mirrors the existing repo convention in
  `mnist_lipschitz/distance.py` of keeping numerically sensitive conventions
  isolated in one place and checkpoint-tested before use.

### Phase 3 — Distance function (Stage 3)

- Concatenate per-stream signature vectors into one feature vector per image.
- Distance = Euclidean on this concatenated vector (basic version).
- PCA or covariance-norm projection is a considered refinement, not required for
  the first pass.

### Phase 4 — Sanity check (Stage 4)

- Within-digit vs. cross-digit distance check, same protocol as used for the
  existing Euclidean/Mahalanobis metrics in `mnist_lipschitz`: same-label pairs
  should be closer than different-label pairs on average, for both Method A and
  Method B.
- Cheap, label-based, no trained model needed — run before anything else in
  Phase 5 to confirm the basic construction isn't broken.

### Phase 5 — Imperceptible adversarial examples + Lipschitz ratio (Stage 5)

- Generate adversarial perturbations (FGSM/PGD, small enough to be visually
  imperceptible) on a sample of test images, reusing the existing attack code in
  `mnist_lipschitz/adversarial/attacks.py` rather than re-deriving it.
- Compute the Lipschitz ratio (existing numerator: `margin_difference` from the
  trained model; new denominator: path/stream signature distance) between each
  original and its perturbed version, for both Method A and Method B.
- Compare against the same ratio computed with plain Euclidean pixel distance, on
  the same pairs.

### Phase 6 — Integration into existing pipeline (Stage 6)

- New notebook, per project convention (no reusable logic in the notebook
  itself — thin driver only, matching every other notebook in the repo).
- Plug into `run_ratio_distribution_analysis` (in `mnist_lipschitz/adversarial/`)
  as new denominator options, one per method, reusing the existing harness,
  galleries, and numerator/denominator decomposition rather than building a
  parallel one.

### Phase 7 — Baseline diagnostics and results (Stage 7)

- Run the Phase 4 and Phase 5 checks on both methods at their fixed/basic
  settings.
- Produce the standard comparison galleries and ratio distributions.
- Decide, from these results, whether one method, both, or neither shows enough
  promise to justify Phase 8's sweep cost — same "don't promote just because it
  runs" discipline the repo already applies elsewhere (e.g. the
  `local_patch_cross_terms` negative finding).

### Phase 8 — Extensive hyperparameter sweep (Stage 8, only after Phase 7 looks sane)

Sweep together, not staged one-at-a-time:

- **Method A**: number of pixel locations K, patch-statistic choice
  (`mode="top1"` vs `"all3"`), truncation level, rescaling factor r.
- **Method B**: number of lines, points per line, straight vs. curved joins,
  truncation level, rescaling factor r.
- Tune/optimise based on which configurations best separate within- vs.
  cross-digit distance (Phase 4 check) and best detect the imperceptible
  adversarial shift (Phase 5 check).

### Phase 9 — Document (Stage 9)

- Negative and positive results both documented explicitly, per project
  convention (root `CLAUDE.md`'s checkpoint-gating section, and the
  `mnist_lipschitz/README.md` precedent for negative findings).
- Comparison galleries maintained throughout, not just at the end.
