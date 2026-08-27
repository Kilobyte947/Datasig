# Implementation Plan: Signature-Based Image Distance — Phase 1 (Stream Construction Only)

## Instructions for Claude Code

Create a new folder `signature_distance/` at the repository root and save a copy of
this plan file into it as `PLAN.md` before writing any code. Work through the
checkpoints in order. **Each checkpoint has a test gate: do not proceed to the next
checkpoint until all tests for the current one pass.** Report which checkpoint you
are on and the test results at each gate.

## Scope boundary — read carefully

**Implement Checkpoints 0–3 only.** Checkpoints 4–6 are documented here so the code
you write has the right interfaces for what comes next, but they must **NOT** be
implemented in this phase. Specifically, in this phase:

- Do NOT compute signatures (truncated or otherwise).
- Do NOT install or import `iisignature`, `signatory`, or `sigkernel`.
- Do NOT compute any distances or evaluation metrics.
- Stop after Checkpoint 3's test gate passes and produce the summary described there.

## Context

This experiment builds candidate distance metrics on MNIST images using path
signatures, for eventual comparison against the Euclidean and Mahalanobis distances
already used in the Lipschitz/adversarial experiments (Experiment 2). Phase 1
constructs the *streams* (paths) from images under two methods. Later phases will
compute truncated signatures of these streams and evaluate the induced distance via
the between-class / within-class mean-distance ratio, the full 10×10 inter-class
distance matrix, same/different-pair AUC, and 1-NN accuracy — all against Euclidean
and Mahalanobis baselines.

### Method 1 — patch singular-value stream (time-augmented)

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

### Method 2 — horizontal-line vector stream

Treat each of the 28 rows of the image as a point in R²⁸ and form the stream by
traversing rows top to bottom. Use **all 28 rows** (no subsampling — the stream is
short and subsampling would need separate justification).

Design decisions:

- Stream shape per image: `(28, 28)` float32; entry `[i, :]` is row i of the image.
- Implement an `axis` parameter: `axis="rows"` (default) and `axis="cols"`
  (transpose; traverse columns left to right). Both are cheap and the column
  variant will be needed for the symmetrisation comparison later.
- No time augmentation is needed here (the stream is already high-dimensional),
  but do not normalise or rescale pixel values beyond the standard [0, 1] load —
  normalisation choices belong to the signature phase and must not be baked in
  here.

## Repository layout to create

```
signature_distance/
├── PLAN.md                  # copy of this file
├── streams.py               # stream construction (this phase)
├── data_pool.py             # fixed evaluation pool loading
├── tests/
│   ├── test_data_pool.py
│   └── test_streams.py
└── artifacts/
    └── pixel_order_seed0.npy   # created by Checkpoint 2
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

---

## Checkpoint 2 — Method 1 streams (`streams.py`)

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
    """Method 1 stream construction.

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

Test gate (`tests/test_streams.py`, Method 1 section):

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

---

## Checkpoint 3 — Method 2 streams (`streams.py`)

Function to implement:

```python
def row_stream(images: torch.Tensor, axis: str = "rows") -> torch.Tensor:
    """Method 2 stream construction.

    images: (N, 28, 28) float32
    returns: (N, 28, 28) float32.
      axis="rows": stream[n, i, :] is row i of image n (top to bottom).
      axis="cols": stream[n, j, :] is column j of image n (left to right).
    """
```

Test gate (`tests/test_streams.py`, Method 2 section):

1. **Identity check**: for `axis="rows"`, `row_stream(x)[n, i]` equals
   `x[n, i, :]` exactly; for `axis="cols"`, equals `x[n, :, j]`.
2. **Transpose consistency**: `row_stream(x, "cols")` equals
   `row_stream(x.transpose(1, 2), "rows")`.
3. **No mutation**: input tensor is unchanged; output is a copy or a safe view
   (document which).
4. **Determinism**: trivially deterministic — assert two calls identical.

Final task for this phase (after the gate passes): write a short
`signature_distance/PHASE1_SUMMARY.md` stating (a) the exact shapes produced by
each method on the default pool (1000 images), (b) wall-clock time to build all
streams on CPU, and (c) confirmation that no signature/distance code was written.
Then STOP.

---

## Checkpoint 4 — Truncated signatures (DO NOT IMPLEMENT IN THIS PHASE)

For reference only. A future phase will add `signatures.py`:

- Method 1 streams (dim 2): truncated signature to depth 4 (30 terms) via
  `iisignature` or `signatory`.
- Method 2 streams (dim 28): truncated signature to depth 2 (812 terms);
  depth 3 (~22k terms) only if memory allows, as an ablation.
- Normalisation decisions (per-level scaling, path rescaling) are made HERE,
  not in stream construction.

## Checkpoint 5 — Distances and baselines (DO NOT IMPLEMENT IN THIS PHASE)

Pairwise Euclidean distance between signature vectors; import Euclidean and
Mahalanobis pixel-space baselines from `lipschitz_diagnostics.py`.

## Checkpoint 6 — Evaluation (DO NOT IMPLEMENT IN THIS PHASE)

On the fixed pool: between/within-class mean-distance ratio (headline),
full 10×10 inter-class mean-distance matrix, same/different-pair AUC,
1-NN accuracy. Uncertainty via resampling the image pool (not pairs).
