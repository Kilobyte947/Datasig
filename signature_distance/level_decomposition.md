# Spec: level-wise decomposition of signature distances

Implement a new **additive, read-only diagnostic** in `signature_distance/`
that answers one question:

> Is the within- vs. cross-digit signal in Method A's (and Method B's)
> signature distance carried entirely by the level-1 terms, or do the
> higher-order levels contribute?

This is a diagnostic only. It must not change any existing distance,
stream, or signature behaviour. No model training is involved — this is a
label-based check on the same Phase 4 sample.

---

## Why this matters (read before implementing)

`signature_of_stream` output for `width=2, depth=4` has 31 dimensions:
index 0 is the constant term `1.0`, and indices `1:3` are always the path's
exact net displacement `stream[-1] - stream[0]`, regardless of depth or
intermediate path shape.

Method A's stream is `(t, sigma1)`. The time channel is
`arange(n)/(n-1)`, so **Δt = 1.0 identically for every image** and
contributes exactly zero to any pairwise distance. That means Method A's
entire level-1 content is a single scalar:

    Δσ₁ = sigma1(anchor 63) − sigma1(anchor 0)

i.e. the difference of two 3×3 patch singular values out of 64 anchors.

If the distance turns out to be level-1 dominated, then Method A's Phase 4
ratio of 1.176 is being produced by that one scalar and the signature
machinery is doing no work. That would change what we do next, so we check
it before investing in any stream-construction changes.

---

## Checkpoint 1 — level index utilities

New module: `signature_distance/level_decomposition.py`

Implement:

- `level_slices(depth, width=2) -> dict[int, slice]`
  Returns `{0: slice(0,1), 1: slice(1,1+width), 2: ...}` mapping each
  signature level to its index block. Level `n` occupies `width**n`
  entries. For `width=2, depth=4` this must give block sizes
  `1, 2, 4, 8, 16` totalling 31.

- `mask_signature_levels(sig, levels, depth, width=2) -> array`
  Returns a copy of `sig` with every level **not** in `levels` zeroed out.
  **Zero, do not slice** — preserving the full 31-dim (and Method B's
  496-dim concatenated) layout means the existing distance functions can
  be reused unmodified. Must handle a trailing-axis-of-31 array of any
  leading batch shape, and Method B's `(N, 16, 31)` before concatenation.

### Gate 1 (tests must pass before continuing)

`signature_distance/tests/test_level_decomposition.py`:

1. `level_slices(4, 2)` partitions `range(0, 31)` exactly — no gaps, no
   overlap, correct block sizes.
2. `mask_signature_levels` preserves shape and dtype; zeroed positions are
   exactly the complement of the requested levels.
3. Orthogonality: for random signature vectors, the sum of squared
   per-level distances equals the total squared distance over levels
   `1..depth`. (The blocks are disjoint coordinates, so this is
   Pythagorean — a cheap correctness check on the slicing.)
4. Closed form: for a straight-line path, the level-1-masked distance
   between two such signatures equals the Euclidean distance between their
   net displacements exactly. Reuse the straight-line construction already
   in `tests/test_signatures_method_b.py` rather than writing a new one.

---

## Checkpoint 2 — the decomposition harness

In the same module:

- `run_level_decomposition(n_per_class=30, seed=0, depth=4) -> dict`

Protocol — must match the existing Phase 4 run exactly so the numbers are
comparable:

1. Load the pool via `load_eval_pool(n_per_class, seed)` (300 images at the
   defaults).
2. Build streams and signatures for **both** methods using the existing
   functions unmodified.
3. Derive `r` per method via `choose_rescale_factor` and apply
   `rescale_signature` — **before** masking. `r` stays independently
   derived per method; do not share it, and do not re-derive it per level
   variant. The variants must differ only in which levels survive.
4. For each variant below, build the feature vector via the existing
   `method_a_feature_vector` / `method_b_feature_vector` (so Method B's
   16-line concatenation still happens at the same point) and call
   `within_vs_cross_digit_distance`.

Variants:

| label | levels kept |
|---|---|
| `all` | 1..depth (constant term excluded) |
| `level1_only` | 1 |
| `level2plus` | 2..depth |
| `level2_only` | 2 |
| `level3_only` | 3 |
| `level4_only` | 4 |

Also report, per method and per level, the **mean fraction of total squared
pairwise distance** contributed by that level, averaged over all pairs.
This is more informative than the ratio alone: a level can carry a
meaningful ratio while contributing almost nothing to the distance
magnitude, and we need to see both.

### Gate 2 — reproduction check (hard stop)

The `all` variant must reproduce the documented Phase 4 numbers to
floating-point tolerance:

| method | r | within | cross | ratio |
|---|---|---|---|---|
| Method A | 1.656 | 14.60 | 17.18 | 1.176 |
| Method B | 2.860 | 28.60 | 33.17 | 1.160 |

If it does not, **stop and report the discrepancy rather than adjusting the
harness to match**. A mismatch means this harness is not reproducing the
Phase 4 protocol, and every downstream number would be untrustworthy.
Note the documented values are rounded, so compare to ~2 decimal places on
the means and ~3 on the ratio.

### Gate 2b — Method A closed-form check

Add to the test file: on a small batch, Method A's `level1_only` pairwise
distance must equal `r * |Δσ₁_i − Δσ₁_j|` exactly, where
`Δσ₁ = stream[-1, 1] - stream[0, 1]`. The Δt component cancels identically
across every pair, so this is an exact equality, not an approximation.
Assert the time-channel endpoints are identical across all images as part
of the same test — if that ever stops holding, this identity silently
breaks.

---

## Checkpoint 3 — write-up

Produce `signature_distance/level_decomposition_summary.md` containing:

- The full variant × method table (within mean, cross mean, ratio).
- The per-level squared-distance contribution table.
- A short verdict paragraph stating plainly which of these holds:
  - **(a)** `level2plus` ratio ≈ 1.00 and `level1_only` ≈ `all` → the
    higher-order signature terms are inert; the distance is a relabelled
    scalar difference.
  - **(b)** `level2plus` ratio is comparable to `all` → the signature is
    contributing genuine higher-order information.
  - **(c)** anything between — report where it actually falls, with numbers.
- Whatever the outcome, record it as an honest finding. Per this project's
  convention, a negative result is carried forward, not treated as a stop
  condition, and does not get argued away in the write-up.

Then add a `### Sub-experiment: signature level decomposition` entry to
`signature_distance/README.md`'s Results section, in the same style as the
existing sub-experiment entries (one-line verdict in the heading, short
paragraph, pointer to the summary file), plus a `level_decomposition.py`
row in the File reference table.

---

## Scope boundaries — do not do these

- Do not change `make_pixel_order`, the pixel visiting order, `K`, or any
  stream construction. Ordering is a separate, later piece of work.
- Do not modify `distances.py`, `streams.py`, or `signatures.py` behaviour.
  If a genuinely necessary hook is missing, add it additively and flag it
  in the summary rather than editing existing logic in place.
- Do not merge Method A and Method B, or share `r` between them.
- Do not train or load any CNN — this diagnostic is label-based only and
  needs no model.
- Do not touch `method_b_adversarial_eval.py`, `per_path_adversarial_eval.py`,
  or `per_path_ratio_robustness_check.py`.
- Do not run `mode="all3"` or any depth other than 4 in this piece of work.

## Run instructions to verify

```bash
.venv/bin/python -m pytest signature_distance/tests/test_level_decomposition.py -v
.venv/bin/python -m pytest signature_distance/tests/ -v   # full suite must still pass
.venv/bin/python -c "from signature_distance.level_decomposition import run_level_decomposition; print(run_level_decomposition())"
```