# Phase 1 Summary — Stream Construction

Checkpoints 0-3 implemented and gated; all 33 tests in `signature_distance/tests/`
pass (`.venv/bin/python -m pytest signature_distance/tests/ -v`).

Method A and Method B naming was reconciled after two independently drafted
plans swapped the labels — see `PLAN.md` and `Method_B.md`. Method B is now
the reference-line stream (`make_reference_lines` + `line_stream`); the
previously implemented row/column vector stream (`row_stream`) is a
superseded draft, kept in the repo but no longer part of the active plan.

## (a) Shapes on the default pool (1000 images, `n_per_class=100, seed=0`)

- `load_eval_pool(n_per_class=100, seed=0)` → `images`: `(1000, 28, 28)` float32
  in `[0, 1]`; `labels`: `(1000,)` int64.
- Method A (`make_pixel_order(k=64, seed=0)` + `patch_sv_stream`, `mode="top1"`):
  `(1000, 64, 2)` float32, columns `[t, sigma1]`.
- Method B (`make_reference_lines()` default `angles_deg=(0, 90),
  counts=(8, 8), points_per_line=32` + `line_stream`):
  `(1000, 16, 32, 2)` float32, columns `[t, intensity]`.
- Superseded draft (`row_stream(images, axis="rows")`): `(1000, 28, 28)` float32.

## (b) Wall-clock time on CPU

Measured with `time.perf_counter()` for a single run (Apple Silicon CPU, no GPU):

- Pool loading (from already-downloaded MNIST files): ~0.007 s
- Method A stream construction (1000 images): ~0.018 s
- Method B stream construction (1000 images, via batched `grid_sample`): ~0.001 s

Both methods are effectively instantaneous on the full default pool.

## (c) Confirmation of scope

No signature computation, no distance/evaluation metric code, and no imports of
`iisignature`, `signatory`, `sigkernel`, `roughpy`, or `roughpy-jax` were written
in this phase. Only `streams.py` (stream construction) and `data_pool.py` (fixed
MNIST pool loading) were implemented, per Checkpoints 0-3 of `PLAN.md`.

Stopping here per the plan's scope boundary.
