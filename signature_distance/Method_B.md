# Method B — Reference-Line Stream: Implementation Plan

## Interfaces

```python
def make_reference_lines(angles_deg: tuple[float, ...] = (0, 90),
                          counts: tuple[int, ...] = (8, 8),
                          points_per_line: int = 32,
                          image_size: int = 28, seed: int = 0) -> torch.Tensor:
    """(sum(counts), points_per_line, 2) float32 tensor of (row, col) sample
    coordinates, continuous (not snapped to the pixel grid), fixed and
    shared across all images. `angles_deg[i]` gets `counts[i]` evenly-spaced
    parallel lines (0 deg = horizontal, 90 deg = vertical); any other split
    (15:0, 0:15, 12:3, ...) is a different call to the same function, not a
    different code path. `seed` reserved for future randomized variants
    (Stage 8); current construction is deterministic without it."""

def line_stream(images: torch.Tensor, lines: torch.Tensor) -> torch.Tensor:
    """images: (N, 28, 28) float32. lines: from make_reference_lines.
    returns: (N, num_lines, points_per_line, 2) float32, columns
    [t, intensity], t identical per line via time_channel(points_per_line).
    Lines stay separate in this output (never concatenated into one raw
    stream) — see "no cross-line concatenation" below.
    """
```

## Line geometry (default)

- 16 lines total: 8 horizontal (rows evenly spaced across image height) +
  8 vertical (columns evenly spaced across image width).
- Built via a single parameterized `make_reference_lines(angles, counts, ...)`
  — horizontal and vertical are two calls with angle=0° and angle=90°,
  not separate implementations. This keeps line-count-per-orientation a
  tunable parameter for Stage 8, not a hardcoded split.
- Each line sampled in a fixed, unambiguous direction (horizontal:
  left→right; vertical: top→bottom).
- No clipping needed in either orientation: every line is fully in-bounds
  regardless of image size or content position.
- `points_per_line = 32` default, per line, deferred tuning to Stage 8.

## Constraint: no cross-line concatenation

Each line's signature computed independently over its own points only,
regardless of orientation. Lines combined by concatenating resulting
signature vectors, never by concatenating raw streams before the
signature step.

This is a Phase 2 concern (recorded here because it constrains Checkpoint 3's
output shape), but it's already satisfied without extra work: `line_stream`
returns `(N, num_lines, points_per_line, 2)`, one stream per line, not a
single flattened stream — so nothing changes here if/when this constraint
is enforced downstream.

## Sampling

- Coordinates are continuous → read intensity via bilinear interpolation,
  batched with `torch.nn.functional.grid_sample` (normalize `(row, col)` to
  `[-1, 1]` once, reuse for every image) — no Python loop over images, same
  vectorization spirit as Method A's batched `unfold` + `svdvals`.

## Time channel

- Per line: `time_channel(points_per_line)` from `streams.py`, broadcast
  across every line (regardless of orientation or count) and all `N`
  images — identical call Method A already uses, per the shared-helper
  reasoning already agreed.

## Artifact

- Save line coordinates to `artifacts/reference_lines_seed0.npy`, same
  pattern as `pixel_order_seed0.npy`.

## Implementation steps

1. `make_reference_lines`: for each `(angle, count)` pair, place `count`
   evenly-spaced parallel lines (horizontal: evenly-spaced rows, each
   sampled left→right; vertical: evenly-spaced columns, each sampled
   top→bottom), `linspace` sample points per line. No clipping logic
   needed — every point is in-bounds by construction. Save artifact.
2. `line_stream`: `grid_sample`-based batched bilinear lookup + time column.
3. Tests (`tests/test_streams.py`, Method B section):
   - Shape: `(N, 16, 32, 2)` for the default `angles_deg=(0, 90),
     counts=(8, 8)`.
   - Parameterization: `angles_deg=(0,), counts=(15,)` (all horizontal),
     `(90,), (15,)` (all vertical), and an uneven mix (e.g. `(0, 90), (12, 3)`)
     all produce the expected line count via the same function — no separate
     code path per orientation.
   - In-bounds: every `(row, col)` in `make_reference_lines` lies in
     `[0, 27]`, for every angle/count combination above.
   - Directionality: horizontal lines' column coordinate is increasing along
     the line; vertical lines' row coordinate is increasing along the line.
   - Determinism: same call → identical lines; same image twice →
     bitwise-identical streams.
   - Interpolation correctness: a sample point exactly on a pixel center
     equals the raw pixel value; constant image → constant intensity at
     every point regardless of line geometry.
   - Time channel: column 0 equals `time_channel(32)` for every line.
   - Line-sensitivity: two different line configs (e.g. different
     `angles_deg`/`counts`) give different streams for a non-constant image.
4. Refresh `PHASE1_SUMMARY.md` once the gate passes (shapes + wall-clock
   time for both methods on the 1000-image pool).
5. Flag `row_stream` for removal — it's the superseded Method 2, per the
   note already in `PLAN.md`. Separate follow-up, not blocking this
   checkpoint.
6. Display, per project convention (module + its own notebook): `plots.py`
   (`plot_reference_lines`, `plot_line_stream`) + `run_experiment.py`
   (`stream_construction_demo`) + `notebook_method_b.ipynb`, all scoped to
   Method B only — no Method A plotting/demo code added here, that's
   Nick's own file(s) to add for Method A.

## Out of scope

No signatures, no distances, no `roughpy-jax` — stop after the test gate
passes, per `PLAN.md`'s Phase 1 scope boundary.
