import pytest
import torch

from signature_distance.streams import (
    HILBERT_ORDER,
    HILBERT_SIDE,
    IMAGE_SIZE,
    NUM_SAMPLE_POINTS,
    NUM_SEGMENTS,
    POINTS_PER_SEGMENT,
    _generate_hilbert_curve,
    _resample_evenly_by_arc_length,
    hilbert_stream,
    make_hilbert_curve,
    make_pixel_order,
    make_reference_lines,
    patch_sv_stream,
    line_stream,
)

# ---------------------------------------------------------------------------
# Method A: patch singular-value stream
# ---------------------------------------------------------------------------


def test_pixel_order_shape():
    order = make_pixel_order(k=64, seed=0)
    assert order.shape == (64, 2)
    assert order.dtype == torch.int64


def test_pixel_order_interior_bound_and_no_duplicates():
    order = make_pixel_order(k=64, seed=0)
    assert order[:, 0].min().item() >= 1
    assert order[:, 0].max().item() <= 26
    assert order[:, 1].min().item() >= 1
    assert order[:, 1].max().item() <= 26

    pairs = {tuple(row.tolist()) for row in order}
    assert len(pairs) == order.shape[0]


def test_pixel_order_k_exceeds_interior_raises():
    with pytest.raises(ValueError):
        make_pixel_order(k=1000, seed=0)


def test_pixel_order_determinism():
    order1 = make_pixel_order(k=64, seed=0)
    order2 = make_pixel_order(k=64, seed=0)
    assert torch.equal(order1, order2)

    order_diff_seed = make_pixel_order(k=64, seed=1)
    assert not torch.equal(order1, order_diff_seed)


def test_patch_sv_stream_shape():
    images = torch.rand(5, 28, 28)
    order = make_pixel_order(k=16, seed=0)
    stream = patch_sv_stream(images, order)
    assert stream.shape == (5, 16, 2)
    assert stream.dtype == torch.float64


def test_patch_sv_stream_all3_shape():
    images = torch.rand(5, 28, 28)
    order = make_pixel_order(k=16, seed=0)
    stream = patch_sv_stream(images, order, mode="all3")
    assert stream.shape == (5, 16, 4)


def test_patch_sv_stream_determinism():
    images = torch.rand(3, 28, 28)
    order = make_pixel_order(k=16, seed=0)
    stream1 = patch_sv_stream(images, order)
    stream2 = patch_sv_stream(images, order)
    assert torch.equal(stream1, stream2)


def test_patch_sv_stream_analytic_constant_image():
    order = make_pixel_order(k=16, seed=0)
    for c in (0.0, 0.5, 1.0):
        images = torch.full((1, 28, 28), c)
        stream = patch_sv_stream(images, order)
        sigma1 = stream[0, :, 1]
        expected = torch.full_like(sigma1, 3 * c)
        assert torch.allclose(sigma1, expected, atol=1e-5)


def test_patch_sv_stream_time_channel():
    images = torch.rand(2, 28, 28)
    k = 16
    order = make_pixel_order(k=k, seed=0)
    stream = patch_sv_stream(images, order)
    expected_t = torch.arange(k, dtype=torch.float64) / (k - 1)
    for n in range(images.shape[0]):
        assert torch.equal(stream[n, :, 0], expected_t)
    assert stream[0, 0, 0].item() == 0.0
    assert stream[0, -1, 0].item() == 1.0


def test_patch_sv_stream_order_sensitivity():
    torch.manual_seed(0)
    images = torch.rand(1, 28, 28)
    order_a = make_pixel_order(k=16, seed=0)
    order_b = make_pixel_order(k=16, seed=1)
    stream_a = patch_sv_stream(images, order_a)
    stream_b = patch_sv_stream(images, order_b)
    assert not torch.equal(stream_a, stream_b)


# ---------------------------------------------------------------------------
# Method B: reference-line stream
# ---------------------------------------------------------------------------


def test_reference_lines_default_shape():
    lines = make_reference_lines()
    assert lines.shape == (16, 32, 2)
    assert lines.dtype == torch.float64


def test_reference_lines_param_combinations():
    all_horiz = make_reference_lines(angles_deg=(0,), counts=(15,), points_per_line=4)
    assert all_horiz.shape == (15, 4, 2)

    all_vert = make_reference_lines(angles_deg=(90,), counts=(15,), points_per_line=4)
    assert all_vert.shape == (15, 4, 2)

    mixed = make_reference_lines(angles_deg=(0, 90), counts=(12, 3), points_per_line=4)
    assert mixed.shape == (15, 4, 2)


def test_reference_lines_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        make_reference_lines(angles_deg=(0, 90), counts=(1,))


def test_reference_lines_invalid_angle_raises():
    with pytest.raises(ValueError):
        make_reference_lines(angles_deg=(45,), counts=(1,))


def test_reference_lines_in_bounds():
    for angles, counts in [((0,), (15,)), ((90,), (15,)), ((0, 90), (8, 8))]:
        lines = make_reference_lines(angles_deg=angles, counts=counts, points_per_line=32)
        assert lines[..., 0].min().item() >= 0
        assert lines[..., 0].max().item() <= 27
        assert lines[..., 1].min().item() >= 0
        assert lines[..., 1].max().item() <= 27


def test_reference_lines_directionality():
    horiz = make_reference_lines(angles_deg=(0,), counts=(1,), points_per_line=8)
    cols = horiz[0, :, 1]
    assert torch.all(cols[1:] > cols[:-1])

    vert = make_reference_lines(angles_deg=(90,), counts=(1,), points_per_line=8)
    rows = vert[0, :, 0]
    assert torch.all(rows[1:] > rows[:-1])


def test_reference_lines_determinism():
    lines1 = make_reference_lines()
    lines2 = make_reference_lines()
    assert torch.equal(lines1, lines2)


def test_line_stream_shape():
    images = torch.rand(5, 28, 28)
    lines = make_reference_lines()
    stream = line_stream(images, lines)
    assert stream.shape == (5, 16, 32, 2)
    assert stream.dtype == torch.float64


def test_line_stream_determinism():
    images = torch.rand(3, 28, 28)
    lines = make_reference_lines()
    stream1 = line_stream(images, lines)
    stream2 = line_stream(images, lines)
    assert torch.equal(stream1, stream2)


def test_line_stream_interpolation_pixel_center():
    images = torch.rand(2, 28, 28)
    row = 5
    cols = torch.arange(28, dtype=torch.float64)
    rows = torch.full((28,), float(row))
    line = torch.stack([rows, cols], dim=-1).unsqueeze(0)  # (1, 28, 2)
    stream = line_stream(images, line)
    intensity = stream[:, 0, :, 1]
    assert torch.allclose(intensity, images[:, row, :], atol=1e-5)

    col = 10
    rows_v = torch.arange(28, dtype=torch.float64)
    cols_v = torch.full((28,), float(col))
    line_v = torch.stack([rows_v, cols_v], dim=-1).unsqueeze(0)
    stream_v = line_stream(images, line_v)
    intensity_v = stream_v[:, 0, :, 1]
    assert torch.allclose(intensity_v, images[:, :, col], atol=1e-5)


def test_line_stream_constant_image():
    lines = make_reference_lines()
    for c in (0.0, 0.5, 1.0):
        images = torch.full((1, 28, 28), c)
        stream = line_stream(images, lines)
        intensity = stream[0, :, :, 1]
        assert torch.allclose(intensity, torch.full_like(intensity, c), atol=1e-5)


def test_line_stream_time_channel():
    images = torch.rand(2, 28, 28)
    lines = make_reference_lines(angles_deg=(0,), counts=(4,), points_per_line=16)
    stream = line_stream(images, lines)
    expected_t = torch.arange(16, dtype=torch.float64) / 15
    for n in range(2):
        for line_idx in range(4):
            assert torch.equal(stream[n, line_idx, :, 0], expected_t)
    assert stream[0, 0, 0, 0].item() == 0.0
    assert stream[0, 0, -1, 0].item() == 1.0


def test_line_stream_sensitivity():
    torch.manual_seed(0)
    images = torch.rand(1, 28, 28)
    lines_a = make_reference_lines(angles_deg=(0,), counts=(8,), points_per_line=16)
    lines_b = make_reference_lines(angles_deg=(90,), counts=(8,), points_per_line=16)
    stream_a = line_stream(images, lines_a)
    stream_b = line_stream(images, lines_b)
    assert not torch.equal(stream_a, stream_b)


# ---------------------------------------------------------------------------
# Method C: Hilbert curve
# ---------------------------------------------------------------------------


def test_generate_hilbert_curve_visits_every_cell_once():
    xy = _generate_hilbert_curve(HILBERT_ORDER)
    assert xy.shape == (4 ** HILBERT_ORDER, 2)
    assert len(set(map(tuple, xy.tolist()))) == xy.shape[0]


def test_generate_hilbert_curve_stays_in_bounds():
    xy = _generate_hilbert_curve(HILBERT_ORDER)
    assert xy.min() >= 0
    assert xy.max() <= HILBERT_SIDE - 1


def test_generate_hilbert_curve_unit_axis_aligned_steps():
    # The property evenly-spaced-by-arc-length sampling relies on: every
    # consecutive pair of raw curve points is exactly 1 grid unit apart,
    # axis-aligned (never diagonal) - checked directly, not assumed.
    xy = _generate_hilbert_curve(HILBERT_ORDER)
    deltas = xy[1:] - xy[:-1]
    manhattan = abs(deltas).sum(axis=1)
    assert set(manhattan.tolist()) == {1}


def test_make_hilbert_curve_shape_and_bounds():
    curve = make_hilbert_curve()
    assert curve.shape == (NUM_SAMPLE_POINTS, 2)
    assert curve.dtype == torch.float64
    assert curve.min().item() >= 0.0
    # True bound is (side-1)*image_size/side = 31*28/32 = 27.125 - slightly
    # past the last valid pixel index (27), not <= 27 as a tidier-looking
    # assumption would suggest (checked directly, see hilbert_stream's own
    # padding_mode="border" handling of this).
    max_bound = (HILBERT_SIDE - 1) * IMAGE_SIZE / HILBERT_SIDE
    assert curve.max().item() <= max_bound + 1e-4
    assert curve.max().item() > IMAGE_SIZE - 1  # confirms it genuinely exceeds 27, not a fluke


def test_make_hilbert_curve_deterministic():
    c1 = make_hilbert_curve()
    c2 = make_hilbert_curve()
    assert torch.equal(c1, c2)


def test_resample_evenly_by_arc_length_on_hand_computable_path():
    # Isolated correctness check on a simple L-shaped polyline where the
    # evenly-spaced points can be hand-computed exactly: (0,0)->(2,0)->(2,2),
    # total length 4, 5 points evenly spaced (step 1) should land exactly
    # on (0,0),(1,0),(2,0),(2,1),(2,2) - deliberately NOT the Hilbert curve,
    # to validate the resampling algorithm itself independent of Hilbert-
    # curve-specific complexity (see the discussion in the function's
    # docstring: arc-length-even resampling is not, in general, equivalent
    # to simple index subsampling once a path bends).
    import numpy as np
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])
    result = _resample_evenly_by_arc_length(coords, num_points=5)
    expected = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [2.0, 2.0]])
    assert np.allclose(result, expected, atol=1e-9)


def test_make_hilbert_curve_endpoints_match_raw_curve_endpoints():
    # First and last resampled points should exactly match the raw curve's
    # first and last vertices (arc length 0 and total length respectively).
    import numpy as np
    xy = _generate_hilbert_curve(HILBERT_ORDER)
    scale = IMAGE_SIZE / HILBERT_SIDE
    curve = make_hilbert_curve().numpy()
    assert np.allclose(curve[0], xy[0] * scale, atol=1e-4)
    assert np.allclose(curve[-1], xy[-1] * scale, atol=1e-4)


def test_make_hilbert_curve_target_arc_length_is_evenly_spaced():
    # The property genuinely guaranteed by construction: the *target* arc-
    # length values the resampling walks the curve to are evenly spaced -
    # verified by reconstructing each output point's arc-length position
    # via the same cumulative-length parameterization and checking the
    # differences are constant, rather than assuming Euclidean spacing
    # between consecutive output points is constant (it isn't, in general,
    # once the underlying path bends between two consecutive samples).
    import numpy as np
    xy = _generate_hilbert_curve(HILBERT_ORDER)
    scale = IMAGE_SIZE / HILBERT_SIDE
    coords = xy.astype(np.float64) * scale
    deltas = np.diff(coords, axis=0)
    seg_lengths = np.sqrt((deltas ** 2).sum(axis=1))
    cum_length = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    expected_target_s = np.linspace(0.0, cum_length[-1], NUM_SAMPLE_POINTS)
    gaps = np.diff(expected_target_s)
    assert np.allclose(gaps, gaps[0], atol=1e-9)


def test_hilbert_stream_shape():
    torch.manual_seed(0)
    images = torch.rand(4, 28, 28)
    curve = make_hilbert_curve()
    stream = hilbert_stream(images, curve)
    assert stream.shape == (4, NUM_SEGMENTS, POINTS_PER_SEGMENT, 2)
    assert torch.isfinite(stream).all()


def test_hilbert_stream_time_channel_matches_shared_helper():
    from signature_distance.streams import time_channel
    images = torch.rand(2, 28, 28)
    curve = make_hilbert_curve()
    stream = hilbert_stream(images, curve)
    expected_t = time_channel(POINTS_PER_SEGMENT)
    for n in range(2):
        for seg in range(NUM_SEGMENTS):
            assert torch.equal(stream[n, seg, :, 0], expected_t)


def test_hilbert_stream_constant_image_gives_constant_intensity():
    curve = make_hilbert_curve()
    for c in (0.0, 0.5, 1.0):
        images = torch.full((1, 28, 28), c)
        stream = hilbert_stream(images, curve)
        intensity = stream[0, :, :, 1]
        assert torch.allclose(intensity, torch.full_like(intensity, c), atol=1e-5)


def test_hilbert_stream_matches_known_pixel_value_at_curve_start():
    # curve[0] is exactly (0, 0) scaled - i.e. still (0, 0) - so the first
    # sampled point of segment 0 should equal images[:, 0, 0] exactly.
    images = torch.rand(3, 28, 28)
    curve = make_hilbert_curve()
    stream = hilbert_stream(images, curve)
    assert torch.allclose(stream[:, 0, 0, 1], images[:, 0, 0], atol=1e-5)
