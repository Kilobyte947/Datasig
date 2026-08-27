import pytest
import torch

from signature_distance.streams import (
    make_pixel_order,
    make_reference_lines,
    patch_sv_stream,
    line_stream,
    row_stream,
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
    assert stream.dtype == torch.float32


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
    expected_t = torch.arange(k, dtype=torch.float32) / (k - 1)
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
    assert lines.dtype == torch.float32


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
    assert stream.dtype == torch.float32


def test_line_stream_determinism():
    images = torch.rand(3, 28, 28)
    lines = make_reference_lines()
    stream1 = line_stream(images, lines)
    stream2 = line_stream(images, lines)
    assert torch.equal(stream1, stream2)


def test_line_stream_interpolation_pixel_center():
    images = torch.rand(2, 28, 28)
    row = 5
    cols = torch.arange(28, dtype=torch.float32)
    rows = torch.full((28,), float(row))
    line = torch.stack([rows, cols], dim=-1).unsqueeze(0)  # (1, 28, 2)
    stream = line_stream(images, line)
    intensity = stream[:, 0, :, 1]
    assert torch.allclose(intensity, images[:, row, :], atol=1e-5)

    col = 10
    rows_v = torch.arange(28, dtype=torch.float32)
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
    expected_t = torch.arange(16, dtype=torch.float32) / 15
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
# Superseded Method B draft: row/column vector stream (see PLAN.md)
# ---------------------------------------------------------------------------


def test_row_stream_identity_rows():
    images = torch.rand(4, 28, 28)
    stream = row_stream(images, axis="rows")
    assert stream.shape == (4, 28, 28)
    for n in range(4):
        for i in range(28):
            assert torch.equal(stream[n, i], images[n, i, :])


def test_row_stream_identity_cols():
    images = torch.rand(4, 28, 28)
    stream = row_stream(images, axis="cols")
    assert stream.shape == (4, 28, 28)
    for n in range(4):
        for j in range(28):
            assert torch.equal(stream[n, j], images[n, :, j])


def test_row_stream_transpose_consistency():
    images = torch.rand(4, 28, 28)
    cols_direct = row_stream(images, axis="cols")
    cols_via_transpose = row_stream(images.transpose(1, 2), axis="rows")
    assert torch.equal(cols_direct, cols_via_transpose)


def test_row_stream_no_mutation():
    images = torch.rand(4, 28, 28)
    original = images.clone()
    _ = row_stream(images, axis="rows")
    _ = row_stream(images, axis="cols")
    assert torch.equal(images, original)


def test_row_stream_determinism():
    images = torch.rand(4, 28, 28)
    stream1 = row_stream(images, axis="rows")
    stream2 = row_stream(images, axis="rows")
    assert torch.equal(stream1, stream2)
