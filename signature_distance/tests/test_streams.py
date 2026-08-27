import torch

from signature_distance.streams import make_pixel_order, patch_sv_stream, row_stream

# ---------------------------------------------------------------------------
# Method 1: patch singular-value stream
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
    import pytest
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
# Method 2: horizontal-line vector stream
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
