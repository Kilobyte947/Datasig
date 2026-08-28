import torch

from signature_distance.signatures import signature_of_stream
from signature_distance.streams import make_pixel_order, patch_sv_stream


def test_signature_shape():
    stream = torch.rand(3, 8, 2)
    sig = signature_of_stream(stream, depth=4)
    # truncated tensor algebra size for width=2, depth=4: 1+2+4+8+16 = 31
    assert sig.shape == (3, 31)
    assert sig.dtype == torch.float32


def test_signature_constant_term_is_one():
    stream = torch.rand(4, 8, 2)
    sig = signature_of_stream(stream, depth=4)
    assert torch.allclose(sig[:, 0], torch.ones(4), atol=1e-5)


def test_signature_level1_equals_net_displacement():
    # Level-1 (depth-1) signature terms are exactly the path's net
    # displacement, independent of the path taken in between - a standard
    # closed-form identity, checked here before trusting the pipeline on
    # real streams (same "checkpoint-gating" pattern as the rest of the repo).
    torch.manual_seed(0)
    stream = torch.rand(5, 10, 2)
    sig = signature_of_stream(stream, depth=4)
    expected = stream[:, -1, :] - stream[:, 0, :]
    assert torch.allclose(sig[:, 1:3], expected, atol=1e-4)


def test_signature_no_nan_or_inf():
    torch.manual_seed(1)
    stream = torch.rand(6, 16, 2)
    sig = signature_of_stream(stream, depth=4)
    assert torch.isfinite(sig).all()


def test_signature_batched_matches_unbatched():
    torch.manual_seed(2)
    stream = torch.rand(4, 12, 2)
    batched = signature_of_stream(stream, depth=4)
    for i in range(stream.shape[0]):
        single = signature_of_stream(stream[i : i + 1], depth=4)
        assert torch.allclose(batched[i], single[0], atol=1e-5)


def test_signature_on_real_patch_sv_stream():
    images = torch.rand(3, 28, 28)
    order = make_pixel_order(k=16, seed=0)
    stream = patch_sv_stream(images, order)  # (3, 16, 2)
    sig = signature_of_stream(stream, depth=4)
    assert sig.shape == (3, 31)
    assert torch.isfinite(sig).all()
    expected = stream[:, -1, :] - stream[:, 0, :]
    assert torch.allclose(sig[:, 1:3], expected, atol=1e-4)
