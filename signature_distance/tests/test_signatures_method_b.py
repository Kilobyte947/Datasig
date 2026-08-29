import torch

from signature_distance.signatures import signature_of_stream
from signature_distance.streams import line_stream, make_reference_lines


def test_signature_straight_line_matches_tensor_exponential():
    # A straight line's truncated signature has an exact closed form:
    # exp(v) = sum_n v^{tensor n} / n!, with v the net displacement - not
    # just the depth-1 term, every level. This checks the full depth-4
    # output against that closed form, not just the identity every path
    # satisfies (see test_signature_level1_equals_net_displacement).
    stream = torch.linspace(0, 1, 5).unsqueeze(0).unsqueeze(-1).expand(1, 5, 2).clone()
    sig = signature_of_stream(stream, depth=4)
    v = 1.0  # net displacement per coordinate
    expected = torch.tensor(
        [1.0]
        + [v] * 2
        + [v**2 / 2] * 4
        + [v**3 / 6] * 8
        + [v**4 / 24] * 16
    )
    assert torch.allclose(sig[0], expected, atol=1e-4)


def test_signature_l_shape_matches_hand_computed_area_term():
    # (0,0) -> (1,0) -> (1,1): two orthogonal unit segments. Straight-line
    # checks alone can't catch a bug in how consecutive *different-direction*
    # segments are combined (a straight line's own area term is zero
    # regardless), so this checks a nonzero cross/"area" term (index 4,
    # word "12" in the depth-2 tensor basis) against a hand-computed value:
    # exp(v1) * exp(v2) with v1=(1,0), v2=(0,1) gives depth-2 term
    # v1(x)v1/2 + v1(x)v2 + v2(x)v2/2 = [0.5, 0, 0, 0] + [0, 1, 0, 0] +
    # [0, 0, 0, 0.5] = [0.5, 1, 0, 0.5].
    stream = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]])
    sig = signature_of_stream(stream, depth=2)
    expected = torch.tensor([1.0, 1.0, 1.0, 0.5, 1.0, 0.0, 0.5])
    assert torch.allclose(sig[0], expected, atol=1e-4)


def test_signature_on_real_line_stream():
    images = torch.rand(3, 28, 28)
    lines = make_reference_lines()
    stream = line_stream(images, lines)  # (3, 16, 32, 2)

    for line_idx in range(stream.shape[1]):
        one_line = stream[:, line_idx]  # (3, 32, 2)
        sig = signature_of_stream(one_line, depth=4)
        assert sig.shape == (3, 31)
        assert torch.isfinite(sig).all()
        expected = one_line[:, -1, :] - one_line[:, 0, :]
        assert torch.allclose(sig[:, 1:3], expected, atol=1e-4)
