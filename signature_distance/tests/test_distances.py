import torch

from signature_distance.distances import (
    choose_rescale_factor,
    method_a_feature_vector,
    method_b_feature_vector,
    pairwise_euclidean_distance,
    per_line_distances,
    rescale_signature,
    within_vs_cross_digit_distance,
)


def _fake_signature(batch, depth=4, width=2):
    dim = sum(width ** n for n in range(depth + 1))
    return torch.rand(batch, dim)


def test_rescale_signature_level0_untouched():
    sig = _fake_signature(3)
    rescaled = rescale_signature(sig, r=2.0, depth=4)
    assert torch.allclose(rescaled[:, 0], sig[:, 0])


def test_rescale_signature_known_scaling():
    sig = torch.ones(1, 31)  # depth=4, width=2: sizes [1,2,4,8,16]
    rescaled = rescale_signature(sig, r=2.0, depth=4)
    expected = torch.cat([
        torch.full((1,), 1.0), torch.full((2,), 2.0), torch.full((4,), 4.0),
        torch.full((8,), 8.0), torch.full((16,), 16.0),
    ], dim=0).unsqueeze(0)
    assert torch.allclose(rescaled, expected)


def test_rescale_signature_wrong_dim_raises():
    sig = torch.rand(2, 10)
    try:
        rescale_signature(sig, r=1.5, depth=4)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_choose_rescale_factor_flattens_decay():
    # A signature with clean geometric decay by factor 0.5 per level should
    # yield r close to 2.0 (the exact inverse), and applying it should make
    # every level's mean magnitude equal.
    torch.manual_seed(0)
    sizes = [1, 2, 4, 8, 16]
    blocks = []
    for n, size in enumerate(sizes):
        blocks.append(torch.full((5, size), 0.5 ** n))
    sig = torch.cat(blocks, dim=1)
    r = choose_rescale_factor(sig, depth=4)
    assert abs(r - 2.0) < 1e-4

    rescaled = rescale_signature(sig, r=r, depth=4)
    idx = 0
    mags = []
    for size in sizes:
        mags.append(rescaled[:, idx:idx + size].abs().mean().item())
        idx += size
    for m in mags:
        assert abs(m - mags[0]) < 1e-4


def test_method_a_feature_vector_is_identity():
    sig = _fake_signature(4)
    assert torch.equal(method_a_feature_vector(sig), sig)


def test_method_b_feature_vector_concatenates_lines():
    line_sigs = torch.rand(5, 16, 31)
    vec = method_b_feature_vector(line_sigs)
    assert vec.shape == (5, 16 * 31)
    assert torch.equal(vec[0], line_sigs[0].reshape(-1))


def test_pairwise_euclidean_distance_properties():
    vectors = torch.rand(6, 10)
    dist = pairwise_euclidean_distance(vectors)
    assert dist.shape == (6, 6)
    assert torch.allclose(torch.diagonal(dist), torch.zeros(6), atol=1e-5)
    assert torch.allclose(dist, dist.T, atol=1e-5)
    assert (dist >= 0).all()


def test_within_vs_cross_digit_distance_separates_clusters():
    # Two well-separated clusters (labels 0 and 1) - within-cluster distance
    # should be far smaller than cross-cluster distance.
    torch.manual_seed(0)
    cluster0 = torch.randn(10, 5) * 0.01 + torch.zeros(5)
    cluster1 = torch.randn(10, 5) * 0.01 + torch.full((5,), 10.0)
    vectors = torch.cat([cluster0, cluster1], dim=0)
    labels = torch.cat([torch.zeros(10), torch.ones(10)]).to(torch.int64)

    result = within_vs_cross_digit_distance(vectors, labels)
    assert result["within_digit_mean"] < result["cross_digit_mean"]
    assert result["ratio_cross_over_within"] > 10


def test_per_line_distances_shape_and_zero_for_identical():
    torch.manual_seed(0)
    sig1 = torch.rand(5, 16, 31)
    d_self = per_line_distances(sig1, sig1)
    assert d_self.shape == (5, 16)
    assert torch.allclose(d_self, torch.zeros(5, 16), atol=1e-6)


def test_per_line_distances_matches_manual_per_line_norm():
    torch.manual_seed(0)
    sig1 = torch.rand(3, 16, 31)
    sig2 = torch.rand(3, 16, 31)
    d = per_line_distances(sig1, sig2)
    for n in range(3):
        for i in range(16):
            expected = (sig1[n, i] - sig2[n, i]).norm()
            assert torch.allclose(d[n, i], expected, atol=1e-5)


def test_per_line_distances_combine_in_quadrature_to_merged_distance():
    # Concatenating the 16 lines then taking one Euclidean norm is
    # mathematically identical to combining the 16 per-line norms in
    # quadrature: ||concat(v_1..v_16)|| == sqrt(sum_i ||v_i||^2). Verifies
    # per_line_distances and method_b_feature_vector are two views of the
    # same underlying signatures, not independently-drifting code paths.
    torch.manual_seed(0)
    sig1 = torch.rand(4, 16, 31)
    sig2 = torch.rand(4, 16, 31)

    merged1 = method_b_feature_vector(sig1)
    merged2 = method_b_feature_vector(sig2)
    merged_dist = (merged1 - merged2).norm(dim=1)

    per_line = per_line_distances(sig1, sig2)
    recombined = (per_line ** 2).sum(dim=1).sqrt()

    assert torch.allclose(merged_dist, recombined, atol=1e-4)
