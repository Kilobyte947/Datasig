import torch

from signature_distance.data_pool import load_eval_pool
from signature_distance.distances import choose_rescale_factor, rescale_signature
from signature_distance.level_decomposition import (
    level_slices,
    mask_signature_levels,
    run_level_decomposition,
)
from signature_distance.signatures import signature_of_stream
from signature_distance.streams import make_pixel_order, patch_sv_stream


# ---------------------------------------------------------------------------
# Gate 1 - level index utilities
# ---------------------------------------------------------------------------

def test_level_slices_partitions_range_exactly():
    slices = level_slices(4, width=2)
    assert list(slices.keys()) == [0, 1, 2, 3, 4]

    sizes = [s.stop - s.start for s in slices.values()]
    assert sizes == [1, 2, 4, 8, 16]

    covered = []
    for s in slices.values():
        covered.extend(range(s.start, s.stop))
    assert covered == list(range(31))  # no gaps, no overlap, in order

    assert slices[0] == slice(0, 1)
    assert slices[1] == slice(1, 3)
    assert slices[4] == slice(15, 31)


def test_mask_signature_levels_preserves_shape_dtype_and_zeros_complement():
    torch.manual_seed(0)
    sig = torch.rand(5, 31, dtype=torch.float64)
    masked = mask_signature_levels(sig, [1, 3], depth=4)

    assert masked.shape == sig.shape
    assert masked.dtype == sig.dtype

    slices = level_slices(4, width=2)
    for level, s in slices.items():
        if level in (1, 3):
            assert torch.equal(masked[..., s], sig[..., s])
        else:
            assert torch.equal(masked[..., s], torch.zeros_like(sig[..., s]))


def test_mask_signature_levels_handles_method_b_shape_before_concatenation():
    # Method B's per-line signatures are (N, num_lines, sig_dim) before
    # distances.method_b_feature_vector concatenates them.
    torch.manual_seed(0)
    sig = torch.rand(3, 16, 31)
    masked = mask_signature_levels(sig, [2], depth=4)

    assert masked.shape == sig.shape
    slices = level_slices(4, width=2)
    assert torch.equal(masked[..., slices[2]], sig[..., slices[2]])
    assert torch.equal(masked[..., slices[1]], torch.zeros_like(sig[..., slices[1]]))
    assert torch.equal(masked[..., slices[4]], torch.zeros_like(sig[..., slices[4]]))


def test_orthogonality_of_per_level_squared_distances():
    # The level blocks are disjoint coordinates, so summing each level's own
    # squared distance (levels 1..depth, masked one at a time) must equal
    # the total squared distance of the level-1..depth-masked vector -
    # Pythagorean, a cheap correctness check on the slicing itself.
    torch.manual_seed(0)
    depth = 4
    sig = torch.randn(6, 31)

    total_masked = mask_signature_levels(sig, list(range(1, depth + 1)), depth=depth)
    diffs = total_masked.unsqueeze(0) - total_masked.unsqueeze(1)  # (6, 6, 31)
    total_sq = (diffs ** 2).sum(dim=-1)  # (6, 6)

    per_level_sq_sum = torch.zeros(6, 6)
    for level in range(1, depth + 1):
        masked = mask_signature_levels(sig, [level], depth=depth)
        d = masked.unsqueeze(0) - masked.unsqueeze(1)
        per_level_sq_sum = per_level_sq_sum + (d ** 2).sum(dim=-1)

    assert torch.allclose(total_sq, per_level_sq_sum, atol=1e-5)


def test_level1_only_matches_net_displacement_for_straight_line():
    # Reuses the straight-line construction from test_signatures_method_b.py
    # (test_signature_straight_line_matches_tensor_exponential): a
    # straight-line path's signature has an exact closed form, and its
    # level-1 block is exactly the net displacement stream[-1] - stream[0].
    v1, v2 = 1.0, 2.5
    stream1 = torch.linspace(0, v1, 5).unsqueeze(0).unsqueeze(-1).expand(1, 5, 2).clone()
    stream2 = torch.linspace(0, v2, 5).unsqueeze(0).unsqueeze(-1).expand(1, 5, 2).clone()
    stream = torch.cat([stream1, stream2], dim=0)  # (2, 5, 2)

    depth = 4
    sig = signature_of_stream(stream, depth=depth)
    masked = mask_signature_levels(sig, [1], depth=depth)

    net_disp = stream[:, -1] - stream[:, 0]  # (2, 2)
    expected_dist = (net_disp[0] - net_disp[1]).norm()
    actual_dist = (masked[0] - masked[1]).norm()

    assert torch.allclose(actual_dist, expected_dist, atol=1e-5)


# ---------------------------------------------------------------------------
# Checkpoint 2 - the decomposition harness
# ---------------------------------------------------------------------------

def test_run_level_decomposition_structure():
    # Uses the real Phase 4 sample size (n_per_class=30), not a smaller one
    # for speed: at very small samples, Method B's level1_only variant can
    # hit an *exact* zero within-digit distance (every reference line starts
    # and ends at the image border, where MNIST intensity is ~0 for nearly
    # every image - see the level_fraction finding below), which would
    # divide by zero in within_vs_cross_digit_distance. That's a real
    # property of the data, not something to work around by changing
    # distances.py - so this test just runs at a sample size large enough
    # to avoid it, same as the reproduction gate below.
    result = run_level_decomposition(n_per_class=30, seed=0, depth=4)

    assert result["n_images"] == 300
    expected_variants = {
        "all", "level1_only", "level2plus", "level2_only", "level3_only", "level4_only",
    }
    assert set(result["variants"].keys()) == expected_variants

    for label, entry in result["variants"].items():
        for method in ("method_a", "method_b"):
            d = entry[method]
            assert "within_digit_mean" in d and "cross_digit_mean" in d
            assert "ratio_cross_over_within" in d

    for method in ("method_a", "method_b"):
        fractions = result["level_fraction"][method]
        assert set(fractions.keys()) == {1, 2, 3, 4}
        assert abs(sum(fractions.values()) - 1.0) < 1e-4


def test_pixel_order_seed_isolates_pixel_order_from_pool():
    # Default (pixel_order_seed=None) must exactly match passing
    # pixel_order_seed=seed explicitly - this is the behaviour Gate 2's
    # reproduction check below depends on.
    default_result = run_level_decomposition(n_per_class=30, seed=0, depth=4)
    explicit_result = run_level_decomposition(n_per_class=30, seed=0, depth=4, pixel_order_seed=0)
    assert default_result["pixel_order_seed"] == 0
    assert explicit_result["pixel_order_seed"] == 0
    assert default_result["r_a"] == explicit_result["r_a"]
    assert (
        default_result["variants"]["all"]["method_a"]
        == explicit_result["variants"]["all"]["method_a"]
    )

    # Same pool (seed=0), different pixel order -> Method A's results
    # change, but Method B's must not (make_reference_lines doesn't depend
    # on a seed at all).
    varied_result = run_level_decomposition(n_per_class=30, seed=0, depth=4, pixel_order_seed=1)
    assert varied_result["pixel_order_seed"] == 1
    assert varied_result["r_a"] != default_result["r_a"]
    assert varied_result["r_b"] == default_result["r_b"]
    assert (
        varied_result["variants"]["all"]["method_b"]
        == default_result["variants"]["all"]["method_b"]
    )


def test_all_variant_reproduces_documented_phase4_numbers():
    # Gate 2 (hard stop): the `all` variant must match the documented Phase
    # 4 table (README.md / level_decomposition.md) to floating-point
    # tolerance, since it uses the exact same protocol as
    # run_experiment.sanity_check_demo (same pool, same streams/signatures,
    # same independently-derived r, applied before masking). Documented
    # values are rounded, so compare to ~2 decimal places on the means and
    # ~3 on the ratio.
    result = run_level_decomposition(n_per_class=30, seed=0, depth=4)
    all_a = result["variants"]["all"]["method_a"]
    all_b = result["variants"]["all"]["method_b"]

    assert abs(result["r_a"] - 1.656) < 0.01
    assert abs(all_a["within_digit_mean"] - 14.60) < 0.01
    assert abs(all_a["cross_digit_mean"] - 17.18) < 0.01
    assert abs(all_a["ratio_cross_over_within"] - 1.176) < 0.002

    assert abs(result["r_b"] - 2.860) < 0.01
    assert abs(all_b["within_digit_mean"] - 28.60) < 0.01
    assert abs(all_b["cross_digit_mean"] - 33.17) < 0.01
    assert abs(all_b["ratio_cross_over_within"] - 1.160) < 0.002


def test_method_a_level1_only_closed_form_on_real_batch():
    # Gate 2b: Method A's level1_only pairwise distance must equal
    # r * |delta_sigma1_i - delta_sigma1_j| exactly - the Δt component of
    # the level-1 block cancels identically across every pair, since the
    # time channel is generated by identical code regardless of image.
    images, labels = load_eval_pool(n_per_class=2, seed=0)  # small batch
    order = make_pixel_order(k=64, seed=0)
    stream = patch_sv_stream(images, order)  # (N, 64, 2), columns [t, sigma1]

    # Premise the closed form depends on: the time-channel endpoints are
    # identical across every image.
    assert torch.allclose(stream[:, 0, 0], stream[0, 0, 0].expand(stream.shape[0]))
    assert torch.allclose(stream[:, -1, 0], stream[0, -1, 0].expand(stream.shape[0]))

    depth = 4
    sig_raw = signature_of_stream(stream, depth=depth)
    r = choose_rescale_factor(sig_raw, depth=depth)
    sig = rescale_signature(sig_raw, r=r, depth=depth)
    masked = mask_signature_levels(sig, [1], depth=depth)

    delta_sigma1 = stream[:, -1, 1] - stream[:, 0, 1]  # (N,)

    n = images.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            actual = (masked[i] - masked[j]).norm()
            expected = r * (delta_sigma1[i] - delta_sigma1[j]).abs()
            assert torch.allclose(actual, expected, atol=1e-4)
