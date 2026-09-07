import torch

from signature_distance.data_pool import load_eval_pool
from signature_distance.distances import (
    METHOD_A_PIXEL_ORDER,
    METHOD_A_R,
    METHOD_B_BORDER_LINE_INDICES,
    METHOD_B_INFORMATIVE_LINE_INDICES,
    METHOD_B_WINNER_LINES,
    _safe_within_vs_cross,
    choose_rescale_factor,
    evaluate_hilbert_depths,
    level_slices,
    margin,
    mask_signature_levels,
    method_a_feature_vector,
    method_a_signature_distance,
    method_b_feature_vector,
    method_b_per_line_cross_within,
    method_b_signature_distance,
    method_c_per_segment_cross_within,
    pairwise_euclidean_distance,
    per_line_distances,
    pixel_euclidean_cross_within,
    pixel_euclidean_distance,
    rescale_signature,
    run_cross_within_comparison,
    run_level_decomposition,
    run_per_line_auc_diagnostic,
    within_vs_cross_digit_distance,
)
from signature_distance.signatures import signature_of_stream
from signature_distance.streams import NUM_SEGMENTS, make_pixel_order, patch_sv_stream


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


# ---------------------------------------------------------------------------
# Margin, pixel/signature distances - shared numerator/denominator building
# blocks
# ---------------------------------------------------------------------------


def test_margin_matches_hand_computed_logits():
    import torch.nn as nn

    class FixedLogits(nn.Module):
        def forward(self, x):
            n = x.shape[0]
            return torch.tensor([[1.0, 5.0, 3.0]]).expand(n, 3).clone()

    model = FixedLogits()
    x = torch.rand(2, 1, 28, 28)
    y = torch.tensor([1, 2])  # true class is the max for sample 0, not for sample 1
    m = margin(model, x, y)
    # sample 0: true logit 5.0, runner-up 3.0 -> margin 2.0
    # sample 1: true logit 3.0, runner-up 5.0 -> margin -2.0
    assert torch.allclose(m, torch.tensor([2.0, -2.0]), atol=1e-5)


def test_pixel_euclidean_distance_zero_for_identical():
    x = torch.rand(3, 1, 28, 28)
    d = pixel_euclidean_distance(x, x)
    assert torch.allclose(d, torch.zeros(3), atol=1e-6)


def test_method_a_signature_distance_zero_for_identical_positive_otherwise():
    torch.manual_seed(0)
    images = torch.rand(3, 28, 28)
    d_self = method_a_signature_distance(images, images)
    assert torch.allclose(d_self, torch.zeros(3, dtype=d_self.dtype), atol=1e-4)

    other = torch.rand(3, 28, 28)
    d_other = method_a_signature_distance(images, other)
    assert (d_other > 0).all()


def test_method_a_signature_distance_matches_default_r_from_sanity_check():
    # METHOD_A_R is hardcoded from run_experiment.sanity_check_demo's Phase 4
    # run (seed=0) rather than re-derived here - this just checks the two
    # don't silently drift apart.
    images, _ = load_eval_pool(n_per_class=30, seed=0)
    sig_raw = signature_of_stream(patch_sv_stream(images, METHOD_A_PIXEL_ORDER), depth=4)
    r = choose_rescale_factor(sig_raw, depth=4)
    assert abs(r - METHOD_A_R) < 1e-6


def test_method_b_signature_distance_zero_for_identical_positive_otherwise():
    torch.manual_seed(0)
    images = torch.rand(3, 28, 28)
    d_self = method_b_signature_distance(images, images)
    assert torch.allclose(d_self, torch.zeros(3, dtype=d_self.dtype), atol=1e-4)

    other = torch.rand(3, 28, 28)
    d_other = method_b_signature_distance(images, other)
    assert (d_other > 0).all()


# ---------------------------------------------------------------------------
# Per-line distance diagnostic (Method B) - AUC ranking of the 16 individual
# lines
# ---------------------------------------------------------------------------


def test_run_per_line_auc_diagnostic_structure():
    # Small sample for speed - correctness of the underlying distance math
    # is already covered above; this just checks the diagnostic's own
    # orchestration/reporting shape.
    result = run_per_line_auc_diagnostic(n_per_class=3, seed=0)

    assert result["n_images"] == 30
    assert "merged" in result["measures"]
    line_keys = [k for k in result["measures"] if k.startswith("line_")]
    assert len(line_keys) == 16

    for key, entry in result["measures"].items():
        assert 0.0 <= entry["auc"] <= 1.0
        assert 0.0 <= entry["fpr_at_tpr90"] <= 1.0

    # ranked is sorted descending by AUC
    aucs = [entry["auc"] for _, entry in result["ranked"]]
    assert aucs == sorted(aucs, reverse=True)
    assert len(result["ranked"]) == 17

    assert isinstance(result["best_individual_line_beats_merged"], bool)


# ---------------------------------------------------------------------------
# Same/different-digit AUC per depth, for Method C's Hilbert segments
# (Stage A cheap screen - no model training)
# ---------------------------------------------------------------------------


def test_evaluate_hilbert_depths_structure():
    result = evaluate_hilbert_depths(n_per_class=3, seed=0, depths=(2, 3))
    assert set(result.keys()) == {2, 3}
    for depth, entry in result.items():
        assert entry["n_segments"] == NUM_SEGMENTS
        assert len(entry["segment_aucs"]) == NUM_SEGMENTS
        assert 0.0 <= entry["best_auc"] <= 1.0


def test_evaluate_hilbert_depths_prefix_shortcut_matches_direct():
    # Same correctness check used for Method B's sweep: results for a
    # given depth via the max-depth-then-slice shortcut must exactly
    # match computing that depth directly as the only requested depth.
    via_shortcut = evaluate_hilbert_depths(n_per_class=3, seed=0, depths=(2, 4))
    direct = evaluate_hilbert_depths(n_per_class=3, seed=0, depths=(2,))
    assert abs(via_shortcut[2]["r"] - direct[2]["r"]) < 1e-6
    for a, b in zip(via_shortcut[2]["segment_aucs"], direct[2]["segment_aucs"]):
        assert abs(a - b) < 1e-9


# ---------------------------------------------------------------------------
# Border/informative line partition (Method B, winner geometry) - shared
# constant sanity check
# ---------------------------------------------------------------------------


def test_border_and_informative_indices_partition_all_16_lines():
    assert len(METHOD_B_BORDER_LINE_INDICES) == 2
    assert len(METHOD_B_INFORMATIVE_LINE_INDICES) == 14
    assert set(METHOD_B_BORDER_LINE_INDICES) | set(METHOD_B_INFORMATIVE_LINE_INDICES) == set(range(16))
    assert set(METHOD_B_BORDER_LINE_INDICES).isdisjoint(METHOD_B_INFORMATIVE_LINE_INDICES)


# ---------------------------------------------------------------------------
# Cross/within-digit distance ratio, per-path (never merged), for Method B's
# winning configuration and Method C, plus a plain pixel-Euclidean baseline
# ---------------------------------------------------------------------------


def test_safe_within_vs_cross_handles_zero_division_gracefully():
    # Real, discovered failure mode: a structurally degenerate line (e.g.
    # Method B's border lines) produces the IDENTICAL vector for every
    # image, giving within-digit distance exactly 0 - within_vs_cross_
    # digit_distance itself raises ZeroDivisionError on this; the wrapper
    # must catch it and report a clearly-flagged degenerate result instead
    # of crashing the whole comparison.
    vectors = torch.zeros(10, 5)
    labels = torch.randint(0, 3, (10,))
    result = _safe_within_vs_cross(vectors, labels)
    assert result["degenerate"] is True
    assert result["ratio_cross_over_within"] != result["ratio_cross_over_within"]  # NaN

    # Non-degenerate case still works normally through the same wrapper.
    torch.manual_seed(0)
    cluster0 = torch.randn(5, 5) * 0.01
    cluster1 = torch.randn(5, 5) * 0.01 + 10.0
    vectors2 = torch.cat([cluster0, cluster1], dim=0)
    labels2 = torch.cat([torch.zeros(5), torch.ones(5)]).to(torch.int64)
    result2 = _safe_within_vs_cross(vectors2, labels2)
    assert result2["degenerate"] is False
    assert result2["ratio_cross_over_within"] > 10


def test_pixel_euclidean_cross_within_separates_synthetic_clusters():
    # Two well-separated pixel-space clusters - within should be far
    # smaller than cross, mirroring this file's own synthetic sanity check
    # for the underlying (reused, unmodified) function.
    torch.manual_seed(0)
    cluster0 = torch.rand(10, 28, 28) * 0.05
    cluster1 = torch.rand(10, 28, 28) * 0.05 + 0.9
    images = torch.cat([cluster0, cluster1], dim=0)
    labels = torch.cat([torch.zeros(10), torch.ones(10)]).to(torch.int64)

    result = pixel_euclidean_cross_within(images, labels)
    assert result["within_digit_mean"] < result["cross_digit_mean"]
    assert result["ratio_cross_over_within"] > 2


def test_method_b_per_line_cross_within_structure():
    torch.manual_seed(0)
    images = torch.rand(30, 28, 28)
    labels = torch.randint(0, 10, (30,))
    result = method_b_per_line_cross_within(images, labels)

    n_lines = METHOD_B_WINNER_LINES.shape[0]
    assert result["num_lines"] == n_lines
    assert set(result["per_line"].keys()) == set(range(n_lines))
    for entry in result["per_line"].values():
        assert entry["ratio_cross_over_within"] > 0
    # the border-adjacent lines are always excluded from the aggregate
    # stats (structural convention, not just whichever happen to be
    # degenerate on this particular sample - see the function's own
    # docstring for why relying on degeneracy alone is sample-fragile)
    assert set(result["degenerate_lines"]) == {0, 15}
    included = {i: v for i, v in result["per_line"].items() if i not in result["degenerate_lines"]}
    assert result["mean_ratio_over_lines"] == sum(
        v["ratio_cross_over_within"] for v in included.values()
    ) / len(included)
    assert result["best_line_ratio"] >= result["mean_ratio_over_lines"] >= result["worst_line_ratio"]
    assert "ratio_cross_over_within" in result["merged"]


def test_method_c_per_segment_cross_within_structure():
    torch.manual_seed(0)
    images = torch.rand(30, 28, 28)
    labels = torch.randint(0, 10, (30,))
    result = method_c_per_segment_cross_within(images, labels)

    assert result["num_segments"] == NUM_SEGMENTS
    assert set(result["per_segment"].keys()) == set(range(NUM_SEGMENTS))
    for entry in result["per_segment"].values():
        assert entry["ratio_cross_over_within"] > 0
    assert result["best_segment_ratio"] >= result["mean_ratio_over_segments"] >= result["worst_segment_ratio"]
    assert "ratio_cross_over_within" in result["merged"]


def test_run_cross_within_comparison_smoke():
    # Small/fast: no model training or attack generation involved at all
    # (this is a pure label-based check on clean images), so no need for
    # a tiny/expensive-training-style smoke test - just a smaller sample.
    # Real MNIST images (not synthetic) - this is what originally caught
    # the degenerate-border-line zero-division bug, so it's kept as the
    # regression check for it: Method B's 2 border lines (winner geometry
    # is 16h+0v) should show up as degenerate (constant across every real
    # digit image), and the aggregate stats must still come out finite
    # over the remaining ones.
    result = run_cross_within_comparison(n_per_class=3, seed=0, verbose=False)
    assert result["n_images"] == 30
    assert result["pixel"]["ratio_cross_over_within"] > 0
    assert len(result["method_b"]["degenerate_lines"]) == 2
    assert result["method_b"]["mean_ratio_over_lines"] > 0
    assert result["method_c"]["mean_ratio_over_segments"] > 0


# ---------------------------------------------------------------------------
# Level-wise decomposition of Method A/B signature distances
# ---------------------------------------------------------------------------


# Gate 1 - level index utilities

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
    # signature_of_stream returns float32 regardless of the package's float64
    # default (JAX's own precision ceiling, see signatures.py's docstring) -
    # torch.allclose requires matching dtypes unlike +/-/*, so promote explicitly.
    actual_dist = (masked[0] - masked[1]).norm().to(expected_dist.dtype)

    assert torch.allclose(actual_dist, expected_dist, atol=1e-5)


# Checkpoint 2 - the decomposition harness

def test_run_level_decomposition_structure():
    # Uses the real Phase 4 sample size (n_per_class=30), not a smaller one
    # for speed: at very small samples, Method B's level1_only variant can
    # hit an *exact* zero within-digit distance (every reference line starts
    # and ends at the image border, where MNIST intensity is ~0 for nearly
    # every image - see the level_fraction finding below), which would
    # divide by zero in within_vs_cross_digit_distance. That's a real
    # property of the data, not something to work around by changing this
    # file - so this test just runs at a sample size large enough to avoid
    # it, same as the reproduction gate below.
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
    # 4 table (README.md / README.md) to floating-point
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
            expected = r * (delta_sigma1[i] - delta_sigma1[j]).abs()
            # signature_of_stream returns float32 regardless of the package's
            # float64 default (JAX's own precision ceiling) - torch.allclose
            # requires matching dtypes unlike +/-/*, so promote explicitly.
            actual = (masked[i] - masked[j]).norm().to(expected.dtype)
            assert torch.allclose(actual, expected, atol=1e-4)
