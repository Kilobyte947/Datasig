import torch

from signature_distance.cross_within_per_path import (
    _safe_within_vs_cross,
    method_b_per_line_cross_within,
    method_c_per_segment_cross_within,
    pixel_euclidean_cross_within,
    run_cross_within_comparison,
)
from signature_distance.hilbert_stream import NUM_SEGMENTS
from signature_distance.pgd_adversarial_eval import METHOD_B_WINNER_LINES


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
    # smaller than cross, mirroring test_distances.py's own synthetic
    # sanity check for the underlying (reused, unmodified) function.
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
    # the 4 border-adjacent lines are always excluded from the aggregate
    # stats (structural convention, not just whichever happen to be
    # degenerate on this particular sample - see the function's own
    # docstring for why relying on degeneracy alone is sample-fragile)
    assert set(result["degenerate_lines"]) == {0, 11, 12, 15}
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
    # regression check for it: Method B's 4 border lines should show up
    # as degenerate (constant across every real digit image), and the
    # aggregate stats must still come out finite over the remaining ones.
    result = run_cross_within_comparison(n_per_class=3, seed=0, verbose=False)
    assert result["n_images"] == 30
    assert result["pixel"]["ratio_cross_over_within"] > 0
    assert len(result["method_b"]["degenerate_lines"]) == 4
    assert result["method_b"]["mean_ratio_over_lines"] > 0
    assert result["method_c"]["mean_ratio_over_segments"] > 0
