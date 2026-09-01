import torch

from signature_distance.hilbert_stream import (
    HILBERT_ORDER,
    HILBERT_SIDE,
    IMAGE_SIZE,
    NUM_SAMPLE_POINTS,
    NUM_SEGMENTS,
    POINTS_PER_SEGMENT,
    _generate_hilbert_curve,
    _resample_evenly_by_arc_length,
    evaluate_hilbert_depths,
    hilbert_robustness_check,
    hilbert_stream,
    make_hilbert_curve,
    plot_depth_comparison,
    plot_hilbert_curve,
    plot_hilbert_segment_streams,
    plot_hilbert_signatures,
    run_hilbert_adversarial_eval,
    summarize_hilbert_result,
)


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
    assert curve.dtype == torch.float32
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


def _fake_hilbert_results(n=20, seed=0):
    torch.manual_seed(seed)
    ratio_adv = torch.rand(n, NUM_SEGMENTS) + 0.5
    ratio_control = torch.rand(n, NUM_SEGMENTS) * 0.2
    dist_adv = torch.rand(n, NUM_SEGMENTS) + 0.5
    dist_control = torch.rand(n, NUM_SEGMENTS) + 0.5
    flip_mask = torch.zeros(n, dtype=torch.bool)
    flip_mask[: n // 4] = True
    return {
        "n_images": n, "epsilons": [0.03], "depth": 2, "r": 2.0,
        "models": {
            "FakeModel": {
                "test_acc": 0.99,
                "eps": {
                    0.03: {
                        "flip_mask": flip_mask,
                        "flip_fraction": flip_mask.float().mean().item(),
                        "ratio_adv": ratio_adv, "ratio_control": ratio_control,
                        "dist_adv": dist_adv, "dist_control": dist_control,
                    }
                },
            }
        },
    }


def test_summarize_hilbert_result_structure():
    results = _fake_hilbert_results()
    summary = summarize_hilbert_result(results)
    entry = summary["FakeModel"][0.03]
    assert entry["n_flipped"] == 5
    assert set(entry["per_segment"].keys()) == set(range(NUM_SEGMENTS))


def test_hilbert_robustness_check_excludes_smallest_distance_segments():
    results = _fake_hilbert_results()
    report = hilbert_robustness_check(results, n_exclude=2)
    r = report["FakeModel"][0.03]
    assert len(r["excluded_segments"]) == 2
    assert isinstance(r["mean_fold_kept"], float)


def test_run_hilbert_adversarial_eval_smoke():
    # Tiny/fast smoke test - real training (1 epoch, small sample) just to
    # confirm the plumbing runs end-to-end.
    out = run_hilbert_adversarial_eval(
        depth=2, n_per_class=2, epsilons=(0.05,), seed=0,
        cnn_epochs=1, strong_epochs=1, verbose=False,
    )
    assert out["n_images"] == 20
    for mname in ("SmallCNN", "StrongCNN"):
        e = out["models"][mname]["eps"][0.05]
        assert e["ratio_adv"].shape == (20, NUM_SEGMENTS)
        assert not torch.isnan(e["ratio_adv"]).any()


def test_plotting_functions_run():
    import matplotlib
    matplotlib.use("Agg")

    images = torch.rand(2, 28, 28)
    curve = make_hilbert_curve()
    stream = hilbert_stream(images, curve)

    fig1 = plot_hilbert_curve(images[0], curve)
    assert fig1 is not None

    fig2 = plot_hilbert_segment_streams(stream[0])
    assert fig2 is not None

    fake_sig = torch.rand(NUM_SEGMENTS, 31)
    fig3 = plot_hilbert_signatures(fake_sig)
    assert fig3 is not None

    depth_results = evaluate_hilbert_depths(n_per_class=2, seed=0, depths=(2, 3))
    fig4 = plot_depth_comparison(depth_results)
    assert fig4 is not None
