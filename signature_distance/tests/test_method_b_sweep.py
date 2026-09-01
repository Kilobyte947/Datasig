import torch

from signature_distance.method_b_sweep import (
    build_stream,
    cubic_spline_refine,
    evaluate_config,
    per_line_aucs,
    run_stage_b_validation,
    signature_dim,
)


def test_signature_dim_matches_known_values():
    assert signature_dim(2) == 1 + 2 + 4
    assert signature_dim(4) == 1 + 2 + 4 + 8 + 16
    assert signature_dim(6) == 1 + 2 + 4 + 8 + 16 + 32 + 64


def test_cubic_spline_refine_passes_through_original_points_shape():
    torch.manual_seed(0)
    stream = torch.stack([
        torch.linspace(0, 1, 8).expand(3, 8),
        torch.rand(3, 8),
    ], dim=-1)
    refined = cubic_spline_refine(stream, upsample_factor=4)
    assert refined.shape == (3, 32, 2)
    # endpoints should match closely (spline interpolates exactly at knots)
    assert torch.allclose(refined[:, 0, 1], stream[:, 0, 1], atol=1e-3)
    assert torch.allclose(refined[:, -1, 1], stream[:, -1, 1], atol=1e-3)


def test_build_stream_linear_matches_existing_line_stream_shape():
    images = torch.rand(4, 28, 28)
    stream = build_stream(images, (0, 90), (8, 8), points_per_line=16, interpolation="linear")
    assert stream.shape == (4, 16, 16, 2)


def test_build_stream_cubic_upsamples():
    images = torch.rand(4, 28, 28)
    stream = build_stream(images, (0, 90), (8, 8), points_per_line=16, interpolation="cubic", cubic_upsample=4)
    assert stream.shape == (4, 16, 64, 2)


def test_build_stream_unknown_interpolation_raises():
    images = torch.rand(2, 28, 28)
    try:
        build_stream(images, (0, 90), (8, 8), points_per_line=16, interpolation="quadratic")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_per_line_aucs_shape():
    torch.manual_seed(0)
    sig = torch.rand(10, 4, 7)  # (N, num_lines, sig_dim)
    labels = torch.randint(0, 3, (10,))
    aucs = per_line_aucs(sig, labels)
    assert len(aucs) == 4
    assert all(0.0 <= a <= 1.0 for a in aucs)


def test_evaluate_config_depth_prefix_matches_direct_computation():
    # The whole point of evaluate_config's speed shortcut: depth-3 results
    # sliced from a depth-6 computation must exactly match what you'd get
    # computing depth=3 directly (already verified for signature_of_stream
    # itself elsewhere - this checks evaluate_config's usage of that fact
    # end-to-end, including the per-depth rescale + AUC steps).
    torch.manual_seed(0)
    images = torch.rand(6, 28, 28)
    labels = torch.randint(0, 3, (6,))

    by_depth = evaluate_config(images, labels, (0, 90), (8, 8), points_per_line=8,
                                depths=(2, 3), interpolation="linear", max_depth=3)

    direct = evaluate_config(images, labels, (0, 90), (8, 8), points_per_line=8,
                              depths=(2,), interpolation="linear", max_depth=2)

    assert abs(by_depth[2]["r"] - direct[2]["r"]) < 1e-6
    for a, b in zip(by_depth[2]["line_aucs"], direct[2]["line_aucs"]):
        assert abs(a - b) < 1e-9


def test_run_stage_b_validation_smoke():
    # Tiny/fast smoke test: real training (1 epoch, small sample) just to
    # confirm the plumbing (shared perturbations across finalists, per-line
    # distances/ratios per finalist) runs end-to-end without error and
    # produces well-formed output - not meant to validate the actual
    # numbers (that's the real Stage B run's job).
    finalists = [
        {"name": "f1", "angles_deg": (0, 90), "counts": (8, 8), "points_per_line": 8,
         "depth": 2, "interpolation": "linear"},
        {"name": "f2", "angles_deg": (0,), "counts": (16,), "points_per_line": 8,
         "depth": 2, "interpolation": "linear"},
    ]
    out = run_stage_b_validation(finalists, n_per_class=2, epsilons=(0.05,), seed=0,
                                  cnn_epochs=1, strong_epochs=1, verbose=False)

    assert out["n_images"] == 20
    assert set(out["results"].keys()) == {"f1", "f2"}
    for fname in ("f1", "f2"):
        for mname in ("SmallCNN", "StrongCNN"):
            e = out["results"][fname]["models"][mname]["eps"][0.05]
            assert e["ratio_adv"].shape == (20, 16)
            assert e["ratio_control"].shape == (20, 16)
            # NaN would mean an actual bug (0/0); occasional +inf from a
            # near-zero (but real, positive) distance is an accepted
            # property of the pre-existing ratio=num/dist formula this
            # reuses unmodified, more likely at this smoke test's
            # deliberately tiny/degenerate scale than in a real run.
            assert not torch.isnan(e["ratio_adv"]).any()
            assert not torch.isnan(e["ratio_control"]).any()
