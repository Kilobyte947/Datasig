import torch

from signature_distance.headline_bootstrap import (
    bootstrap_quantile_ci,
    check_overlap,
    collect_headline_bootstrap,
    plot_headline_ci,
)


def test_bootstrap_quantile_ci_reasonable_on_synthetic_data():
    torch.manual_seed(0)
    # 200 images x 12 lines, values roughly uniform on [0, 10] - true P90
    # of a Uniform(0,10) is 9.0.
    ratio_matrix = torch.rand(200, 12) * 10.0
    result = bootstrap_quantile_ci(ratio_matrix, quantile=0.90, n_bootstrap=500, ci_level=0.90, seed=0)

    assert abs(result["point_estimate"] - 9.0) < 1.0
    assert result["ci_low"] < result["point_estimate"] < result["ci_high"]
    assert result["ci_low"] < result["ci_high"]
    assert result["n_images"] == 200
    assert result["n_bootstrap"] == 500


def test_bootstrap_quantile_ci_is_deterministic():
    torch.manual_seed(0)
    ratio_matrix = torch.rand(50, 8) * 5.0
    r1 = bootstrap_quantile_ci(ratio_matrix, n_bootstrap=200, seed=42)
    r2 = bootstrap_quantile_ci(ratio_matrix, n_bootstrap=200, seed=42)
    assert r1["ci_low"] == r2["ci_low"]
    assert r1["ci_high"] == r2["ci_high"]


def test_bootstrap_ci_narrower_with_more_images_same_distribution():
    torch.manual_seed(0)
    small_n = torch.rand(30, 12) * 10.0
    large_n = torch.rand(300, 12) * 10.0
    r_small = bootstrap_quantile_ci(small_n, n_bootstrap=500, seed=0)
    r_large = bootstrap_quantile_ci(large_n, n_bootstrap=500, seed=0)
    width_small = r_small["ci_high"] - r_small["ci_low"]
    width_large = r_large["ci_high"] - r_large["ci_low"]
    assert width_large < width_small


def _fake_bootstrap_data(overlap_b_clean=False, overlap_c_clean=True):
    def entry(point, half_width):
        return {"point_estimate": point, "ci_low": point - half_width, "ci_high": point + half_width,
                "ci_level": 0.90, "n_bootstrap": 500, "n_images": 200, "boot_std": half_width / 2}

    small_b_clean = entry(5.0, 0.5) if not overlap_b_clean else entry(5.0, 3.0)
    strong_b_clean = entry(12.0, 0.5) if not overlap_b_clean else entry(6.0, 3.0)

    return {
        "primary_eps": 0.03, "quantile": 0.90, "ci_level": 0.90, "n_bootstrap": 500,
        "models": {
            "SmallCNN": {
                "method_b": {"clean": small_b_clean, "adv": entry(22.0, 2.0)},
                "method_c": {"clean": entry(1.5, 2.0) if overlap_c_clean else entry(1.5, 0.2),
                             "adv": entry(8.0, 1.0)},
            },
            "StrongCNN": {
                "method_b": {"clean": strong_b_clean, "adv": entry(32.0, 2.0)},
                "method_c": {"clean": entry(3.9, 2.0) if overlap_c_clean else entry(3.9, 0.2),
                             "adv": entry(10.5, 1.0)},
            },
        },
    }


def test_check_overlap_detects_non_overlapping_pair():
    data = _fake_bootstrap_data(overlap_b_clean=False)
    result = check_overlap(data)
    assert result["method_b"]["clean"]["overlap"] is False


def test_check_overlap_detects_overlapping_pair():
    data = _fake_bootstrap_data(overlap_c_clean=True)
    result = check_overlap(data)
    assert result["method_c"]["clean"]["overlap"] is True


def test_plot_headline_ci_runs():
    import matplotlib
    matplotlib.use("Agg")
    fig = plot_headline_ci(_fake_bootstrap_data())
    assert fig is not None


def test_collect_headline_bootstrap_smoke():
    # Tiny/fast smoke test - real training (1 epoch, small sample, few
    # bootstrap resamples) just to confirm the combined driver runs
    # end-to-end and every one of the 8 CIs is well-formed.
    data = collect_headline_bootstrap(
        n_per_class=2, epsilons=(0.05,), primary_eps=0.05, seed=0,
        cnn_epochs=1, strong_epochs=1, n_bootstrap=20, verbose=False,
    )
    assert set(data["models"].keys()) == {"SmallCNN", "StrongCNN"}
    for mname, entry in data["models"].items():
        for method_key in ("method_b", "method_c"):
            for cond in ("clean", "adv"):
                r = entry[method_key][cond]
                assert r["ci_low"] <= r["point_estimate"] <= r["ci_high"]
                assert r["n_images"] == 20

    overlap = check_overlap(data)
    for method_key in ("method_b", "method_c"):
        for cond in ("clean", "adv"):
            assert isinstance(overlap[method_key][cond]["overlap"], bool)
