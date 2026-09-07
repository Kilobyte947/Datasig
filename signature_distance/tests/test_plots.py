import torch

from signature_distance.distances import evaluate_hilbert_depths
from signature_distance.plots import (
    plot_depth_comparison,
    plot_headline_ci,
    plot_headline_punchline,
    plot_hilbert_curve,
    plot_hilbert_segment_streams,
    plot_hilbert_signatures,
    plot_hilbert_spike_gallery,
    plot_per_line_bar,
    plot_ratio_distribution,
    plot_reference_lines_with_metric,
    plot_spike_comparison,
    plot_spike_gallery,
)
from signature_distance.streams import NUM_SEGMENTS, hilbert_stream, make_hilbert_curve, make_reference_lines


def test_plot_reference_lines_with_metric_runs():
    import matplotlib
    matplotlib.use("Agg")
    image = torch.rand(28, 28)
    lines = make_reference_lines(angles_deg=(0, 90), counts=(12, 4))
    metric_values = torch.rand(lines.shape[0])
    fig = plot_reference_lines_with_metric(image, lines, metric_values, colorbar_label="fold-ratio")
    assert fig is not None


def test_plot_ratio_distribution_runs_with_tensors_and_arrays():
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    adv = torch.rand(50) + 1.0
    ctrl = torch.rand(50) * 0.2
    fig = plot_ratio_distribution(adv, ctrl)
    assert fig is not None

    fig2 = plot_ratio_distribution(np.array(adv), np.array(ctrl))
    assert fig2 is not None


def test_plot_per_line_bar_runs_and_highlights():
    import matplotlib
    matplotlib.use("Agg")
    values = {i: float(i) for i in range(5)}
    fig = plot_per_line_bar(values, highlight_key=3, ylabel="fold-ratio")
    assert fig is not None


# ---------------------------------------------------------------------------
# Method C: Hilbert curve
# ---------------------------------------------------------------------------


def test_hilbert_plotting_functions_run():
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


# ---------------------------------------------------------------------------
# Spike galleries (Method B, Method C, and the two side by side)
# ---------------------------------------------------------------------------


def _fake_per_path_results(n=20, num_lines=16, seed=0):
    # Method B border-line indices used elsewhere (0, 15) for the
    # "simulate near-zero border distances" realism touch below.
    border_line_indices = (0, 15)
    torch.manual_seed(seed)
    ratio_adv = torch.rand(n, num_lines) + 0.1
    ratio_control = torch.rand(n, num_lines) * 0.2
    dist_adv = torch.rand(n, num_lines) + 0.5
    dist_adv[:, list(border_line_indices)] *= 0.01  # simulate near-zero border distances
    flip_mask = torch.zeros(n, dtype=torch.bool)
    flip_mask[: n // 4] = True  # first quarter "flip"
    return {
        "n_images": n, "epsilons": [0.03], "labels": torch.randint(0, 10, (n,)),
        "images": torch.rand(n, 28, 28),
        "models": {
            "FakeModel": {
                "test_acc": 0.99,
                "eps": {
                    0.03: {
                        "flip_mask": flip_mask,
                        "flip_fraction": flip_mask.float().mean().item(),
                        "ratio_adv": ratio_adv, "ratio_control": ratio_control,
                        "dist_adv": dist_adv, "dist_control": torch.rand(n, num_lines) + 0.5,
                        "x_adv": torch.rand(n, 28, 28), "x_control": torch.rand(n, 28, 28),
                    }
                },
            }
        },
    }


def test_plot_spike_gallery_runs():
    import matplotlib
    matplotlib.use("Agg")
    results = _fake_per_path_results()
    fig = plot_spike_gallery(results, "FakeModel", 0.03, pair_idx=0)
    assert fig is not None


def _fake_hilbert_results_with_images(n=20, seed=0):
    torch.manual_seed(seed)
    curve = make_hilbert_curve()
    images = torch.rand(n, 28, 28)
    x_adv = torch.rand(n, 28, 28)
    ratio_adv = torch.rand(n, NUM_SEGMENTS) + 0.1
    ratio_control = torch.rand(n, NUM_SEGMENTS) * 0.2
    flip_mask = torch.zeros(n, dtype=torch.bool)
    flip_mask[: n // 4] = True
    return {
        "n_images": n, "epsilons": [0.03], "depth": 3, "r": 2.5,
        "images": images, "labels": torch.randint(0, 10, (n,)), "curve": curve,
        "models": {
            "FakeModel": {
                "test_acc": 0.99,
                "eps": {
                    0.03: {
                        "flip_mask": flip_mask,
                        "flip_fraction": flip_mask.float().mean().item(),
                        "ratio_adv": ratio_adv, "ratio_control": ratio_control,
                        "dist_adv": torch.rand(n, NUM_SEGMENTS) + 0.5,
                        "dist_control": torch.rand(n, NUM_SEGMENTS) + 0.5,
                        "x_adv": x_adv, "x_control": torch.rand(n, 28, 28),
                    }
                },
            }
        },
    }


def _fake_method_b_results_matching(x_adv, images, labels, n=20, seed=0):
    torch.manual_seed(seed)
    ratio_adv = torch.rand(n, 16) + 0.1
    ratio_control = torch.rand(n, 16) * 0.2
    flip_mask = torch.zeros(n, dtype=torch.bool)
    flip_mask[: n // 4] = True
    return {
        "n_images": n, "epsilons": [0.03], "labels": labels, "images": images,
        "models": {
            "FakeModel": {
                "test_acc": 0.99,
                "eps": {
                    0.03: {
                        "flip_mask": flip_mask,
                        "flip_fraction": flip_mask.float().mean().item(),
                        "ratio_adv": ratio_adv, "ratio_control": ratio_control,
                        "dist_adv": torch.rand(n, 16) + 0.5, "dist_control": torch.rand(n, 16) + 0.5,
                        "x_adv": x_adv, "x_control": torch.rand(n, 28, 28),
                    }
                },
            }
        },
    }


def test_plot_hilbert_spike_gallery_runs():
    import matplotlib
    matplotlib.use("Agg")
    results = _fake_hilbert_results_with_images()
    fig = plot_hilbert_spike_gallery(results, "FakeModel", 0.03, pair_idx=0)
    assert fig is not None


def test_plot_spike_comparison_runs_on_matching_pairs():
    import matplotlib
    matplotlib.use("Agg")
    results_c = _fake_hilbert_results_with_images()
    x_adv = results_c["models"]["FakeModel"]["eps"][0.03]["x_adv"]
    results_b = _fake_method_b_results_matching(x_adv, results_c["images"], results_c["labels"])

    fig = plot_spike_comparison(results_b, results_c, "FakeModel", 0.03, pair_idx=0)
    assert fig is not None


def test_plot_spike_comparison_rejects_mismatched_pairs():
    import matplotlib
    matplotlib.use("Agg")
    results_c = _fake_hilbert_results_with_images()
    mismatched_x_adv = torch.rand(20, 28, 28)  # deliberately NOT results_c's x_adv
    results_b = _fake_method_b_results_matching(mismatched_x_adv, results_c["images"], results_c["labels"])

    try:
        plot_spike_comparison(results_b, results_c, "FakeModel", 0.03, pair_idx=0)
        assert False, "expected an assertion error on mismatched perturbed images"
    except AssertionError as e:
        assert "differ" in str(e)


# ---------------------------------------------------------------------------
# Headline punchline (FGSM/PGD, Method B vs. Method C)
# ---------------------------------------------------------------------------


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


def test_plot_headline_ci_runs():
    import matplotlib
    matplotlib.use("Agg")
    fig = plot_headline_ci(_fake_bootstrap_data())
    assert fig is not None


def _fake_headline_data():
    return {
        "epsilons": [0.02, 0.03, 0.05], "primary_eps": 0.03, "quantile": 0.90,
        "models": {
            "SmallCNN": {
                "clean_test_acc": 0.9824,
                "adv_acc_by_eps": {0.02: 0.965, 0.03: 0.96, 0.05: 0.93},
                "method_b": {"clean_quantile": 1.2, "adv_quantile": 8.5},
                "method_b_all16": {"clean_quantile": 1.3, "adv_quantile": 9.4},
                "method_c": {"clean_quantile": 1.1, "adv_quantile": 6.9},
            },
            "StrongCNN": {
                "clean_test_acc": 0.9936,
                "adv_acc_by_eps": {0.02: 0.98, 0.03: 0.975, 0.05: 0.955},
                "method_b": {"clean_quantile": 0.9, "adv_quantile": 4.2},
                "method_b_all16": {"clean_quantile": 1.0, "adv_quantile": 4.9},
                "method_c": {"clean_quantile": 0.8, "adv_quantile": 3.1},
            },
        },
    }


def test_plot_headline_punchline_runs():
    import matplotlib
    matplotlib.use("Agg")
    fig = plot_headline_punchline(_fake_headline_data())
    assert fig is not None
