import torch

from signature_distance.per_path_adversarial_eval import (
    BEST_LINE_INDEX,
    BORDER_LINE_INDICES,
    INFORMATIVE_LINE_INDICES,
    per_line_rescaled_signatures,
    plot_spike_gallery,
    spike_analysis,
    summarize_informative_subset,
)


def test_border_and_informative_indices_partition_all_16_lines():
    assert len(BORDER_LINE_INDICES) == 4
    assert len(INFORMATIVE_LINE_INDICES) == 12
    assert set(BORDER_LINE_INDICES) | set(INFORMATIVE_LINE_INDICES) == set(range(16))
    assert set(BORDER_LINE_INDICES).isdisjoint(INFORMATIVE_LINE_INDICES)
    assert BEST_LINE_INDEX in INFORMATIVE_LINE_INDICES


def test_per_line_rescaled_signatures_shape():
    torch.manual_seed(0)
    images = torch.rand(3, 28, 28)
    sig = per_line_rescaled_signatures(images)
    assert sig.shape == (3, 16, 31)  # depth=4, width=2 -> 31; 16 lines
    assert torch.isfinite(sig).all()


def _fake_results(n=20, num_lines=16, seed=0):
    torch.manual_seed(seed)
    ratio_adv = torch.rand(n, num_lines) + 0.1
    ratio_control = torch.rand(n, num_lines) * 0.2
    dist_adv = torch.rand(n, num_lines) + 0.5
    dist_adv[:, list(BORDER_LINE_INDICES)] *= 0.01  # simulate near-zero border distances
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


def test_summarize_informative_subset_structure():
    results = _fake_results()
    summary = summarize_informative_subset(results)
    assert "FakeModel" in summary
    entry = summary["FakeModel"][0.03]
    assert entry["n_flipped"] == 5  # n=20, n//4 flipped
    assert set(entry["per_line"].keys()) == set(INFORMATIVE_LINE_INDICES)
    for i in INFORMATIVE_LINE_INDICES:
        assert entry["per_line"][i]["mean_ratio_adv_flipped"] is not None


def test_summarize_informative_subset_handles_zero_flips():
    results = _fake_results()
    results["models"]["FakeModel"]["eps"][0.03]["flip_mask"] = torch.zeros(20, dtype=torch.bool)
    summary = summarize_informative_subset(results)
    entry = summary["FakeModel"][0.03]
    assert entry["n_flipped"] == 0
    for i in INFORMATIVE_LINE_INDICES:
        assert entry["per_line"][i]["mean_ratio_adv_flipped"] is None


def test_spike_analysis_only_considers_informative_lines():
    results = _fake_results()
    analysis = spike_analysis(results)
    entry = analysis["FakeModel"][0.03]
    assert set(entry["argmax_counts_adv"].keys()) == set(INFORMATIVE_LINE_INDICES)
    assert 0.0 <= entry["entropy_adv_bits"] <= entry["max_entropy_bits"] + 1e-6
    # border distances were deliberately shrunk in the fixture - confirm the
    # module's own sanity check correctly measures that gap
    assert entry["border_line_mean_distance"] < entry["informative_line_mean_distance"]


def test_plot_spike_gallery_runs():
    import matplotlib
    matplotlib.use("Agg")
    results = _fake_results()
    fig = plot_spike_gallery(results, "FakeModel", 0.03, pair_idx=0)
    assert fig is not None
