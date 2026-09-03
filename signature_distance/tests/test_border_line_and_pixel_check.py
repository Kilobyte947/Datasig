import torch

from signature_distance.border_line_and_pixel_check import (
    INFORMATIVE_LINE_INDICES,
    run_border_and_pixel_check,
    summarize_border_and_pixel_check,
)
from signature_distance.headline_plot import METHOD_B_BORDER_LINE_INDICES


def test_informative_and_border_indices_partition_all_16_lines():
    assert len(METHOD_B_BORDER_LINE_INDICES) == 4
    assert len(INFORMATIVE_LINE_INDICES) == 12
    assert set(METHOD_B_BORDER_LINE_INDICES) | set(INFORMATIVE_LINE_INDICES) == set(range(16))


def _fake_results(n=20, seed=0):
    torch.manual_seed(seed)
    flip_mask = torch.zeros(n, dtype=torch.bool)
    flip_mask[: n // 4] = True

    ratio_b_adv = torch.rand(n, 16) + 0.5
    ratio_b_control = torch.rand(n, 16) * 0.2
    dist_b_adv = torch.rand(n, 16) + 0.5
    # simulate border lines having a much smaller adversarial distance
    dist_b_adv[:, list(METHOD_B_BORDER_LINE_INDICES)] *= 0.01

    return {
        "n_images": n, "epsilons": [0.03],
        "models": {
            "FakeModel": {
                "test_acc": 0.99,
                "eps": {
                    0.03: {
                        "flip_mask": flip_mask, "flip_fraction": flip_mask.float().mean().item(),
                        "ratio_b_adv": ratio_b_adv, "ratio_b_control": ratio_b_control,
                        "dist_b_adv": dist_b_adv, "dist_b_control": torch.rand(n, 16) + 0.5,
                        "ratio_pixel_adv": torch.rand(n) + 0.5, "ratio_pixel_control": torch.rand(n) * 0.2,
                        "dist_pixel_adv": torch.rand(n) + 0.5, "dist_pixel_control": torch.rand(n) + 0.5,
                    }
                },
            }
        },
    }


def test_summarize_reports_border_lines_with_smaller_distance_than_informative():
    results = _fake_results()
    summary = summarize_border_and_pixel_check(results)
    bvi = summary["border_vs_informative"]
    assert bvi["border_mean_dist_adv"] < bvi["informative_mean_dist_adv"]


def test_summarize_structure():
    results = _fake_results()
    summary = summarize_border_and_pixel_check(results)
    assert summary["method_b_all16"]["total"] == 16
    assert summary["border_vs_informative"]["method_b_12line_total"] == 12
    assert summary["pixel"]["total"] == 1
    assert set(summary["border_vs_informative"]["per_line_mean_fold"].keys()) == set(range(16))


def test_run_border_and_pixel_check_smoke():
    # Tiny/fast smoke test - real training (1 epoch, small sample) just to
    # confirm the combined driver runs end-to-end and produces sane shapes
    # for both Method B (all 16 lines) and pixel-Euclidean.
    out = run_border_and_pixel_check(
        n_per_class=2, epsilons=(0.05,), seed=0,
        cnn_epochs=1, strong_epochs=1, verbose=False,
    )
    assert out["n_images"] == 20
    for mname in ("SmallCNN", "StrongCNN"):
        e = out["models"][mname]["eps"][0.05]
        assert e["ratio_b_adv"].shape == (20, 16)
        assert e["ratio_pixel_adv"].shape == (20,)
        assert not torch.isnan(e["ratio_b_adv"]).any()
        assert not torch.isnan(e["ratio_pixel_adv"]).any()

    summary = summarize_border_and_pixel_check(out)
    assert summary["method_b_all16"]["total"] > 0
    assert summary["pixel"]["total"] > 0
