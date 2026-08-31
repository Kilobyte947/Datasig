import torch

from signature_distance.per_path_adversarial_eval import BORDER_LINE_INDICES, INFORMATIVE_LINE_INDICES
from signature_distance.per_path_ratio_robustness_check import (
    EXCLUDED_LINES,
    ROBUST_LINE_INDICES,
    _pearson,
    fold_ratio_robustness,
)


def test_robust_line_indices_excludes_9_and_14():
    assert EXCLUDED_LINES == (9, 14)
    assert set(ROBUST_LINE_INDICES) == set(INFORMATIVE_LINE_INDICES) - {9, 14}
    assert len(ROBUST_LINE_INDICES) == 10


def test_pearson_known_cases():
    assert abs(_pearson([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-9
    assert abs(_pearson([1, 2, 3], [3, 2, 1]) - (-1.0)) < 1e-9
    assert _pearson([1, 1, 1], [1, 2, 3]) == 0.0  # zero variance in x - defined as 0, not NaN


def _fake_results(n=20, num_lines=16, seed=0):
    torch.manual_seed(seed)
    ratio_adv = torch.rand(n, num_lines) * 5 + 1.0
    ratio_control = torch.rand(n, num_lines) * 0.5 + 0.1
    dist_adv = torch.rand(n, num_lines) + 0.5
    dist_control = torch.rand(n, num_lines) + 0.5
    flip_mask = torch.zeros(n, dtype=torch.bool)
    flip_mask[: n // 4] = True
    return {
        "n_images": n, "epsilons": [0.03],
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


def test_fold_ratio_robustness_structure_and_line_counts():
    results = _fake_results()
    report = fold_ratio_robustness(results)
    r = report["FakeModel"][0.03]

    assert r["n_flipped"] == 5
    assert set(r["fold_12"].keys()) == set(INFORMATIVE_LINE_INDICES)
    assert set(r["fold_10"].keys()) == set(ROBUST_LINE_INDICES)
    assert 9 not in r["fold_10"] and 14 not in r["fold_10"]
    assert set(r["baseline_dist_10"].keys()) == set(ROBUST_LINE_INDICES)
    assert -1.0 <= r["dist_fold_correlation_10"] <= 1.0


def test_fold_ratio_robustness_handles_zero_flips():
    results = _fake_results()
    results["models"]["FakeModel"]["eps"][0.03]["flip_mask"] = torch.zeros(20, dtype=torch.bool)
    report = fold_ratio_robustness(results)
    assert report["FakeModel"][0.03]["n_flipped"] == 0
    assert "fold_12" not in report["FakeModel"][0.03]


def test_excluding_lines_cannot_introduce_a_new_adv_lt_control_violation():
    # If every line in the 12-line set has adv > control, the 10-line
    # subset (strictly fewer lines) must too - removing entries from an
    # all-True set can't make it False. Direct correctness check on the
    # reported boolean, not just an assumption.
    results = _fake_results()
    report = fold_ratio_robustness(results)
    r = report["FakeModel"][0.03]
    all_12_survive = all(v > 1.0 for v in r["fold_12"].values())
    if all_12_survive:
        assert r["all_10_survive_adv_gt_control"] is True
