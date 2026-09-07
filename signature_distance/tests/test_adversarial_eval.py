import torch

from signature_distance.adversarial_eval import (
    BEST_LINE_INDEX,
    BORDER_LINE_INDICES,
    EXCLUDED_LINES,
    INFORMATIVE_LINE_INDICES,
    ROBUST_LINE_INDICES,
    _pearson,
    fold_ratio_robustness,
    hilbert_robustness_check,
    per_line_rescaled_signatures,
    pgd_fold_summary,
    run_border_and_pixel_check,
    run_hilbert_adversarial_eval,
    run_hilbert_adversarial_eval_with_images,
    run_method_a_adversarial_evaluation,
    run_method_b_adversarial_evaluation,
    run_pgd_comparison,
    run_per_path_adversarial_eval,
    run_robustness_report,
    spike_analysis,
    summarize_border_and_pixel_check,
    summarize_hilbert_result,
    summarize_informative_subset,
)
from signature_distance.distances import METHOD_B_BORDER_LINE_INDICES, METHOD_B_WINNER_LINES
from signature_distance.streams import NUM_SEGMENTS


# ---------------------------------------------------------------------------
# Method A
# ---------------------------------------------------------------------------
# No dedicated smoke test currently exists for run_method_a_adversarial_evaluation
# itself - only its constituent pieces (method_a_signature_distance, margin,
# fgsm_attack) are tested, in test_distances.py/test_attacks.py.


# ---------------------------------------------------------------------------
# Method B
# ---------------------------------------------------------------------------
# No dedicated smoke test currently exists for run_method_b_adversarial_evaluation
# itself either - same situation as Method A above.


# ---------------------------------------------------------------------------
# Method C / Hilbert
# ---------------------------------------------------------------------------


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
    # Tiny/fast smoke test - loads the canonical checkpoint (no training),
    # small sample, just to confirm the plumbing runs end-to-end.
    out = run_hilbert_adversarial_eval(
        depth=2, n_per_class=2, epsilons=(0.05,), seed=0, verbose=False,
    )
    assert out["n_images"] == 20
    for mname in ("SmallCNN", "StrongCNN"):
        e = out["models"][mname]["eps"][0.05]
        assert e["ratio_adv"].shape == (20, NUM_SEGMENTS)
        assert not torch.isnan(e["ratio_adv"]).any()


def test_run_hilbert_adversarial_eval_with_images_matches_unmodified_function():
    # Checkpoint: the image-retaining function must reproduce
    # run_hilbert_adversarial_eval's own ratio/distance numbers exactly
    # given the same seed/params - not just "runs without error." Tiny/fast:
    # canonical checkpoint (no training), 2 images/class, 1 epsilon.
    kwargs = dict(depth=2, n_per_class=2, epsilons=(0.05,), seed=0, verbose=False)
    baseline = run_hilbert_adversarial_eval(**kwargs)
    with_images = run_hilbert_adversarial_eval_with_images(**kwargs)

    for mname in ("SmallCNN", "StrongCNN"):
        e_base = baseline["models"][mname]["eps"][0.05]
        e_img = with_images["models"][mname]["eps"][0.05]
        assert torch.allclose(e_base["ratio_adv"], e_img["ratio_adv"])
        assert torch.allclose(e_base["ratio_control"], e_img["ratio_control"])
        assert torch.allclose(e_base["dist_adv"], e_img["dist_adv"])
        assert torch.equal(e_base["flip_mask"], e_img["flip_mask"])


def test_run_hilbert_adversarial_eval_with_images_shares_pairs_with_method_b():
    # Checkpoint for the gallery's core premise: Method B's and Method C's
    # eval pools/trained-models/FGSM perturbations are bit-identical for a
    # given pair_idx/model/eps under matched seed/params - not assumed.
    kwargs = dict(n_per_class=2, epsilons=(0.05,), seed=0, verbose=False)
    results_b = run_per_path_adversarial_eval(**kwargs)
    results_c = run_hilbert_adversarial_eval_with_images(depth=2, **kwargs)

    assert torch.equal(results_b["labels"], results_c["labels"])
    assert torch.allclose(results_b["images"], results_c["images"])
    for mname in ("SmallCNN", "StrongCNN"):
        x_adv_b = results_b["models"][mname]["eps"][0.05]["x_adv"]
        x_adv_c = results_c["models"][mname]["eps"][0.05]["x_adv"]
        assert torch.allclose(x_adv_b, x_adv_c, atol=1e-6)


# ---------------------------------------------------------------------------
# PGD (Method B + Method C)
# ---------------------------------------------------------------------------


def test_run_pgd_comparison_smoke():
    # Tiny/fast smoke test - loads the canonical checkpoint (no training),
    # small sample, few PGD steps, just to confirm the combined Method B/C
    # driver runs end-to-end and produces sane shapes for both methods.
    out = run_pgd_comparison(
        n_per_class=2, epsilons=(0.05,), seed=0, pgd_steps=2, verbose=False,
    )
    assert out["n_images"] == 20
    n_lines = METHOD_B_WINNER_LINES.shape[0]
    for mname in ("SmallCNN", "StrongCNN"):
        e_b = out["method_b"]["models"][mname]["eps"][0.05]
        e_c = out["method_c"]["models"][mname]["eps"][0.05]
        assert e_b["ratio_adv"].shape == (20, n_lines)
        assert e_c["ratio_adv"].shape == (20, NUM_SEGMENTS)
        assert not torch.isnan(e_b["ratio_adv"]).any()
        assert not torch.isnan(e_c["ratio_adv"]).any()
        # both methods evaluated the SAME perturbation - same flip mask
        assert torch.equal(e_b["flip_mask"], e_c["flip_mask"])
        assert e_b["flip_fraction"] == e_c["flip_fraction"]


def test_pgd_fold_summary_structure():
    torch.manual_seed(0)
    n = 20
    flip_mask = torch.zeros(n, dtype=torch.bool)
    flip_mask[:5] = True
    fake_results = {
        "n_images": n, "epsilons": [0.03], "pgd_steps": 10,
        "method_b": {"r": 2.5, "depth": 2, "n_lines": 16, "models": {
            "FakeModel": {"test_acc": 0.99, "eps": {0.03: {
                "flip_mask": flip_mask, "flip_fraction": 0.25, "fgsm_flip_fraction": 0.2,
                "ratio_adv": torch.rand(n, 16) + 0.1, "ratio_control": torch.rand(n, 16) * 0.2,
                "dist_adv": torch.rand(n, 16) + 0.5, "dist_control": torch.rand(n, 16) + 0.5,
            }}},
        }},
        "method_c": {"r": 2.5, "depth": 3, "n_segments": 16, "models": {
            "FakeModel": {"test_acc": 0.99, "eps": {0.03: {
                "flip_mask": flip_mask, "flip_fraction": 0.25, "fgsm_flip_fraction": 0.2,
                "ratio_adv": torch.rand(n, 16) + 0.1, "ratio_control": torch.rand(n, 16) * 0.2,
                "dist_adv": torch.rand(n, 16) + 0.5, "dist_control": torch.rand(n, 16) + 0.5,
            }}},
        }},
    }
    summary = pgd_fold_summary(fake_results)
    for key in ("method_b", "method_c"):
        assert "overall_mean_fold" in summary[key]
        assert summary[key]["overall_total"] == 16
        entry = summary[key]["by_model_eps"]["FakeModel"][0.03]
        assert entry["n_flipped"] == 5
        assert entry["fgsm_flip_fraction"] == 0.2


# ---------------------------------------------------------------------------
# Per-path (Method B)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Per-path robustness check (fold_ratio_robustness, run_robustness_report) -
# merged from what was a separate per_path_ratio_robustness_check.py module
# ---------------------------------------------------------------------------


def test_robust_line_indices_excludes_9_and_14():
    assert EXCLUDED_LINES == (9, 14)
    assert set(ROBUST_LINE_INDICES) == set(INFORMATIVE_LINE_INDICES) - {9, 14}
    assert len(ROBUST_LINE_INDICES) == 10


def test_pearson_known_cases():
    assert abs(_pearson([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-9
    assert abs(_pearson([1, 2, 3], [3, 2, 1]) - (-1.0)) < 1e-9
    assert _pearson([1, 1, 1], [1, 2, 3]) == 0.0  # zero variance in x - defined as 0, not NaN


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


# ---------------------------------------------------------------------------
# Border-line and pixel check (Method B, winner geometry)
# ---------------------------------------------------------------------------


def test_informative_and_border_indices_partition_all_16_lines_winner_geometry():
    # Same partition property as the default-geometry check above, but for
    # Method B's winner geometry (16h+0v) - a genuinely different index set
    # (METHOD_B_BORDER_LINE_INDICES from distances.py), not a duplicate.
    from signature_distance.distances import METHOD_B_INFORMATIVE_LINE_INDICES
    assert len(METHOD_B_BORDER_LINE_INDICES) == 2
    assert len(METHOD_B_INFORMATIVE_LINE_INDICES) == 14
    assert set(METHOD_B_BORDER_LINE_INDICES) | set(METHOD_B_INFORMATIVE_LINE_INDICES) == set(range(16))


def _fake_border_check_results(n=20, seed=0):
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
    results = _fake_border_check_results()
    summary = summarize_border_and_pixel_check(results)
    bvi = summary["border_vs_informative"]
    assert bvi["border_mean_dist_adv"] < bvi["informative_mean_dist_adv"]


def test_summarize_border_and_pixel_check_structure():
    results = _fake_border_check_results()
    summary = summarize_border_and_pixel_check(results)
    assert summary["method_b_all16"]["total"] == 16
    assert summary["border_vs_informative"]["method_b_informative_total"] == 14
    assert summary["pixel"]["total"] == 1
    assert set(summary["border_vs_informative"]["per_line_mean_fold"].keys()) == set(range(16))


def test_run_border_and_pixel_check_smoke():
    # Tiny/fast smoke test - loads the canonical checkpoint (no training),
    # small sample, just to confirm the combined driver runs end-to-end and
    # produces sane shapes for both Method B (all 16 lines) and pixel-Euclidean.
    out = run_border_and_pixel_check(
        n_per_class=2, epsilons=(0.05,), seed=0, verbose=False,
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
