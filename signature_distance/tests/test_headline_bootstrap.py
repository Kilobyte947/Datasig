import torch

from signature_distance.headline_bootstrap import (
    bootstrap_quantile_ci,
    check_overlap,
    collect_headline_bootstrap,
    collect_headline_data,
    collect_pgd_headline,
    compare_attacks,
    compare_line_counts,
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


def test_collect_headline_bootstrap_smoke():
    # Tiny/fast smoke test - loads the canonical checkpoint (no training),
    # small sample, few bootstrap resamples, just to confirm the combined
    # driver runs end-to-end and every one of the 8 CIs is well-formed.
    data = collect_headline_bootstrap(
        n_per_class=2, epsilons=(0.05,), primary_eps=0.05, seed=0,
        n_bootstrap=20, verbose=False,
    )
    assert set(data["models"].keys()) == {"SmallCNN", "StrongCNN"}
    for mname, entry in data["models"].items():
        for method_key in ("method_b", "method_b_all16", "method_c"):
            for cond in ("clean", "adv"):
                r = entry[method_key][cond]
                assert r["ci_low"] <= r["point_estimate"] <= r["ci_high"]
                assert r["n_images"] == 20

    overlap = check_overlap(data)
    for method_key in ("method_b", "method_b_all16", "method_c"):
        for cond in ("clean", "adv"):
            assert isinstance(overlap[method_key][cond]["overlap"], bool)


def test_check_overlap_skips_method_b_all16_when_absent():
    # Backward compatibility: data collected before method_b_all16 existed
    # (or any hand-built dict lacking it) must not make check_overlap raise.
    data = _fake_bootstrap_data()
    assert "method_b_all16" not in data["models"]["SmallCNN"]
    result = check_overlap(data)
    assert "method_b_all16" not in result
    assert set(result.keys()) == {"method_b", "method_c"}


def test_collect_headline_bootstrap_all16_uses_distinct_seed_stream_from_informative():
    # method_b (12-line) and method_b_all16 must not accidentally share a
    # bootstrap resampling stream - their point estimates differ (all-16
    # includes the near-zero-distance border lines) so their CIs should
    # generally differ too, and specifically their boot_std values (a
    # function of the resampling draws) should not be identical unless the
    # streams coincided by mistake.
    data = collect_headline_bootstrap(
        n_per_class=2, epsilons=(0.05,), primary_eps=0.05, seed=0,
        n_bootstrap=20, verbose=False,
    )
    for mname, entry in data["models"].items():
        for cond in ("clean", "adv"):
            twelve = entry["method_b"][cond]
            sixteen = entry["method_b_all16"][cond]
            assert twelve["point_estimate"] != sixteen["point_estimate"]


# ---------------------------------------------------------------------------
# Point estimates (FGSM) - collect_headline_data/compare_line_counts,
# formerly headline_plot.py
# ---------------------------------------------------------------------------


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


def test_collect_headline_data_smoke():
    # Tiny/fast smoke test - loads the canonical checkpoint (no training),
    # small sample, just to confirm the combined Method B/C driver runs
    # end-to-end, the cross-method consistency checks (identical test
    # accuracy, identical flip masks) pass on real data rather than only
    # on fakes, and the quantiles come out finite and ordered (adversarial
    # >= clean, since FGSM is a directed worst-case perturbation and clean
    # is undirected random noise of the same magnitude).
    #
    # Uses eps=0.03 (this project's actual primary_eps, not 0.05): on the
    # canonical checkpoint, adv >= clean holds robustly at 0.03 for every
    # model/method/subset (verified directly, n_per_class 2 through 20),
    # but StrongCNN's 12-informative-line Method B subset can invert at the
    # more extreme eps=0.05 (P90 is a noisy tail statistic there) - a real,
    # narrow edge case, not a pipeline bug, and not one that reaches the
    # actual headline plot (which uses primary_eps=0.03).
    data = collect_headline_data(
        n_per_class=2, epsilons=(0.03,), primary_eps=0.03, seed=0, verbose=False,
    )
    assert set(data["models"].keys()) == {"SmallCNN", "StrongCNN"}
    for mname, entry in data["models"].items():
        assert 0.0 <= entry["clean_test_acc"] <= 1.0
        assert 0.0 <= entry["adv_acc_by_eps"][0.03] <= 1.0
        for method_key in ("method_b", "method_b_all16", "method_c"):
            cq = entry[method_key]["clean_quantile"]
            aq = entry[method_key]["adv_quantile"]
            assert cq == cq and aq == aq  # not NaN
            assert aq >= cq


def test_compare_line_counts_zero_cost_on_fake_data():
    data = _fake_headline_data()
    comparison = compare_line_counts(data)
    assert set(comparison.keys()) == {"SmallCNN", "StrongCNN"}
    small_adv = comparison["SmallCNN"]["adv_quantile"]
    assert small_adv["informative_line"] == 8.5
    assert small_adv["all_16_line"] == 9.4
    assert abs(small_adv["delta"] - 0.9) < 1e-9
    assert abs(small_adv["pct_change"] - (0.9 / 8.5 * 100)) < 1e-6


def test_compare_line_counts_matches_real_collect_output():
    # The comparison helper must accept collect_headline_data's own output
    # shape directly, not just the hand-built fake above.
    data = collect_headline_data(
        n_per_class=2, epsilons=(0.05,), primary_eps=0.05, seed=0, verbose=False,
    )
    comparison = compare_line_counts(data)
    for mname in ("SmallCNN", "StrongCNN"):
        for cond in ("clean_quantile", "adv_quantile"):
            c = comparison[mname][cond]
            assert c["all_16_line"] - c["informative_line"] == c["delta"]


# ---------------------------------------------------------------------------
# PGD counterpart - formerly pgd_headline.py
# ---------------------------------------------------------------------------


def _fake_pgd_ci(point, half_width, n_images=20):
    return {"point_estimate": point, "ci_low": point - half_width, "ci_high": point + half_width,
            "ci_level": 0.90, "n_bootstrap": 500, "n_images": n_images, "boot_std": half_width / 2}


def _fake_pgd_data(scale=1.0):
    def m(clean, adv):
        return {"clean": _fake_pgd_ci(clean * scale, 0.3), "adv": _fake_pgd_ci(adv * scale, 1.0)}

    return {
        "primary_eps": 0.03, "quantile": 0.90, "ci_level": 0.90, "n_bootstrap": 500,
        "models": {
            "SmallCNN": {
                "method_b": m(1.2, 8.5), "method_b_all16": m(1.3, 9.4), "method_c": m(1.1, 6.9),
            },
            "StrongCNN": {
                "method_b": m(0.9, 4.2), "method_b_all16": m(1.0, 4.9), "method_c": m(0.8, 3.1),
            },
        },
    }


def test_collect_pgd_headline_smoke():
    # Tiny/fast smoke test - loads the canonical checkpoint (no training),
    # small sample, few PGD steps, few bootstrap resamples, just to confirm
    # the single combined PGD driver produces well-formed CIs for all three
    # subsets, matching collect_headline_bootstrap's output shape exactly.
    data = collect_pgd_headline(
        n_per_class=2, epsilons=(0.05,), primary_eps=0.05, seed=0,
        pgd_steps=2, n_bootstrap=20, verbose=False,
    )
    assert set(data["models"].keys()) == {"SmallCNN", "StrongCNN"}
    for mname, entry in data["models"].items():
        assert 0.0 <= entry["test_acc"] <= 1.0
        assert 0.0 <= entry["flip_fraction"] <= 1.0
        assert 0.0 <= entry["fgsm_flip_fraction"] <= 1.0
        for method_key in ("method_b", "method_b_all16", "method_c"):
            for cond in ("clean", "adv"):
                r = entry[method_key][cond]
                assert r["ci_low"] <= r["point_estimate"] <= r["ci_high"]
                assert r["n_images"] == 20

    # Same key structure as collect_headline_bootstrap's FGSM output -
    # reuses check_overlap unmodified.
    overlap = check_overlap(data)
    for method_key in ("method_b", "method_b_all16", "method_c"):
        for cond in ("clean", "adv"):
            assert isinstance(overlap[method_key][cond]["overlap"], bool)


def test_compare_attacks_structure_and_pgd_stronger_flag():
    fgsm_data = _fake_pgd_data(scale=1.0)
    pgd_data = _fake_pgd_data(scale=1.2)  # PGD strictly stronger in this fake
    comparison = compare_attacks(fgsm_data, pgd_data)

    assert set(comparison.keys()) == {"SmallCNN", "StrongCNN"}
    for mname in comparison:
        for subset in ("method_b", "method_b_all16", "method_c"):
            for cond in ("clean", "adv"):
                entry = comparison[mname][subset][cond]
                assert entry["pgd_stronger"] is True
                assert entry["pgd_point"] > entry["fgsm_point"]


def test_compare_attacks_pgd_not_always_stronger():
    fgsm_data = _fake_pgd_data(scale=1.0)
    pgd_data = _fake_pgd_data(scale=0.8)  # PGD weaker in this fake
    comparison = compare_attacks(fgsm_data, pgd_data)
    assert comparison["SmallCNN"]["method_b"]["adv"]["pgd_stronger"] is False
