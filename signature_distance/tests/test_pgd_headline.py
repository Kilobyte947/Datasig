from signature_distance.headline_bootstrap import check_overlap
from signature_distance.pgd_headline import collect_pgd_headline, compare_attacks


def _fake_ci(point, half_width, n_images=20):
    return {"point_estimate": point, "ci_low": point - half_width, "ci_high": point + half_width,
            "ci_level": 0.90, "n_bootstrap": 500, "n_images": n_images, "boot_std": half_width / 2}


def _fake_data(scale=1.0):
    def m(clean, adv):
        return {"clean": _fake_ci(clean * scale, 0.3), "adv": _fake_ci(adv * scale, 1.0)}

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
    # Tiny/fast smoke test - real training (1 epoch, small sample, few PGD
    # steps, few bootstrap resamples) just to confirm the single combined
    # PGD driver produces well-formed CIs for all three subsets, matching
    # collect_headline_bootstrap's output shape exactly.
    data = collect_pgd_headline(
        n_per_class=2, epsilons=(0.05,), primary_eps=0.05, seed=0,
        cnn_epochs=1, strong_epochs=1, pgd_steps=2, n_bootstrap=20, verbose=False,
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

    # Same key structure as headline_bootstrap's FGSM output - reuses
    # check_overlap unmodified.
    overlap = check_overlap(data)
    for method_key in ("method_b", "method_b_all16", "method_c"):
        for cond in ("clean", "adv"):
            assert isinstance(overlap[method_key][cond]["overlap"], bool)


def test_compare_attacks_structure_and_pgd_stronger_flag():
    fgsm_data = _fake_data(scale=1.0)
    pgd_data = _fake_data(scale=1.2)  # PGD strictly stronger in this fake
    comparison = compare_attacks(fgsm_data, pgd_data)

    assert set(comparison.keys()) == {"SmallCNN", "StrongCNN"}
    for mname in comparison:
        for subset in ("method_b", "method_b_all16", "method_c"):
            for cond in ("clean", "adv"):
                entry = comparison[mname][subset][cond]
                assert entry["pgd_stronger"] is True
                assert entry["pgd_point"] > entry["fgsm_point"]


def test_compare_attacks_pgd_not_always_stronger():
    fgsm_data = _fake_data(scale=1.0)
    pgd_data = _fake_data(scale=0.8)  # PGD weaker in this fake
    comparison = compare_attacks(fgsm_data, pgd_data)
    assert comparison["SmallCNN"]["method_b"]["adv"]["pgd_stronger"] is False
