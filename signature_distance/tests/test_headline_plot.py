import torch

from signature_distance.headline_plot import (
    METHOD_B_BORDER_LINE_INDICES,
    METHOD_B_INFORMATIVE_LINE_INDICES,
    collect_headline_data,
    plot_headline_punchline,
)


def test_border_and_informative_indices_partition_all_16_lines():
    assert len(METHOD_B_BORDER_LINE_INDICES) == 4
    assert len(METHOD_B_INFORMATIVE_LINE_INDICES) == 12
    assert set(METHOD_B_BORDER_LINE_INDICES) | set(METHOD_B_INFORMATIVE_LINE_INDICES) == set(range(16))
    assert set(METHOD_B_BORDER_LINE_INDICES).isdisjoint(METHOD_B_INFORMATIVE_LINE_INDICES)


def _fake_headline_data():
    return {
        "epsilons": [0.02, 0.03, 0.05], "primary_eps": 0.03, "quantile": 0.90,
        "models": {
            "SmallCNN": {
                "clean_test_acc": 0.9824,
                "adv_acc_by_eps": {0.02: 0.965, 0.03: 0.96, 0.05: 0.93},
                "method_b": {"clean_quantile": 1.2, "adv_quantile": 8.5},
                "method_c": {"clean_quantile": 1.1, "adv_quantile": 6.9},
            },
            "StrongCNN": {
                "clean_test_acc": 0.9936,
                "adv_acc_by_eps": {0.02: 0.98, 0.03: 0.975, 0.05: 0.955},
                "method_b": {"clean_quantile": 0.9, "adv_quantile": 4.2},
                "method_c": {"clean_quantile": 0.8, "adv_quantile": 3.1},
            },
        },
    }


def test_plot_headline_punchline_runs():
    import matplotlib
    matplotlib.use("Agg")
    fig = plot_headline_punchline(_fake_headline_data())
    assert fig is not None


def test_collect_headline_data_smoke():
    # Tiny/fast smoke test - real training (1 epoch, small sample) just to
    # confirm the combined Method B/C driver runs end-to-end, the
    # cross-method consistency checks (identical test accuracy, identical
    # flip masks) pass on real data rather than only on fakes, and the
    # quantiles come out finite and ordered (adversarial >= clean, since
    # FGSM is a directed worst-case perturbation and clean is undirected
    # random noise of the same magnitude).
    data = collect_headline_data(
        n_per_class=2, epsilons=(0.05,), primary_eps=0.05, seed=0,
        cnn_epochs=1, strong_epochs=1, verbose=False,
    )
    assert set(data["models"].keys()) == {"SmallCNN", "StrongCNN"}
    for mname, entry in data["models"].items():
        assert 0.0 <= entry["clean_test_acc"] <= 1.0
        assert 0.0 <= entry["adv_acc_by_eps"][0.05] <= 1.0
        for method_key in ("method_b", "method_c"):
            cq = entry[method_key]["clean_quantile"]
            aq = entry[method_key]["adv_quantile"]
            assert cq == cq and aq == aq  # not NaN
            assert aq >= cq
