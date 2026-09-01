import torch

from signature_distance.hilbert_stream import (
    NUM_SEGMENTS,
    make_hilbert_curve,
    run_hilbert_adversarial_eval,
)
from signature_distance.method_c_gallery import (
    plot_hilbert_spike_gallery,
    plot_spike_comparison,
    run_hilbert_adversarial_eval_with_images,
)
from signature_distance.per_path_adversarial_eval import run_per_path_adversarial_eval


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


def test_run_hilbert_adversarial_eval_with_images_matches_unmodified_function():
    # Checkpoint: the new image-retaining function must reproduce
    # hilbert_stream.run_hilbert_adversarial_eval's own ratio/distance
    # numbers exactly given the same seed/params - not just "runs without
    # error." Tiny/fast: 1 epoch, 2 images/class, 1 epsilon.
    kwargs = dict(depth=2, n_per_class=2, epsilons=(0.05,), seed=0,
                  cnn_epochs=1, strong_epochs=1, verbose=False)
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
    kwargs = dict(n_per_class=2, epsilons=(0.05,), seed=0, cnn_epochs=1, strong_epochs=1, verbose=False)
    results_b = run_per_path_adversarial_eval(**kwargs)
    results_c = run_hilbert_adversarial_eval_with_images(depth=2, **kwargs)

    assert torch.equal(results_b["labels"], results_c["labels"])
    assert torch.allclose(results_b["images"], results_c["images"])
    for mname in ("SmallCNN", "StrongCNN"):
        x_adv_b = results_b["models"][mname]["eps"][0.05]["x_adv"]
        x_adv_c = results_c["models"][mname]["eps"][0.05]["x_adv"]
        assert torch.allclose(x_adv_b, x_adv_c, atol=1e-6)
