"""Method C (Hilbert-curve) per-path spike gallery, plus side-by-side
comparison figures against Method B's existing gallery on the same image
pairs.

Closes a parity gap: Method B has `per_path_adversarial_eval.plot_spike_gallery`
(original image, perturbed image, overlay showing which reference line spiked
hardest); Method C had no equivalent. This module adds the same style of
gallery for Method C's 16 Hilbert segments, plus a combined figure that puts
both methods' overlays side by side on the same pair.

Visualization only - no new attack, metric, or evaluation logic. The FGSM/
random-control pipeline below is the identical one already used and
validated by `hilbert_stream.run_hilbert_adversarial_eval`
(`SmallCNN`/`StrongCNN`/`train_classifier`/`fgsm_attack`/
`random_noise_perturbation`/`margin`/`pixel_euclidean_distance` from
`method_b_adversarial_eval.py`; `make_hilbert_curve`/
`per_segment_rescaled_signatures` from `hilbert_stream.py`); the only
difference is that `images`/`labels`/`x_adv`/`x_control` are retained here
for plotting, since `run_hilbert_adversarial_eval` itself doesn't need them
and doesn't keep them. Given seed=0, this reproduces that function's own
ratio/distance results exactly (checked directly in tests below) rather
than being assumed to.

Method B's side is read straight from `run_per_path_adversarial_eval`
(`per_path_adversarial_eval.py`), unmodified.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from signature_distance.data_pool import load_eval_pool
from signature_distance.distances import choose_rescale_factor, per_line_distances, rescale_signature
from signature_distance.hilbert_stream import (
    NUM_SEGMENTS,
    POINTS_PER_SEGMENT,
    hilbert_stream as compute_hilbert_stream,
    make_hilbert_curve,
    per_segment_rescaled_signatures,
)
from signature_distance.method_b_adversarial_eval import (
    SmallCNN,
    StrongCNN,
    fgsm_attack,
    load_mnist_train_test,
    margin,
    pixel_euclidean_distance,
    random_noise_perturbation,
    train_classifier,
)
from signature_distance.per_path_adversarial_eval import INFORMATIVE_LINE_INDICES, METHOD_B_LINES
from signature_distance.signatures import signature_of_stream

RESULTS_DIR = Path(__file__).parent / "results"


def run_hilbert_adversarial_eval_with_images(depth: int = 3, n_per_class: int = 20,
                                              epsilons=(0.02, 0.03, 0.05), seed: int = 0,
                                              cnn_epochs: int = 3, strong_epochs: int = 3,
                                              verbose: bool = True) -> dict:
    """Same training/attack pipeline as
    `hilbert_stream.run_hilbert_adversarial_eval`, additionally retaining
    `images`, `labels`, `x_adv`, `x_control`, and the fixed `curve` for
    gallery plotting. No new adversarial generation or metric."""
    torch.manual_seed(seed)
    train_loader, test_loader = load_mnist_train_test()

    models = {}
    for name, model, epochs in [("SmallCNN", SmallCNN(), cnn_epochs), ("StrongCNN", StrongCNN(), strong_epochs)]:
        if verbose:
            print(f"Training {name} ({epochs} epochs)...")
        trained, _, test_acc = train_classifier(model, train_loader, test_loader, epochs=epochs, verbose=verbose)
        trained.eval()
        models[name] = {"model": trained, "test_acc": test_acc}
        if verbose:
            print(f"  {name}: test_acc={test_acc:.4f}")

    images, labels = load_eval_pool(n_per_class=n_per_class, seed=seed)
    images_c = images.unsqueeze(1)
    generator = torch.Generator().manual_seed(seed)

    curve = make_hilbert_curve()
    sig_orig_raw = torch.stack(
        [signature_of_stream(compute_hilbert_stream(images, curve)[:, i], depth=depth) for i in range(NUM_SEGMENTS)],
        dim=1,
    )
    r = choose_rescale_factor(sig_orig_raw, depth=depth)
    sig_orig = rescale_signature(sig_orig_raw, r=r, depth=depth)

    results = {
        "n_images": images.shape[0], "epsilons": list(epsilons), "depth": depth, "r": r,
        "images": images, "labels": labels, "curve": curve, "models": {},
    }

    for name, info in models.items():
        model = info["model"]
        model_result = {"test_acc": info["test_acc"], "eps": {}}

        for eps in epsilons:
            x_adv_c = fgsm_attack(model, images_c, labels, eps)
            x_adv = x_adv_c.squeeze(1)
            fgsm_l2 = pixel_euclidean_distance(images_c, x_adv_c)
            x_control_c = random_noise_perturbation(images_c, fgsm_l2, generator=generator)
            x_control = x_control_c.squeeze(1)

            with torch.no_grad():
                margin_orig = margin(model, images_c, labels)
                margin_adv = margin(model, x_adv_c, labels)
                margin_control = margin(model, x_control_c, labels)
                preds_adv = model(x_adv_c).argmax(dim=1)

            num_adv = (margin_orig - margin_adv).abs()
            num_control = (margin_orig - margin_control).abs()

            sig_adv = per_segment_rescaled_signatures(x_adv, curve, depth, r)
            sig_control = per_segment_rescaled_signatures(x_control, curve, depth, r)

            dist_adv = per_line_distances(sig_orig, sig_adv)
            dist_control = per_line_distances(sig_orig, sig_control)

            ratio_adv = num_adv.unsqueeze(1) / dist_adv
            ratio_control = num_control.unsqueeze(1) / dist_control

            model_result["eps"][eps] = {
                "flip_mask": preds_adv != labels,
                "flip_fraction": (preds_adv != labels).float().mean().item(),
                "ratio_adv": ratio_adv, "ratio_control": ratio_control,
                "dist_adv": dist_adv, "dist_control": dist_control,
                "x_adv": x_adv, "x_control": x_control,
            }

        results["models"][name] = model_result

    return results


def plot_hilbert_spike_gallery(results: dict, model_name: str, eps: float, pair_idx: int,
                                title: str = None, save_path=None):
    """Method C equivalent of `per_path_adversarial_eval.plot_spike_gallery`:
    original image, perturbed image, and the Hilbert curve overlaid with the
    single largest-ratio segment drawn thick/red, the rest thin/gray. All 16
    segments are eligible (unlike Method B's gallery, which excludes 4
    structural border lines) - Stage A's depth sweep found every Hilbert
    segment carries above-chance signal, so there's no degenerate subset to
    exclude here."""
    e = results["models"][model_name]["eps"][eps]
    image = results["images"][pair_idx]
    x_adv = e["x_adv"][pair_idx]
    curve = results["curve"]

    ratio_row = e["ratio_adv"][pair_idx]  # (16,)
    spike_segment = int(ratio_row.argmax())
    spike_ratio = float(ratio_row.max())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"original (label {int(results['labels'][pair_idx])})")
    axes[0].axis("off")

    axes[1].imshow(x_adv, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"FGSM-perturbed (eps={eps})")
    axes[1].axis("off")

    axes[2].imshow(x_adv, cmap="gray", vmin=0, vmax=1)
    axes[2].plot(curve[:, 1], curve[:, 0], color="lightgray", linewidth=0.8, alpha=0.7, zorder=1)
    seg_start = spike_segment * POINTS_PER_SEGMENT
    seg_end = seg_start + POINTS_PER_SEGMENT
    axes[2].plot(curve[seg_start:seg_end, 1], curve[seg_start:seg_end, 0],
                 color="red", linewidth=2.5, zorder=3)
    axes[2].set_title(f"segment {spike_segment} spikes (ratio={spike_ratio:.2f})")
    axes[2].axis("off")

    fig.suptitle(title or f"{model_name}, pair {pair_idx}: which segment spikes")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_spike_comparison(results_b: dict, results_c: dict, model_name: str, eps: float,
                           pair_idx: int, title: str = None, save_path=None):
    """Side-by-side comparison on the SAME original/perturbed pair: original
    image, Method B's spike overlay (largest-ratio reference line), Method
    C's spike overlay (largest-ratio Hilbert segment) - lets a reader see
    directly whether the two methods' signal concentrates on the same
    region of the image or not.

    `results_b`/`results_c` must come from the same eval pool/seed/model/eps
    (checked via an assert on the perturbed image itself, not assumed)."""
    e_b = results_b["models"][model_name]["eps"][eps]
    e_c = results_c["models"][model_name]["eps"][eps]

    x_adv_b = e_b["x_adv"][pair_idx]
    x_adv_c = e_c["x_adv"][pair_idx]
    assert torch.allclose(x_adv_b, x_adv_c, atol=1e-6), (
        "Method B and Method C perturbed images differ for this pair - "
        "results_b/results_c must come from the same eval pool/seed/model."
    )

    image = results_b["images"][pair_idx]
    curve = results_c["curve"]

    idx_tensor = torch.tensor(INFORMATIVE_LINE_INDICES)
    ratio_row_b = e_b["ratio_adv"][pair_idx, idx_tensor]
    spike_line_b = int(idx_tensor[ratio_row_b.argmax()])
    spike_ratio_b = float(ratio_row_b.max())

    ratio_row_c = e_c["ratio_adv"][pair_idx]
    spike_segment_c = int(ratio_row_c.argmax())
    spike_ratio_c = float(ratio_row_c.max())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"original (label {int(results_b['labels'][pair_idx])})")
    axes[0].axis("off")

    axes[1].imshow(x_adv_b, cmap="gray", vmin=0, vmax=1)
    for i in range(METHOD_B_LINES.shape[0]):
        line = METHOD_B_LINES[i]
        if i == spike_line_b:
            axes[1].plot(line[:, 1], line[:, 0], color="red", linewidth=2.5, zorder=3)
        else:
            axes[1].plot(line[:, 1], line[:, 0], color="lightgray", linewidth=0.8, alpha=0.7, zorder=1)
    axes[1].set_title(f"Method B: line {spike_line_b} spikes (ratio={spike_ratio_b:.2f})")
    axes[1].axis("off")

    axes[2].imshow(x_adv_c, cmap="gray", vmin=0, vmax=1)
    axes[2].plot(curve[:, 1], curve[:, 0], color="lightgray", linewidth=0.8, alpha=0.7, zorder=1)
    seg_start = spike_segment_c * POINTS_PER_SEGMENT
    seg_end = seg_start + POINTS_PER_SEGMENT
    axes[2].plot(curve[seg_start:seg_end, 1], curve[seg_start:seg_end, 0],
                 color="red", linewidth=2.5, zorder=3)
    axes[2].set_title(f"Method C: segment {spike_segment_c} spikes (ratio={spike_ratio_c:.2f})")
    axes[2].axis("off")

    fig.suptitle(title or f"{model_name}, eps={eps}, pair {pair_idx}: Method B vs. Method C")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
