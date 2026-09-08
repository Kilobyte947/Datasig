"""Demo drivers for signature_distance: build streams and signatures for a small set of sample
images and save display figures to results/, for the notebooks to show.
"""

from pathlib import Path
import torch
from signature_distance import plots
torch.set_default_dtype(torch.float64)
from signature_distance.data_pool import load_eval_pool
from signature_distance.distances import (
    choose_rescale_factor,
    method_a_feature_vector,
    method_b_feature_vector,
    rescale_signature,
    within_vs_cross_digit_distance,
)
from signature_distance.signatures import signature_of_stream
from signature_distance.streams import (
    line_stream,
    make_pixel_order,
    make_reference_lines,
    patch_sv_stream,
)

RESULTS_DIR = Path(__file__).parent / "results"
SIGNATURE_DEPTH = 4


def stream_construction_demo(n_digits: int = 3, seed: int = 0) -> dict:
    """Builds Method B streams and per-line signatures for one sample image per digit. Saves an
    overlay, stream, and per-line-signature plot per digit to results/. Returns the raw tensors and
    figure handles."""
    RESULTS_DIR.mkdir(exist_ok=True)
    images, labels = load_eval_pool(n_per_class=100, seed=seed)

    lines = make_reference_lines()

    figures = {}
    for digit in range(n_digits):
        idx = (labels == digit).nonzero()[0].item()
        image = images[idx]

        stream = line_stream(image.unsqueeze(0), lines)[0]  # (16, 32, 2)
        # One signature per line, kept separate
        sig = signature_of_stream(stream, depth=SIGNATURE_DEPTH)  # (16, sig_dim)

        figures[f"digit{digit}_overlay"] = plots.plot_reference_lines(
            image, lines, title=f"Method B reference lines (digit {digit})",
            save_path=RESULTS_DIR / f"digit{digit}_method_b_overlay.png")
        figures[f"digit{digit}_stream"] = plots.plot_line_stream(
            stream, title=f"Method B streams (digit {digit})",
            save_path=RESULTS_DIR / f"digit{digit}_method_b_stream.png")
        figures[f"digit{digit}_signature"] = plots.plot_line_signatures(
            sig, title=f"Method B per-line signatures (digit {digit}, depth={SIGNATURE_DEPTH})",
            save_path=RESULTS_DIR / f"digit{digit}_method_b_signature.png")

    return {
        "images": images, "labels": labels,
        "lines": lines,
        "figures": figures,
    }


def method_a_demo(n_digits: int = 3, seed: int = 0) -> dict:
    """Builds Method A streams and signatures for one sample image per digit. Saves an overlay,
    stream, and signature plot per digit to results/. Returns the raw tensors and figure handles."""
    RESULTS_DIR.mkdir(exist_ok=True)
    images, labels = load_eval_pool(n_per_class=100, seed=seed)

    pixel_order = make_pixel_order(k=64, seed=seed)

    figures = {}
    for digit in range(n_digits):
        idx = (labels == digit).nonzero()[0].item()
        image = images[idx]

        stream = patch_sv_stream(image.unsqueeze(0), pixel_order)[0]  # (64, 2)
        sig = signature_of_stream(stream.unsqueeze(0), depth=SIGNATURE_DEPTH)[0]  # (sig_dim,)

        figures[f"digit{digit}_overlay"] = plots.plot_pixel_order(
            image, pixel_order, title=f"Method A pixel order (digit {digit})",
            save_path=RESULTS_DIR / f"digit{digit}_method_a_overlay.png")
        figures[f"digit{digit}_stream"] = plots.plot_patch_sv_stream(
            stream, title=f"Method A stream (digit {digit})",
            save_path=RESULTS_DIR / f"digit{digit}_method_a_stream.png")
        figures[f"digit{digit}_signature"] = plots.plot_signature(
            sig, title=f"Method A signature (digit {digit}, depth={SIGNATURE_DEPTH})",
            save_path=RESULTS_DIR / f"digit{digit}_method_a_signature.png")

    return {
        "images": images, "labels": labels,
        "pixel_order": pixel_order,
        "figures": figures,
    }


def sanity_check_demo(n_per_class: int = 30, seed: int = 0, depth: int = SIGNATURE_DEPTH) -> dict:
    """Within-digit vs cross-digit mean distance, for Method A and Method B independently, on a
    sample of images. Rescales each method's signatures with its own empirically-derived r, builds
    each method's per-image feature vector, and runs the check on both raw and rescaled signatures.
    Returns each method's results and chosen r value."""
    images, labels = load_eval_pool(n_per_class=n_per_class, seed=seed)

    order = make_pixel_order(k=64, seed=seed)
    sig_a_raw = signature_of_stream(patch_sv_stream(images, order), depth=depth)
    r_a = choose_rescale_factor(sig_a_raw, depth=depth)
    sig_a = rescale_signature(sig_a_raw, r=r_a, depth=depth)
    vec_a_raw = method_a_feature_vector(sig_a_raw)
    vec_a = method_a_feature_vector(sig_a)

    lines = make_reference_lines()
    stream_b = line_stream(images, lines)
    sig_b_raw = torch.stack(
        [signature_of_stream(stream_b[:, i], depth=depth) for i in range(stream_b.shape[1])],
        dim=1,
    )
    r_b = choose_rescale_factor(sig_b_raw, depth=depth)
    sig_b = rescale_signature(sig_b_raw, r=r_b, depth=depth)
    vec_b_raw = method_b_feature_vector(sig_b_raw)
    vec_b = method_b_feature_vector(sig_b)

    return {
        "n_images": images.shape[0],
        "method_a": {
            "r": r_a,
            "raw": within_vs_cross_digit_distance(vec_a_raw, labels),
            "rescaled": within_vs_cross_digit_distance(vec_a, labels),
        },
        "method_b": {
            "r": r_b,
            "raw": within_vs_cross_digit_distance(vec_b_raw, labels),
            "rescaled": within_vs_cross_digit_distance(vec_b, labels),
        },
    }
