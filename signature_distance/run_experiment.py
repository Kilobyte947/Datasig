"""Demo drivers for signature_distance: build streams (and now signatures)
for a small set of sample images and save display figures to results/, for
the notebooks to show. Both methods' demos live here, matching the rest of
the project's convention (one run_experiment.py per package, many `_demo`/
`run_*` functions - see e.g. mnist_lipschitz/run_experiment.py).
"""

from pathlib import Path

import torch

from signature_distance import plots
from signature_distance.data_pool import load_eval_pool
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
    """Build Method B streams and per-line signatures for one sample image
    per digit, for digits `0..n_digits-1`, from the default eval pool.
    Saves an overlay + stream + per-line-signature plot per digit to
    results/, and returns the raw tensors and figure handles.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    images, labels = load_eval_pool(n_per_class=100, seed=seed)

    lines = make_reference_lines()

    figures = {}
    for digit in range(n_digits):
        idx = (labels == digit).nonzero()[0].item()
        image = images[idx]

        stream = line_stream(image.unsqueeze(0), lines)[0]  # (16, 32, 2)
        # One signature per line, kept separate (no cross-line concatenation
        # before the signature step - see Method_B.md).
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
    """Build Method A streams and signatures for one sample image per
    digit, for digits `0..n_digits-1`, from the default eval pool. Saves
    an overlay + stream + signature plot per digit to results/, and
    returns the raw tensors and figure handles.
    """
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
