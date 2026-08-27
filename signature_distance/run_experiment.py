"""Method B Phase 1 driver: builds reference-line streams for a small set of
demo images and saves display figures to results/, for the notebook to show.
No signature or distance computation happens here - see PLAN.md's Phase 1
scope boundary.

Method B only: doesn't build or import anything Method A-specific beyond the
shared `time_channel` helper already used inside `line_stream` - that's
Nick's side of the work.
"""

from pathlib import Path

from signature_distance import plots
from signature_distance.data_pool import load_eval_pool
from signature_distance.streams import line_stream, make_reference_lines

RESULTS_DIR = Path(__file__).parent / "results"


def stream_construction_demo(n_digits: int = 3, seed: int = 0) -> dict:
    """Build Method B streams for one sample image per digit, for digits
    `0..n_digits-1`, from the default eval pool. Saves an overlay + stream
    plot per digit to results/, and returns the raw tensors and figure
    handles for further inspection.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    images, labels = load_eval_pool(n_per_class=100, seed=seed)

    lines = make_reference_lines()

    figures = {}
    for digit in range(n_digits):
        idx = (labels == digit).nonzero()[0].item()
        image = images[idx]

        stream = line_stream(image.unsqueeze(0), lines)[0]

        figures[f"digit{digit}_overlay"] = plots.plot_reference_lines(
            image, lines, title=f"Method B reference lines (digit {digit})",
            save_path=RESULTS_DIR / f"digit{digit}_method_b_overlay.png")
        figures[f"digit{digit}_stream"] = plots.plot_line_stream(
            stream, title=f"Method B streams (digit {digit})",
            save_path=RESULTS_DIR / f"digit{digit}_method_b_stream.png")

    return {
        "images": images, "labels": labels,
        "lines": lines,
        "figures": figures,
    }
