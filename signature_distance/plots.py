"""Plotting for signature_distance's Method B (reference-line stream) Phase 1
output.

Pure plotting: takes already-computed data (images, lines, streams) and
produces matplotlib figures, optionally saved to disk. No stream/data
computation happens here - see streams.py / data_pool.py.

Method B only: Method A's own plotting is Nick's, not built here, to avoid
duplicating/overstepping into that side of the work.
"""

import matplotlib.pyplot as plt
import torch


def plot_reference_lines(image: torch.Tensor, lines: torch.Tensor,
                          title: str = None, save_path=None):
    """Method B: image with reference lines overlaid, colored by
    orientation (horizontal vs. vertical, inferred per line from whether
    its row-coordinate or column-coordinate range is larger)."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    for line in lines:
        rows, cols = line[:, 0], line[:, 1]
        horizontal = (rows.max() - rows.min()) < (cols.max() - cols.min())
        color = "tab:orange" if horizontal else "tab:cyan"
        ax.plot(cols, rows, color=color, linewidth=1.2, alpha=0.85)
    ax.set_title(title or "Method B: reference lines")
    ax.axis("off")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_line_stream(stream: torch.Tensor, title: str = None, save_path=None):
    """Method B: intensity vs. t for every line of one image's
    (num_lines, points_per_line, 2) stream, one curve per line."""
    fig, ax = plt.subplots(figsize=(5, 3))
    num_lines = stream.shape[0]
    cmap = plt.get_cmap("viridis")
    for i in range(num_lines):
        color = cmap(i / max(num_lines - 1, 1))
        ax.plot(stream[i, :, 0], stream[i, :, 1], color=color, alpha=0.8, linewidth=1)
    ax.set_xlabel("t")
    ax.set_ylabel("intensity")
    ax.set_title(title or "Method B streams (one curve per line)")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
