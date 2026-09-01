"""Plotting for signature_distance: Method A (patch singular-value stream),
Method B (reference-line stream), and the shared signature output.

Pure plotting: takes already-computed data (images, pixel orders/lines,
streams, signatures) and produces matplotlib figures, optionally saved to
disk. No stream/signature computation happens here - see streams.py /
data_pool.py / signatures.py.
"""

import matplotlib.pyplot as plt
import torch


def plot_pixel_order(image: torch.Tensor, pixel_order: torch.Tensor,
                      title: str = None, save_path=None):
    """Method A: image with sampled (row, col) locations overlaid, colored
    by visiting order (t)."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    order_idx = torch.arange(pixel_order.shape[0])
    sc = ax.scatter(pixel_order[:, 1], pixel_order[:, 0], c=order_idx,
                     cmap="viridis", s=25, edgecolors="white", linewidths=0.5)
    ax.set_title(title or "Method A: patch pixel order")
    ax.axis("off")
    fig.colorbar(sc, ax=ax, label="visit order (t)", fraction=0.046)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_patch_sv_stream(stream: torch.Tensor, title: str = None, save_path=None):
    """Method A: sigma1 vs. t for one image's (K, 2) stream."""
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(stream[:, 0], stream[:, 1], marker="o", markersize=3)
    ax.set_xlabel("t")
    ax.set_ylabel("sigma1")
    ax.set_title(title or "Method A stream")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_signature(sig: torch.Tensor, title: str = None, save_path=None):
    """Method-agnostic: bar chart of one signature vector's coefficients
    (index 0 is always the constant term, 1.0)."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(sig.shape[0]), sig, color="tab:blue")
    ax.set_xlabel("signature term index")
    ax.set_ylabel("value")
    ax.set_title(title or "Signature")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_line_signatures(sig_batch: torch.Tensor, title: str = None, save_path=None):
    """Method B: heatmap of per-line signatures, one row per line.

    sig_batch: (num_lines, signature_dim), e.g. the 16 independent
    per-line signatures for one image (never concatenated into one raw
    stream before this point - each row is its own line's signature)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(sig_batch, aspect="auto", cmap="viridis")
    ax.set_xlabel("signature term index")
    ax.set_ylabel("line index")
    ax.set_title(title or "Method B: per-line signatures")
    fig.colorbar(im, ax=ax, label="value", fraction=0.046)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


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


def plot_per_line_auc_ranking(ranked: list, title: str = None, save_path=None):
    """Method B: bar chart of same/different-digit AUC per measure (16
    individual lines + the merged 496-dim distance), ranked highest to
    lowest, from `per_line_diagnostics.run_per_line_auc_diagnostic`'s
    `ranked` output. The merged bar is colored differently so it's easy to
    see how many individual lines rank above/below it."""
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [name for name, _ in ranked]
    aucs = [entry["auc"] for _, entry in ranked]
    colors = ["tab:orange" if name == "merged" else "tab:blue" for name in names]
    ax.bar(range(len(names)), aucs, color=colors)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (AUC=0.5)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("same/different-digit AUC")
    ax.set_title(title or "Method B: per-line vs. merged distance AUC")
    ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_reference_lines_with_metric(image: torch.Tensor, lines: torch.Tensor,
                                      metric_values, title: str = None,
                                      colorbar_label: str = "metric",
                                      cmap: str = "plasma", save_path=None):
    """Method B: image with reference lines overlaid, each colored by a
    per-line metric (e.g. fold-ratio, AUC) instead of orientation - shows
    directly where on the image the highest/lowest-scoring lines sit.

    metric_values: sequence of length lines.shape[0], one value per line,
    in the same order as `lines`."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    values = torch.as_tensor(metric_values, dtype=torch.float32)
    norm = plt.Normalize(vmin=values.min().item(), vmax=values.max().item())
    cmap_obj = plt.get_cmap(cmap)
    for i in range(lines.shape[0]):
        line = lines[i]
        color = cmap_obj(norm(values[i].item()))
        ax.plot(line[:, 1], line[:, 0], color=color, linewidth=2.5)
    ax.set_title(title or "Method B: reference lines colored by metric")
    ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=colorbar_label, fraction=0.046)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_ratio_distribution(ratio_adv, ratio_control, title: str = None, save_path=None):
    """Histogram of adversarial vs. control ratio values for one distance
    measure - shows whether it separates genuinely adversarial
    perturbations from equally-sized random ones (bimodal, minimal
    overlap = clear separation)."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    adv = ratio_adv.detach().cpu().numpy() if torch.is_tensor(ratio_adv) else ratio_adv
    ctrl = ratio_control.detach().cpu().numpy() if torch.is_tensor(ratio_control) else ratio_control
    ax.hist(adv, bins=20, alpha=0.6, label="adversarial (FGSM)", color="tab:red")
    ax.hist(ctrl, bins=20, alpha=0.6, label="control (random noise)", color="tab:blue")
    ax.set_xlabel("ratio")
    ax.set_ylabel("count")
    ax.set_title(title or "Ratio distribution: adversarial vs. control")
    ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_per_line_bar(values: dict, title: str = None, ylabel: str = "value",
                       highlight_key=None, save_path=None):
    """Generic labeled bar chart over a {label: value} dict - e.g. per-line
    fold-ratios. `highlight_key`, if given, colors that one bar
    differently (e.g. the single best-performing line)."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    labels = list(values.keys())
    vals = list(values.values())
    colors = ["tab:orange" if k == highlight_key else "tab:blue" for k in labels]
    ax.bar(range(len(labels)), vals, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([str(k) for k in labels], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title or "")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
