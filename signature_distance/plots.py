"""Plotting for signature_distance: 
Method A (patch singular-value stream),
Method B (reference-line stream), 
Method C (Hilbert curve), 
and the headline/spike-gallery figures built on top of the adversarial-eval results.
"""

import matplotlib.pyplot as plt
import torch
from signature_distance.adversarial_eval import INFORMATIVE_LINE_INDICES
from signature_distance.distances import METHOD_B_LINES
from signature_distance.streams import POINTS_PER_SEGMENT

torch.set_default_dtype(torch.float64)


def plot_pixel_order(image: torch.Tensor, pixel_order: torch.Tensor,
                      title: str = None, save_path=None):
    """Method A: image with sampled pixel locations overlaid, coloured by visiting order."""
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
    """Method A: singular value vs position for one image's stream."""
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(stream[:, 0], stream[:, 1], marker="o", markersize=3)
    ax.set_xlabel("t")
    ax.set_ylabel("sigma1")
    ax.set_title(title or "Method A stream")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_signature(sig: torch.Tensor, title: str = None, save_path=None):
    """Bar chart of one signature vector's coefficients."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(sig.shape[0]), sig, color="blue")
    ax.set_xlabel("signature term index")
    ax.set_ylabel("value")
    ax.set_title(title or "Signature")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_line_signatures(sig_batch: torch.Tensor, title: str = None, save_path=None):
    """Method B: heatmap of per-line signatures, one row per line."""
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
    """Method B: image with reference lines overlaid, coloured by orientation."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    for line in lines:
        rows, cols = line[:, 0], line[:, 1]
        horizontal = (rows.max() - rows.min()) < (cols.max() - cols.min())
        color = "orange" if horizontal else "cyan"
        ax.plot(cols, rows, color=color, linewidth=1.2, alpha=0.85)
    ax.set_title(title or "Method B: reference lines")
    ax.axis("off")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_line_stream(stream: torch.Tensor, title: str = None, save_path=None):
    """Method B: intensity vs position for every line of one image's stream, one curve per line."""
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
    """Method B: same/different-digit AUC per measure (16 lines plus the merged distance), ranked
    highest to lowest, with the merged bar coloured separately."""
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [name for name, _ in ranked]
    aucs = [entry["auc"] for _, entry in ranked]
    colors = ["orange" if name == "merged" else "blue" for name in names]
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
    """Method B: image with reference lines overlaid, each coloured by a given per-line metric (e.g.
    fold-ratio, AUC)."""
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
    """Histogram of adversarial vs control ratio values for one distance measure."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    adv = ratio_adv.detach().cpu().numpy() if torch.is_tensor(ratio_adv) else ratio_adv
    ctrl = ratio_control.detach().cpu().numpy() if torch.is_tensor(ratio_control) else ratio_control
    ax.hist(adv, bins=20, alpha=0.6, label="adversarial (FGSM)", color="red")
    ax.hist(ctrl, bins=20, alpha=0.6, label="control (random noise)", color="blue")
    ax.set_xlabel("ratio")
    ax.set_ylabel("count")
    ax.set_title(title or "Ratio distribution: adversarial vs. control")
    ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_per_line_bar(values: dict, title: str = None, ylabel: str = "value",
                       highlight_key=None, save_path=None):
    """Labelled bar chart over a {label: value} dict, with an optional single bar highlighted."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    labels = list(values.keys())
    vals = list(values.values())
    colors = ["orange" if k == highlight_key else "blue" for k in labels]
    ax.bar(range(len(labels)), vals, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([str(k) for k in labels], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title or "")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Method C: Hilbert curve
# ---------------------------------------------------------------------------


def plot_hilbert_curve(image: torch.Tensor, curve: torch.Tensor,
                        title: str = None, save_path=None):
    """Image with the Hilbert curve overlaid, coloured by position along the curve, with the 16
    segment boundaries marked."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    cmap = plt.get_cmap("viridis")
    num_points = curve.shape[0]
    for i in range(num_points - 1):
        color = cmap(i / (num_points - 1))
        ax.plot(curve[i:i + 2, 1], curve[i:i + 2, 0], color=color, linewidth=1.2)
    for seg in range(0, num_points, POINTS_PER_SEGMENT):
        ax.scatter(curve[seg, 1], curve[seg, 0], color="red", s=12, zorder=3)
    ax.set_title(title or "Method C: Hilbert curve (16 segment starts marked)")
    ax.axis("off")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_hilbert_segment_streams(stream: torch.Tensor, title: str = None, save_path=None):
    """Intensity vs position for every segment of one image's Hilbert stream, one curve per segment."""
    fig, ax = plt.subplots(figsize=(5, 3))
    num_segments = stream.shape[0]
    cmap = plt.get_cmap("viridis")
    for i in range(num_segments):
        color = cmap(i / max(num_segments - 1, 1))
        ax.plot(stream[i, :, 0], stream[i, :, 1], color=color, alpha=0.8, linewidth=1)
    ax.set_xlabel("t")
    ax.set_ylabel("intensity")
    ax.set_title(title or "Method C streams (one curve per segment)")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_hilbert_signatures(sig: torch.Tensor, title: str = None, save_path=None):
    """Heatmap of all 16 Hilbert segments' signatures, one row per segment."""
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(sig, aspect="auto", cmap="viridis")
    ax.set_xlabel("signature term index")
    ax.set_ylabel("segment index")
    ax.set_title(title or "Method C: per-segment signatures")
    fig.colorbar(im, ax=ax, label="value", fraction=0.046)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_depth_comparison(depth_results: dict, title: str = None, save_path=None):
    """Bar chart of best-segment and mean AUC per depth."""
    depths = sorted(depth_results.keys())
    best = [depth_results[d]["best_auc"] for d in depths]
    mean = [depth_results[d]["mean_auc"] for d in depths]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    width = 0.35
    x = range(len(depths))
    ax.bar([i - width / 2 for i in x], best, width, label="best segment", color="blue")
    ax.bar([i + width / 2 for i in x], mean, width, label="mean over segments", color="orange")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (AUC=0.5)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(d) for d in depths])
    ax.set_xlabel("truncation depth")
    ax.set_ylabel("same/different-digit AUC")
    ax.set_title(title or "Method C: depth mini-sweep")
    ax.legend(fontsize=8)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Headline(FGSM/PGD, Method B vs. Method C)
# ---------------------------------------------------------------------------

def plot_headline_punchline(data: dict, title: str = None, save_path=None):
    """Three-panel figure: clean test accuracy, adversarial accuracy, and the headline quantile local
    Lipschitz estimate, grouped by model, Method B as bars and Method C as an overlaid marker."""
    models = list(data["models"].keys())
    eps = data["primary_eps"]
    q_pct = int(round(data["quantile"] * 100))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: clean accuracy
    clean_acc = [data["models"][m]["clean_test_acc"] * 100 for m in models]
    bars = axes[0].bar(models, clean_acc, color="green")
    for bar, v in zip(bars, clean_acc):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=10)
    axes[0].set_ylabel("accuracy (%)")
    axes[0].set_title("Clean test accuracy")
    axes[0].set_ylim(0, 105)

    # Panel 2: adversarial accuracy at the primary epsilon
    adv_acc = [data["models"][m]["adv_acc_by_eps"][eps] * 100 for m in models]
    bars = axes[1].bar(models, adv_acc, color="red")
    for bar, v in zip(bars, adv_acc):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=10)
    axes[1].set_ylabel("accuracy (%)")
    axes[1].set_title(f"FGSM adversarial accuracy (eps={eps})")
    axes[1].set_ylim(0, 105)

    # Panel 3: high-quantile local Lipschitz estimate (Method B bars, Method C markers)
    x = range(len(models))
    width = 0.35
    clean_q = [data["models"][m]["method_b"]["clean_quantile"] for m in models]
    adv_q = [data["models"][m]["method_b"]["adv_quantile"] for m in models]
    clean_q_c = [data["models"][m]["method_c"]["clean_quantile"] for m in models]
    adv_q_c = [data["models"][m]["method_c"]["adv_quantile"] for m in models]

    bars_clean = axes[2].bar([i - width / 2 for i in x], clean_q, width, label="clean (Method B)", color="blue")
    bars_adv = axes[2].bar([i + width / 2 for i in x], adv_q, width, label="adversarial (Method B)", color="orange")
    for bar, v in zip(bars_clean, clean_q):
        axes[2].text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    for bar, v in zip(bars_adv, adv_q):
        axes[2].text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    axes[2].scatter([i - width / 2 for i in x], clean_q_c, marker="D", color="navy", zorder=3, label="clean (Method C)")
    axes[2].scatter([i + width / 2 for i in x], adv_q_c, marker="D", color="darkred", zorder=3, label="adversarial (Method C)")

    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(models)
    axes[2].set_ylabel(f"P{q_pct} local Lipschitz ratio")
    axes[2].set_title(f"P{q_pct} local Lipschitz estimate (eps={eps})")
    axes[2].legend(fontsize=7, loc="upper left")

    fig.suptitle(title or "Clean accuracy, adversarial accuracy, and high-quantile local Lipschitz estimate")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_headline_ci(data: dict, title: str = None, save_path=None):
    """Two-panel figure (clean, adversarial): grouped bar chart of Method B's P90 point estimate with
    bootstrap CI for SmallCNN vs StrongCNN, with Method C's point estimate and CI overlaid as
    error-barred markers."""
    models = list(data["models"].keys())
    ci_pct = int(round(data["ci_level"] * 100))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, cond, cond_label in zip(axes, ("clean", "adv"), ("clean", "adversarial")):
        b_point = [data["models"][m]["method_b"][cond]["point_estimate"] for m in models]
        b_lo = [data["models"][m]["method_b"][cond]["ci_low"] for m in models]
        b_hi = [data["models"][m]["method_b"][cond]["ci_high"] for m in models]
        b_err = [[p - lo for p, lo in zip(b_point, b_lo)], [hi - p for p, hi in zip(b_point, b_hi)]]

        c_point = [data["models"][m]["method_c"][cond]["point_estimate"] for m in models]
        c_lo = [data["models"][m]["method_c"][cond]["ci_low"] for m in models]
        c_hi = [data["models"][m]["method_c"][cond]["ci_high"] for m in models]
        c_err = [[p - lo for p, lo in zip(c_point, c_lo)], [hi - p for p, hi in zip(c_point, c_hi)]]

        x = range(len(models))
        ax.bar(x, b_point, yerr=b_err, capsize=6, color="blue", alpha=0.75, label="Method B")
        ax.errorbar(x, c_point, yerr=c_err, fmt="D", color="darkred", capsize=6, label="Method C")
        ax.set_xticks(list(x))
        ax.set_xticklabels(models)
        ax.set_ylabel("P90 local Lipschitz ratio")
        ax.set_title(f"{cond_label} (eps={data['primary_eps']})")
        ax.legend(fontsize=8)

    fig.suptitle(title or f"P90 local Lipschitz estimate with {ci_pct}% bootstrap CI ({data['n_bootstrap']} resamples)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Spike galleries (Method B, Method C, and the two side by side)
# ---------------------------------------------------------------------------


def plot_spike_gallery(results: dict, model_name: str, eps: float, pair_idx: int,
                        title: str = None, save_path=None):
    """Original and perturbed image with the 16 reference lines overlaid, the single largest-ratio
    informative line drawn thick, the rest thin."""
    e = results["models"][model_name]["eps"][eps]
    image = results["images"][pair_idx]
    x_adv = e["x_adv"][pair_idx]

    idx_tensor = torch.tensor(INFORMATIVE_LINE_INDICES)
    ratio_row = e["ratio_adv"][pair_idx, idx_tensor]
    spike_line = int(idx_tensor[ratio_row.argmax()])
    spike_ratio = float(ratio_row.max())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"original (label {int(results['labels'][pair_idx])})")
    axes[0].axis("off")

    axes[1].imshow(x_adv, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"FGSM-perturbed (eps={eps})")
    axes[1].axis("off")

    axes[2].imshow(x_adv, cmap="gray", vmin=0, vmax=1)
    for i in range(METHOD_B_LINES.shape[0]):
        line = METHOD_B_LINES[i]
        if i == spike_line:
            axes[2].plot(line[:, 1], line[:, 0], color="red", linewidth=2.5, zorder=3)
        else:
            axes[2].plot(line[:, 1], line[:, 0], color="lightgray", linewidth=0.8, alpha=0.7, zorder=1)
    axes[2].set_title(f"line {spike_line} spikes (ratio={spike_ratio:.2f})")
    axes[2].axis("off")

    fig.suptitle(title or f"{model_name}, pair {pair_idx}: which path spikes")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_hilbert_spike_gallery(results: dict, model_name: str, eps: float, pair_idx: int,
                                title: str = None, save_path=None):
    """Method C equivalent of plot_spike_gallery: original and perturbed image with the Hilbert curve
    overlaid, the single largest-ratio segment drawn thick, the rest thin."""
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
    """Side-by-side comparison on the same pair: original image, Method B's spike overlay, and Method
    C's spike overlay, for comparing where each method's signal concentrates."""
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
