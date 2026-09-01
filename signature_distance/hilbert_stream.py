"""Method C: Hilbert-curve stream construction and evaluation.

A single space-filling curve through the image, in place of Method B's 16
straight reference lines. Order-5 Hilbert curve (32x32 grid, 1024 cells),
scaled into the 28x28 image domain, sampled at 512 points evenly spaced
along the curve's arc length, then cut into 16 contiguous segments of 32
points each - matching Method B's current-default total point budget
(16 lines x 32 points) and points-per-segment (32), so a comparison
between the two methods isn't confounded by differing sample budgets.

Per-path only, per the convention established for Method B: the 16
segments are never merged into one vector before comparison - each gets
its own signature and its own distance/ratio, examined as a collection.

Reuses existing infrastructure unmodified: `time_channel` (streams.py) for
each segment's time coordinate, the same batched `grid_sample` bilinear
sampling technique `line_stream` uses (not called directly - Method C
samples along one curve rather than per-line, so needs its own grid_sample
call, but the technique and conventions - align_corners=True,
padding_mode="border" - are identical), `signature_of_stream`
(signatures.py), `choose_rescale_factor`/`rescale_signature`/
`per_line_distances`/`auc_for_distance` (distances.py), `signature_dim`
(method_b_sweep.py), and the adversarial infrastructure
(`SmallCNN`/`StrongCNN`/`train_classifier`/`fgsm_attack`/`margin`/
`random_noise_perturbation`/`pixel_euclidean_distance`/
`load_mnist_train_test`) from `method_b_adversarial_eval.py`.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from signature_distance.data_pool import load_eval_pool
from signature_distance.distances import (
    auc_for_distance,
    choose_rescale_factor,
    per_line_distances,
    rescale_signature,
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
from signature_distance.method_b_sweep import signature_dim
from signature_distance.signatures import signature_of_stream
from signature_distance.streams import time_channel

HILBERT_ORDER = 5
HILBERT_SIDE = 2 ** HILBERT_ORDER  # 32
IMAGE_SIZE = 28
NUM_SAMPLE_POINTS = 512
NUM_SEGMENTS = 16
POINTS_PER_SEGMENT = NUM_SAMPLE_POINTS // NUM_SEGMENTS  # 32

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
RESULTS_DIR = Path(__file__).parent / "results"


def _resample_evenly_by_arc_length(coords: np.ndarray, num_points: int) -> np.ndarray:
    """Given an (M, 2) polyline (consecutive vertices), return (num_points, 2)
    points evenly spaced along its arc length, from the first vertex to the
    last (both endpoints included), via cumulative-length parameterization
    and linear interpolation. Note this is genuinely arc-length-based, not
    equivalent to simple index subsampling of `coords` in general - even
    though every raw Hilbert-curve step below has the same length, the
    curve still bends between steps, so a resampled point can land at a
    corner or partway along a straight run depending on where its target
    arc-length falls; only the *target* arc-length values are evenly
    spaced by construction, not necessarily the Euclidean spacing between
    consecutive resampled points when the path curves between them.
    """
    deltas = np.diff(coords, axis=0)
    seg_lengths = np.sqrt((deltas ** 2).sum(axis=1))
    cum_length = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cum_length[-1]

    target_s = np.linspace(0.0, total_length, num_points)
    x = np.interp(target_s, cum_length, coords[:, 0])
    y = np.interp(target_s, cum_length, coords[:, 1])
    return np.stack([x, y], axis=1)


def _generate_hilbert_curve(order: int) -> np.ndarray:
    """(4**order, 2) int64 array of (x, y) grid coordinates, 0 <= x, y <
    2**order, visited in standard Hilbert-curve order (Wikipedia's
    index-to-xy algorithm). Verified directly (see tests) to visit every
    cell exactly once, stay in bounds, and take only unit axis-aligned
    steps between consecutive points.
    """
    n_cells = 4 ** order
    side = 2 ** order
    xy = np.zeros((n_cells, 2), dtype=np.int64)
    for d in range(n_cells):
        t = d
        s = 1
        x = y = 0
        while s < side:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        xy[d] = (x, y)
    return xy


def make_hilbert_curve(order: int = HILBERT_ORDER, image_size: int = IMAGE_SIZE,
                        num_points: int = NUM_SAMPLE_POINTS) -> torch.Tensor:
    """(num_points, 2) float32 tensor of (row, col) continuous coordinates,
    evenly spaced along the arc length of an order-`order` Hilbert curve,
    scaled by `image_size / 2**order` into the image domain. Fixed,
    deterministic - no seed needed (no randomness anywhere in this
    construction, unlike Method B's reference lines which reserve a seed
    parameter for a variant that was never used).

    Every point lies in `[0, (2**order - 1) * image_size/2**order]` by
    construction - for the defaults (order=5, image_size=28) that's
    `[0, 27.125]`, very slightly past the last valid pixel index (27), not
    strictly within it. `hilbert_stream`'s `grid_sample` call uses
    `padding_mode="border"`, so this clamps to the border pixel rather than
    erroring or extrapolating - a deliberate, checked choice, not an
    oversight (see the shape/bounds test, which checks the true bound
    directly rather than assuming a tidier one).
    """
    side = 2 ** order
    xy = _generate_hilbert_curve(order)  # (4**order, 2) int, grid coords
    scale = image_size / side
    coords = xy.astype(np.float64) * scale  # (4**order, 2) continuous [row, col]

    out = _resample_evenly_by_arc_length(coords, num_points).astype(np.float32)
    return torch.from_numpy(out)


def hilbert_stream(images: torch.Tensor, curve_points: torch.Tensor) -> torch.Tensor:
    """Method C stream construction: samples image intensity along the
    fixed Hilbert curve via batched bilinear interpolation (same
    `grid_sample` technique/conventions as `line_stream`), then cuts the
    point sequence into `NUM_SEGMENTS` contiguous segments.

    images: (N, 28, 28) float32
    curve_points: (num_points, 2) float32 [row, col], from make_hilbert_curve.
    returns: (N, num_segments, points_per_segment, 2) float32, columns
             [t, intensity] - t via time_channel per segment (same
             convention as Method B's per-line streams). Segments stay
             separate in this output - never concatenated (no
             cross-segment concatenation, same rule as Method B's lines).
    """
    n, h, w = images.shape
    num_points = curve_points.shape[0]

    x_norm = 2 * curve_points[:, 1] / (w - 1) - 1
    y_norm = 2 * curve_points[:, 0] / (h - 1) - 1
    grid = torch.stack([x_norm, y_norm], dim=-1)  # (num_points, 2)
    grid = grid.view(1, 1, num_points, 2).expand(n, 1, num_points, 2)

    intensity = F.grid_sample(
        images.unsqueeze(1), grid, mode="bilinear",
        align_corners=True, padding_mode="border",
    ).squeeze(1).squeeze(1)  # (N, num_points)

    num_segments = NUM_SEGMENTS
    points_per_segment = num_points // num_segments
    intensity = intensity.reshape(n, num_segments, points_per_segment)

    t = time_channel(points_per_segment).view(1, 1, points_per_segment).expand(
        n, num_segments, points_per_segment
    )
    return torch.stack([t, intensity], dim=-1).to(torch.float32)


# ---------------------------------------------------------------------------
# Stage A: cheap depth mini-sweep (no model training)
# ---------------------------------------------------------------------------

DEPTH_VARIANTS = (2, 3, 4)


def evaluate_hilbert_depths(n_per_class: int = 15, seed: int = 0,
                             depths=DEPTH_VARIANTS) -> dict:
    """Same/different-digit AUC per segment, for each depth in `depths`,
    computed via the same max-depth-then-prefix-slice shortcut used for
    Method B's sweep (verified there to be numerically exact) - the
    expensive signature step runs once, at the maximum depth, and every
    lower depth is sliced from that single result.
    """
    images, labels = load_eval_pool(n_per_class=n_per_class, seed=seed)
    curve = make_hilbert_curve()
    stream = hilbert_stream(images, curve)  # (N, 16, 32, 2)
    num_segments = stream.shape[1]

    max_depth = max(depths)
    sig_max = torch.stack(
        [signature_of_stream(stream[:, i], depth=max_depth) for i in range(num_segments)], dim=1
    )

    n = images.shape[0]
    iu, ju = torch.triu_indices(n, n, offset=1)
    same = (labels[iu] == labels[ju]).numpy().astype(int)

    results_by_depth = {}
    for depth in depths:
        dim = signature_dim(depth)
        sig_raw = sig_max[:, :, :dim]
        r = choose_rescale_factor(sig_raw, depth=depth)
        sig = rescale_signature(sig_raw, r=r, depth=depth)

        segment_aucs = []
        for i in range(num_segments):
            d = torch.cdist(sig[:, i], sig[:, i], p=2)[iu, ju].numpy()
            segment_aucs.append(auc_for_distance(same, d)["auc"])

        n_chance = sum(1 for a in segment_aucs if a <= 0.505)
        results_by_depth[depth] = {
            "r": r, "segment_aucs": segment_aucs,
            "best_auc": max(segment_aucs), "mean_auc": sum(segment_aucs) / len(segment_aucs),
            "n_segments": num_segments, "n_chance": n_chance,
            "n_informative": num_segments - n_chance,
        }

    return results_by_depth


# ---------------------------------------------------------------------------
# Stage B: full per-path adversarial evaluation
# ---------------------------------------------------------------------------


def per_segment_rescaled_signatures(images: torch.Tensor, curve: torch.Tensor,
                                     depth: int, r: float) -> torch.Tensor:
    stream = hilbert_stream(images, curve)
    num_segments = stream.shape[1]
    sig = torch.stack(
        [signature_of_stream(stream[:, i], depth=depth) for i in range(num_segments)], dim=1
    )
    return rescale_signature(sig, r=r, depth=depth)


def run_hilbert_adversarial_eval(depth: int, n_per_class: int = 20,
                                  epsilons=(0.02, 0.03, 0.05), seed: int = 0,
                                  cnn_epochs: int = 3, strong_epochs: int = 3,
                                  verbose: bool = True) -> dict:
    """Full per-path adversarial/control evaluation for Method C, same
    framework as per_path_adversarial_eval.run_per_path_adversarial_eval
    (FGSM, matched random control, margin-difference numerator, per-segment
    distances - reused unmodified via method_b_adversarial_eval.py/
    distances.py imports), applied to the Hilbert-curve segments instead
    of Method B's reference lines. All 16 segments are used (no a priori
    "informative subset" exclusion - unlike Method B's structural border
    lines, there's no a priori reason any particular segment index would
    be chance-level for a space-filling curve; Stage A's own AUC screen is
    the empirical check for that, not an assumption carried in here).
    """
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
        [signature_of_stream(hilbert_stream(images, curve)[:, i], depth=depth) for i in range(NUM_SEGMENTS)], dim=1
    )
    r = choose_rescale_factor(sig_orig_raw, depth=depth)
    sig_orig = rescale_signature(sig_orig_raw, r=r, depth=depth)

    results = {"n_images": images.shape[0], "epsilons": list(epsilons), "depth": depth, "r": r, "models": {}}

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
            }

        results["models"][name] = model_result

    return results


def summarize_hilbert_result(results: dict) -> dict:
    """Per model/epsilon, per segment: mean ratio on genuinely adversarial
    (flipped) pairs vs. mean ratio on the matched control pairs - same
    definition as Method B's per-path fold-ratio finding, over all 16
    segments (no exclusion)."""
    summary = {}
    for name, mres in results["models"].items():
        summary[name] = {}
        for eps, e in mres["eps"].items():
            flip_idx = e["flip_mask"].nonzero(as_tuple=True)[0]
            n_flipped = flip_idx.shape[0]
            if n_flipped == 0:
                summary[name][eps] = {"n_flipped": 0}
                continue
            per_segment = {}
            for i in range(NUM_SEGMENTS):
                per_segment[i] = {
                    "mean_ratio_adv_flipped": e["ratio_adv"][flip_idx, i].mean().item(),
                    "mean_ratio_control_matched": e["ratio_control"][flip_idx, i].mean().item(),
                }
            summary[name][eps] = {"n_flipped": n_flipped, "per_segment": per_segment}
    return summary


def hilbert_robustness_check(results: dict, n_exclude: int = 2) -> dict:
    """Same spirit as Method B's fold_ratio_robustness: identifies the
    `n_exclude` segments with the smallest mean baseline distance (the
    same kind of scale confound flagged for Method B's lines 9/14 -
    checked here rather than assumed absent) and reports whether the
    aggregate fold-ratio survives their exclusion.
    """
    summary = summarize_hilbert_result(results)
    report = {}

    for name, mres in results["models"].items():
        report[name] = {}
        for eps, e in mres["eps"].items():
            s = summary[name][eps]
            n_flipped = s["n_flipped"]
            if n_flipped == 0:
                report[name][eps] = {"n_flipped": 0}
                continue

            baseline_dist = {
                i: ((e["dist_adv"][:, i].mean() + e["dist_control"][:, i].mean()) / 2).item()
                for i in range(NUM_SEGMENTS)
            }
            excluded = sorted(baseline_dist, key=baseline_dist.get)[:n_exclude]
            kept = [i for i in range(NUM_SEGMENTS) if i not in excluded]

            fold_all = {
                i: s["per_segment"][i]["mean_ratio_adv_flipped"] / s["per_segment"][i]["mean_ratio_control_matched"]
                for i in range(NUM_SEGMENTS)
            }
            fold_kept = {i: fold_all[i] for i in kept}

            report[name][eps] = {
                "n_flipped": n_flipped,
                "excluded_segments": excluded,
                "mean_fold_all": sum(fold_all.values()) / len(fold_all),
                "mean_fold_kept": sum(fold_kept.values()) / len(fold_kept),
                "all_kept_survive_adv_gt_control": all(v > 1.0 for v in fold_kept.values()),
            }

    return report


# ---------------------------------------------------------------------------
# Plotting - pure display, no computation, matching plots.py's convention
# (kept here rather than in plots.py since these are Method C-specific,
# same precedent as per_path_adversarial_eval.py's own plot_spike_gallery).
# ---------------------------------------------------------------------------


def plot_hilbert_curve(image: torch.Tensor, curve: torch.Tensor,
                        title: str = None, save_path=None):
    """Image with the Hilbert curve overlaid, colored by position along
    the curve (dark to light = start to end), with the 16 segment
    boundaries marked."""
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
    """Intensity vs. t for every segment of one image's Hilbert stream,
    one curve per segment - same style as Method B's per-line stream plot."""
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
    """Heatmap of all 16 segments' signatures, one row per segment - same
    style as Method B's per-line signature heatmap."""
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
    """Bar chart of best-segment and mean AUC per depth, from
    evaluate_hilbert_depths' output."""
    depths = sorted(depth_results.keys())
    best = [depth_results[d]["best_auc"] for d in depths]
    mean = [depth_results[d]["mean_auc"] for d in depths]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    width = 0.35
    x = range(len(depths))
    ax.bar([i - width / 2 for i in x], best, width, label="best segment", color="tab:blue")
    ax.bar([i + width / 2 for i in x], mean, width, label="mean over segments", color="tab:orange")
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
