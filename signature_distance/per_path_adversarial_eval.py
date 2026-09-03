"""Per-path Lipschitz ratios under adversarial perturbation, plus a
robustness check on the resulting fold-ratio finding.

Corrects the framing of the earlier merged-distance evaluation: don't merge
the 16 reference-line paths into any single combined score (not the 496-dim
concatenation, not max/top-k/weighted-sum). Instead, treat the 16 paths the
way the pixel-space Lipschitz work (Experiment 1/2) treats individual
pixels - as separate coordinates, each producing its own local Lipschitz
ratio, examined as a collection, never reduced to one number. Directly
mirrors the gradient-norm estimator's per-pixel sensitivity approach, with
paths standing in for pixels.

The robustness check at the bottom of this file (`fold_ratio_robustness`,
`run_robustness_report`) was originally a separate, read-only module kept
apart deliberately so it was visually obvious nothing in this file's own
evaluation logic was being touched while it was being checked; now that
that history is recorded in README.md's "Method B: Reference-Line Signature
Distance" section, it lives here as an additional
function instead of a separate file - still read-only over an
already-computed `run_per_path_adversarial_eval` result, still no new
signature computation.

Reuses existing, UNMODIFIED infrastructure - nothing in the imported
modules is changed by this file:
  - `SmallCNN`/`StrongCNN`/`train_classifier`/`fgsm_attack`/
    `random_noise_perturbation`/`margin`/`pixel_euclidean_distance`/
    `load_mnist_train_test`/`METHOD_B_LINES`/`METHOD_B_R`/`SIGNATURE_DEPTH`
    from `method_b_adversarial_eval.py`.
  - `per_line_distances` from `distances.py`.
  - `line_stream`/`signature_of_stream`/`rescale_signature` (the same
    computation `method_b_adversarial_eval.method_b_signature_distance`
    performs internally before concatenating - `per_line_rescaled_signatures`
    below exposes it pre-concatenation, since that's exactly what per-path
    ratios need; no new signature computation).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from signature_distance.data_pool import load_eval_pool
from signature_distance.distances import per_line_distances, rescale_signature
from signature_distance.method_b_adversarial_eval import (
    METHOD_B_LINES,
    METHOD_B_R,
    SIGNATURE_DEPTH,
    SmallCNN,
    StrongCNN,
    fgsm_attack,
    load_mnist_train_test,
    margin,
    pixel_euclidean_distance,
    random_noise_perturbation,
    train_classifier,
)
from signature_distance.signatures import signature_of_stream
from signature_distance.streams import line_stream

RESULTS_DIR = Path(__file__).parent / "results"

# The 4 lines running exactly along the image border (row/col 0 or 27) -
# structural consequence of make_reference_lines()'s default
# angles_deg=(0, 90), counts=(8, 8): index 0/7 are the first/last of the 8
# horizontal lines (rows via linspace(0, 27, 8)), index 8/15 the first/last
# of the 8 vertical lines. Identified by per_line_diagnostics.py's AUC
# ranking as carrying zero same/different-digit signal (AUC == 0.5000
# exactly, every one of them) - MNIST digits essentially never touch the
# border, so these run through background regardless of the image.
BORDER_LINE_INDICES = (0, 7, 8, 15)
INFORMATIVE_LINE_INDICES = tuple(i for i in range(16) if i not in BORDER_LINE_INDICES)
# per_line_diagnostics.py's single highest-AUC individual line.
BEST_LINE_INDEX = 6


def per_line_rescaled_signatures(images: torch.Tensor, depth: int = SIGNATURE_DEPTH,
                                  r: float = METHOD_B_R) -> torch.Tensor:
    """(N, num_lines, sig_dim) rescaled per-line signatures for a batch of
    images, stopping one step before method_b_signature_distance's
    concatenation - reuses line_stream/signature_of_stream/rescale_signature
    unchanged."""
    num_lines = METHOD_B_LINES.shape[0]
    stream = line_stream(images, METHOD_B_LINES)
    sig = torch.stack(
        [signature_of_stream(stream[:, i], depth=depth) for i in range(num_lines)], dim=1
    )
    return rescale_signature(sig, r=r, depth=depth)


def run_per_path_adversarial_eval(n_per_class: int = 20, epsilons=(0.02, 0.03, 0.05),
                                   seed: int = 0, cnn_epochs: int = 3, strong_epochs: int = 3,
                                   verbose: bool = True) -> dict:
    """Same sample/models/attack setup as method_b_adversarial_eval.run_adversarial_evaluation
    (trains SmallCNN and StrongCNN fresh, same FGSM epsilons and magnitude-
    matched random control), but computes 16 SEPARATE per-line ratios per
    pair instead of one merged ratio - the per-line signatures and distances
    aren't retained by that function's own return value, so they're
    recomputed here via the same underlying calls, not duplicated logic.
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

    sig_orig = per_line_rescaled_signatures(images)  # (N, num_lines, sig_dim) - computed once

    results = {"n_images": images.shape[0], "epsilons": list(epsilons), "models": {},
               "labels": labels, "images": images}

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

            num_adv = (margin_orig - margin_adv).abs()      # (N,)
            num_control = (margin_orig - margin_control).abs()

            sig_adv = per_line_rescaled_signatures(x_adv)
            sig_control = per_line_rescaled_signatures(x_control)

            dist_adv = per_line_distances(sig_orig, sig_adv)          # (N, num_lines)
            dist_control = per_line_distances(sig_orig, sig_control)  # (N, num_lines)

            ratio_adv = num_adv.unsqueeze(1) / dist_adv          # (N, num_lines) - never merged
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


def summarize_informative_subset(results: dict) -> dict:
    """Per model/epsilon, per informative line (the 12 non-border lines,
    line_6 highlighted separately): mean ratio over the GENUINELY adversarial
    pairs (prediction actually flipped) vs. mean ratio over the SAME index
    subset's control pairs (same images, for a matched, apples-to-apples
    comparison - not the full 200-image control set, since the flipped
    subset is a specific, often small, harder-to-classify slice)."""
    summary = {}
    for name, mres in results["models"].items():
        summary[name] = {}
        for eps, e in mres["eps"].items():
            flip_idx = e["flip_mask"].nonzero(as_tuple=True)[0]
            n_flipped = flip_idx.shape[0]
            per_line = {}
            for i in INFORMATIVE_LINE_INDICES:
                if n_flipped == 0:
                    per_line[i] = {"mean_ratio_adv_flipped": None, "mean_ratio_control_matched": None}
                else:
                    per_line[i] = {
                        "mean_ratio_adv_flipped": e["ratio_adv"][flip_idx, i].mean().item(),
                        "mean_ratio_control_matched": e["ratio_control"][flip_idx, i].mean().item(),
                    }
            summary[name][eps] = {"n_flipped": n_flipped, "per_line": per_line}
    return summary


def spike_analysis(results: dict) -> dict:
    """Per model/epsilon: for every pair, which of the 12 INFORMATIVE lines
    has the largest ratio (argmax over ratio_adv restricted to
    INFORMATIVE_LINE_INDICES - the border lines are excluded here
    specifically because their near-constant, near-zero-distance signatures
    make their ratio a numerically degenerate near-zero-denominator blowup,
    not a meaningful "spike"; verified/reported below). Reports the
    distribution of which line wins most often, for both adversarial and
    control pairs, plus each distribution's entropy (bits) - a more
    concentrated/peaked distribution (lower entropy) means the perturbation
    consistently spikes the same one or two lines; a flatter distribution
    (entropy close to log2(12) ~= 3.58 bits, uniform over 12 lines) means it
    spreads roughly evenly.
    """
    import math

    idx_tensor = torch.tensor(INFORMATIVE_LINE_INDICES)
    analysis = {}
    for name, mres in results["models"].items():
        analysis[name] = {}
        for eps, e in mres["eps"].items():
            ratio_adv_informative = e["ratio_adv"][:, idx_tensor]      # (N, 12)
            ratio_control_informative = e["ratio_control"][:, idx_tensor]

            argmax_adv = idx_tensor[ratio_adv_informative.argmax(dim=1)]
            argmax_control = idx_tensor[ratio_control_informative.argmax(dim=1)]

            def _distribution_and_entropy(argmax_indices):
                counts = {i: 0 for i in INFORMATIVE_LINE_INDICES}
                for v in argmax_indices.tolist():
                    counts[v] += 1
                n = argmax_indices.shape[0]
                probs = [c / n for c in counts.values() if c > 0]
                entropy = -sum(p * math.log2(p) for p in probs)
                return counts, entropy

            counts_adv, entropy_adv = _distribution_and_entropy(argmax_adv)
            counts_control, entropy_control = _distribution_and_entropy(argmax_control)

            # Sanity check on the degenerate-denominator concern: are border
            # lines' distances actually much smaller than informative lines'?
            border_dist_mean = e["dist_adv"][:, list(BORDER_LINE_INDICES)].mean().item()
            informative_dist_mean = e["dist_adv"][:, idx_tensor].mean().item()

            analysis[name][eps] = {
                "argmax_counts_adv": counts_adv, "entropy_adv_bits": entropy_adv,
                "argmax_counts_control": counts_control, "entropy_control_bits": entropy_control,
                "max_entropy_bits": math.log2(len(INFORMATIVE_LINE_INDICES)),
                "border_line_mean_distance": border_dist_mean,
                "informative_line_mean_distance": informative_dist_mean,
            }

    return analysis


def plot_spike_gallery(results: dict, model_name: str, eps: float, pair_idx: int,
                        title: str = None, save_path=None):
    """Original image, perturbed image, and the 16 reference lines overlaid
    with the single largest-ratio INFORMATIVE line for this specific pair
    drawn thick/red, the rest thin/gray - the interpretable per-pair
    output: is the adversarial change concentrated on one path or not,
    shown directly on the image it happened to."""
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


# ---------------------------------------------------------------------------
# Robustness check: does the fold-ratio finding above survive excluding
# lines 9 and 14 (flagged, in the spike-count analysis, as having
# systematically smaller baseline distances than the other informative
# lines regardless of perturbation type)? Read-only over an already-computed
# `run_per_path_adversarial_eval` result - no new signature computation.
# ---------------------------------------------------------------------------

EXCLUDED_LINES = (9, 14)
ROBUST_LINE_INDICES = tuple(i for i in INFORMATIVE_LINE_INDICES if i not in EXCLUDED_LINES)


def _pearson(xs, ys) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def fold_ratio_robustness(results: dict) -> dict:
    """Per model/epsilon: per-line fold-ratio (mean adversarial ratio on
    genuinely flipped pairs / mean control ratio on the same pairs - same
    definition as the fold-ratio finding above), reported for both the
    original 12-line informative set and the 10-line set with lines 9 and
    14 excluded, side by side - plus each line's baseline distance (mean of
    dist_adv and dist_control, i.e. not perturbation-direction-dependent)
    and the Pearson correlation between baseline distance and fold-ratio
    across the 10-line set, to check whether the scale confound extends
    beyond lines 9/14.
    """
    summary = summarize_informative_subset(results)
    report = {}

    for name, mres in results["models"].items():
        report[name] = {}
        for eps, e in mres["eps"].items():
            s = summary[name][eps]
            n_flipped = s["n_flipped"]
            if n_flipped == 0:
                report[name][eps] = {"n_flipped": 0}
                continue

            fold_12 = {
                i: s["per_line"][i]["mean_ratio_adv_flipped"] / s["per_line"][i]["mean_ratio_control_matched"]
                for i in INFORMATIVE_LINE_INDICES
            }
            fold_10 = {i: fold_12[i] for i in ROBUST_LINE_INDICES}

            baseline_dist_10 = {
                i: ((e["dist_adv"][:, i].mean() + e["dist_control"][:, i].mean()) / 2).item()
                for i in ROBUST_LINE_INDICES
            }

            xs = [baseline_dist_10[i] for i in ROBUST_LINE_INDICES]
            ys = [fold_10[i] for i in ROBUST_LINE_INDICES]

            report[name][eps] = {
                "n_flipped": n_flipped,
                "fold_12": fold_12,
                "fold_10": fold_10,
                "mean_fold_12": sum(fold_12.values()) / len(fold_12),
                "mean_fold_10": sum(fold_10.values()) / len(fold_10),
                "min_fold_10": min(fold_10.values()),
                "max_fold_10": max(fold_10.values()),
                "all_10_survive_adv_gt_control": all(v > 1.0 for v in fold_10.values()),
                "baseline_dist_10": baseline_dist_10,
                "dist_fold_correlation_10": _pearson(xs, ys),
            }

    return report


def run_robustness_report(n_per_class: int = 20, epsilons=(0.02, 0.03, 0.05), seed: int = 0,
                           cnn_epochs: int = 3, strong_epochs: int = 3, verbose: bool = True) -> dict:
    """Reproducible entry point: regenerates results via
    run_per_path_adversarial_eval (deterministic given seed=0, verified
    bit-for-bit reproducible across independent prior runs) and runs
    fold_ratio_robustness on it. Prints a summary table.
    """
    results = run_per_path_adversarial_eval(
        n_per_class=n_per_class, epsilons=epsilons, seed=seed,
        cnn_epochs=cnn_epochs, strong_epochs=strong_epochs, verbose=verbose,
    )
    report = fold_ratio_robustness(results)

    for name, per_eps in report.items():
        for eps, r in per_eps.items():
            if r["n_flipped"] == 0:
                print(f"{name} eps={eps}: no flips, skipped")
                continue
            print(f"=== {name} eps={eps} (n_flipped={r['n_flipped']}) ===")
            print(f"  mean fold-ratio, 12 lines: {r['mean_fold_12']:.2f}x")
            print(f"  mean fold-ratio, 10 lines (excl. {EXCLUDED_LINES}): {r['mean_fold_10']:.2f}x")
            print(f"  min/max (10-line): {r['min_fold_10']:.2f}x / {r['max_fold_10']:.2f}x")
            print(f"  all 10 lines still adv > control: {r['all_10_survive_adv_gt_control']}")
            print(f"  dist-vs-fold correlation (10-line): {r['dist_fold_correlation_10']:.3f}")
            print()

    return {"results": results, "report": report}
