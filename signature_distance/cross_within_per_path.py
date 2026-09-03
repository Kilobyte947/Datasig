"""Cross/within-digit distance ratio, per-path (never merged), for Method
B's winning configuration and Method C, plus a plain pixel-Euclidean
baseline computed the same way, on the same images.

Closes a gap identified in conversation: this project never directly
compared per-path signature distance against pixel-Euclidean distance on
a same/different-digit basis. The only same/different-digit numbers that
existed before this module were (a) `per_line_diagnostics.py`'s AUC
ranking, which never included a pixel baseline, and (b) the original
Phase 4 sanity check (`distances.within_vs_cross_digit_distance`), which
only ever measured the *merged* signature distance and cited pixel's own
ratio (~1.13) as an external reference from a different project
(`mnist_lipschitz`), not something computed in this codebase.

Reuses `distances.within_vs_cross_digit_distance` UNMODIFIED (the exact
Phase 4 check) - applied per line/segment instead of after merging, which
is all "per-path" means here: the same function, called once per line/
segment on that line/segment's own (N, sig_dim) vectors, instead of once
on the concatenated (N, num_lines * sig_dim) vector. No model training or
adversarial generation - this is a label-based sanity check on clean
images only, same protocol and same cost as Phase 4.

Sample: n_per_class=30 (300 images), seed=0, matching Phase 4's own
convention exactly, so the merged-distance numbers reported here are
directly comparable to Phase 4's historical 1.176 (Method A) / 1.160
(Method B, original config) figures.
"""

import statistics
from pathlib import Path

import torch

from signature_distance.data_pool import load_eval_pool
from signature_distance.distances import (
    choose_rescale_factor,
    method_b_feature_vector,
    rescale_signature,
    within_vs_cross_digit_distance,
)
from signature_distance.headline_plot import METHOD_B_BORDER_LINE_INDICES
from signature_distance.hilbert_stream import NUM_SEGMENTS, hilbert_stream, make_hilbert_curve
from signature_distance.pgd_adversarial_eval import (
    METHOD_B_WINNER_DEPTH,
    METHOD_B_WINNER_LINES,
    METHOD_C_DEPTH,
)
from signature_distance.signatures import signature_of_stream
from signature_distance.streams import line_stream

RESULTS_DIR = Path(__file__).parent / "results"


def _safe_within_vs_cross(vectors: torch.Tensor, labels: torch.Tensor) -> dict:
    """Wraps `within_vs_cross_digit_distance` (unmodified) with a guard
    against the real, discovered failure mode: a structurally degenerate
    line/segment (e.g. Method B's border lines, which sit on image rows/
    columns MNIST digits never touch) produces the IDENTICAL signature for
    every image regardless of digit, giving a within-digit distance of
    exactly 0 and a division by zero - not a coding error to swallow
    silently, but a real, informative outcome (this line/segment carries
    no same/different-digit signal at all) worth reporting as such rather
    than crashing the whole comparison."""
    try:
        result = within_vs_cross_digit_distance(vectors, labels)
    except ZeroDivisionError:
        return {
            "within_digit_mean": 0.0, "cross_digit_mean": float("nan"),
            "ratio_cross_over_within": float("nan"), "degenerate": True,
        }
    result["degenerate"] = False
    return result


def pixel_euclidean_cross_within(images: torch.Tensor, labels: torch.Tensor) -> dict:
    """Baseline: plain flattened-pixel Euclidean distance, same
    within_vs_cross_digit_distance check as everything else here."""
    flat = images.reshape(images.shape[0], -1)
    return within_vs_cross_digit_distance(flat, labels)


def method_b_per_line_cross_within(images: torch.Tensor, labels: torch.Tensor,
                                    lines: torch.Tensor = METHOD_B_WINNER_LINES,
                                    depth: int = METHOD_B_WINNER_DEPTH) -> dict:
    """Per-line (never merged) cross/within ratio for Method B's winning
    configuration (12h+4v, depth=2 by default), plus the same check on the
    merged (concatenated) vector for direct reference against the
    per-line numbers and against Phase 4's historical merged figure.

    The 4 structurally border-adjacent lines (`headline_plot.
    METHOD_B_BORDER_LINE_INDICES`) are excluded from the aggregate
    (mean/median/best/worst) stats, not just whichever ones happen to
    trigger `_safe_within_vs_cross`'s zero-division guard - checked
    directly (not assumed) that relying on the guard alone is sample-size
    fragile: at n_per_class=3 all 4 border lines are exactly zero-
    signature for every image (ZeroDivisionError, caught); at
    n_per_class=30, two of the four (11, 15) are each touched by exactly
    1 of 300 images, avoiding the exact zero-division but producing a
    ratio of precisely 1.0 - a single-outlier-image artifact, not
    meaningful same/different-digit signal, so excluded on the same
    border-adjacency grounds as the other two rather than kept because it
    happened not to crash."""
    num_lines = lines.shape[0]
    stream = line_stream(images, lines)
    sig_raw = torch.stack(
        [signature_of_stream(stream[:, i], depth=depth) for i in range(num_lines)], dim=1
    )
    r = choose_rescale_factor(sig_raw, depth=depth)
    sig = rescale_signature(sig_raw, r=r, depth=depth)

    per_line = {i: _safe_within_vs_cross(sig[:, i], labels) for i in range(num_lines)}
    merged_vec = method_b_feature_vector(sig)
    merged = _safe_within_vs_cross(merged_vec, labels)

    excluded_lines = sorted(set(METHOD_B_BORDER_LINE_INDICES) | {i for i, v in per_line.items() if v["degenerate"]})
    ratios = [v["ratio_cross_over_within"] for i, v in per_line.items() if i not in excluded_lines]
    return {
        "r": r, "depth": depth, "num_lines": num_lines,
        "per_line": per_line, "merged": merged, "degenerate_lines": excluded_lines,
        "mean_ratio_over_lines": sum(ratios) / len(ratios) if ratios else float("nan"),
        "median_ratio_over_lines": statistics.median(ratios) if ratios else float("nan"),
        "best_line_ratio": max(ratios) if ratios else float("nan"),
        "worst_line_ratio": min(ratios) if ratios else float("nan"),
    }


def method_c_per_segment_cross_within(images: torch.Tensor, labels: torch.Tensor,
                                       curve: torch.Tensor = None,
                                       depth: int = METHOD_C_DEPTH) -> dict:
    """Per-segment (never merged) cross/within ratio for Method C's
    Hilbert-curve construction, plus the merged-vector reference number.
    No border-segment exclusion, matching how Method C's own numbers are
    reported everywhere else in this project (Stage A found no
    structurally degenerate segment)."""
    if curve is None:
        curve = make_hilbert_curve()
    stream = hilbert_stream(images, curve)
    sig_raw = torch.stack(
        [signature_of_stream(stream[:, i], depth=depth) for i in range(NUM_SEGMENTS)], dim=1
    )
    r = choose_rescale_factor(sig_raw, depth=depth)
    sig = rescale_signature(sig_raw, r=r, depth=depth)

    per_segment = {i: _safe_within_vs_cross(sig[:, i], labels) for i in range(NUM_SEGMENTS)}
    merged_vec = sig.reshape(sig.shape[0], -1)  # same concatenation convention as method_b_feature_vector
    merged = _safe_within_vs_cross(merged_vec, labels)

    degenerate_segments = [i for i, v in per_segment.items() if v["degenerate"]]
    ratios = [v["ratio_cross_over_within"] for i, v in per_segment.items() if not v["degenerate"]]
    return {
        "r": r, "depth": depth, "num_segments": NUM_SEGMENTS,
        "per_segment": per_segment, "merged": merged, "degenerate_segments": degenerate_segments,
        "mean_ratio_over_segments": sum(ratios) / len(ratios) if ratios else float("nan"),
        "median_ratio_over_segments": statistics.median(ratios) if ratios else float("nan"),
        "best_segment_ratio": max(ratios) if ratios else float("nan"),
        "worst_segment_ratio": min(ratios) if ratios else float("nan"),
    }


def run_cross_within_comparison(n_per_class: int = 30, seed: int = 0, verbose: bool = True) -> dict:
    """Full comparison: pixel-Euclidean baseline, Method B per-line
    (winning config), Method C per-segment - same 300-image pool (Phase
    4's own n_per_class=30 convention), same underlying
    within_vs_cross_digit_distance check throughout."""
    images, labels = load_eval_pool(n_per_class=n_per_class, seed=seed)

    pixel = pixel_euclidean_cross_within(images, labels)
    method_b = method_b_per_line_cross_within(images, labels)
    method_c = method_c_per_segment_cross_within(images, labels)

    if verbose:
        print(f"pixel-Euclidean:        ratio={pixel['ratio_cross_over_within']:.4f}")
        print(f"Method B merged (winner config): ratio={method_b['merged']['ratio_cross_over_within']:.4f}")
        if method_b["degenerate_lines"]:
            print(f"  (degenerate/border lines excluded from per-line stats: {method_b['degenerate_lines']})")
        print(f"Method B per-line mean:  ratio={method_b['mean_ratio_over_lines']:.4f}  "
              f"(best={method_b['best_line_ratio']:.4f}, worst={method_b['worst_line_ratio']:.4f})")
        print(f"Method C merged:         ratio={method_c['merged']['ratio_cross_over_within']:.4f}")
        if method_c["degenerate_segments"]:
            print(f"  (degenerate segments excluded from per-segment stats: {method_c['degenerate_segments']})")
        print(f"Method C per-segment mean: ratio={method_c['mean_ratio_over_segments']:.4f}  "
              f"(best={method_c['best_segment_ratio']:.4f}, worst={method_c['worst_segment_ratio']:.4f})")

    return {"n_images": images.shape[0], "pixel": pixel, "method_b": method_b, "method_c": method_c}
