"""Distance functions over signature vectors, the within/cross-digit sanity check, and every
distance diagnostic built on top of them that needs no trained model.

Method A, Method B, and Method C are always kept as separate distance functions, never combined
into one metric.
"""

import statistics

import torch
from sklearn.metrics import roc_auc_score, roc_curve

from signature_distance.signatures import signature_of_stream
from signature_distance.streams import (
    NUM_SEGMENTS,
    hilbert_stream,
    line_stream,
    make_hilbert_curve,
    make_pixel_order,
    make_reference_lines,
    patch_sv_stream,
)

torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------
# Fixed geometry / configuration constants, shared across this file,
# adversarial_eval.py, headline_bootstrap.py, and method_b_sweep.py.
# ---------------------------------------------------------------------------

SIGNATURE_DEPTH = 4
# Fixed from the Phase 4 sanity check (run_experiment.sanity_check_demo),
METHOD_A_R = 1.6562803550838685
METHOD_B_R = 2.8597598377587485

# Method B's current winning configuration (Stage 8 sweep): 16 horizontal + 0 vertical lines, depth=2. 
METHOD_B_WINNER_DEPTH = 2
# Method C's settled depth
METHOD_C_DEPTH = 3  

# Method B's border lines (the first and last of the 16) are degenerate: they have the same signature for every image
METHOD_B_BORDER_LINE_INDICES = (0, 15)
METHOD_B_INFORMATIVE_LINE_INDICES = tuple(i for i in range(16) if i not in METHOD_B_BORDER_LINE_INDICES)

METHOD_A_PIXEL_ORDER = make_pixel_order(k=64, seed=0)  # fixed geometry, shared across every call
METHOD_B_LINES = make_reference_lines()  # original 8h+8v geometry, fixed, shared across every call
METHOD_B_WINNER_LINES = make_reference_lines(angles_deg=(0,), counts=(16,), points_per_line=32)


def _level_sizes(width: int, depth: int) -> list:
    """Number of coefficients at each signature level 0..depth, for the given tensor width."""
    return [width ** n for n in range(depth + 1)]

def _level_slices(width: int, depth: int) -> list:
    """Index range of each signature level 0..depth within a flattened signature vector."""
    idx = 0
    slices = []
    for size in _level_sizes(width, depth):
        slices.append((idx, idx + size))
        idx += size
    return slices


def rescale_signature(sig: torch.Tensor, r: float, depth: int, width: int = 2) -> torch.Tensor:
    """Scales each level-n block of a signature by r**n, correcting for the roughly 1/n! decay in raw
    signature magnitude with level."""
    slices = _level_slices(width, depth)
    expected_dim = slices[-1][1]
    if sig.shape[-1] != expected_dim:
        raise ValueError(
            f"sig last dim {sig.shape[-1]} != expected {expected_dim} "
            f"for width={width}, depth={depth}"
        )
    out = sig.clone()
    for n, (a, b) in enumerate(slices):
        out[..., a:b] = out[..., a:b] * (r ** n)
    return out


def choose_rescale_factor(sig: torch.Tensor, depth: int, width: int = 2) -> float:
    """Derives r empirically from a batch of signatures: the geometric mean of the level-to-level
    magnitude ratio, inverted, so that r**n roughly equalises level magnitudes. Requires depth >= 2."""
    if depth < 2:
        raise ValueError("choose_rescale_factor needs depth >= 2")
    slices = _level_slices(width, depth)
    mags = [sig[..., a:b].abs().mean().item() for a, b in slices]
    ratios = [mags[i] / mags[i - 1] for i in range(2, len(mags))]
    geo_mean = 1.0
    for ratio in ratios:
        geo_mean *= ratio
    geo_mean **= 1 / len(ratios)
    return 1.0 / geo_mean


def method_a_feature_vector(sig: torch.Tensor) -> torch.Tensor:
    """Method A's per-image feature vector: the signature itself, unchanged."""
    return sig


def method_b_feature_vector(line_sigs: torch.Tensor) -> torch.Tensor:
    """Concatenates Method B's 16 independent per-line signatures into one feature vector per image.
    line_sigs is (N, num_lines, sig_dim); returns (N, num_lines * sig_dim)."""
    n = line_sigs.shape[0]
    return line_sigs.reshape(n, -1)


def pairwise_euclidean_distance(vectors: torch.Tensor) -> torch.Tensor:
    """Pairwise Euclidean distance matrix for a batch of feature vectors."""
    return torch.cdist(vectors, vectors, p=2)


def per_line_distances(sig1: torch.Tensor, sig2: torch.Tensor) -> torch.Tensor:
    """16 separate per-line Euclidean distances between two images' signatures, rather than merging
    into one vector first. Floored at 1e-12 so a degenerate (identical) line signature can't produce
    a division-by-zero ratio downstream; real distances in this project's data are always far above
    this floor."""
    return (sig1 - sig2).norm(dim=-1).clamp_min(1e-12)


def within_vs_cross_digit_distance(vectors: torch.Tensor, labels: torch.Tensor) -> dict:
    """Mean pairwise distance for same-digit pairs vs different-digit pairs, over a sample. A
    meaningful distance should show within-digit pairs closer than cross-digit pairs."""
    dist = pairwise_euclidean_distance(vectors)
    n = vectors.shape[0]
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    diag = torch.eye(n, dtype=torch.bool, device=vectors.device)
    within_mask = same & ~diag
    cross_mask = ~same

    within_mean = dist[within_mask].mean().item()
    cross_mean = dist[cross_mask].mean().item()
    # Floored at 1e-12 (same convention as per_line_distances above) so a
    # structurally degenerate line/vector set can't raise
    # ZeroDivisionError or return a meaningless huge ratio - a no-op for
    # every non-degenerate cases
    return {
        "within_digit_mean": within_mean,
        "cross_digit_mean": cross_mean,
        "ratio_cross_over_within": cross_mean / max(within_mean, 1e-12),
    }


def auc_for_distance(same, dist_values, tpr_target: float = None) -> dict:
    """Same/different-label AUC for one distance measure, treating negative distance as a same-label
    classifier score."""
    scores = -dist_values
    result = {"auc": float(roc_auc_score(same, scores))}
    if tpr_target is not None:
        fpr, tpr, thresh = roc_curve(same, scores)
        idx = next((k for k, t in enumerate(tpr) if t >= tpr_target), len(tpr) - 1)
        pct = int(tpr_target * 100)
        result[f"fpr_at_tpr{pct}"] = float(fpr[idx])
        result[f"distance_threshold_at_tpr{pct}"] = float(-thresh[idx])
    return result


# ---------------------------------------------------------------------------
# Margin, pixel/signature distances - shared numerator/denominator building
# blocks for the adversarial-eval drivers in adversarial_eval.py.
# ---------------------------------------------------------------------------


def margin(model, x, y_true) -> torch.Tensor:
    """logit[y_true] - max(logit[j] for j != y_true), per example."""
    logits = model(x)
    true_logit = logits.gather(1, y_true.unsqueeze(1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, y_true.unsqueeze(1), float("-inf"))
    runner_up_logit = masked.max(dim=1).values
    return true_logit - runner_up_logit


def pixel_euclidean_distance(x1, x2) -> torch.Tensor:
    """Plain Euclidean distance in flat pixel space."""
    n = x1.shape[0]
    return (x1.reshape(n, -1) - x2.reshape(n, -1)).norm(dim=1)


def method_a_signature_distance(images1: torch.Tensor, images2: torch.Tensor,
                                 depth: int = SIGNATURE_DEPTH, r: float = METHOD_A_R) -> torch.Tensor:
    """Method A's full distance pipeline: fixed pixel order, patch singular-value stream, signature,
    rescale, Euclidean distance. images1, images2 are (N, 28, 28) in [0, 1]."""
    def _feature_vector(images):
        stream = patch_sv_stream(images, METHOD_A_PIXEL_ORDER)  # (N, 64, 2)
        sig = signature_of_stream(stream, depth=depth)  # (N, sig_dim)
        sig = rescale_signature(sig, r=r, depth=depth)
        return method_a_feature_vector(sig)  # (N, sig_dim) - identity

    vec1 = _feature_vector(images1)
    vec2 = _feature_vector(images2)
    return (vec1 - vec2).norm(dim=1)


def method_b_signature_distance(images1: torch.Tensor, images2: torch.Tensor,
                                 depth: int = SIGNATURE_DEPTH, r: float = METHOD_B_R) -> torch.Tensor:
    """Method B's full distance pipeline: fixed reference lines, per-line signature, rescale,
    concatenation, Euclidean distance. images1, images2 are (N, 28, 28) in [0, 1]."""
    num_lines = METHOD_B_LINES.shape[0]

    def _feature_vector(images):
        stream = line_stream(images, METHOD_B_LINES)  # (N, num_lines, points_per_line, 2)
        sig = torch.stack(
            [signature_of_stream(stream[:, i], depth=depth) for i in range(num_lines)], dim=1
        )  # (N, num_lines, sig_dim)
        sig = rescale_signature(sig, r=r, depth=depth)
        return method_b_feature_vector(sig)  # (N, num_lines * sig_dim)

    vec1 = _feature_vector(images1)
    vec2 = _feature_vector(images2)
    return (vec1 - vec2).norm(dim=1)


# ---------------------------------------------------------------------------
# Per-line distance diagnostic (Method B) - AUC ranking of the 16 individual
# lines against the merged 496-dim concatenated distance.
# ---------------------------------------------------------------------------

def _line_orientation_label(lines: torch.Tensor, line_idx: int) -> str:
    """Whether a reference line is horizontal or vertical."""
    line = lines[line_idx]
    rows, cols = line[:, 0], line[:, 1]
    horizontal = (rows.max() - rows.min()) < (cols.max() - cols.min())
    if horizontal:
        return f"horizontal (row={rows[0].item():.1f})"
    return f"vertical (col={cols[0].item():.1f})"


def run_per_line_auc_diagnostic(n_per_class: int = 30, seed: int = 0,
                                 depth: int = SIGNATURE_DEPTH,
                                 tpr_target: float = 0.90) -> dict:
    """Same/different-digit AUC for each of Method B's 16 individual lines and for the merged
    distance, on the same sample. Returns the ranked AUC results."""
    from signature_distance.data_pool import load_eval_pool

    images, labels = load_eval_pool(n_per_class=n_per_class, seed=seed)
    lines = METHOD_B_LINES
    num_lines = lines.shape[0]

    stream = line_stream(images, lines)  # (N, num_lines, points_per_line, 2)
    sig_raw = torch.stack(
        [signature_of_stream(stream[:, i], depth=depth) for i in range(num_lines)], dim=1
    )  # (N, num_lines, sig_dim)
    r = choose_rescale_factor(sig_raw, depth=depth)
    sig = rescale_signature(sig_raw, r=r, depth=depth)  # (N, num_lines, sig_dim), same as sanity_check_demo

    # Merged distance matrix - existing pipeline, unmodified.
    merged_vec = method_b_feature_vector(sig)
    merged_dist_matrix = torch.cdist(merged_vec, merged_vec, p=2)

    # Per-line distance matrices - one (N, N) matrix per line, using cdist
    # per line (equivalent to per_line_distances, batched over pairs).
    per_line_dist_matrices = torch.stack(
        [torch.cdist(sig[:, i], sig[:, i], p=2) for i in range(num_lines)], dim=0
    )  # (num_lines, N, N)

    n = images.shape[0]
    iu, ju = torch.triu_indices(n, n, offset=1)  # every unordered pair once, no self-pairs
    same = (labels[iu] == labels[ju]).numpy().astype(int)

    measures = {}
    merged_pair_dist = merged_dist_matrix[iu, ju].numpy()
    measures["merged"] = auc_for_distance(same, merged_pair_dist, tpr_target)

    for i in range(num_lines):
        line_pair_dist = per_line_dist_matrices[i][iu, ju].numpy()
        entry = auc_for_distance(same, line_pair_dist, tpr_target)
        entry["orientation"] = _line_orientation_label(lines, i)
        measures[f"line_{i}"] = entry

    ranked = sorted(measures.items(), key=lambda kv: kv[1]["auc"], reverse=True)

    return {
        "n_images": n, "n_pairs": int(iu.shape[0]), "r": r, "depth": depth,
        "tpr_target": tpr_target,
        "measures": measures,
        "ranked": ranked,
        "best_individual_line_beats_merged": ranked[0][0] != "merged",
    }


# ---------------------------------------------------------------------------
# Same/different-digit AUC per depth, for Method C's Hilbert segments
# ---------------------------------------------------------------------------

HILBERT_DEPTH_VARIANTS = (2, 3, 4)

def evaluate_hilbert_depths(n_per_class: int = 30, seed: int = 0,
                             depths=HILBERT_DEPTH_VARIANTS) -> dict:
    """Same/different-digit AUC per Hilbert segment, at each candidate depth. Computes the signature
    once at the maximum depth and slices lower depths from it."""
    from signature_distance.data_pool import load_eval_pool
    from signature_distance.method_b_sweep import signature_dim  # lazy: method_b_sweep imports this module

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
# Cross/within-digit distance ratio, per-path (never merged), for Method B's
# winning configuration and Method C, plus a plain pixel-Euclidean baseline
# computed the same way, on the same images.
# ---------------------------------------------------------------------------

_DEGENERATE_THRESHOLD = 1e-9

def _safe_within_vs_cross(vectors: torch.Tensor, labels: torch.Tensor) -> dict:
    """within_vs_cross_digit_distance, guarded against a structurally degenerate line or segment
    (one producing an identical signature regardless of digit, e.g. Method B's border lines) —
    flagged rather than silently reported as a near-zero within-digit distance."""
    result = within_vs_cross_digit_distance(vectors, labels)
    if result["within_digit_mean"] <= _DEGENERATE_THRESHOLD:
        return {
            "within_digit_mean": result["within_digit_mean"], "cross_digit_mean": float("nan"),
            "ratio_cross_over_within": float("nan"), "degenerate": True,
        }
    result["degenerate"] = False
    return result


def pixel_euclidean_cross_within(images: torch.Tensor, labels: torch.Tensor) -> dict:
    """Baseline within/cross-digit distance check on raw flattened pixels."""
    flat = images.reshape(images.shape[0], -1)
    return within_vs_cross_digit_distance(flat, labels)


def method_b_per_line_cross_within(images: torch.Tensor, labels: torch.Tensor,
                                    lines: torch.Tensor = None,
                                    depth: int = METHOD_B_WINNER_DEPTH) -> dict:
    """Per-line and merged within/cross-digit distance ratio for Method B's winning configuration.
    Structurally border-adjacent lines are excluded from the aggregate per-line statistics."""
    if lines is None:
        lines = METHOD_B_WINNER_LINES
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
    """Per-segment and merged within/cross-digit distance ratio for Method C. No segment exclusion,
    since no structurally degenerate segment was found."""
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
    """Full within/cross-digit comparison: pixel-Euclidean baseline, Method B per-line, and Method C
    per-segment, on the same image pool."""
    from signature_distance.data_pool import load_eval_pool

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


# ---------------------------------------------------------------------------
# Level-wise decomposition of Method A/B signature distances - is the
# within/cross-digit signal carried entirely by the level-1 terms, or do
# higher-order levels (2..depth) contribute?
# ---------------------------------------------------------------------------

def level_slices(depth: int, width: int = 2) -> dict:
    """Index range of each signature level 0..depth within a signature_of_stream output."""
    slices = {}
    idx = 0
    for n in range(depth + 1):
        size = width ** n
        slices[n] = slice(idx, idx + size)
        idx += size
    return slices


def mask_signature_levels(sig: torch.Tensor, levels, depth: int,
                           width: int = 2) -> torch.Tensor:
    """Returns a copy of sig with every level not in levels zeroed out, keeping the full signature
    layout so existing feature/distance functions can be reused unmodified."""
    slices = level_slices(depth, width=width)
    out = torch.zeros_like(sig)
    for n in levels:
        s = slices[n]
        out[..., s] = sig[..., s]
    return out


def _per_level_fraction(sig: torch.Tensor, depth: int, feature_fn) -> dict:
    """Mean fraction of total squared pairwise distance contributed by each signature level, averaged
    over all unique pairs in a sample."""
    n = sig.shape[0]
    iu, ju = torch.triu_indices(n, n, offset=1)

    sq_dist_per_level = {}
    for level in range(1, depth + 1):
        masked = mask_signature_levels(sig, [level], depth=depth)
        vec = feature_fn(masked)
        vec = vec.reshape(n, -1)
        diff = vec[iu] - vec[ju]
        sq_dist_per_level[level] = (diff ** 2).sum(dim=-1)  # (n_pairs,)

    total = sum(sq_dist_per_level.values())
    return {
        level: (sq_dist_per_level[level] / total).mean().item()
        for level in sq_dist_per_level
    }


def _variant_levels(depth: int) -> dict:
    """The set of level-inclusion variants (all, level1_only, level2plus, etc.) tested by
    run_level_decomposition."""
    return {
        "all": list(range(1, depth + 1)),
        "level1_only": [1],
        "level2plus": list(range(2, depth + 1)),
        "level2_only": [2],
        "level3_only": [3],
        "level4_only": [4],
    }


def run_level_decomposition(n_per_class: int = 30, seed: int = 0,
                             depth: int = SIGNATURE_DEPTH,
                             pixel_order_seed: int = None) -> dict:
    """Level-wise decomposition of the within/cross-digit sanity check, for Method A and Method B
    independently: for each level-inclusion variant, the within/cross ratio and each method's
    per-level contribution to total distance."""
    from signature_distance.data_pool import load_eval_pool

    images, labels = load_eval_pool(n_per_class=n_per_class, seed=seed)

    order_seed = seed if pixel_order_seed is None else pixel_order_seed
    order = make_pixel_order(k=64, seed=order_seed)
    sig_a_raw = signature_of_stream(patch_sv_stream(images, order), depth=depth)
    r_a = choose_rescale_factor(sig_a_raw, depth=depth)
    sig_a = rescale_signature(sig_a_raw, r=r_a, depth=depth)

    lines = make_reference_lines()
    stream_b = line_stream(images, lines)
    sig_b_raw = torch.stack(
        [signature_of_stream(stream_b[:, i], depth=depth) for i in range(stream_b.shape[1])],
        dim=1,
    )
    r_b = choose_rescale_factor(sig_b_raw, depth=depth)
    sig_b = rescale_signature(sig_b_raw, r=r_b, depth=depth)

    variant_results = {}
    for label, levels in _variant_levels(depth).items():
        vec_a = method_a_feature_vector(mask_signature_levels(sig_a, levels, depth=depth))
        vec_b = method_b_feature_vector(mask_signature_levels(sig_b, levels, depth=depth))
        variant_results[label] = {
            "levels": levels,
            "method_a": within_vs_cross_digit_distance(vec_a, labels),
            "method_b": within_vs_cross_digit_distance(vec_b, labels),
        }

    return {
        "n_images": images.shape[0],
        "pixel_order_seed": order_seed,
        "r_a": r_a,
        "r_b": r_b,
        "variants": variant_results,
        "level_fraction": {
            "method_a": _per_level_fraction(sig_a, depth, method_a_feature_vector),
            "method_b": _per_level_fraction(sig_b, depth, method_b_feature_vector),
        },
    }
