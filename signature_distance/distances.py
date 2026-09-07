"""Distance functions over signature vectors (Phase 3), the within/cross
-digit sanity check (Phase 4) - see README.md - and every distance-diagnostic
built directly on top of them that needs no trained model (cross/within-
digit per-path breakdown, level-wise decomposition, per-line AUC ranking,
Method C's per-depth AUC screen). Shared by all three methods; Method A,
Method B, and Method C are always kept as separate distance functions,
never combined into one metric (per README.md's "two candidate distance
functions, not one" note, extended to three with Method C).

No model training or adversarial perturbation happens in this file - every
function here operates on clean images/signatures only, or on
already-computed tensors handed in by a caller. Adversarial evaluation
(which trains models and runs attacks, including the border-line/pixel
check) lives in `adversarial_eval.py`.
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
# not re-derived here - Method A/B's distance functions aren't changed by
# this file, only evaluated.
METHOD_A_R = 1.6562803550838685
METHOD_B_R = 2.8597598377587485

# Method B's current winning configuration (Stage 8 sweep, README.md):
# 16 horizontal + 0 vertical lines, depth=2. Re-derived on the canonical
# mnist_example checkpoint during the pre-publication audit - "12h+4v,
# depth=2" was the winner under the throwaway 3-epoch models Stage B
# originally trained; on the canonical checkpoint, 16h+0v beats it on
# BOTH models (SmallCNN and StrongCNN) with fewer exceptions, not just a
# split decision as it was before, so it replaces 12h+4v as the winner.
METHOD_B_WINNER_DEPTH = 2
METHOD_C_DEPTH = 3  # Method C's settled depth (README.md).

# 16h+0v has only 2 structurally border-adjacent lines (the first and last
# of the 16 horizontal lines, rows 0 and 27) - no vertical lines at all,
# so no border-column lines the way 12h+4v's 4 border lines had.
METHOD_B_BORDER_LINE_INDICES = (0, 15)
METHOD_B_INFORMATIVE_LINE_INDICES = tuple(i for i in range(16) if i not in METHOD_B_BORDER_LINE_INDICES)

METHOD_A_PIXEL_ORDER = make_pixel_order(k=64, seed=0)  # fixed geometry, shared across every call
METHOD_B_LINES = make_reference_lines()  # original 8h+8v geometry, fixed, shared across every call
METHOD_B_WINNER_LINES = make_reference_lines(angles_deg=(0,), counts=(16,), points_per_line=32)


def _level_sizes(width: int, depth: int) -> list:
    return [width ** n for n in range(depth + 1)]


def _level_slices(width: int, depth: int) -> list:
    idx = 0
    slices = []
    for size in _level_sizes(width, depth):
        slices.append((idx, idx + size))
        idx += size
    return slices


def rescale_signature(sig: torch.Tensor, r: float, depth: int, width: int = 2) -> torch.Tensor:
    """Scale each level-n block of a (..., sig_dim) signature by r**n.

    Signature terms decay ~1/n! with depth, so raw level-4 coefficients are
    tiny next to level-1 (verified empirically: Method A/B raw level-4
    magnitudes are roughly 4-50x smaller than level-1) - without this,
    Euclidean distance on the raw signature would mostly just measure the
    depth-1 terms. sig's last dimension must equal
    sum(width**n for n in 0..depth).
    """
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
    """Derive r empirically from a batch of already-computed signatures:
    the geometric mean of the level-to-level magnitude ratio across levels
    2..depth (skipping the level 0->1 step, since level 0 is always the
    trivial constant 1.0, not a meaningful decay rate), then r = 1 / that
    ratio so that r**n roughly equalizes level magnitudes. Requires
    depth >= 2. Run once per method (not shared across methods - their raw
    signature scales differ, and they're always compared as two separate
    distance functions, never combined).
    """
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
    """Method A: the (rescaled) signature is already the full per-image
    feature vector - no concatenation needed. Identity, kept only for
    interface symmetry with method_b_feature_vector."""
    return sig


def method_b_feature_vector(line_sigs: torch.Tensor) -> torch.Tensor:
    """Method B: concatenate the 16 independent per-line (rescaled)
    signatures into one feature vector per image. This is the first point
    the 16 lines combine - deferred until after the signature step, never
    before (see README.md's "no cross-line concatenation" rule).

    line_sigs: (N, num_lines, sig_dim) -> (N, num_lines * sig_dim).
    """
    n = line_sigs.shape[0]
    return line_sigs.reshape(n, -1)


def pairwise_euclidean_distance(vectors: torch.Tensor) -> torch.Tensor:
    """(N, D) feature vectors -> (N, N) pairwise Euclidean distance matrix."""
    return torch.cdist(vectors, vectors, p=2)


def per_line_distances(sig1: torch.Tensor, sig2: torch.Tensor) -> torch.Tensor:
    """16 separate per-line Euclidean distances, instead of merging into one
    concatenated vector first - compares line i to line i directly ("path
    by path", per the method's original framing), never fusing signal
    across lines before computing a distance. Deliberately the step right
    before `method_b_feature_vector`'s concatenation, not a replacement for
    it - both are kept, this is an additive diagnostic.

    Use on already-rescaled per-line signatures (e.g. via
    rescale_signature), same as before concatenation in the existing
    pipeline - no change to that step, just stop one step earlier.

    sig1, sig2: (..., num_lines, sig_dim).
    returns: (..., num_lines).

    Floored at 1e-12 (matching mnist_example/estimators.py's convention)
    so a degenerate zero-distance line - e.g. a border line whose signature
    is identical for both images - can't produce a division-by-zero/Inf
    ratio in a downstream Lipschitz-ratio computation. A no-op on every
    non-degenerate case in the project's reported numbers (verified: real
    per-line distances are always many orders of magnitude above this
    floor), so this doesn't change any previously reported figure.
    """
    return (sig1 - sig2).norm(dim=-1).clamp_min(1e-12)


def within_vs_cross_digit_distance(vectors: torch.Tensor, labels: torch.Tensor) -> dict:
    """Cheap, label-based sanity check (README.md Phase 4): mean pairwise
    Euclidean distance for same-digit pairs vs. different-digit pairs, over
    the given sample. No model needed - run before anything downstream
    (adversarial/Lipschitz evaluation, sweeps). A meaningful distance
    should show within-digit pairs closer than cross-digit pairs.
    """
    dist = pairwise_euclidean_distance(vectors)
    n = vectors.shape[0]
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    diag = torch.eye(n, dtype=torch.bool, device=vectors.device)
    within_mask = same & ~diag
    cross_mask = ~same

    within_mean = dist[within_mask].mean().item()
    cross_mean = dist[cross_mask].mean().item()
    # Floored at 1e-12 (same convention as per_line_distances above) so a
    # structurally degenerate line/vector set (e.g. Method B's border lines,
    # whose signature is identical across every image) can't raise
    # ZeroDivisionError or return a meaningless huge ratio - a no-op for
    # every non-degenerate case actually reported in this project.
    return {
        "within_digit_mean": within_mean,
        "cross_digit_mean": cross_mean,
        "ratio_cross_over_within": cross_mean / max(within_mean, 1e-12),
    }


def auc_for_distance(same, dist_values, tpr_target: float = None) -> dict:
    """Same/different-label AUC for one distance measure, treating
    `-distance` as a same-label classifier score. Shared by
    `run_per_line_auc_diagnostic` below (per-line vs. merged AUC ranking) and
    method_b_sweep.py (per-line AUC across the hyperparameter grid) - both
    previously computed this identically but independently, once each.

    same: (n_pairs,) array-like, 1 if the pair shares a label, 0 otherwise.
    dist_values: (n_pairs,) that measure's distance for each pair - both
    typically built via `torch.triu_indices` over an (N, N) distance
    matrix, by the caller (kept local to each caller since it's a couple
    of trivial indexing lines, not worth abstracting further).
    tpr_target: if given (e.g. 0.90), also returns the FPR and distance
    threshold at that TPR operating point; omitted by default so callers
    that only need the AUC (e.g. a large sweep) don't pay for
    `roc_curve` unnecessarily.
    """
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
    """logit[y_true] - max(logit[j] for j != y_true), per example - the
    scalar this project's Lipschitz-ratio numerator is always built from,
    not raw logits or cross-entropy loss."""
    logits = model(x)
    true_logit = logits.gather(1, y_true.unsqueeze(1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, y_true.unsqueeze(1), float("-inf"))
    runner_up_logit = masked.max(dim=1).values
    return true_logit - runner_up_logit


def pixel_euclidean_distance(x1, x2) -> torch.Tensor:
    """Baseline denominator: plain Euclidean distance in flat pixel space.
    x1, x2: (N, ...) same shape."""
    n = x1.shape[0]
    return (x1.reshape(n, -1) - x2.reshape(n, -1)).norm(dim=1)


def method_a_signature_distance(images1: torch.Tensor, images2: torch.Tensor,
                                 depth: int = SIGNATURE_DEPTH, r: float = METHOD_A_R) -> torch.Tensor:
    """Method A's own pipeline - make_pixel_order (once, module-level) ->
    patch_sv_stream -> signature_of_stream -> rescale_signature ->
    method_a_feature_vector (identity) -> Euclidean.

    images1, images2: (N, 28, 28) in [0, 1].
    """
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
    """Method B's own pipeline - make_reference_lines (once, module-level)
    -> line_stream -> signature_of_stream (per line) -> rescale_signature ->
    method_b_feature_vector (concatenate the 16 lines) -> Euclidean.

    images1, images2: (N, 28, 28) in [0, 1].
    """
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
    """Fresh, small re-derivation of the same horizontal/vertical test
    plots.py's plot_reference_lines already uses for coloring - reported
    here as a label, not a plot."""
    line = lines[line_idx]
    rows, cols = line[:, 0], line[:, 1]
    horizontal = (rows.max() - rows.min()) < (cols.max() - cols.min())
    if horizontal:
        return f"horizontal (row={rows[0].item():.1f})"
    return f"vertical (col={cols[0].item():.1f})"


def run_per_line_auc_diagnostic(n_per_class: int = 30, seed: int = 0,
                                 depth: int = SIGNATURE_DEPTH,
                                 tpr_target: float = 0.90) -> dict:
    """Reuses the Phase 4 sanity-check sample (n_per_class per digit, same
    seed/pool as run_experiment.sanity_check_demo). For every pair of
    images: the existing merged 496-dim distance (unchanged - same
    rescale-then-concatenate-then-Euclidean pipeline), and 16 separate
    per-line distances (rescale, then stop before concatenation).

    For each of the resulting 17 distance measures, treats -distance as a
    same/different-digit classifier score and computes an ROC curve + AUC
    over all pairs, plus the FPR and distance threshold at `tpr_target`
    (default 90%) TPR. Returns a dict with per-measure results, ranked by
    AUC, plus each line's orientation/position for the secondary
    line-ranking question.
    """
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
# (Stage A cheap screen - no model training).
# ---------------------------------------------------------------------------

HILBERT_DEPTH_VARIANTS = (2, 3, 4)


def evaluate_hilbert_depths(n_per_class: int = 30, seed: int = 0,
                             depths=HILBERT_DEPTH_VARIANTS) -> dict:
    """Same/different-digit AUC per segment, for each depth in `depths`,
    computed via the same max-depth-then-prefix-slice shortcut used for
    Method B's sweep (verified there to be numerically exact) - the
    expensive signature step runs once, at the maximum depth, and every
    lower depth is sliced from that single result.

    Default n_per_class=30 matches what signatures_formation.ipynb's Method C
    section actually calls this with to produce README.md's Stage A depth-sweep table
    (0.5654/0.6485 at depth 2, etc.) - an earlier default of 15 here did
    not reproduce that table (gave 0.5722/0.6874 instead), a reproducibility
    trap for anyone calling this with its bare defaults.
    """
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

_DEGENERATE_THRESHOLD = 1e-9  # comfortably above this file's own 1e-12 division floor


def _safe_within_vs_cross(vectors: torch.Tensor, labels: torch.Tensor) -> dict:
    """Wraps `within_vs_cross_digit_distance` (unmodified) with a guard
    against the real, discovered failure mode: a structurally degenerate
    line/segment (e.g. Method B's border lines, which sit on image rows/
    columns MNIST digits never touch) produces the IDENTICAL signature for
    every image regardless of digit, giving a within-digit distance of
    (near) zero - not a coding error to swallow silently, but a real,
    informative outcome (this line/segment carries no same/different-digit
    signal at all) worth reporting as such rather than folding it into the
    aggregate stats as if it were a real ratio.

    Degeneracy is detected by checking `within_digit_mean` directly against
    `_DEGENERATE_THRESHOLD`, not by catching a ZeroDivisionError from
    `within_vs_cross_digit_distance` - that function floors its own
    division, so it no longer raises on an exactly-zero within-mean;
    checking the value directly here also catches the near-zero case a
    bare exception never would have."""
    result = within_vs_cross_digit_distance(vectors, labels)
    if result["within_digit_mean"] <= _DEGENERATE_THRESHOLD:
        return {
            "within_digit_mean": result["within_digit_mean"], "cross_digit_mean": float("nan"),
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
                                    lines: torch.Tensor = None,
                                    depth: int = METHOD_B_WINNER_DEPTH) -> dict:
    """Per-line (never merged) cross/within ratio for Method B's winning
    configuration (16h+0v, depth=2 by default), plus the same check on the
    merged (concatenated) vector for direct reference against the
    per-line numbers and against Phase 4's historical merged figure.

    The structurally border-adjacent lines (`METHOD_B_BORDER_LINE_INDICES`)
    are excluded from the aggregate (mean/median/best/worst) stats, not just
    whichever ones happen to trigger `_safe_within_vs_cross`'s zero-division
    guard - relying on the guard alone is sample-size fragile: a border line
    can sit at exact-zero signature for every image at a small sample, or
    escape the exact-zero case at a larger sample while still only being
    touched by a handful of outlier images (a ratio near 1.0 that's a
    single-image artifact, not real same/different-digit signal) - excluded
    on border-adjacency grounds either way, not kept just because it
    happened not to crash."""
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
    """Map each signature level 0..depth to its index block in a
    `signature_of_stream(..., depth=depth)` output. Level n occupies
    `width**n` entries, in level order (0 is the constant term, 1 is net
    displacement, etc.) - matches signatures.py's output layout exactly.

    For width=2, depth=4: {0: slice(0,1), 1: slice(1,3), 2: slice(3,7),
    3: slice(7,15), 4: slice(15,31)}, partitioning range(0, 31) exactly.
    """
    slices = {}
    idx = 0
    for n in range(depth + 1):
        size = width ** n
        slices[n] = slice(idx, idx + size)
        idx += size
    return slices


def mask_signature_levels(sig: torch.Tensor, levels, depth: int,
                           width: int = 2) -> torch.Tensor:
    """Return a copy of `sig` with every level not in `levels` zeroed out.

    Zeroes rather than slices, so the output keeps the full signature
    layout (31-dim for width=2/depth=4, or Method B's (N, 16, 31) before
    concatenation) - existing distance functions (`method_a_feature_vector`,
    `method_b_feature_vector`, `within_vs_cross_digit_distance`) can be
    reused unmodified on the result. Works on any leading batch shape,
    since the level blocks are slices of the trailing axis.

    sig: (..., sum(width**n for n in 0..depth)).
    levels: iterable of level indices (0..depth) to keep.
    """
    slices = level_slices(depth, width=width)
    out = torch.zeros_like(sig)
    for n in levels:
        s = slices[n]
        out[..., s] = sig[..., s]
    return out


def _per_level_fraction(sig: torch.Tensor, depth: int, feature_fn) -> dict:
    """Mean fraction of total squared pairwise distance contributed by each
    level 1..depth, averaged over all unique pairs (upper triangle, no
    self-pairs). `feature_fn` is `method_a_feature_vector` or
    `method_b_feature_vector`, applied after masking so Method B's
    16-line concatenation happens at the same point as everywhere else.

    Because level blocks are disjoint coordinates, the per-level squared
    distances sum exactly to the total squared distance (Pythagorean - the
    same orthogonality checked directly in tests/test_distances.py) -
    level 0 is excluded from that total since it's the constant 1.0 term for
    every image and so contributes exactly zero to any pairwise distance.
    """
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
    """Level-wise decomposition of the Phase 4 within/cross-digit sanity
    check, for both methods independently. Protocol matches
    `run_experiment.sanity_check_demo` exactly (same pool, same stream/
    signature construction, same independently-derived rescale factor `r`
    applied before any masking) so the `all` variant's numbers are directly
    comparable to the documented Phase 4 table.

    `r` is derived once per method (never shared, never re-derived per
    level variant) - the variants below differ only in which levels survive
    `mask_signature_levels`, not in `r` or anything upstream of masking.

    `pixel_order_seed`, if given, builds Method A's `make_pixel_order` with
    a seed independent of `seed` (which still controls the eval pool) - lets
    the pixel-order-sensitivity question ("Method A's pixel visiting order
    is a random sample, not a spatially coherent walk") be checked while
    holding the image sample fixed, isolating the effect of ordering from
    sample-to-sample variance. Defaults to `None`, which reuses `seed` for
    the pixel order too - the original, unchanged behaviour. Method B is
    unaffected either way - `make_reference_lines`'s line geometry doesn't
    depend on a seed (see streams.py's docstring).

    Returns r per method, each variant's within/cross/ratio for both
    methods, and each method's per-level mean fraction of total squared
    pairwise distance.
    """
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
