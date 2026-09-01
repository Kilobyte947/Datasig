"""Distance functions over signature vectors (Phase 3) and the within/cross
-digit sanity check (Phase 4) - see PLAN.md. Shared by both methods; Method A
and Method B are always kept as two separate distance functions, never
combined into one metric (per PLAN.md's "two candidate distance functions,
not one" note).
"""

import torch
from sklearn.metrics import roc_auc_score, roc_curve


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
    before (see Method_B.md's "no cross-line concatenation" rule).

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
    """
    return (sig1 - sig2).norm(dim=-1)


def within_vs_cross_digit_distance(vectors: torch.Tensor, labels: torch.Tensor) -> dict:
    """Cheap, label-based sanity check (PLAN.md Phase 4): mean pairwise
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
    return {
        "within_digit_mean": within_mean,
        "cross_digit_mean": cross_mean,
        "ratio_cross_over_within": cross_mean / within_mean,
    }


def auc_for_distance(same, dist_values, tpr_target: float = None) -> dict:
    """Same/different-label AUC for one distance measure, treating
    `-distance` as a same-label classifier score. Shared by
    per_line_diagnostics.py (per-line vs. merged AUC ranking) and
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
