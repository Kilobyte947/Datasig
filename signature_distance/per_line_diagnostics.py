"""Per-line distance breakdown + same/different-digit AUC diagnostic for
Method B - a Phase 4 extension, additive only.

Tests the method's original "path by path" framing (compare reference line
i to reference line i directly, ask whether images are close on EACH path)
against the merged 496-dim concatenated distance already built and
documented (distances.py's Phase 3/4, method_b_adversarial_eval.py):
concatenating 16 lines' signatures before taking one Euclidean distance
risks diluting a strong signal on a few informative lines with noise from
lines that carry little information (e.g. lines crossing mostly
background). This module doesn't change or replace the merged pipeline -
it's a diagnostic run alongside it, using the same underlying per-line
signatures the merged distance is already built from.
"""

import torch

from signature_distance.data_pool import load_eval_pool
from signature_distance.distances import (
    auc_for_distance,
    choose_rescale_factor,
    method_b_feature_vector,
    per_line_distances,
    rescale_signature,
)
from signature_distance.signatures import signature_of_stream
from signature_distance.streams import line_stream, make_reference_lines

SIGNATURE_DEPTH = 4


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
    per-line distances (new - rescale, then stop before concatenation).

    For each of the resulting 17 distance measures, treats -distance as a
    same/different-digit classifier score and computes an ROC curve + AUC
    over all pairs, plus the FPR and distance threshold at `tpr_target`
    (default 90%) TPR. Returns a dict with per-measure results, ranked by
    AUC, plus each line's orientation/position for the secondary
    line-ranking question.
    """
    images, labels = load_eval_pool(n_per_class=n_per_class, seed=seed)
    lines = make_reference_lines()
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

    # Per-line distance matrices - new: one (N, N) matrix per line, using
    # per_line_distances via cdist per line (equivalent, batched over pairs).
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
