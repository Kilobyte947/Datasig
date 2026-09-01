"""Level-wise decomposition of Method A/B signature distances - a read-only
diagnostic, additive alongside the existing Phase 3/4 pipeline (see
`level_decomposition.md` for the full spec/rationale).

Answers one question: is the within- vs. cross-digit signal in Method A's
(and Method B's) signature distance carried entirely by the level-1 terms,
or do the higher-order levels (2..depth) contribute? `signature_of_stream`'s
output always has index 0 as the constant term `1.0` and indices `1:3` as
the path's exact net displacement (`stream[-1] - stream[0]`) regardless of
depth - so if the distance turns out to be level-1 dominated, the signature
machinery downstream of that single scalar difference is doing no work.

Does not change any existing distance/stream/signature behaviour - reuses
`distances.py`/`streams.py`/`signatures.py` unmodified, and does not merge
Method A and Method B or share their independently-derived rescale factor
`r` (same convention as `distances.py`).
"""

import torch

from signature_distance.data_pool import load_eval_pool
from signature_distance.distances import (
    choose_rescale_factor,
    method_a_feature_vector,
    method_b_feature_vector,
    rescale_signature,
    within_vs_cross_digit_distance,
)
from signature_distance.signatures import signature_of_stream
from signature_distance.streams import (
    line_stream,
    make_pixel_order,
    make_reference_lines,
    patch_sv_stream,
)

SIGNATURE_DEPTH = 4


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
    same orthogonality checked directly in test_level_decomposition.py) -
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
    comparable to the documented Phase 4 table - see Gate 2 in
    `level_decomposition.md`.

    `r` is derived once per method (never shared, never re-derived per
    level variant) - the variants below differ only in which levels survive
    `mask_signature_levels`, not in `r` or anything upstream of masking.

    `pixel_order_seed`, if given, builds Method A's `make_pixel_order` with
    a seed independent of `seed` (which still controls the eval pool) - lets
    the pixel-order-sensitivity question flagged in README.md ("Method A's
    pixel visiting order is a random sample, not a spatially coherent walk")
    be checked while holding the image sample fixed, isolating the effect of
    ordering from sample-to-sample variance. Defaults to `None`, which reuses
    `seed` for the pixel order too - the original, unchanged behaviour (Gate
    2's reproduction check depends on this default). Method B is unaffected
    either way - `make_reference_lines`'s line geometry doesn't depend on a
    seed (see streams.py's docstring).

    Returns r per method, each variant's within/cross/ratio for both
    methods, and each method's per-level mean fraction of total squared
    pairwise distance.
    """
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
