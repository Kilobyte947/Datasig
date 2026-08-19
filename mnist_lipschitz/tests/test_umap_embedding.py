import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mnist_lipschitz.data import load_mnist, get_dev_subset
from mnist_lipschitz.umap_embedding import (
    fit_umap_embedding,
    make_umap_embed_fn,
    make_umap_euclidean_distance_fn,
    knn_label_purity,
)

# UMAP requires n_neighbors < n_samples; kept small so the test suite doesn't
# pay full-scale UMAP fitting cost (numba JIT warmup dominates the first fit
# in a process either way, ~4s measured, ~0.3s for every fit after).
N_POINTS = 300
N_NEIGHBORS = 10


def _dev_subset(seed=0, n=N_POINTS):
    train = load_mnist(train=True)
    return get_dev_subset(train, n=n, seed=seed)


def test_fit_is_unsupervised_no_labels_parameter():
    """fit_umap_embedding must have no way to pass class labels into
    .fit() -- unsupervised only, by design (see module docstring:
    supervised UMAP is an open question deferred pending confirmation of
    which variant was intended, not implemented here)."""
    import inspect
    params = inspect.signature(fit_umap_embedding).parameters
    assert not any(name in params for name in ("y", "labels", "target"))


def test_output_shape_matches_n_components():
    dev = _dev_subset()
    for n_components in (2, 5):
        reducer = fit_umap_embedding(dev.x_flat, n_components=n_components,
                                      seed=0, n_neighbors=N_NEIGHBORS)
        embed_fn = make_umap_embed_fn(reducer)
        out = embed_fn(dev.x_flat)
        assert out.shape == (N_POINTS, n_components)
        assert out.dtype == torch.get_default_dtype()


def test_same_seed_reproduces_the_same_embedding():
    dev = _dev_subset()
    reducer_a = fit_umap_embedding(dev.x_flat, n_components=5, seed=0, n_neighbors=N_NEIGHBORS)
    reducer_b = fit_umap_embedding(dev.x_flat, n_components=5, seed=0, n_neighbors=N_NEIGHBORS)
    embed_fn_a, embed_fn_b = make_umap_embed_fn(reducer_a), make_umap_embed_fn(reducer_b)
    out_a, out_b = embed_fn_a(dev.x_flat), embed_fn_b(dev.x_flat)
    assert torch.allclose(out_a, out_b)


def test_different_seed_gives_a_different_embedding():
    dev = _dev_subset()
    reducer_a = fit_umap_embedding(dev.x_flat, n_components=5, seed=0, n_neighbors=N_NEIGHBORS)
    reducer_b = fit_umap_embedding(dev.x_flat, n_components=5, seed=1, n_neighbors=N_NEIGHBORS)
    embed_fn_a, embed_fn_b = make_umap_embed_fn(reducer_a), make_umap_embed_fn(reducer_b)
    out_a, out_b = embed_fn_a(dev.x_flat), embed_fn_b(dev.x_flat)
    assert not torch.allclose(out_a, out_b)


def test_transform_works_on_new_held_out_points_not_just_refitting():
    """The whole point of embed_fn/distance_fn reuse in this project is
    fit-once-transform-many -- checked directly against points the reducer
    was never fit on, not just re-embedding the fit data."""
    train_dev = _dev_subset(seed=0)
    reducer = fit_umap_embedding(train_dev.x_flat, n_components=5, seed=0, n_neighbors=N_NEIGHBORS)
    embed_fn = make_umap_embed_fn(reducer)

    test = load_mnist(train=False)
    held_out = test.x_flat[:20]  # disjoint from the fit data (train vs. test split)
    out = embed_fn(held_out)
    assert out.shape == (20, 5)
    assert torch.isfinite(out).all()

    # Reproducible on held-out points too, not just the fit data.
    out2 = embed_fn(held_out)
    assert torch.allclose(out, out2)


def test_euclidean_distance_fn_matches_direct_computation():
    dev = _dev_subset()
    reducer = fit_umap_embedding(dev.x_flat, n_components=5, seed=0, n_neighbors=N_NEIGHBORS)
    distance_fn = make_umap_euclidean_distance_fn(reducer)
    embed_fn = make_umap_embed_fn(reducer)

    x, y = dev.x_flat[:5], dev.x_flat[5:10]
    dist = distance_fn(x, y)
    expected = (embed_fn(x) - embed_fn(y)).norm(p=2, dim=-1)
    assert torch.allclose(dist, expected)
    assert dist.shape == (5,)


def test_transform_is_not_batch_composition_invariant_motivating_the_cache():
    """Documents the actual reason make_umap_euclidean_distance_fn's
    x_reference cache exists (not just speed): UMAP's .transform() gives
    DIFFERENT coordinates for the same row depending on what else is in
    the same call's batch -- deterministic for a fixed batch composition
    (test_same_seed_reproduces_the_same_embedding), but not invariant to
    it. Confirmed directly here on held-out points (not the reducer's own
    fit data -- transform() special-cases a query batch that exactly
    matches the fit data's size by returning the fit-time embedding
    directly, which would mask this effect)."""
    dev = _dev_subset()
    reducer = fit_umap_embedding(dev.x_flat, n_components=5, seed=0, n_neighbors=N_NEIGHBORS)
    embed_fn = make_umap_embed_fn(reducer)

    test = load_mnist(train=False)
    alone = embed_fn(test.x_flat[[0]])
    as_part_of_larger_batch = embed_fn(test.x_flat[:50])[0:1]
    assert not torch.allclose(alone, as_part_of_larger_batch)


def test_cached_distance_fn_gives_batch_composition_independent_results():
    """The property the x_reference cache actually guarantees: the same
    pair of points gets the same distance regardless of which other pairs
    happen to be gathered into the same distance_fn call alongside it --
    exactly what the previous test shows the *uncached* path does not
    provide. Uses held-out points as x_reference (matching real usage,
    e.g. run_ratio_distribution_analysis's x_pool) rather than the
    reducer's own fit data, so this doesn't rely on transform()'s
    fit-data-sized-batch shortcut -- the cache is built from one genuine
    transform() call on the reference pool, same as it would be in
    practice."""
    dev = _dev_subset()
    reducer = fit_umap_embedding(dev.x_flat, n_components=5, seed=0, n_neighbors=N_NEIGHBORS)

    test = load_mnist(train=False)
    reference_pool = test.x_flat[:250]
    cached_fn = make_umap_euclidean_distance_fn(reducer, x_reference=reference_pool)

    dist_alone = cached_fn(reference_pool[[0]], reference_pool[[10]])

    ii = torch.cat([torch.tensor([0]), torch.arange(20, 120)])
    jj = torch.cat([torch.tensor([10]), torch.arange(120, 220)])
    dist_in_large_batch = cached_fn(reference_pool[ii], reference_pool[jj])[0:1]

    assert torch.allclose(dist_alone, dist_in_large_batch)


def test_cached_distance_fn_falls_back_for_rows_outside_reference():
    """Rows not in x_reference must still work (via the uncached fallback),
    not raise or silently return garbage."""
    dev = _dev_subset()
    reducer = fit_umap_embedding(dev.x_flat, n_components=5, seed=0, n_neighbors=N_NEIGHBORS)

    small_reference = dev.x_flat[:50]
    cached_fn = make_umap_euclidean_distance_fn(reducer, x_reference=small_reference)

    test = load_mnist(train=False)
    outside_rows = test.x_flat[:5]  # not in small_reference at all
    dist = cached_fn(outside_rows, dev.x_flat[:5])
    assert dist.shape == (5,)
    assert torch.isfinite(dist).all()


def test_knn_label_purity_near_one_for_perfectly_separated_clusters():
    """Sanity check on the metric itself, independent of any real UMAP
    embedding: 10 tight, well-separated clusters (one per digit) should
    give purity close to 1.0."""
    torch.manual_seed(0)
    points, labels = [], []
    for digit in range(10):
        center = torch.full((2,), digit * 100.0)
        points.append(center + torch.randn(20, 2) * 0.01)
        labels.append(torch.full((20,), digit, dtype=torch.long))
    embedded = torch.cat(points)
    labels = torch.cat(labels)

    purity = knn_label_purity(embedded, labels, k=5)
    assert purity > 0.99


def test_knn_label_purity_near_chance_for_randomly_shuffled_labels():
    """Same well-separated clusters, but with labels randomly shuffled so
    they no longer correspond to cluster membership -- purity should drop
    to roughly the 10-class chance baseline (0.1), not stay near 1.0."""
    torch.manual_seed(0)
    points, true_labels = [], []
    for digit in range(10):
        center = torch.full((2,), digit * 100.0)
        points.append(center + torch.randn(20, 2) * 0.01)
        true_labels.append(torch.full((20,), digit, dtype=torch.long))
    embedded = torch.cat(points)

    generator = torch.Generator().manual_seed(0)
    shuffled_labels = torch.randint(0, 10, (200,), generator=generator)

    purity = knn_label_purity(embedded, shuffled_labels, k=5)
    assert purity < 0.3
