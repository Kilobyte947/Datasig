"""UMAP as an alternative distance metric for the Lipschitz ratio-distribution analysis: fits a
learned low-dimensional embedding, then measures plain Euclidean distance in that embedded space.

Unsupervised only — no labels are passed to UMAP's fit, so the embedding reflects unlabelled
geometric structure rather than being shaped by class labels directly.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
import torch
import umap
from mnist_example.distance import knn_label_purity
from mnist_example.estimators import euclidean_distance_fn
torch.set_default_dtype(torch.float64)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SEED = 0


def fit_umap_embedding(x_flat, n_components=5, seed=SEED, n_neighbors=15, min_dist=0.1, verbose=False):
    """Fits an unsupervised UMAP embedding on x_flat (N, 784), reducing to n_components dimensions.
    Intended for a modest subset, not the full training set. seed fixes random_state, giving 
    bit-identical embeddings across repeated fits. Returns the fitted UMAP object."""
    x_np = x_flat.detach().cpu().numpy()
    reducer = umap.UMAP(n_components=n_components, random_state=seed,
                         n_neighbors=n_neighbors, min_dist=min_dist, verbose=verbose)
    reducer.fit(x_np)
    return reducer


def make_umap_embed_fn(reducer):
    """Wraps a fitted UMAP reducer as an embed_fn: maps raw (N, 784) pixel vectors to (N, n_components)
    embedded coordinates. Works on any points, not just ones the reducer was fit on."""
    def embed_fn(x_flat):
        x_np = x_flat.detach().cpu().numpy()
        embedded = reducer.transform(x_np)
        return torch.as_tensor(embedded, dtype=torch.get_default_dtype())
    return embed_fn


def make_umap_euclidean_distance_fn(reducer, x_reference=None):
    """Builds a distance_fn(x, y) that measures Euclidean distance in a fitted UMAP's embedded space,
    for use with the ratio-distribution pipeline.
    
    UMAP's transform is not invariant to batch composition — the same point embeds to different
    coordinates depending on what else is in its batch. This function embeds x_reference once and
    caches by exact row value, so the same pair of points always gets the same distance regardless of
    which batch they're later gathered into. Pass the full pool the caller will draw pairs from as
    x_reference. Falls back to an uncached transform for any row not found in the cache."""
    embed_fn = make_umap_embed_fn(reducer)
    cache = {}
    if x_reference is not None:
        embedded_ref = embed_fn(x_reference)
        for row, emb_row in zip(x_reference, embedded_ref):
            cache[row.numpy().tobytes()] = emb_row

    def _embed_with_cache(x):
        keys = [row.numpy().tobytes() for row in x]
        if cache and all(k in cache for k in keys):
            return torch.stack([cache[k] for k in keys])
        return embed_fn(x)

    def distance_fn(x, y):
        return euclidean_distance_fn(_embed_with_cache(x), _embed_with_cache(y))

    return distance_fn