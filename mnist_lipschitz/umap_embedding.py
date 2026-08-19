"""UMAP as an alternative distance metric for the Lipschitz ratio-distribution
analysis: instead of a hand-built distance formula -- plain Euclidean, a
ridge-regularized Mahalanobis distance fit from the pixel covariance
(`distance.py`), or a fixed pixel-space feature map (`embeddings.py`) -- fit
a *learned* low-dimensional embedding that explicitly optimizes for
preserving local neighborhood structure, then measure plain Euclidean
distance in that embedded space. Follows the same `embed_fn` convention
`embeddings.py`'s `elementwise_embedding`/`local_patch_cross_terms` use, so
a fitted UMAP embedding can be dropped into the same
`run_experiment.py::run_ratio_distribution_analysis` pipeline used for
every other metric comparison in this project, for direct numeric
comparability.

**Unsupervised UMAP only, deliberately.** UMAP also supports a *supervised*
mode (`umap.UMAP(...).fit(x, y)`, using class labels to shape the embedding
directly), which would very plausibly produce cleaner per-digit clustering
almost by construction -- but that changes the question being asked: an
embedding built *from* the labels is a different (and less interesting,
for a distance-metric-validity question) thing to then measure margin
sensitivity against than one built purely from unlabeled geometric
structure, the same way a supervised distance metric would need separate
justification anywhere else in this project. Which variant is actually
wanted here is an open question flagged for Nick/Terry, not decided in
this module -- `fit_umap_embedding` below never passes labels to `.fit()`,
and nothing in this module implements the supervised path. See
`notebook_umap.ipynb`'s intro markdown for the same note in context.

**Why plain Euclidean on the embedded coordinates, not a further
Mahalanobis layer.** Every other embedding in this project
(`elementwise_embedding`, `local_patch_cross_terms`) is a fixed, purely
geometric feature map with no notion of "closeness" baked in on its own --
Mahalanobis reweighting on top of it is what does the actual metric
learning. UMAP is different: its whole fitting objective *is* to place
points so that Euclidean distance in the embedded space already reflects
local neighborhood similarity. Layering a second, covariance-based
reweighting on top would re-derive (and could partially undo) structure
UMAP already optimized for, not add anything -- so `distance.py`'s
Mahalanobis machinery is deliberately not threaded through here.

**A real segfault, and the workaround.** A plain `umap.UMAP(...).fit()`
call crashed outright (exit code 139) when run after this project's other
torch-heavy modules were already imported -- numba (UMAP's JIT backend)
and torch's own OpenMP thread pool conflict the same way this project
already documented for sklearn's `NearestNeighbors`
(`run_experiment.py::run_ratio_distribution_analysis`). Confirmed directly:
forcing single-threaded execution via `OMP_NUM_THREADS`/`NUMBA_NUM_THREADS`
before UMAP is used removes the crash entirely and costs nothing extra --
passing a fixed `random_state` already forces UMAP itself down to
single-threaded internally (see its own `UserWarning`), so this doesn't
trade away any real parallelism. Set once at import time here (every
UMAP call in this module hits the same risk), rather than scoped around a
single call the way the sklearn workaround is -- matches this module's
narrower, UMAP-only surface area.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
import torch
import umap
from sklearn.neighbors import NearestNeighbors

from mnist_lipschitz.estimators import euclidean_distance_fn

torch.set_default_dtype(torch.float64)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SEED = 0


def fit_umap_embedding(x_flat, n_components=5, seed=SEED, n_neighbors=15, min_dist=0.1, verbose=False):
    """Fits an **unsupervised** UMAP embedding on `x_flat` (N, 784) raw
    pixel vectors, reducing to `n_components` dimensions -- no labels are
    passed to `.fit()` (see module docstring for why).

    Intended to be fit on a modest subset (a few thousand points via
    `data.py::get_dev_subset`), not the full 60k training set -- UMAP
    fitting cost scales with dataset size, and this project's convention
    elsewhere (`run_embedding_degree_sweep`'s epsilon-selection pool, this
    module's own segfault workaround above) is to keep exploratory
    new-metric work on a manageable subset rather than discovering a
    runtime/memory ceiling mid-run. `n_neighbors`/`min_dist` are UMAP's own
    standard knobs (local neighborhood size and how tightly points are
    allowed to pack in the embedding); left at UMAP's own defaults here,
    not tuned for this dataset.

    `seed` fixes `random_state`, which also forces UMAP down to
    single-threaded internally (a `UserWarning` UMAP prints itself) --
    confirmed directly to give bit-identical embeddings across repeated
    fits with the same seed and data (`tests/test_umap_embedding.py`).

    **Fit/evaluate separation is the caller's responsibility, and matters**:
    this function fits on exactly whatever `x_flat` it's given -- nothing
    here prevents a caller from fitting on the same points a downstream
    ratio-distribution analysis then evaluates, which would let the
    embedding "see" the evaluation points during its own optimization
    (UMAP's fit directly shapes where each fit-time point lands) before
    supposedly-held-out distances are measured on them. Verified directly
    (not just assumed) that `notebook_umap.ipynb` avoids this: `dev`, the
    subset passed here, is drawn from `data.py::load_mnist(train=True)`;
    every downstream evaluation -- the Step 2 validation subset, the Step
    3 ratio-distribution analysis, and the `min_dist` sweep -- operates on
    `load_mnist(train=False)`-derived subsets exclusively, reached only via
    `make_umap_embed_fn`/`make_umap_euclidean_distance_fn`'s `.transform()`
    calls, never by fitting a second reducer on evaluation data. Train and
    test are MNIST's own disjoint splits, so this is genuine held-out
    evaluation, not a different subset of the same pool. A future caller
    fitting this on data it also means to evaluate distances on downstream
    would reintroduce exactly this leakage risk -- keep the fit subset and
    every evaluation subset on opposite sides of the train/test split.

    Returns the fitted `umap.UMAP` object -- has `.transform()` for
    embedding new/held-out points without refitting (see
    `make_umap_embed_fn` below).
    """
    x_np = x_flat.detach().cpu().numpy()
    reducer = umap.UMAP(n_components=n_components, random_state=seed,
                         n_neighbors=n_neighbors, min_dist=min_dist, verbose=verbose)
    reducer.fit(x_np)
    return reducer


def make_umap_embed_fn(reducer):
    """Wraps a fitted UMAP `reducer` as this project's `embed_fn`
    convention (see `embeddings.py`): `embed_fn(x_flat)` maps raw (N, 784)
    pixel vectors to (N, `reducer.n_components`) embedded coordinates, as a
    plain tensor -- usable anywhere this project threads an `embed_fn`
    through (though see `make_umap_euclidean_distance_fn` below for the
    metric actually used with `run_ratio_distribution_analysis`, which
    wraps this with a lookup cache rather than calling it directly on
    every batch).

    Uses `.transform()`, not `.fit_transform()` -- works on any points,
    including ones the reducer was never fit on (checked directly in
    `tests/test_umap_embedding.py`, not just assumed), the same
    fit-once/reuse-on-new-points contract every other fitted object in
    this project follows (a trained model, a fitted precision matrix).
    """
    def embed_fn(x_flat):
        x_np = x_flat.detach().cpu().numpy()
        embedded = reducer.transform(x_np)
        return torch.as_tensor(embedded, dtype=torch.get_default_dtype())
    return embed_fn


def make_umap_euclidean_distance_fn(reducer, x_reference=None):
    """Builds a `distance_fn(x, y) = euclidean_distance_fn(embed_fn(x),
    embed_fn(y))` usable directly with
    `run_experiment.py::run_ratio_distribution_analysis` and the rest of
    `estimators.py`'s pairwise machinery.

    **This cache is not just a performance optimization -- it's what makes
    the resulting "distance" a well-defined function of two points at
    all.** UMAP's `.transform()` is deterministic for a *fixed batch
    composition*, but **not invariant to batch composition**: the same row
    embedded alone vs. as part of a larger batch gives measurably
    different coordinates (confirmed directly,
    `tests/test_umap_embedding.py` -- not a small numerical artifact,
    differences of similar magnitude to the embedding's own coordinate
    range). `run_ratio_distribution_analysis` calls `distance_fn` with
    different, differently-sized, differently-composed *gathered* batches
    of pairs at different points in its pipeline (the near-neighbor pairs,
    then separately the ~500k all-pairs) -- calling `.transform()` fresh
    inside `distance_fn` on each of those ad-hoc batches would silently
    make "the distance between point A and point B" depend on which
    *other* points happened to be batched alongside them in that
    particular call, not just on A and B themselves. `x_reference`, given
    once up front, is embedded in a single canonical call and cached by
    exact row value (`.tobytes()` as the key -- exact, not approximate,
    since gathered rows are bit-identical copies of the original tensor,
    never recomputed); every subsequent `distance_fn` call then reuses
    that one fixed embedding regardless of batch shape, so the same pair
    of points always gets the same distance. Pass the *pool* the caller
    will draw its subset from (e.g. `test.x_flat`, matching
    `run_ratio_distribution_analysis`'s own `x_pool` argument) so every
    row it ends up gathering is covered. Falls back to a direct
    (uncached) `.transform()` call for any row not found in the cache --
    consistent for repeat queries of that exact fallback batch, but not
    across different ad-hoc batches, for the reason above -- so
    `x_reference` should cover the caller's actual pool whenever the
    batch-composition-independence property matters, which it does for
    `run_ratio_distribution_analysis`.
    """
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


def knn_label_purity(embedded, labels, k=5):
    """Quantitative supplement to the visual per-digit-clustering check
    (`plots.py::plot_umap_embedding_scatter`) -- this project's convention
    elsewhere (e.g. the epsilon-selection stability check, checkpoint-gating
    generally, see `CLAUDE.md`) is to back a visual/qualitative read with a
    number, not rely on either alone.

    For each point, the fraction of its `k` nearest neighbors (Euclidean,
    in the given embedded space, excluding itself) that share its true
    label, averaged over every point. A well-clustered-by-digit embedding
    should score well above the 10-class chance baseline (0.1, if every
    class were equally likely and neighbors uninformative); a
    scattered/mixed embedding should sit close to it.

    `embedded`: (N, d) array/tensor of embedded coordinates. `labels`:
    (N,) integer true labels, same order.
    """
    embedded_np = embedded.detach().cpu().numpy() if hasattr(embedded, "detach") else np.asarray(embedded)
    labels_np = labels.detach().cpu().numpy() if hasattr(labels, "detach") else np.asarray(labels)

    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(embedded_np)
    _, neighbor_idx = nn.kneighbors(embedded_np)
    neighbor_idx = neighbor_idx[:, 1:]  # drop each point's own (zero-distance) self-match

    matches = (labels_np[neighbor_idx] == labels_np[:, None]).mean()
    return float(matches)
