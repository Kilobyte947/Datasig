# `local_patch_cross_terms` under plain Euclidean distance (follow-up)

Confirmatory follow-up to the `local_patch_cross_terms` negative result in `README.md`
("A structurally different embedding: local spatial cross-terms (negative result)"), where
epsilon selection under Mahalanobis distance failed categorically at every regularization level
tested. That failure leaves open whether the embedding itself carries no useful locality signal,
or whether it was specifically the covariance-based reweighting that was unstable. This closes
that question: run under plain Euclidean distance, with no covariance/precision/epsilon involved
at all, to see whether the locality signal survives on its own.

This was a one-off confirmatory check, not the start of a new sweep — no new permanent driver
function was added to `run_experiment.py`. It was run as a standalone scratch script; a
reproducible version of that script is included at the bottom of this file.

## Setup

- **Model**: a freshly trained `LogisticRegressionModel` (15 epochs, Adam, `lr=1e-3`, seed=0) —
  `train_acc=92.87%`, `test_acc=92.70%`, matching this README's existing logistic-regression
  baseline exactly (same seed/data), confirming this is an apples-to-apples reference model.
- **Embedding**: `embeddings.py::local_patch_cross_terms`, wrapped as
  `embed_fn(x) = local_patch_cross_terms(x.reshape(*x.shape[:-1], 28, 28))` — maps 784-d raw
  pixels to 3920-d (784 raw pixels + 4×784 spatial cross-terms).
- **Distance**: `euclidean_distance_fn` applied directly to the embedded feature vectors —
  `dist(x, y) = euclidean_distance_fn(embed_fn(x), embed_fn(y))`. No covariance, precision matrix,
  or epsilon anywhere in this run.
- **Analysis**: `run_experiment.py::run_ratio_distribution_analysis`, unmodified, on a 1000-point
  class-stratified subset of the MNIST test set, `k=5` near-neighbors (5000 near-neighbor pairs,
  computed exhaustively — no cap needed, cheap at this scale).
- **All-pairs cap**: `max_pairs=20000` (a random subsample of the ~499,500 total pairs among 1000
  points, via `pairwise_lipschitz_all`'s existing `max_pairs` safety valve). `local_patch_cross_terms`
  is 3920-d — 5x raw pixels' 784-d — and this README already documents the same embedding
  exhausting this machine's memory at the full pair count under Mahalanobis distance (forcing a
  cutback to `n=300` there). Peak RSS for this run stayed under 1.5GB at `max_pairs=20000`, versus
  an estimated ~78GB for the uncapped 499,500-pair computation at this embedding's dimensionality —
  comfortably safe.
- **Seed**: 0 throughout, matching this package's convention.

## Result

| Setup | All-pairs mean | Near-neighbor mean | Near/all | Direction |
|---|---|---|---|---|
| Raw-pixel, Euclidean, logistic regression (README) | 0.310 | 0.349 | 1.13 | near-neighbor **above** (+13%) |
| `local_patch_cross_terms`, Mahalanobis (epsilon=1 fallback, n=300; README) | 0.288 | 0.246 | 0.854 | near-neighbor **below** (−14.6%) |
| **`local_patch_cross_terms`, plain Euclidean (this run, n=1000, max_pairs=20000)** | **0.165** | **0.171** | **1.039** | near-neighbor **above** (+3.9%) |

(`all_pairs_max=1.299`, `near_neighbor_max=1.830`, `n_all_pairs=19,978`, `n_near_neighbor_pairs=5,000`.)

**Conclusion**: the locality signal does show up under plain Euclidean distance on the cross-term
features on its own — near-neighbor pairs sit above the all-pairs mean, same direction as raw-pixel
Euclidean — so the earlier epsilon-selection instability was a Mahalanobis/covariance-reweighting
problem, not evidence the embedding itself is locally uninformative. The signal is markedly weaker
than raw pixels' own Euclidean signal (+3.9% vs. +13%), so this embedding doesn't add value over
raw pixels here. And even the heavily-damped `epsilon=1` Mahalanobis fallback — already documented
as behaving considerably more like Euclidean distance than a genuine data-fit metric — still
reverses the near/all sign relative to true Euclidean distance on the same features (−14.6% vs.
+3.9%), generalizing this project's "self-referential covariance reweighting shrinks/reverses the
signal" finding beyond just full-strength or numerically unstable Mahalanobis: even a weak,
near-Euclidean-strength one flips it.

## Methodological asterisk

The Mahalanobis row and this Euclidean row cap for memory in two *different* ways, so their
all-pairs numbers are directionally but not bit-for-bit comparable:

- The Mahalanobis `epsilon=1` fallback capped by **point count** — a smaller pool (n=300, ~44,850
  pairs), computed exhaustively over all pairs in that smaller pool.
- This Euclidean run capped by **pair count** — the full requested 1000-point pool, but only a
  20,000-pair random subsample of its ~499,500 total pairs.

The near-neighbor side is on firmer footing across the two (5,000 pairs here vs. ~1,500 in the
Mahalanobis fallback, both drawn from their respective full point pools), but the all-pairs means
should be read as "same ballpark, same sign" rather than a precise like-for-like comparison.

## Reproducing this

Not wired into `run_experiment.py`/`main()` — this is the scratch script, included here so the
result is reproducible without depending on a session-specific temp path:

```python
import torch

from mnist_lipschitz.data import load_mnist, make_loader
from mnist_lipschitz.models import LogisticRegressionModel, train_classifier
from mnist_lipschitz.estimators import euclidean_distance_fn
from mnist_lipschitz.embeddings import local_patch_cross_terms
from mnist_lipschitz.run_experiment import run_ratio_distribution_analysis, SEED

MAX_PAIRS = 20000  # see "Setup" above for the memory reasoning

torch.manual_seed(SEED)
train = load_mnist(train=True)
test = load_mnist(train=False)
train_flat = make_loader(train.x_flat, train.y, batch_size=256, shuffle=True, seed=SEED)
test_flat = make_loader(test.x_flat, test.y, batch_size=1000, shuffle=False)

lr_model, lr_train_acc, lr_test_acc = train_classifier(
    LogisticRegressionModel(), train_flat, test_flat, epochs=15, lr=1e-3, verbose=True)

def embed_fn(x):
    return local_patch_cross_terms(x.reshape(*x.shape[:-1], 28, 28))

def embedded_euclidean_distance_fn(x, y):
    return euclidean_distance_fn(embed_fn(x), embed_fn(y))

result = run_ratio_distribution_analysis(
    lr_model, "logistic_regression", "local_patch_cross_terms_euclidean",
    test.x_flat, test.y, embedded_euclidean_distance_fn,
    n_points=1000, k_neighbors=5, max_pairs=MAX_PAIRS, seed=SEED, verbose=True)

print(result["summary"])
```
