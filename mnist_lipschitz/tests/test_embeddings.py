import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mnist_lipschitz.embeddings import elementwise_embedding
from mnist_lipschitz.distance import svd_ridge_precision, make_mahalanobis_distance_fn


def test_degree_1_is_identity():
    torch.manual_seed(0)
    x = torch.randn(10, 784)
    embedded = elementwise_embedding(x, degree=1)
    assert embedded.shape == (10, 784)
    assert torch.equal(embedded, x)


def test_elementwise_no_cross_terms_small_example():
    """Hand-checkable example: 2 points, 3 pixels, degree 3. Confirms the exact block layout
    [x, x**2, x**3] concatenated along the feature axis, with no cross-pixel product terms."""
    x = torch.tensor([[1.0, 2.0, 3.0],
                       [-1.0, 0.5, 4.0]])
    embedded = elementwise_embedding(x, degree=3)

    assert embedded.shape == (2, 9)
    expected = torch.cat([x, x ** 2, x ** 3], dim=-1)
    assert torch.equal(embedded, expected)

    # explicit spot checks, not just self-referential against x**k
    assert torch.equal(embedded[0], torch.tensor([1.0, 2.0, 3.0, 1.0, 4.0, 9.0, 1.0, 8.0, 27.0]))
    assert torch.equal(embedded[1], torch.tensor([-1.0, 0.5, 4.0, 1.0, 0.25, 16.0, -1.0, 0.125, 64.0]))


def test_degree_generalizes_shape():
    torch.manual_seed(1)
    x = torch.randn(5, 784)
    for degree in (1, 2, 3):
        embedded = elementwise_embedding(x, degree)
        assert embedded.shape == (5, 784 * degree)


def test_embedded_mahalanobis_matches_raw_pixel_at_degree_1():
    """Checkpoint: since elementwise_embedding(x, 1) is the identity, a Mahalanobis distance_fn built with
    embed_fn=elementwise_embedding(., 1) must exactly reproduce the existing raw-pixel distance_fn (embed_fn=None)
    on the same precision matrix -- confirms make_mahalanobis_distance_fn's new embed_fn path is correct before
    trusting it at degree > 1, where there's no such direct raw-pixel comparison available."""
    torch.manual_seed(2)
    x_flat = torch.randn(200, 20)  # small synthetic "pixel" matrix, N >= d
    precision = svd_ridge_precision(x_flat, epsilon=0.1)

    embed_fn = lambda x: elementwise_embedding(x, degree=1)
    raw_distance_fn = make_mahalanobis_distance_fn(precision)
    embedded_distance_fn = make_mahalanobis_distance_fn(precision, embed_fn=embed_fn)

    x = torch.randn(15, 20)
    y = torch.randn(15, 20)
    assert torch.allclose(raw_distance_fn(x, y), embedded_distance_fn(x, y))


def test_make_mahalanobis_distance_fn_default_embed_fn_none_unchanged():
    """Regression check: embed_fn defaulting to None must leave the pre-existing raw-pixel behavior exactly
    unchanged (project convention -- see CLAUDE.md)."""
    torch.manual_seed(3)
    x_flat = torch.randn(50, 10)
    precision = svd_ridge_precision(x_flat, epsilon=1.0)
    distance_fn = make_mahalanobis_distance_fn(precision)

    x = torch.randn(6, 10)
    y = torch.randn(6, 10)
    diff = x - y
    expected = torch.einsum("ni,ij,nj->n", diff, precision, diff).clamp_min(0.0).sqrt()
    assert torch.allclose(distance_fn(x, y), expected)
