import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mnist_lipschitz.data import load_mnist, make_loader
from mnist_lipschitz.models import (
    LogisticRegressionModel,
    SmallMLP,
    SmallCNN,
    train_classifier,
    margin_fn,
)

# Accuracy thresholds are deliberately set with headroom below what was
# actually observed during development (LR ~92-93%, MLP ~97-98%,
# CNN ~98.6%), not at the ceiling -- the point of this test is "did
# training actually work," not "hit an exact number," so it shouldn't be
# flaky against normal seed-to-seed variance.
LR_THRESHOLD = 0.90
MLP_THRESHOLD = 0.96
CNN_THRESHOLD = 0.98


def _loaders():
    train = load_mnist(train=True)
    test = load_mnist(train=False)
    train_flat = make_loader(train.x_flat, train.y, batch_size=256, shuffle=True, seed=0)
    test_flat = make_loader(test.x_flat, test.y, batch_size=1000, shuffle=False)
    train_img = make_loader(train.x_image, train.y, batch_size=256, shuffle=True, seed=0)
    test_img = make_loader(test.x_image, test.y, batch_size=1000, shuffle=False)
    return train_flat, test_flat, train_img, test_img


def test_logistic_regression_trains_above_threshold():
    train_flat, test_flat, _, _ = _loaders()
    torch.manual_seed(0)
    model = LogisticRegressionModel()
    model, train_acc, test_acc = train_classifier(model, train_flat, test_flat, epochs=15, lr=1e-3, verbose=False)
    assert test_acc > LR_THRESHOLD, f"logistic regression test acc {test_acc:.4f} <= {LR_THRESHOLD}"


def test_mlp_trains_above_threshold():
    train_flat, test_flat, _, _ = _loaders()
    torch.manual_seed(0)
    model = SmallMLP(hidden_sizes=(128,))
    model, train_acc, test_acc = train_classifier(model, train_flat, test_flat, epochs=15, lr=1e-3, verbose=False)
    assert test_acc > MLP_THRESHOLD, f"MLP test acc {test_acc:.4f} <= {MLP_THRESHOLD}"


def test_cnn_trains_above_threshold():
    _, _, train_img, test_img = _loaders()
    torch.manual_seed(0)
    model = SmallCNN()
    model, train_acc, test_acc = train_classifier(model, train_img, test_img, epochs=8, lr=1e-3, verbose=False)
    assert test_acc > CNN_THRESHOLD, f"CNN test acc {test_acc:.4f} <= {CNN_THRESHOLD}"


def test_margin_fn_matches_manual_computation():
    torch.manual_seed(0)
    model = LogisticRegressionModel()
    x = torch.randn(16, 784)
    y = torch.randint(0, 10, (16,))

    margins = margin_fn(model, x, y)
    assert margins.shape == (16,)

    logits = model(x).detach()
    for i in range(16):
        true_logit = logits[i, y[i]].item()
        other_logits = torch.cat([logits[i, :y[i]], logits[i, y[i] + 1:]])
        expected = true_logit - other_logits.max().item()
        assert abs(margins[i].item() - expected) < 1e-9


def test_margin_fn_is_differentiable_wrt_x():
    torch.manual_seed(0)
    model = LogisticRegressionModel()
    x = torch.randn(4, 784, requires_grad=True)
    y = torch.randint(0, 10, (4,))
    margins = margin_fn(model, x, y)
    margins.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()


def test_cnn_forward_matches_head_of_extractor():
    """The extractor/head split must be a pure refactor: forward(x) has to
    equal head(extractor(x)) exactly, for every input, not just on average."""
    torch.manual_seed(0)
    model = SmallCNN()
    x = torch.rand(8, 1, 28, 28)

    direct = model(x)
    composed = model.head(model.extractor(x))
    assert torch.equal(direct, composed)


def test_cnn_extractor_and_head_shapes():
    torch.manual_seed(0)
    model = SmallCNN()
    x = torch.rand(5, 1, 28, 28)

    features = model.extractor(x)
    assert features.shape == (5, 32 * 7 * 7)

    logits = model.head(features)
    assert logits.shape == (5, 10)


def test_cnn_conv_channels_parameter_changes_width_but_preserves_default():
    torch.manual_seed(0)
    default_model = SmallCNN()
    assert default_model.head.in_features == 32 * 7 * 7  # unchanged default behavior

    torch.manual_seed(0)
    narrow_model = SmallCNN(conv_channels=(4, 8))
    assert narrow_model.head.in_features == 8 * 7 * 7
    x = torch.rand(3, 1, 28, 28)
    out = narrow_model(x)
    assert out.shape == (3, 10)
