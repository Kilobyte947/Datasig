import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mnist_lipschitz.data import load_mnist, make_loader, get_dev_subset
from mnist_lipschitz.models import (
    LogisticRegressionModel,
    SmallMLP,
    SmallCNN,
    StrongCNN,
    train_classifier,
    margin_fn,
)
from mnist_lipschitz.augmentation import random_affine_augment

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


def test_strong_cnn_forward_shape():
    torch.manual_seed(0)
    model = StrongCNN()
    model.eval()
    x = torch.rand(5, 1, 28, 28)
    out = model(x)
    assert out.shape == (5, 10)


def test_strong_cnn_dropout_rates_are_configurable():
    torch.manual_seed(0)
    model = StrongCNN(dropout_conv=0.1, dropout_fc=0.2)
    dropout2d_ps = [m.p for m in model.features if isinstance(m, torch.nn.Dropout2d)]
    dropout_ps = [m.p for m in model.classifier if isinstance(m, torch.nn.Dropout)]
    assert dropout2d_ps == [0.1, 0.1]
    assert dropout_ps == [0.2]


def test_strong_cnn_trains_on_small_subset_with_augmentation_and_scheduler():
    """Fast smoke test for the new code paths (StrongCNN, augment_fn,
    lr_scheduler_fn) -- not a full accuracy check. The real ~99.3%+ target
    accuracy is validated by run_experiment.py's
    run_stronger_cnn_raw_mnist_experiment on the full STRONG_CNN_CONFIG
    (25 epochs, full 60k training set); training that config here would
    make the test suite far too slow."""
    train = load_mnist(train=True)
    test = load_mnist(train=False)
    train_dev = get_dev_subset(train, 2000, seed=0)
    test_dev = get_dev_subset(test, 500, seed=1)
    train_img = make_loader(train_dev.x_image, train_dev.y, batch_size=128, shuffle=True, seed=0)
    test_img = make_loader(test_dev.x_image, test_dev.y, batch_size=500, shuffle=False)

    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    augment_fn = lambda x: random_affine_augment(x, degrees=10.0, translate=0.1, generator=generator)
    lr_scheduler_fn = lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3)

    model, train_acc, test_acc = train_classifier(
        StrongCNN(), train_img, test_img, epochs=3, lr=1e-3, verbose=False,
        augment_fn=augment_fn, lr_scheduler_fn=lr_scheduler_fn)

    assert 0.0 <= train_acc <= 1.0
    assert 0.0 <= test_acc <= 1.0
    assert test_acc > 0.5  # well above chance (1/10) after 3 epochs -- sanity that training actually worked


def test_train_classifier_default_augment_and_scheduler_are_none_and_unchanged():
    """augment_fn/lr_scheduler_fn default to None -- confirms passing them
    explicitly as None reproduces the exact pre-existing train_classifier
    behavior (same convention as distance_fn/embed_fn elsewhere in this
    project: leaving a new pluggable parameter unset must not change
    existing behavior)."""
    train = load_mnist(train=True)
    test = load_mnist(train=False)
    train_dev = get_dev_subset(train, 500, seed=0)
    test_dev = get_dev_subset(test, 200, seed=1)

    # Fresh loaders per run: make_loader's shuffle generator advances its
    # internal state as batches are drawn, so reusing one DataLoader across
    # both training runs would make the second run continue from wherever
    # the first left off instead of reproducing the same batch order.
    def _fresh_loaders():
        train_flat = make_loader(train_dev.x_flat, train_dev.y, batch_size=128, shuffle=True, seed=0)
        test_flat = make_loader(test_dev.x_flat, test_dev.y, batch_size=200, shuffle=False)
        return train_flat, test_flat

    torch.manual_seed(0)
    train_flat_a, test_flat_a = _fresh_loaders()
    _, train_acc_a, test_acc_a = train_classifier(
        LogisticRegressionModel(), train_flat_a, test_flat_a, epochs=2, lr=1e-3, verbose=False)

    torch.manual_seed(0)
    train_flat_b, test_flat_b = _fresh_loaders()
    _, train_acc_b, test_acc_b = train_classifier(
        LogisticRegressionModel(), train_flat_b, test_flat_b, epochs=2, lr=1e-3, verbose=False,
        augment_fn=None, lr_scheduler_fn=None)

    assert train_acc_a == train_acc_b
    assert test_acc_a == test_acc_b


def test_lr_scheduler_fn_steps_once_per_epoch_and_lowers_lr():
    torch.manual_seed(0)
    train = load_mnist(train=True)
    test = load_mnist(train=False)
    train_dev = get_dev_subset(train, 200, seed=0)
    test_dev = get_dev_subset(test, 100, seed=1)
    train_flat = make_loader(train_dev.x_flat, train_dev.y, batch_size=64, shuffle=True, seed=0)
    test_flat = make_loader(test_dev.x_flat, test_dev.y, batch_size=100, shuffle=False)

    seen_lrs = []

    class _RecordingStepLR:
        def __init__(self, optimizer):
            self.optimizer = optimizer
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        def step(self):
            self.scheduler.step()
            seen_lrs.append(self.optimizer.param_groups[0]["lr"])

    train_classifier(
        LogisticRegressionModel(), train_flat, test_flat, epochs=3, lr=1e-3, verbose=False,
        lr_scheduler_fn=_RecordingStepLR)

    assert len(seen_lrs) == 3
    assert seen_lrs == sorted(seen_lrs, reverse=True)
    assert seen_lrs[-1] < 1e-3
