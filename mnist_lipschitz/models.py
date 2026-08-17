"""The three classifiers under study, their training loop, and the margin
function the Lipschitz estimators are actually applied to.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)

# float64 is required throughout this project (see toy_lipschitz's convention
# of avoiding float32 noise in true-vs-estimate comparisons), and PyTorch's
# MPS backend does not support float64 -- so despite Apple-Silicon MPS being
# available, this always resolves to CPU on this machine. CUDA (when present,
# e.g. on Colab) does support float64 and will be used automatically.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LogisticRegressionModel(nn.Module):
    """Single linear layer, 784 -> 10. Margin is exactly linear in x, which
    makes this the one model with a closed-form Lipschitz constant
    (||w_true - w_runner_up||_2 for a fixed class pair) -- see
    estimators.py's checkpoint test.
    """

    def __init__(self, input_dim=784, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


class SmallMLP(nn.Module):
    """One or two hidden layers, ReLU by default.

    Activation choice: toy_lipschitz used tanh throughout for continuity
    with its smooth closed-form ground truth. There is no such ground truth
    here, and ReLU is the standard choice for MNIST classifiers (faster to
    train, no vanishing-gradient concern at this depth) -- so ReLU is the
    default, with tanh still available via `activation` for anyone who wants
    to compare. See README's Design decisions section.
    """

    def __init__(self, input_dim=784, hidden_sizes=(128,), num_classes=10, activation="relu"):
        super().__init__()
        act_cls = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]
        dims = [input_dim] + list(hidden_sizes)
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(act_cls())
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    """Two conv+pool blocks, then a small FC head. Deliberately small/fast --
    this is a diagnostics project, not an accuracy benchmark.

    Split into two explicitly separate, independently callable submodules --
    `extractor` (everything up to and including the flatten) and `head` (the
    final linear layer, raw logits) -- so the layer-decomposition
    sub-experiment (layer_decomposition.py) can evaluate each in isolation
    (`model.extractor(x)`, `model.head(features)`) without forward hooks.
    `forward(x)` is unchanged in behavior: `model(x) == model.head(model.extractor(x))`
    exactly (see tests/test_layer_decomposition.py), and every constructor
    argument and the trained-accuracy behavior are unaffected by this split
    -- it's a pure module-structure refactor, channel sizes/kernel/pooling
    are identical to before.
    """

    def __init__(self, num_classes=10, conv_channels=(16, 32)):
        super().__init__()
        c1, c2 = conv_channels
        self.extractor = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),    # (1,28,28) -> (c1,28,28)
            nn.ReLU(),
            nn.MaxPool2d(2),                                # -> (c1,14,14)
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),    # -> (c2,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),                                # -> (c2,7,7)
            nn.Flatten(start_dim=1),                        # -> (c2*7*7,)
        )
        self.head = nn.Linear(c2 * 7 * 7, num_classes)

    def forward(self, x):
        return self.head(self.extractor(x))


class StrongCNN(nn.Module):
    """Higher-capacity CNN aimed at near-state-of-the-art MNIST accuracy
    (target ~99.3%+ test accuracy), built as a stronger baseline ahead of
    the data-cleaning experiment -- NOT a replacement for `SmallCNN`, which
    stays exactly as-is as the original, deliberately modest baseline (see
    mnist_lipschitz/README.md's three-model capacity comparison).

    Fixed architecture (paired with `STRONG_CNN_CONFIG` below for the
    exact training recipe) -- recorded here, not just in a notebook cell,
    because the later data-cleaning experiment reuses this exact
    architecture unchanged and needs something durable to point to:

      Conv2d(1->32, 3x3, pad=1) -> BatchNorm2d(32) -> ReLU
      Conv2d(32->32, 3x3, pad=1) -> BatchNorm2d(32) -> ReLU
      MaxPool2d(2) -> Dropout2d(p=dropout_conv)                      # (32,14,14)
      Conv2d(32->64, 3x3, pad=1) -> BatchNorm2d(64) -> ReLU
      Conv2d(64->64, 3x3, pad=1) -> BatchNorm2d(64) -> ReLU
      MaxPool2d(2) -> Dropout2d(p=dropout_conv)                      # (64,7,7)
      Flatten -> Linear(64*7*7 -> 256) -> BatchNorm1d(256) -> ReLU -> Dropout(p=dropout_fc)
      Linear(256 -> num_classes)

    No extractor/head split (unlike `SmallCNN`) -- that split exists
    specifically to support `layer_decomposition.py`'s sub-experiment,
    which this model isn't part of.
    """

    def __init__(self, num_classes=10, dropout_conv=0.25, dropout_fc=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(dropout_conv),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(dropout_conv),
            nn.Flatten(start_dim=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_fc),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# Exact training recipe for StrongCNN's raw-MNIST baseline -- kept as one
# importable, durable config (not a notebook cell) so the later
# data-cleaning experiment can reuse it byte-for-byte. Consumed by
# run_experiment.py::run_stronger_cnn_raw_mnist_experiment, which builds
# an augmentation.random_affine_augment closure from the augment_* keys and
# a torch.optim.lr_scheduler.CosineAnnealingLR from the lr_scheduler_* keys,
# and passes both to train_classifier below via its augment_fn/
# lr_scheduler_fn parameters.
STRONG_CNN_CONFIG = {
    "epochs": 25,
    "lr": 1e-3,
    "batch_size": 256,
    "optimizer": "adam",
    "lr_scheduler": "cosine_annealing",
    "lr_scheduler_t_max": 25,        # == epochs: one full cosine cycle over the whole run
    "lr_scheduler_eta_min": 1e-5,
    "augment_degrees": 10.0,          # max +/- rotation, degrees
    "augment_translate": 0.1,         # max +/- shift, fraction of image size
    "dropout_conv": 0.25,
    "dropout_fc": 0.5,
    "conv_channels": (32, 32, 64, 64),
    "fc_hidden": 256,
}


class FlattenedInputWrapper(nn.Module):
    """Wraps a model that expects (N, 1, 28, 28) image input (i.e. SmallCNN)
    so it instead accepts (N, 784) flat input, reshaping internally.

    estimators.py samples perturbation directions and computes distances in
    flat 784-d pixel space uniformly across all three models -- this lets
    the CNN be handed to those same estimator functions unchanged (same
    contract as the logistic regression / MLP models, which are already
    flat), rather than special-casing image-shaped input inside the
    estimators themselves.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x.reshape(x.shape[0], 1, 28, 28))


def train_classifier(model, train_loader, test_loader, epochs, lr, device=DEVICE, verbose=True,
                      augment_fn=None, lr_scheduler_fn=None):
    """Plain cross-entropy + Adam training loop. Returns (model, train_acc, test_acc).

    `augment_fn` (optional): called as `augment_fn(x)` on each training
    batch's input before the forward pass -- e.g.
    `augmentation.random_affine_augment` -- applied only during training,
    never at eval time. Left `None` (the default), this is exactly the
    original unaugmented loop: every pre-existing caller (logistic
    regression, MLP, the original `SmallCNN`) is unaffected.

    `lr_scheduler_fn` (optional): called once as `lr_scheduler_fn(optimizer)`
    to build a scheduler object (e.g. a `torch.optim.lr_scheduler`
    instance), whose `.step()` is called once per epoch, after that
    epoch's batches (not per batch). Left `None` (the default), lr stays
    fixed at `lr` throughout, exactly as before.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler_fn(optimizer) if lr_scheduler_fn is not None else None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if augment_fn is not None:
                x = augment_fn(x)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
        if scheduler is not None:
            scheduler.step()
        if verbose:
            lr_str = f"  lr={optimizer.param_groups[0]['lr']:.2e}" if scheduler is not None else ""
            print(f"  epoch {epoch:2d}/{epochs}  train loss {running_loss / n_seen:.4f}{lr_str}", flush=True)

    train_acc = evaluate_accuracy(model, train_loader, device)
    test_acc = evaluate_accuracy(model, test_loader, device)
    return model, train_acc, test_acc


@torch.no_grad()
def evaluate_accuracy(model, loader, device=DEVICE):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return correct / total


def margin_fn(model, x, y_true):
    """logit[y_true] - max(logit[j] for j != y_true), per example.

    This is the natural classifier analogue of a scalar regression output:
    a single real number per input, and it's what robustness actually
    depends on (margin crossing zero = the predicted class flips). The
    Lipschitz estimators in estimators.py are applied to this function, not
    to raw logits.
    """
    logits = model(x)
    true_logit = logits.gather(1, y_true.unsqueeze(1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, y_true.unsqueeze(1), float("-inf"))
    runner_up_logit = masked.max(dim=1).values
    return true_logit - runner_up_logit
