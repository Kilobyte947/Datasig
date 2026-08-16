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


def train_classifier(model, train_loader, test_loader, epochs, lr, device=DEVICE, verbose=True):
    """Plain cross-entropy + Adam training loop. Returns (model, train_acc, test_acc)."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
        if verbose:
            print(f"  epoch {epoch:2d}/{epochs}  train loss {running_loss / n_seen:.4f}")

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
