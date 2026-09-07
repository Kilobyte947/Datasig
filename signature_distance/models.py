"""Classifier models, training loop, and MNIST train/test loading for
signature_distance's adversarial-evaluation infrastructure.

SmallCNN/StrongCNN are imported directly from `mnist_example.models`, not
redefined here - a deliberate choice, the opposite of this project's
earlier isolation convention (see git history / mnist_example's own
"Shared model checkpoint" README section for the full rationale): this
project and `mnist_example` are meant to evaluate literally the same
trained weights, not just the same architecture independently retrained,
so every adversarial-eval driver in this package that wants "the"
SmallCNN/StrongCNN loads `mnist_example`'s one shared, cached checkpoint
via `train_or_load_small_cnn`/`train_or_load_strong_cnn` below, rather than
training its own copy. `train_classifier`/`evaluate_accuracy`/
`load_mnist_train_test` remain local to this package - they're used
whenever a driver still wants a fresh, independently-trained model (the
seed/hyperparameter sweeps), not shared infrastructure with mnist_example.

This whole package uses `torch.float64` (see the project-wide convention
in `mnist_example`/`toy_example`), so the shared checkpoint - itself
trained in float64 - loads here with no precision cast at all: this
module's `SmallCNN`/`StrongCNN` instances and mnist_example's are
numerically identical, not just architecturally identical.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from mnist_example.models import STRONG_CNN_CONFIG, SmallCNN, StrongCNN

torch.set_default_dtype(torch.float64)

DATA_ROOT = Path(__file__).parent / "data"
MNIST_EXAMPLE_CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "mnist_example" / "checkpoints"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_mnist_train_test(batch_size: int = 256):
    """Fresh MNIST train/test loader (torchvision only).

    `transforms.ToTensor()` always returns float32, regardless of torch's
    global default dtype - not something a transform argument fixes, so the
    cast to `torch.get_default_dtype()` happens per-batch in
    `train_classifier`/`evaluate_accuracy` below instead, right before the
    model sees it. This package's models (the shared mnist_example
    checkpoint) are float64, so an uncast float32 batch would raise a dtype
    mismatch the moment the model's forward pass runs on it.
    """
    to_tensor = transforms.ToTensor()
    train_ds = datasets.MNIST(root=str(DATA_ROOT), train=True, download=True, transform=to_tensor)
    test_ds = datasets.MNIST(root=str(DATA_ROOT), train=False, download=True, transform=to_tensor)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=512, shuffle=False)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_classifier(model, train_loader, test_loader, epochs: int, lr: float = 1e-3,
                      device: str = "cpu", verbose: bool = True):
    """Plain cross-entropy + Adam training loop. Returns (model, train_acc, test_acc)."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, n_seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device=device, dtype=torch.get_default_dtype()), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
        if verbose:
            print(f"  epoch {epoch}/{epochs}  train loss {running_loss / n_seen:.4f}", flush=True)

    train_acc = evaluate_accuracy(model, train_loader, device)
    test_acc = evaluate_accuracy(model, test_loader, device)
    return model, train_acc, test_acc


@torch.no_grad()
def evaluate_accuracy(model, loader, device: str = "cpu") -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device=device, dtype=torch.get_default_dtype()), y.to(device)
        correct += (model(x).argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# Shared checkpoint (mnist_example) - cache-aside, load-only
# ---------------------------------------------------------------------------


def train_or_load_small_cnn(seed: int = 0, checkpoint_dir=MNIST_EXAMPLE_CHECKPOINT_DIR, verbose: bool = True):
    """Loads mnist_example's canonical, already-trained SmallCNN checkpoint
    (`mnist_example.models.train_or_load_small_cnn`'s own cache-aside
    training is the one and only place this weight is ever produced -
    this function never trains, only loads). Raises FileNotFoundError with
    a clear message if the checkpoint doesn't exist yet - run
    mnist_example's own training first (e.g.
    `mnist_example.models.train_or_load_small_cnn()`, or `main()`).

    Returns (model, train_acc, test_acc), model in eval mode - same shape
    as mnist_example's own function.
    """
    checkpoint_path = Path(checkpoint_dir) / "small_cnn_state_dict.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Shared SmallCNN checkpoint not found at {checkpoint_path}. "
            "Train it via mnist_example first, e.g.: "
            "`from mnist_example.models import train_or_load_small_cnn; train_or_load_small_cnn()`."
        )
    state = torch.load(checkpoint_path, weights_only=True)
    model = SmallCNN()
    model.load_state_dict(state["model_state_dict"])
    train_acc, test_acc = state["train_acc"], state["test_acc"]
    if verbose:
        print(f"[checkpoint] loaded SmallCNN from {checkpoint_path} "
              f"(train_acc={train_acc:.4f}  test_acc={test_acc:.4f})")
    model.eval()
    return model, train_acc, test_acc


def train_or_load_strong_cnn(seed: int = 0, checkpoint_dir=MNIST_EXAMPLE_CHECKPOINT_DIR, verbose: bool = True):
    """Loads mnist_example's canonical, already-trained StrongCNN
    checkpoint - same load-only contract as `train_or_load_small_cnn`
    above. StrongCNN's BatchNorm/Dropout make independent training runs
    not bit-reproducible even at a fixed seed, so this shared checkpoint
    is the only way this package and mnist_example see literally the same
    StrongCNN, not just the same architecture.

    Raises FileNotFoundError if the checkpoint doesn't exist yet - run
    mnist_example's own training first (e.g.
    `mnist_example.models.train_or_load_strong_cnn()`).
    """
    checkpoint_path = Path(checkpoint_dir) / "strong_cnn_state_dict.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Shared StrongCNN checkpoint not found at {checkpoint_path}. "
            "Train it via mnist_example first, e.g.: "
            "`from mnist_example.models import train_or_load_strong_cnn; train_or_load_strong_cnn()`."
        )
    state = torch.load(checkpoint_path, weights_only=True)
    model = StrongCNN(dropout_conv=STRONG_CNN_CONFIG["dropout_conv"], dropout_fc=STRONG_CNN_CONFIG["dropout_fc"])
    model.load_state_dict(state["model_state_dict"])
    train_acc, test_acc = state["train_acc"], state["test_acc"]
    if verbose:
        print(f"[checkpoint] loaded StrongCNN from {checkpoint_path} "
              f"(train_acc={train_acc:.4f}  test_acc={test_acc:.4f})")
    model.eval()
    return model, train_acc, test_acc
