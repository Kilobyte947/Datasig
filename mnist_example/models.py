"""The three classifiers under study, their training loop, and the margin function the Lipschitz
estimators are applied to."""

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from mnist_example.augmentation import random_affine_augment
from mnist_example.data import load_mnist, make_loader
torch.set_default_dtype(torch.float64)
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LogisticRegressionModel(nn.Module):
    """Single linear layer, 784 -> 10."""
    def __init__(self, input_dim=784, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


class SmallMLP(nn.Module):
    """One or two hidden layers, ReLU by default (tanh also available)."""
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
    """Two conv+pool blocks, then a small linear head. Split into extractor (everything up to the
    flatten) and head (the final linear layer), so each can be evaluated in isolation."""
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
    """Higher-capacity CNN: four conv layers with batch norm and dropout, then a two-layer classifier
    head with batch norm and dropout, targeting near-state-of-the-art MNIST accuracy."""
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

STRONG_CNN_CONFIG = {
    "epochs": 25,
    "lr": 1e-3,
    "batch_size": 256,
    "optimizer": "adam",
    "lr_scheduler": "cosine_annealing",
    "lr_scheduler_t_max": 25,
    "lr_scheduler_eta_min": 1e-5,
    "augment_degrees": 10.0, 
    "augment_translate": 0.1,
    "dropout_conv": 0.25,
    "dropout_fc": 0.5,
    "conv_channels": (32, 32, 64, 64),
    "fc_hidden": 256,
}


class FlattenedInputWrapper(nn.Module):
    """Wraps a model that expects (N, 1, 28, 28) image input so it instead accepts (N, 784) flat input."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x.reshape(x.shape[0], 1, 28, 28))


def train_classifier(model, train_loader, test_loader, epochs, lr, device=DEVICE, verbose=True,
                      augment_fn=None, lr_scheduler_fn=None):
    """Cross-entropy + Adam training loop. augment_fn, if given, is applied to each training batch's input. 
    lr_scheduler_fn, if given, builds a scheduler stepped once per epoch. 
    Returns (model, train_acc, test_acc)."""
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
    """logit[y_true] - max(logit[j] for j != y_true), per example. The function the Lipschitz 
    estimators are applied to."""
    logits = model(x)
    true_logit = logits.gather(1, y_true.unsqueeze(1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, y_true.unsqueeze(1), float("-inf"))
    runner_up_logit = masked.max(dim=1).values
    return true_logit - runner_up_logit


def train_or_load_small_cnn(seed=0, checkpoint_dir=CHECKPOINT_DIR, verbose=True):
    """Loads a cached SmallCNN if one exists at checkpoint_dir, otherwise trains one with the
    project's standard recipe and saves it. checkpoint_dir=None forces a fresh, uncached run. 
    Returns (model, train_acc, test_acc)."""
    train = load_mnist(train=True)
    test = load_mnist(train=False)
    train_loader = make_loader(train.x_image, train.y, batch_size=256, shuffle=True, seed=seed)
    test_loader = make_loader(test.x_image, test.y, batch_size=1000, shuffle=False)

    checkpoint_path = checkpoint_dir / "small_cnn_state_dict.pt" if checkpoint_dir is not None else None
    if checkpoint_path is not None and checkpoint_path.exists():
        state = torch.load(checkpoint_path, weights_only=True)
        model = SmallCNN()
        model.load_state_dict(state["model_state_dict"])
        train_acc, test_acc = state["train_acc"], state["test_acc"]
        if verbose:
            print(f"[checkpoint] loaded SmallCNN from {checkpoint_path} "
                  f"(train_acc={train_acc:.4f}  test_acc={test_acc:.4f})")
    else:
        torch.manual_seed(seed)
        model = SmallCNN()
        model, train_acc, test_acc = train_classifier(
            model, train_loader, test_loader, epochs=8, lr=1e-3, verbose=verbose)
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "train_acc": train_acc, "test_acc": test_acc},
                       checkpoint_path)

    model.eval()
    return model, train_acc, test_acc


def train_or_load_strong_cnn(seed=0, checkpoint_dir=CHECKPOINT_DIR, verbose=True):
    """Loads a cached StrongCNN if one exists at checkpoint_dir, otherwise trains one with
    STRONG_CNN_CONFIG's recipe and saves it. A shared checkpoint matters more here than for SmallCNN,
    since StrongCNN's batch norm and dropout make independent training runs not bit-reproducible even at a fixed seed. 
    Returns (model, train_acc, test_acc)."""
    train = load_mnist(train=True)
    test = load_mnist(train=False)
    train_loader = make_loader(train.x_image, train.y, batch_size=STRONG_CNN_CONFIG["batch_size"],
                                shuffle=True, seed=seed)
    test_loader = make_loader(test.x_image, test.y, batch_size=1000, shuffle=False)

    checkpoint_path = checkpoint_dir / "strong_cnn_state_dict.pt" if checkpoint_dir is not None else None
    model = StrongCNN(dropout_conv=STRONG_CNN_CONFIG["dropout_conv"],
                       dropout_fc=STRONG_CNN_CONFIG["dropout_fc"])

    if checkpoint_path is not None and checkpoint_path.exists():
        state = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        train_acc, test_acc = state["train_acc"], state["test_acc"]
        if verbose:
            print(f"[checkpoint] loaded StrongCNN from {checkpoint_path} "
                  f"(train_acc={train_acc:.4f}  test_acc={test_acc:.4f})")
    else:
        torch.manual_seed(seed)
        augment_generator = torch.Generator().manual_seed(seed)
        augment_fn = lambda x: random_affine_augment(
            x, degrees=STRONG_CNN_CONFIG["augment_degrees"],
            translate=STRONG_CNN_CONFIG["augment_translate"], generator=augment_generator)
        lr_scheduler_fn = lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=STRONG_CNN_CONFIG["lr_scheduler_t_max"],
            eta_min=STRONG_CNN_CONFIG["lr_scheduler_eta_min"])

        model, train_acc, test_acc = train_classifier(
            model, train_loader, test_loader, epochs=STRONG_CNN_CONFIG["epochs"],
            lr=STRONG_CNN_CONFIG["lr"], verbose=verbose, augment_fn=augment_fn,
            lr_scheduler_fn=lr_scheduler_fn)
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "train_acc": train_acc, "test_acc": test_acc},
                       checkpoint_path)

    model.eval()
    return model, train_acc, test_acc
