""" This file defines the three classifiers under study (logistic regression, small MLP, and small CNN), their training loop, and 
the margin function the Lipschitz estimators are actually applied to."""

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64) # needed everywhere to avoid silent float32->float64 upcasting and the resulting performance hit.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LogisticRegressionModel(nn.Module):
    """Single linear layer, 784 inputs straight to 10 output scores. 
    No hidden layers, no nonlinearity at all.
    Margin is exactly linear in x, which makes this the one model with a closed-form Lipschitz constant (||w_true - w_runner_up||_2 for a fixed class pair).
    """
    def __init__(self, input_dim=784, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


class SmallMLP(nn.Module):
    """One or two hidden layers, ReLU by default (faster to
    train, no vanishing-gradient concern at this depth) with tanh available via `activation` . 
    Hidden layer sizes are configurable via `hidden_sizes`.
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
    """Two convolutional + pooling blocks, then a small classifier head.
    CNN needs to be handed (N, 1, 28, 28) image-shaped input instead of flat input (N, 784); 
    FlattenedInputWrapper wraps this model to accept flat input and reshape it internally."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # (1,28,28) -> (16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # -> (16,14,14)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),   # -> (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # -> (32,7,7)
        )
        self.classifier = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


class FlattenedInputWrapper(nn.Module):
    """Internally reshapes (N, 784) flat input to (N, 1, 28, 28) image input for SmallCNN which expects that shape. 
    This lets the CNN be handed to the same Lipschitz estimator functions as the other two models, which are already flat.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x.reshape(x.shape[0], 1, 28, 28))


def train_classifier(model, train_loader, test_loader, epochs, lr, device=DEVICE, verbose=True):
    """Train a classifier model and returns the trained model and its train/test accuracy - (model, train_acc, test_acc)""" 
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

    Collapses the classifier's 10 logits into the one scalar the Lipschitz estimators 
    actually operate on - the classifier analogue of a scalar regression output. 
    Positive and large means confidently correct; crossing zero means the predicted class flips. 
    Robustness is exactly about how easily a small input change pushes this margin across zero,
    which is why every estimator in estimators.py is applied to margin_fn, not to raw logits.
    """
    logits = model(x)
    true_logit = logits.gather(1, y_true.unsqueeze(1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, y_true.unsqueeze(1), float("-inf"))
    runner_up_logit = masked.max(dim=1).values
    return true_logit - runner_up_logit
