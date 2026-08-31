"""Method B adversarial / Lipschitz-ratio evaluation - standalone.

Tests Method B's signature distance as the denominator in the project's
Lipschitz-ratio pipeline (numerator = margin change under a small
adversarial perturbation), compared against plain pixel-Euclidean distance,
on two classifiers of different capacity.

ISOLATION: this module does not import from, or otherwise depend on, any
module in `toy_lipschitz` or `mnist_lipschitz` (their model classes,
adversarial-attack code, margin/distance functions, or data loaders). Only
this project's own new code (`streams.py`, `signatures.py`, `distances.py`,
`data_pool.py`, all in this same `signature_distance` package) and generic
libraries (torch, torchvision, numpy, matplotlib) are used. `SmallCNN`,
`StrongCNN`, `margin`, and `fgsm_attack` below are FRESH REIMPLEMENTATIONS of
ideas that also exist in `mnist_lipschitz/models.py` and
`mnist_lipschitz/adversarial/attacks.py` - each flagged individually in its
own docstring - not imports, and not guaranteed to produce identical trained
weights (different init/seed even with an identical architecture): treat any
comparison against Experiment 2's documented numbers as architecture-level,
not an exact reproduction.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from signature_distance.data_pool import load_eval_pool
from signature_distance.distances import method_b_feature_vector, rescale_signature
from signature_distance.signatures import signature_of_stream
from signature_distance.streams import line_stream, make_reference_lines

DATA_ROOT = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
SIGNATURE_DEPTH = 4
# Fixed from the Phase 4 sanity check (run_experiment.sanity_check_demo),
# not re-derived here - Method B's distance function isn't being changed by
# this task, only evaluated.
METHOD_B_R = 2.8597598377587485
METHOD_B_LINES = make_reference_lines()  # fixed geometry, shared across every call


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_mnist_train_test(batch_size: int = 256):
    """Fresh MNIST train/test loader (torchvision only) - independent of
    mnist_lipschitz.data, per this module's isolation constraint."""
    to_tensor = transforms.ToTensor()
    train_ds = datasets.MNIST(root=str(DATA_ROOT), train=True, download=True, transform=to_tensor)
    test_ds = datasets.MNIST(root=str(DATA_ROOT), train=False, download=True, transform=to_tensor)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=512, shuffle=False)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Models - fresh reimplementations of mnist_lipschitz.models' architectures
# ---------------------------------------------------------------------------


class SmallCNN(nn.Module):
    """Fresh reimplementation of `mnist_lipschitz.models.SmallCNN`'s
    architecture (two conv+pool blocks, then a linear head) - not imported,
    per this module's isolation constraint. Channel sizes/kernel/pooling
    match the original exactly; the extractor/head split (added there for
    `layer_decomposition.py`) is omitted here since it's not needed for this
    evaluation."""

    def __init__(self, num_classes: int = 10, conv_channels=(16, 32)):
        super().__init__()
        c1, c2 = conv_channels
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(start_dim=1),
        )
        self.head = nn.Linear(c2 * 7 * 7, num_classes)

    def forward(self, x):
        return self.head(self.features(x))


class StrongCNN(nn.Module):
    """Fresh reimplementation of `mnist_lipschitz.models.StrongCNN`'s
    architecture (4 conv layers with BatchNorm, MaxPool + Dropout2d, an FC
    head with BatchNorm1d + Dropout) - not imported. Trained here with a
    SIMPLIFIED recipe (fewer epochs, no rotation/translation augmentation,
    no cosine LR schedule) - the original's 25-epoch/augmented recipe
    (`STRONG_CNN_CONFIG`) targeted ~99.3%+ test accuracy; this evaluation
    only needs architecture-level capacity contrast with SmallCNN, not that
    exact number - see `run_adversarial_evaluation`'s reported accuracies
    for what was actually achieved here.
    """

    def __init__(self, num_classes: int = 10, dropout_conv: float = 0.25, dropout_fc: float = 0.5):
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


def train_classifier(model, train_loader, test_loader, epochs: int, lr: float = 1e-3,
                      device: str = "cpu", verbose: bool = True):
    """Plain cross-entropy + Adam training loop - fresh reimplementation of
    the same basic recipe in `mnist_lipschitz.models.train_classifier`
    (without its optional augmentation/LR-scheduler hooks, unused here).
    Returns (model, train_acc, test_acc)."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, n_seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
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
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# Margin, attack, distances
# ---------------------------------------------------------------------------


def margin(model, x, y_true) -> torch.Tensor:
    """logit[y_true] - max(logit[j] for j != y_true), per example. Fresh
    reimplementation of `mnist_lipschitz.models.margin_fn`'s definition -
    the scalar this project's Lipschitz-ratio numerator is always built
    from, not raw logits or cross-entropy loss."""
    logits = model(x)
    true_logit = logits.gather(1, y_true.unsqueeze(1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, y_true.unsqueeze(1), float("-inf"))
    runner_up_logit = masked.max(dim=1).values
    return true_logit - runner_up_logit


def fgsm_attack(model, x, y, epsilon: float) -> torch.Tensor:
    """Single-step FGSM: clip(x + epsilon * sign(grad_x CE(model(x), y)), 0, 1).
    Fresh reimplementation of the standard formula (Goodfellow et al. 2015),
    same convention as `mnist_lipschitz.adversarial.attacks.fgsm_attack`
    (cross-entropy loss, L_inf single step, clip to the valid [0,1] pixel
    range) but not imported. `epsilon=0` returns `x` unchanged."""
    x = x.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    (grad,) = torch.autograd.grad(loss, x)
    x_adv = x.detach() + epsilon * grad.sign()
    return x_adv.clamp(0.0, 1.0)


def random_noise_perturbation(x, l2_budget: torch.Tensor, generator=None) -> torch.Tensor:
    """Control perturbation: random, non-gradient-directed noise with the
    SAME per-example L2 norm as a given FGSM perturbation (`l2_budget`,
    shape (N,)), clipped to [0, 1]. Used to check whether Method B's
    distance separates genuinely *adversarial* shifts from equally-large
    but undirected ones, not just any shift of similar magnitude."""
    n = x.shape[0]
    flat_shape = x.reshape(n, -1).shape
    noise = torch.randn(flat_shape, generator=generator)
    noise = noise / noise.norm(dim=1, keepdim=True) * l2_budget.unsqueeze(1)
    x_control = x.reshape(n, -1) + noise
    return x_control.reshape(x.shape).clamp(0.0, 1.0)


def pixel_euclidean_distance(x1, x2) -> torch.Tensor:
    """Baseline denominator (a): plain Euclidean distance in flat pixel
    space. x1, x2: (N, ...) same shape."""
    n = x1.shape[0]
    return (x1.reshape(n, -1) - x2.reshape(n, -1)).norm(dim=1)


def method_b_signature_distance(images1: torch.Tensor, images2: torch.Tensor,
                                 depth: int = SIGNATURE_DEPTH, r: float = METHOD_B_R) -> torch.Tensor:
    """Denominator (b): Method B's own pipeline - make_reference_lines (once,
    module-level) -> line_stream -> signature_of_stream (per line) ->
    rescale_signature -> method_b_feature_vector (concatenate the 16 lines)
    -> Euclidean. Reuses Method B's own code unmodified (this task's `Out of
    scope` explicitly excludes changing it); nothing here is reimplemented.

    images1, images2: (N, 28, 28) float32 in [0, 1].
    """
    num_lines = METHOD_B_LINES.shape[0]

    def _feature_vector(images):
        stream = line_stream(images, METHOD_B_LINES)  # (N, num_lines, points_per_line, 2)
        sig = torch.stack(
            [signature_of_stream(stream[:, i], depth=depth) for i in range(num_lines)], dim=1
        )  # (N, num_lines, sig_dim)
        sig = rescale_signature(sig, r=r, depth=depth)
        return method_b_feature_vector(sig)  # (N, num_lines * sig_dim)

    vec1 = _feature_vector(images1)
    vec2 = _feature_vector(images2)
    return (vec1 - vec2).norm(dim=1)


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------


def run_adversarial_evaluation(n_per_class: int = 20, epsilons=(0.02, 0.03, 0.05),
                                seed: int = 0, cnn_epochs: int = 3, strong_epochs: int = 3,
                                device: str = "cpu", verbose: bool = True) -> dict:
    """Train SmallCNN and StrongCNN fresh, then for a sample of test images
    (n_per_class per digit, from data_pool.load_eval_pool) and each epsilon:
    generate an FGSM adversarial perturbation and a magnitude-matched random
    control perturbation, compute the margin-change numerator, both
    denominators (pixel-Euclidean and Method B signature distance), and the
    resulting ratios. Returns a nested dict, one entry per model.
    """
    torch.manual_seed(seed)
    train_loader, test_loader = load_mnist_train_test()

    models = {}
    for name, model, epochs in [("SmallCNN", SmallCNN(), cnn_epochs), ("StrongCNN", StrongCNN(), strong_epochs)]:
        if verbose:
            print(f"Training {name} ({epochs} epochs)...")
        trained, train_acc, test_acc = train_classifier(
            model, train_loader, test_loader, epochs=epochs, device=device, verbose=verbose
        )
        trained.eval()
        models[name] = {"model": trained, "train_acc": train_acc, "test_acc": test_acc}
        if verbose:
            print(f"  {name}: train_acc={train_acc:.4f} test_acc={test_acc:.4f}")

    images, labels = load_eval_pool(n_per_class=n_per_class, seed=seed)  # (N, 28, 28)
    images_c = images.unsqueeze(1)  # (N, 1, 28, 28) for model input
    generator = torch.Generator().manual_seed(seed)

    results = {"n_images": images.shape[0], "epsilons": list(epsilons), "models": {}}

    for name, info in models.items():
        model = info["model"]
        model_result = {
            "train_acc": info["train_acc"], "test_acc": info["test_acc"],
            "eps": {},
        }

        for eps in epsilons:
            x_adv_c = fgsm_attack(model, images_c, labels, eps)  # (N,1,28,28)
            x_adv = x_adv_c.squeeze(1)  # (N,28,28), for Method B's stream functions

            fgsm_l2 = pixel_euclidean_distance(images_c, x_adv_c)
            x_control_c = random_noise_perturbation(images_c, fgsm_l2, generator=generator)
            x_control = x_control_c.squeeze(1)

            with torch.no_grad():
                margin_orig = margin(model, images_c, labels)
                margin_adv = margin(model, x_adv_c, labels)
                margin_control = margin(model, x_control_c, labels)

            num_adv = (margin_orig - margin_adv).abs()
            num_control = (margin_orig - margin_control).abs()

            denom_a_adv = pixel_euclidean_distance(images_c, x_adv_c)
            denom_a_control = pixel_euclidean_distance(images_c, x_control_c)
            denom_b_adv = method_b_signature_distance(images, x_adv)
            denom_b_control = method_b_signature_distance(images, x_control)

            with torch.no_grad():
                preds_adv = model(x_adv_c).argmax(dim=1)
            flip_fraction = (preds_adv != labels).float().mean().item()

            eps_result = {
                "flip_fraction": flip_fraction,
                "numerator_adv": num_adv, "numerator_control": num_control,
                "denom_a_adv": denom_a_adv, "denom_a_control": denom_a_control,
                "denom_b_adv": denom_b_adv, "denom_b_control": denom_b_control,
                "ratio_a_adv": num_adv / denom_a_adv, "ratio_b_adv": num_adv / denom_b_adv,
                "ratio_a_control": num_control / denom_a_control,
                "ratio_b_control": num_control / denom_b_control,
                "labels": labels, "x_adv": x_adv, "images": images,
            }

            top10 = eps_result["ratio_b_adv"].topk(min(10, images.shape[0])).indices
            eps_result["top10_pairs"] = [
                (int(labels[i]), int(preds_adv[i]), float(eps_result["ratio_b_adv"][i]))
                for i in top10
            ]

            model_result["eps"][eps] = eps_result

        results["models"][name] = model_result

    return results
