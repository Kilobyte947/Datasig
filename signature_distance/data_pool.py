"""Fixed, reproducible pool of MNIST test images.

Mirrors the pool-based evaluation protocol used in Experiment 2
(`mnist_lipschitz`): a deterministic, class-balanced subset of the MNIST test
set, selected by shuffling indices with a seed and taking the first
`n_per_class` images per class from that shuffled order.
"""

import torch
from torchvision import datasets


def load_eval_pool(n_per_class: int = 100, seed: int = 0,
                    root: str = "./data") -> tuple[torch.Tensor, torch.Tensor]:
    """Return (images, labels).

    images: (10 * n_per_class, 28, 28) float32 in [0, 1]
    labels: (10 * n_per_class,) int64

    Deterministic: for each class, take the first n_per_class images of that
    class in MNIST test-set order after shuffling indices with the given seed.
    Images are sorted by class in the returned tensors (all 0s, then all 1s, ...).
    """
    dataset = datasets.MNIST(root=root, train=False, download=True)
    images_all = dataset.data.to(torch.float32) / 255.0  # (10000, 28, 28)
    labels_all = dataset.targets.to(torch.int64)          # (10000,)

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(images_all.shape[0], generator=generator)
    shuffled_labels = labels_all[perm]

    selected_idx = []
    for c in range(10):
        class_perm_idx = perm[shuffled_labels == c][:n_per_class]
        if class_perm_idx.shape[0] < n_per_class:
            raise ValueError(
                f"class {c} has only {class_perm_idx.shape[0]} images available, "
                f"need {n_per_class}")
        selected_idx.append(class_perm_idx)

    idx = torch.cat(selected_idx)
    return images_all[idx], labels_all[idx]
