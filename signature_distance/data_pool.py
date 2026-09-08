"""Fixed, reproducible, class-balanced pool of MNIST test images."""

import torch
from torchvision import datasets

torch.set_default_dtype(torch.float64)


def load_eval_pool(n_per_class: int = 100, seed: int = 0,
                    root: str = "./data") -> tuple[torch.Tensor, torch.Tensor]:
    """A deterministic, class-balanced subset of the MNIST test set: for each class, the first
    n_per_class images in shuffled order under the given seed. 
    Returns (images, labels), images sorted by class."""
    dataset = datasets.MNIST(root=root, train=False, download=True)
    images_all = dataset.data.to(torch.get_default_dtype()) / 255.0  # (10000, 28, 28)
    labels_all = dataset.targets.to(torch.int64) # (10000,)

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
