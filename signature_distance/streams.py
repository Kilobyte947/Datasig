"""Stream (path) construction from MNIST images, for later signature computation.

Two independent methods for turning a 28x28 image into a path:
  - Method 1 (`make_pixel_order` + `patch_sv_stream`): a time-augmented stream
    of the largest singular value of 3x3 patches, visited in a fixed shared
    pixel order.
  - Method 2 (`row_stream`, added in Checkpoint 3): the image's rows (or
    columns) treated directly as a stream in R^28.

No signature computation happens in this module - see `PLAN.md`.
"""

import torch


def make_pixel_order(k: int = 64, seed: int = 0,
                      image_size: int = 28) -> torch.Tensor:
    """Return (k, 2) int64 tensor of (row, col) locations, sampled uniformly
    WITHOUT replacement from the interior grid [1, image_size-2]^2, in a fixed
    order determined by seed. Raises ValueError if k exceeds the number of
    interior pixels.
    """
    lo, hi = 1, image_size - 2
    n_side = hi - lo + 1
    n_interior = n_side * n_side
    if k > n_interior:
        raise ValueError(f"k={k} exceeds number of interior pixels ({n_interior})")

    generator = torch.Generator().manual_seed(seed)
    flat_idx = torch.randperm(n_interior, generator=generator)[:k]
    rows = flat_idx // n_side + lo
    cols = flat_idx % n_side + lo
    return torch.stack([rows, cols], dim=1).to(torch.int64)


def patch_sv_stream(images: torch.Tensor, pixel_order: torch.Tensor,
                     mode: str = "top1") -> torch.Tensor:
    """Method 1 stream construction.

    images: (N, 28, 28) float32
    pixel_order: (K, 2) int64 from make_pixel_order
    returns: (N, K, 2) float32 for mode="top1", columns [t, sigma1],
             with t = arange(K) / (K - 1).
             (N, K, 4) for mode="all3", columns [t, s1, s2, s3], s1 >= s2 >= s3.
    """
    if mode not in ("top1", "all3"):
        raise ValueError(f"unknown mode: {mode!r}")

    n = images.shape[0]
    k = pixel_order.shape[0]
    offsets = torch.tensor([-1, 0, 1], dtype=torch.int64)

    rows = pixel_order[:, 0]  # (K,)
    cols = pixel_order[:, 1]  # (K,)
    row_grid = (rows.view(k, 1) + offsets.view(1, 3)).view(k, 3, 1).expand(k, 3, 3)
    col_grid = (cols.view(k, 1) + offsets.view(1, 3)).view(k, 1, 3).expand(k, 3, 3)

    patches = images[:, row_grid, col_grid]  # (N, K, 3, 3)
    sigma = torch.linalg.svdvals(patches)  # (N, K, 3), descending

    t = torch.arange(k, dtype=torch.float32) / (k - 1) if k > 1 else torch.zeros(k)
    t = t.view(1, k).expand(n, k)

    if mode == "top1":
        return torch.stack([t, sigma[..., 0]], dim=-1).to(torch.float32)
    return torch.cat([t.unsqueeze(-1), sigma], dim=-1).to(torch.float32)


def row_stream(images: torch.Tensor, axis: str = "rows") -> torch.Tensor:
    """Method 2 stream construction.

    images: (N, 28, 28) float32
    returns: (N, 28, 28) float32.
      axis="rows": stream[n, i, :] is row i of image n (top to bottom).
      axis="cols": stream[n, j, :] is column j of image n (left to right).

    Does not mutate `images`; always returns a copy (via `.clone()` for
    axis="rows", via `.transpose(1, 2).contiguous()` for axis="cols", which
    also copies since the transposed view is non-contiguous).
    """
    if axis == "rows":
        return images.clone()
    if axis == "cols":
        return images.transpose(1, 2).contiguous()
    raise ValueError(f"unknown axis: {axis!r}")
