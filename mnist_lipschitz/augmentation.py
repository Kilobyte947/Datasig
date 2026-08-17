"""Data augmentation for CNN training -- currently just random small
rotations/translations (no flips: MNIST digits are not flip-invariant, and
no shear/scale -- the point is a light augmentation, not an aggressive
one).

Kept as a separate module (mirrors distance.py/embeddings.py's "one
pluggable transform, one file" convention) so it can be swapped in/out of
`models.train_classifier` via the `augment_fn` parameter without touching
training-loop logic.
"""

import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)


def random_affine_augment(x_image, degrees=10.0, translate=0.1, generator=None):
    """Applies an independent random small rotation + translation to each
    image in a batch, via `affine_grid`/`grid_sample` -- fully vectorized
    over the batch (no per-sample Python loop, no PIL round-trip), and
    dtype-preserving (works directly on this project's float64 tensors,
    checked in tests/test_augmentation.py).

    `x_image`: (N, 1, H, W). `degrees`: max absolute rotation, sampled
    uniformly in [-degrees, +degrees] independently per sample. `translate`:
    max absolute shift as a fraction of image size (e.g. 0.1 = up to 10% of
    H/W), sampled uniformly in [-translate, +translate] independently per
    sample per axis.

    `generator`: optional `torch.Generator` for reproducible sampling
    (matches this project's seeding convention elsewhere, e.g.
    `local_perturbation_lipschitz`'s `seed` argument).

    With `degrees=0` and `translate=0` this reduces to the identity map (up
    to float rounding from the grid-sample interpolation) -- checked
    directly in tests/test_augmentation.py, not just assumed.

    Returns a tensor of the same shape/dtype/device as `x_image`.
    """
    N, C, H, W = x_image.shape
    dtype, device = x_image.dtype, x_image.device

    rand_kwargs = dict(dtype=dtype, device=device)
    if generator is not None:
        rand_kwargs["generator"] = generator
    u_angle = torch.rand(N, **rand_kwargs)
    u_tx = torch.rand(N, **rand_kwargs)
    u_ty = torch.rand(N, **rand_kwargs)

    angles = (u_angle * 2 - 1) * degrees * (torch.pi / 180.0)
    tx = (u_tx * 2 - 1) * translate
    ty = (u_ty * 2 - 1) * translate

    cos, sin = torch.cos(angles), torch.sin(angles)
    theta = torch.zeros(N, 2, 3, dtype=dtype, device=device)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 0, 2] = tx * 2  # affine_grid's translation unit is half the image extent
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, 1, 2] = ty * 2

    grid = F.affine_grid(theta, x_image.shape, align_corners=False)
    return F.grid_sample(x_image, grid, align_corners=False, padding_mode="zeros")
