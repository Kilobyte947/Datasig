"""Data augmentation for CNN training: random small rotations and translations."""

import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)


def random_affine_augment(x_image, degrees=10.0, translate=0.1, generator=None):
    """Applies an independent random rotation and translation to each image in a batch. degrees is 
    the max absolute rotation in degrees; translate is the max absolute shift as a fraction of image size. 
    Both sampled uniformly per sample. generator is an optional torch.Generator for reproducible sampling. 
    Returns a tensor of the same shape as x_image."""
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
