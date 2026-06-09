"""Coordinate- and channel-aware transforms for landmark training.
"""

from typing import Callable, Tuple

import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms

from . import landmark_schema as schema


# ---------- atomic ops -------------------------------------------------------

def resize_with_coords(
    img: Image.Image,
    coords: torch.Tensor,
    img_size: int,
) -> Tuple[Image.Image, torch.Tensor, torch.Tensor]:
    """Resize image to (img_size, img_size); rescale coords accordingly.

    Returns:
        img:    resized PIL image.
        coords: [K, 2] in resized pixel space.
        scale:  [2] tensor (sx, sy) = (img_size / W_orig, img_size / H_orig).
    """
    w_orig, h_orig = img.size
    img_r = TF.resize(img, [img_size, img_size])
    sx = img_size / w_orig
    sy = img_size / h_orig
    coords_r = coords.clone().float()
    coords_r[:, 0] = coords_r[:, 0] * sx
    coords_r[:, 1] = coords_r[:, 1] * sy
    return img_r, coords_r, torch.tensor([sx, sy], dtype=torch.float32)


def hflip_with_coords(
    img: Image.Image,
    coords: torch.Tensor,
    img_size: int,
    channel_perm,
) -> Tuple[Image.Image, torch.Tensor]:
    img = TF.hflip(img)
    coords = coords.clone()
    coords[:, 0] = (img_size - 1) - coords[:, 0]
    coords = coords[list(channel_perm)]
    return img, coords


def vflip_with_coords(
    img: Image.Image,
    coords: torch.Tensor,
    img_size: int,
    channel_perm,
) -> Tuple[Image.Image, torch.Tensor]:
    img = TF.vflip(img)
    coords = coords.clone()
    coords[:, 1] = (img_size - 1) - coords[:, 1]
    coords = coords[list(channel_perm)]
    return img, coords


def coords_to_heatmaps(
    coords: torch.Tensor,
    img_size: int,
    sigma: float = 2.0,
) -> torch.Tensor:
    """Render [K, 2] coordinates as a stack of K unnormalised Gaussian heatmaps.

    Each channel's peak is at the corresponding landmark; peak amplitude is 1.0.

    Returns:
        heatmaps: [K, img_size, img_size] float32 tensor.
    """
    K = coords.shape[0]
    device = coords.device

    ys = torch.arange(img_size, device=device, dtype=torch.float32)
    xs = torch.arange(img_size, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [H, W]

    grid_x = grid_x.unsqueeze(0)  # [1, H, W]
    grid_y = grid_y.unsqueeze(0)

    cx = coords[:, 0].view(K, 1, 1)
    cy = coords[:, 1].view(K, 1, 1)

    sq_dist = (grid_x - cx) ** 2 + (grid_y - cy) ** 2  # [K, H, W]
    return torch.exp(-sq_dist / (2.0 * sigma * sigma))


# ---------- image-only normalisation ----------------------------------------

_IMAGE_TO_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


# ---------- callable transforms returned to the dataset ---------------------

def _train_transform(
    img: Image.Image,
    coords: torch.Tensor,
    *,
    img_size: int,
    sigma: float,
    enable_flips: bool,
    enable_jitter: bool,
    p_hflip: float,
    p_vflip: float,
    p_jitter: float,
    jitter,
):
    """Returns (image_tensor, heatmaps, coords_scaled, scale)."""
    img, coords, scale = resize_with_coords(img, coords, img_size)

    if enable_flips:
        if torch.rand(1).item() < p_hflip:
            img, coords = hflip_with_coords(img, coords, img_size, schema.HFLIP_PERM)
        if torch.rand(1).item() < p_vflip:
            img, coords = vflip_with_coords(img, coords, img_size, schema.VFLIP_PERM)

    if enable_jitter and torch.rand(1).item() < p_jitter:
        img = jitter(img)

    img_t = _IMAGE_TO_TENSOR(img)
    heatmaps = coords_to_heatmaps(coords, img_size, sigma=sigma)
    return img_t, heatmaps, coords, scale


def _val_transform(
    img: Image.Image,
    coords: torch.Tensor,
    *,
    img_size: int,
    sigma: float,
):
    img, coords, scale = resize_with_coords(img, coords, img_size)
    img_t = _IMAGE_TO_TENSOR(img)
    heatmaps = coords_to_heatmaps(coords, img_size, sigma=sigma)
    return img_t, heatmaps, coords, scale


def make_train_transform(
    img_size: int = 224,
    sigma: float = 8.0,
    enable_flips: bool = False,
    enable_jitter: bool = True,
    p_hflip: float = 0.5,
    p_vflip: float = 0.5,
    p_jitter: float = 0.2,
) -> Callable:
    jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)

    def f(img, coords):
        return _train_transform(
            img, coords,
            img_size=img_size, sigma=sigma,
            enable_flips=enable_flips, enable_jitter=enable_jitter,
            p_hflip=p_hflip, p_vflip=p_vflip, p_jitter=p_jitter,
            jitter=jitter,
        )
    return f


def make_val_transform(img_size: int = 224, sigma: float = 8.0) -> Callable:
    def f(img, coords):
        return _val_transform(img, coords, img_size=img_size, sigma=sigma)
    return f
