"""Transforms for the USenhance-adapted EnlightenGAN pipeline.
"""

from typing import Callable

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from .networks import compute_attention_map


def _to_pm1(img: Image.Image, image_size: int) -> torch.Tensor:
    """Resize to image_size, ToTensor, normalize to [-1,1] (mean=.5 std=.5)."""
    img = TF.resize(img, [image_size, image_size],
                    interpolation=TF.InterpolationMode.BILINEAR)
    t = TF.to_tensor(img)                 # [0,1]
    return t * 2.0 - 1.0                   # [-1,1]


def make_train_image_transform(image_size: int = 256, p_hflip: float = 0.5) -> Callable:
    """Returns f(pil) -> tensor[-1,1]. Random hflip is applied independently
    per image (the domains are unpaired, so no paired flip is needed)."""
    def f(img: Image.Image) -> torch.Tensor:
        if torch.rand(1).item() < p_hflip:
            img = TF.hflip(img)
        return _to_pm1(img, image_size)
    return f


def make_eval_image_transform(image_size: int = 256) -> Callable:
    """Deterministic resize + [-1,1] normalize (test / holdout)."""
    def f(img: Image.Image) -> torch.Tensor:
        return _to_pm1(img, image_size)
    return f


def attention_for(x_pm1: torch.Tensor) -> torch.Tensor:
    """Per-image attention map from a [-1,1] [3,H,W] (or [B,3,H,W]) tensor.
    Returns [1,H,W] for a 3-D input, [B,1,H,W] for a 4-D input."""
    if x_pm1.dim() == 3:
        return compute_attention_map(x_pm1.unsqueeze(0))[0]
    return compute_attention_map(x_pm1)
