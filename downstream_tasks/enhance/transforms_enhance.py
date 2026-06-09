"""Paired transforms for USenhance low/high-quality image pairs.
"""

from typing import Callable, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def _paired_resize(lq: Image.Image, hq: Image.Image, size: int):
    return (
        TF.resize(lq, [size, size], interpolation=TF.InterpolationMode.BILINEAR),
        TF.resize(hq, [size, size], interpolation=TF.InterpolationMode.BILINEAR),
    )


def _paired_train(
    lq: Image.Image,
    hq: Image.Image,
    *,
    image_size: int,
    p_hflip: float,
    p_rot: float,
    rot_deg: float,
    p_jitter: float,
    jitter: transforms.ColorJitter,
) -> Tuple[torch.Tensor, torch.Tensor]:
    lq, hq = _paired_resize(lq, hq, image_size)

    if torch.rand(1).item() < p_hflip:
        lq = TF.hflip(lq)
        hq = TF.hflip(hq)

    if torch.rand(1).item() < p_rot:
        angle = float((torch.rand(1).item() * 2.0 - 1.0) * rot_deg)
        lq = TF.rotate(lq, angle, interpolation=TF.InterpolationMode.BILINEAR)
        hq = TF.rotate(hq, angle, interpolation=TF.InterpolationMode.BILINEAR)

    # ColorJitter is applied only to LQ (input). HQ is the target — distorting
    # it would change what the model is supposed to produce.
    if torch.rand(1).item() < p_jitter:
        lq = jitter(lq)

    lq_t = TF.to_tensor(lq)
    hq_t = TF.to_tensor(hq)
    lq_t = TF.normalize(lq_t, _IMAGENET_MEAN, _IMAGENET_STD)
    return lq_t, hq_t


def _paired_val(
    lq: Image.Image,
    hq: Image.Image,
    *,
    image_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    lq, hq = _paired_resize(lq, hq, image_size)
    lq_t = TF.to_tensor(lq)
    hq_t = TF.to_tensor(hq)
    lq_t = TF.normalize(lq_t, _IMAGENET_MEAN, _IMAGENET_STD)
    return lq_t, hq_t


def make_train_transform(
    image_size: int = 256,
    p_hflip: float = 0.5,
    p_rot: float = 0.3,
    rot_deg: float = 5.0,
    p_jitter: float = 0.2,
) -> Callable:
    jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1)

    def f(lq, hq):
        return _paired_train(
            lq, hq,
            image_size=image_size,
            p_hflip=p_hflip,
            p_rot=p_rot, rot_deg=rot_deg,
            p_jitter=p_jitter, jitter=jitter,
        )
    return f


def make_val_transform(image_size: int = 256) -> Callable:
    def f(lq, hq):
        return _paired_val(lq, hq, image_size=image_size)
    return f


def make_holdout_transform(image_size: int = 256) -> Callable:
    """Holdout images have no HQ partner — caller passes the same PIL twice or
    builds its own pipeline. Provided here for symmetry."""
    def f(lq):
        lq = TF.resize(lq, [image_size, image_size],
                       interpolation=TF.InterpolationMode.BILINEAR)
        lq_t = TF.to_tensor(lq)
        lq_t = TF.normalize(lq_t, _IMAGENET_MEAN, _IMAGENET_STD)
        return lq_t
    return f
