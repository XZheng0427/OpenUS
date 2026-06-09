"""Image-only transforms for CAMUS LVEF.
"""

from typing import Callable, List

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def _normalize(t: torch.Tensor) -> torch.Tensor:
    return TF.normalize(t, mean=IMAGENET_MEAN, std=IMAGENET_STD)


def make_train_transform(
    img_size: int,
    enable_vflip: bool = False,
    enable_jitter: bool = True,
    jitter_strength: float = 0.2,
) -> Callable[[List[Image.Image]], torch.Tensor]:
    """Returns a callable: ``list[PIL.Image] -> Tensor[4, 3, img_size, img_size]``.

    Augmentation parameters are sampled once per call and applied identically
    to every frame in the list, so all four views of a patient see the same
    flip / jitter realisation.
    """
    jitter = transforms.ColorJitter(
        brightness=jitter_strength,
        contrast=jitter_strength,
    ) if enable_jitter else None

    def _tf(frames: List[Image.Image]) -> torch.Tensor:
        # Sample augmentation params ONCE for the whole patient sample.
        do_vflip = bool(enable_vflip and torch.rand(()).item() < 0.5)
        jitter_params = None
        if jitter is not None:
            # torchvision.ColorJitter.get_params returns the sample order
            # and the per-attribute factors; deterministic when reused.
            jitter_params = transforms.ColorJitter.get_params(
                jitter.brightness, jitter.contrast,
                jitter.saturation, jitter.hue,
            )

        out = []
        for img in frames:
            img = TF.resize(img, [img_size, img_size])
            if do_vflip:
                img = TF.vflip(img)
            if jitter_params is not None:
                fn_idx, b, c, s, h = jitter_params
                for fn_id in fn_idx:
                    if fn_id == 0 and b is not None:
                        img = TF.adjust_brightness(img, b)
                    elif fn_id == 1 and c is not None:
                        img = TF.adjust_contrast(img, c)
                    elif fn_id == 2 and s is not None:
                        img = TF.adjust_saturation(img, s)
                    elif fn_id == 3 and h is not None:
                        img = TF.adjust_hue(img, h)
            t = TF.to_tensor(img)        # [3, H, W] in [0, 1]
            t = _normalize(t)
            out.append(t)
        return torch.stack(out, dim=0)   # [4, 3, H, W]

    return _tf


def make_val_transform(img_size: int) -> Callable[[List[Image.Image]], torch.Tensor]:
    """Returns a callable: ``list[PIL.Image] -> Tensor[4, 3, img_size, img_size]``."""

    def _tf(frames: List[Image.Image]) -> torch.Tensor:
        out = []
        for img in frames:
            img = TF.resize(img, [img_size, img_size])
            t = TF.to_tensor(img)
            t = _normalize(t)
            out.append(t)
        return torch.stack(out, dim=0)

    return _tf
