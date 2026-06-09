"""Loss functions for the image-enhancement task.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# VGG16 features-layer indices in torchvision.models.vgg16().features (inplace
# ReLUs are at indices 1, 3, 6, 8, 11, 13, 15, 18, 20, 22, 25, 27, 29).
_VGG16_RELU_INDEX = {
    "relu1_1":  1, "relu1_2":  3,
    "relu2_1":  6, "relu2_2":  8,
    "relu3_1": 11, "relu3_2": 13, "relu3_3": 15,
    "relu4_1": 18, "relu4_2": 20, "relu4_3": 22,
    "relu5_1": 25, "relu5_2": 27, "relu5_3": 29,
}


class VGGPerceptualLoss(nn.Module):
    """L1 between VGG16 feature maps at a chosen set of ReLU layers.

    Inputs are expected as [B, 3, H, W] tensors in [0, 1] (matches the
    post-sigmoid output of `EnhanceHead` / `EchoCareEnhanceHead`).
    Internally the inputs are re-normalised with ImageNet statistics (VGG was
    pretrained on ImageNet) before being passed through a frozen VGG16
    feature extractor.

    The VGG weights are frozen and the module is set to .eval() permanently.
    Gradients flow only into the prediction branch — the target branch runs
    under torch.no_grad() to halve activation memory.
    """

    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD  = (0.229, 0.224, 0.225)

    def __init__(
        self,
        layers: Tuple[str, ...] = ("relu1_2", "relu2_2", "relu3_3", "relu4_3"),
        layer_weights: Tuple[float, ...] = None,
    ):
        super().__init__()
        # Lazy import so unit tests that don't use perceptual don't pay the
        # cost of importing torchvision.
        from torchvision.models import vgg16, VGG16_Weights

        for layer in layers:
            if layer not in _VGG16_RELU_INDEX:
                raise ValueError(
                    f"unknown VGG layer {layer!r}; available: "
                    f"{sorted(_VGG16_RELU_INDEX)}"
                )
        self.layers = tuple(layers)
        self.layer_indices = tuple(_VGG16_RELU_INDEX[l] for l in layers)
        max_idx = max(self.layer_indices)

        # Use only the slice of features up to the deepest requested layer.
        feats = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.vgg = nn.Sequential(*list(feats.children())[:max_idx + 1])
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.vgg.eval()

        if layer_weights is None:
            layer_weights = (1.0,) * len(layers)
        if len(layer_weights) != len(layers):
            raise ValueError(
                f"len(layer_weights)={len(layer_weights)} != len(layers)={len(layers)}"
            )
        self.register_buffer(
            "_layer_weights",
            torch.tensor(layer_weights, dtype=torch.float32),
        )

        self.register_buffer(
            "_mean",
            torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_std",
            torch.tensor(self._IMAGENET_STD).view(1, 3, 1, 1),
        )

    def train(self, mode: bool = True):
        # Keep VGG in eval mode (frozen BN / no dropout in vgg16.features
        # anyway, but explicit for safety).
        super().train(mode)
        self.vgg.eval()
        return self

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / self._std

    def _extract(self, x: torch.Tensor):
        out = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layer_indices:
                out.append(x)
        return out

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Run VGG in fp32 to dodge AMP precision issues with deep feature maps.
        pred = pred.float()
        target = target.float()
        pred_n = self._normalize(pred)
        target_n = self._normalize(target)

        with torch.no_grad():
            tgt_feats = self._extract(target_n)
        pred_feats = self._extract(pred_n)

        total = pred.new_zeros(())
        for w, pf, tf in zip(self._layer_weights, pred_feats, tgt_feats):
            total = total + w * F.l1_loss(pf, tf)
        return total


class CombinedReconLoss(nn.Module):
    """L1 (or L2) + lambda × perceptual.

    After each forward, exposes:
        self.last_l1   : detached scalar tensor
        self.last_perc : detached scalar tensor
    so the train loop can log per-component values without recomputing.
    """

    def __init__(
        self,
        base_name: str = "l1",
        perceptual_weight: float = 0.1,
        perceptual_layers: Tuple[str, ...] = ("relu1_2", "relu2_2", "relu3_3", "relu4_3"),
    ):
        super().__init__()
        if base_name == "l1":
            self.base = nn.L1Loss()
        elif base_name in ("l2", "mse"):
            self.base = nn.MSELoss()
        else:
            raise ValueError(f"unknown base loss {base_name!r}")
        self.perceptual = VGGPerceptualLoss(layers=perceptual_layers)
        self.perceptual_weight = float(perceptual_weight)
        # Filled in by forward() so the train loop can log components.
        self.register_buffer("_last_l1",   torch.zeros(()), persistent=False)
        self.register_buffer("_last_perc", torch.zeros(()), persistent=False)

    @property
    def last_l1(self) -> torch.Tensor:
        return self._last_l1

    @property
    def last_perc(self) -> torch.Tensor:
        return self._last_perc

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = self.base(pred, target)
        perc = self.perceptual(pred, target)
        self._last_l1.copy_(l1.detach())
        self._last_perc.copy_(perc.detach())
        return l1 + self.perceptual_weight * perc


def build_enhance_loss(
    name: str = "l1",
    perceptual_weight: float = 0.0,
    perceptual_layers: Tuple[str, ...] = ("relu1_2", "relu2_2", "relu3_3", "relu4_3"),
) -> nn.Module:
    """Factory for the enhancement loss.

    perceptual_weight == 0.0  → pure base loss (L1 or L2). Backward-compat
                                with v1 / Phase 0 behaviour.
    perceptual_weight  > 0.0  → CombinedReconLoss(base, λ × VGG perceptual).
    """
    name = name.lower()
    if name not in ("l1", "l2", "mse"):
        raise ValueError(f"unknown loss {name!r}; expected 'l1' or 'l2'")
    if perceptual_weight <= 0.0:
        return nn.L1Loss() if name == "l1" else nn.MSELoss()
    return CombinedReconLoss(
        base_name=name,
        perceptual_weight=perceptual_weight,
        perceptual_layers=perceptual_layers,
    )
