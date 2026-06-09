"""Heatmap loss factory for the landmark pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSEHeatmapLoss(nn.Module):
    """Mean squared error on heatmaps. Wraps ``nn.MSELoss(reduction='mean')``
    so the v4 numerics are preserved bit-for-bit when ``--loss_type mse``."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse(logits, target)


class BCEHeatmapLoss(nn.Module):
    """Binary cross-entropy with logits. Target Gaussians stay in [0, 1]."""

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, target, reduction="mean")


class FocalHeatmapLoss(nn.Module):
    """CenterNet-style focal loss on Gaussian heatmap targets.

    For every pixel with ``target == 1.0`` (the exact peak written by
    ``coords_to_heatmaps``), uses ``-(1 - p)^alpha * log(p)``.
    Elsewhere uses ``-(1 - target)^beta * p^gamma * log(1 - p)``. ``p`` is
    the sigmoid of the prediction logit.

    **Reduction** is ``mean`` over ALL pixels in the batch (not the
    CenterNet ``/N_pos`` convention). This is deliberate: the v5-plan tuned
    ``coord_loss_weight=0.1`` against v4's ``nn.MSELoss(reduction='mean')``
    scale (~1e-3 - 2e-1). The original ``/N_pos`` normalisation produces
    focal loss values in the thousands (24 positives vs ~1.2 M pixels in
    a [B=8, K=24, 224, 224] target), which dwarfs ``coord_loss_weight *
    coord_loss`` by ~5 orders of magnitude and starves the coord-L1
    signal — that's the v5-AB collapse documented in
    ``01_fetal_brain_landmark_RESULTS.md``. Per-pixel mean reduction
    keeps focal loss at v4-comparable O(0.01) so the existing
    ``coord_loss_weight=0.1`` retains its v4-tuned meaning.

    Numerically implemented via ``binary_cross_entropy_with_logits`` with
    explicit per-pixel weights. This avoids ``log(0)`` and works under AMP.
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # p has the same shape as logits; compute with no_grad so the
        # focal modulator doesn't introduce a second autograd path through
        # sigmoid. Gradients flow through the BCE-with-logits term.
        with torch.no_grad():
            p = torch.sigmoid(logits)
            pos_mask = target.eq(1.0)
            # Positive pixels: weight = (1 - p)^alpha
            # Negative pixels: weight = (1 - target)^beta * p^gamma
            weight_pos = (1.0 - p).pow(self.alpha)
            weight_neg = (1.0 - target).pow(self.beta) * p.pow(self.gamma)
            weight = torch.where(pos_mask, weight_pos, weight_neg)

        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return (weight * bce).mean()


def loss_expects_logits(loss_type: str) -> bool:
    """True if soft-argmax should be applied to ``sigmoid(pred)`` rather
    than to raw ``pred`` for this loss family. MSE keeps the v4 path
    (raw logits) so reproduction stays bit-equivalent."""
    return loss_type in {"bce", "focal"}


def make_loss(
    loss_type: str,
    *,
    focal_alpha: float = 2.0,
    focal_beta: float = 4.0,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """Factory used by ``eval_landmark.main``."""
    if loss_type == "mse":
        return MSEHeatmapLoss()
    if loss_type == "bce":
        return BCEHeatmapLoss()
    if loss_type == "focal":
        return FocalHeatmapLoss(alpha=focal_alpha, beta=focal_beta, gamma=focal_gamma)
    raise ValueError(
        f"unknown loss_type={loss_type!r}; use 'mse', 'bce', or 'focal'"
    )
