"""Landmark metrics: soft-argmax decoding + pixel-space MSE/SDR.

All metrics are computed in *original-image* pixel space. The training input
to the network is the resized 224x224 image; predictions decoded from
heatmaps must be rescaled back to the original-image coordinate system using
``meta['scale']`` from the dataset (see dataset_brainbench.py).
"""

from typing import Tuple

import torch
import torch.nn.functional as F


def soft_argmax_2d(heatmaps: torch.Tensor, beta: float = 100.0) -> torch.Tensor:
    """Sub-pixel argmax over spatial dims via temperature-scaled softmax.

    Args:
        heatmaps: [B, K, H, W] real-valued logits / scores.
        beta:     softmax temperature; higher = closer to a hard argmax.

    Returns:
        coords: [B, K, 2] sub-pixel (x, y) in input-image pixel coordinates,
                origin = top-left, x is column, y is row.
    """
    B, K, H, W = heatmaps.shape
    flat = (beta * heatmaps).reshape(B, K, H * W)
    flat = flat - flat.amax(dim=-1, keepdim=True)  # numerical stability
    prob = F.softmax(flat, dim=-1).reshape(B, K, H, W)

    ys = torch.arange(H, device=heatmaps.device, dtype=heatmaps.dtype)
    xs = torch.arange(W, device=heatmaps.device, dtype=heatmaps.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [H, W]

    coord_x = (prob * grid_x).sum(dim=(-1, -2))  # [B, K]
    coord_y = (prob * grid_y).sum(dim=(-1, -2))  # [B, K]
    return torch.stack([coord_x, coord_y], dim=-1)  # [B, K, 2]


def rescale_to_original(coords: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Map coords from input-resolution space back to original-image space.

    Args:
        coords: [B, K, 2] in input-resolution (e.g. 224x224) pixel space.
        scale:  [B, 2] (sx, sy) where sx = input_W / original_W, sy similar.

    Returns:
        coords in original-image pixel space, same shape [B, K, 2].
    """
    inv = 1.0 / scale  # [B, 2]
    return coords * inv.unsqueeze(1)


def mse_pixels(pred_coords: torch.Tensor, gt_coords: torch.Tensor) -> torch.Tensor:
    """Average squared pixel error over all (batch, landmark) pairs.

    Both inputs in [B, K, 2] pixel coordinates. Returns a scalar tensor.
    """
    diff = pred_coords - gt_coords  # [B, K, 2]
    sq = (diff ** 2).sum(dim=-1)    # [B, K] squared Euclidean
    return sq.mean()


def euclid_error(pred_coords: torch.Tensor, gt_coords: torch.Tensor) -> torch.Tensor:
    """Per-landmark Euclidean pixel error, [B, K]."""
    return torch.norm(pred_coords - gt_coords, dim=-1)


def sdr_at(pred_coords: torch.Tensor, gt_coords: torch.Tensor, tau: float) -> torch.Tensor:
    """Successful Detection Rate at threshold tau pixels.

    Fraction of (batch, landmark) pairs whose Euclidean error is <= tau.
    Returns scalar tensor in [0, 1].
    """
    err = euclid_error(pred_coords, gt_coords)  # [B, K]
    return (err <= tau).float().mean()


def decode_and_score(
    pred_heatmaps: torch.Tensor,
    gt_coords_original: torch.Tensor,
    scale: torch.Tensor,
    taus: Tuple[float, ...] = (2.0, 4.0, 10.0),
    beta: float = 100.0,
):
    """End-to-end heatmap → metrics in original-image space.

    Args:
        pred_heatmaps: [B, K, H, W] decoder output at input resolution.
        gt_coords_original: [B, K, 2] in ORIGINAL image pixel coords.
        scale: [B, 2] from dataset meta.
        taus: thresholds in pixels for SDR.
        beta: soft-argmax temperature.

    Returns:
        dict with keys: 'mse', 'sdr_<tau>' for each tau in taus.
    """
    pred_input = soft_argmax_2d(pred_heatmaps, beta=beta)        # [B, K, 2]
    pred_orig = rescale_to_original(pred_input, scale)           # [B, K, 2]
    out = {"mse": mse_pixels(pred_orig, gt_coords_original).item()}
    for tau in taus:
        out[f"sdr_{tau}"] = sdr_at(pred_orig, gt_coords_original, tau).item()
    return out
