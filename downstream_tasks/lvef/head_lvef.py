"""Regression head for LVEF.
"""

from typing import List

import torch
import torch.nn as nn


class LVEFRegressionHead(nn.Module):
    """4-frame concat → 2-layer MLP regression to a scalar.

    Args:
        in_dim_per_view: channel count of the encoder's deepest feature map
            (post-global-pool). 768 for VMamba-small, 2048 for EchoCare Swin.
        n_views: number of frames per patient (4 = ED+ES × CH2+CH4).
        hidden_dim: width of the MLP hidden layer.
        dropout: dropout probability between the MLP layers.
        deepest_stage_index: which entry of the feature list to read from.
            Negative indexing is supported; -1 (default) takes the last
            (deepest) stage, which works for both VMamba (4 stages) and
            EchoCare (5 stages).
    """

    def __init__(
        self,
        in_dim_per_view: int,
        n_views: int = 4,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        deepest_stage_index: int = -1,
    ):
        super().__init__()
        self.in_dim_per_view = in_dim_per_view
        self.n_views = n_views
        self.deepest_stage_index = deepest_stage_index

        self.gap = nn.AdaptiveAvgPool2d(1)
        in_dim = in_dim_per_view * n_views
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feats_per_view: List[List[torch.Tensor]]) -> torch.Tensor:
        """``feats_per_view`` is a list of length ``n_views``; each entry is
        the feature list produced by the encoder for one frame batch.

        Returns: ``[B, 1]`` tensor of predicted EF in % (linear output).
        """
        if len(feats_per_view) != self.n_views:
            raise ValueError(
                f"expected {self.n_views} per-view feature lists, got {len(feats_per_view)}"
            )
        pooled = []
        for view_feats in feats_per_view:
            f = view_feats[self.deepest_stage_index]
            # f: [B, C, H, W]
            v = self.gap(f).flatten(1)               # [B, C]
            pooled.append(v)
        x = torch.cat(pooled, dim=1)                 # [B, C * n_views]
        return self.mlp(x)                            # [B, 1]


# Backbone label -> deepest-stage channel count.
_BACKBONE_DIM = {
    "vmamba_small":  768,
    "echocare_swin": 2048,
}


def build_head(
    backbone: str,
    n_views: int = 4,
    hidden_dim: int = 512,
    dropout: float = 0.1,
    in_dim_per_view: int = None,
) -> LVEFRegressionHead:
    """Factory.

    Pass ``backbone='vmamba_small'`` or ``'echocare_swin'`` for the standard
    dims, or override with ``in_dim_per_view`` directly (useful for ablations
    with a smaller VMamba variant).
    """
    if in_dim_per_view is None:
        if backbone not in _BACKBONE_DIM:
            raise ValueError(
                f"unknown backbone {backbone!r}; pass in_dim_per_view explicitly "
                f"or one of {list(_BACKBONE_DIM)}"
            )
        in_dim_per_view = _BACKBONE_DIM[backbone]
    return LVEFRegressionHead(
        in_dim_per_view=in_dim_per_view,
        n_views=n_views,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
