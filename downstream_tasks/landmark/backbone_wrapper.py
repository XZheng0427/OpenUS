"""Feature-list adapter for Backbone_DINOv2_VSSM_2.
"""

import torch
import torch.nn as nn


class VSSMFeatureExtractor(nn.Module):
    """Returns the 4 spatial feature maps produced by Backbone_DINOv2_VSSM_2.

    Output is a list of length ``len(backbone.out_indices)`` (4 by default),
    each tensor in ``[B, C_i, H_i, W_i]`` layout. For vmamba_small with
    ``patch_size=4`` and a 224x224 input, shapes are:
        [B,  96, 56, 56], [B, 192, 28, 28], [B, 384, 14, 14], [B, 768, 7, 7]
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor):
        b = self.backbone
        x = b.patch_embed(x)
        outs = []
        for i, layer in enumerate(b.layers):
            block_out = layer.blocks(x)
            x = layer.downsample(block_out)
            if i in b.out_indices:
                norm = getattr(b, f"outnorm{i}")
                out = norm(block_out)
                if not b.channel_first:
                    out = out.permute(0, 3, 1, 2)
                outs.append(out.contiguous())
        return outs
