"""Landmark heatmap heads.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from vmamba_models.MambaDecoder import MambaDecoder


class LandmarkHeatmapHead(nn.Module):
    def __init__(
        self,
        num_landmarks: int = 24,
        in_channels=(96, 192, 384, 768),
        img_size: int = 224,
        patch_size: int = 4,
    ):
        super().__init__()
        self.decoder = MambaDecoder(
            num_classes=num_landmarks,
            in_channels=list(in_channels),
            img_size=[img_size, img_size],
            patch_size=patch_size,
        )

    def forward(self, feats_list, orig_hw, lq=None):
        """
        Args:
            feats_list: list of 4 tensors [B, C_i, H_i, W_i] from VSSMFeatureExtractor.
            orig_hw:    (H, W) tuple of the original input resolution.
            lq:         ignored (kept for call-signature parity with LandmarkUNetHead).

        Returns:
            logits: [B, num_landmarks, H, W] heatmap logits at original resolution.
        """
        logits = self.decoder(feats_list)
        return F.interpolate(logits, size=orig_hw, mode="bilinear", align_corners=False)


class LandmarkLinearHead(nn.Module):
    """1x1 conv on a single VSSM feature map, bilinear-upsampled to input res.

    Total trainable params for vmamba_small @ stage 0: 96*24 + 24 = 2,328.

    Args:
        num_landmarks: output channels.
        in_channels:   feature-map channel count for the chosen stage. For
                       vmamba_small: (96, 192, 384, 768) for stages 0..3.
        feature_index: which of the 4 VSSM feature maps to project from.
                       0 = highest resolution (56x56 for 224 input);
                       3 = lowest resolution / deepest semantic (7x7).
                       Default 0 — best resolution for keypoint localisation.
    """

    def __init__(
        self,
        num_landmarks: int = 24,
        in_channels: int = 96,
        feature_index: int = 0,
    ):
        super().__init__()
        self.feature_index = feature_index
        self.conv = nn.Conv2d(in_channels, num_landmarks,
                              kernel_size=1, bias=True)

    def forward(self, feats_list, orig_hw, lq=None):
        x = feats_list[self.feature_index]      # [B, in_channels, H, W]
        logits = self.conv(x)                   # [B, K, H, W]
        return F.interpolate(logits, size=orig_hw,
                             mode="bilinear", align_corners=False)


# ---------- v8-A: U-Net head (ported from UNetEnhanceHead) ----------

class _ConvBlock(nn.Module):
    """2x(Conv3x3 - InstanceNorm - LeakyReLU). Copied verbatim from
    downstream_tasks/enhance/head_enhance.py::_ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _StemConv(nn.Module):
    """Conv stem producing a stride-2 feature from a raw 3-channel image.
    Copied verbatim from downstream_tasks/enhance/head_enhance.py::_StemConv."""

    def __init__(self, in_ch: int = 3, out_ch: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LandmarkUNetHead(nn.Module):
    """5-stage U-Net decoder for VMamba landmark, ported from UNetEnhanceHead.

    Architecture-parity with `downstream_tasks/enhance/head_enhance.py::UNetEnhanceHead`,
    which outperforms `MambaDecoder` on image enhancement with the same
    OpenUS/VMamba encoder. Three deliberate differences for landmark:

      1. ``out_channels = num_landmarks`` (24), not 3 (image channels).
      2. No final ``sigmoid`` — landmark soft-argmax operates on raw logits.
      3. Returns logits at ``orig_hw`` (the input-resolution heatmaps; the
         dataset's per-image `meta['scale']` then maps soft-argmax results
         back to the original 800x600 image space).

    Feature contract (224x224 input, vmamba_small):
        f0_stem  : [B, stem_channels=128, 112, 112]   <- Conv stem on raw input
        f1       : [B,  96,  56,  56]                 <- VSSM stage 0
        f2       : [B, 192,  28,  28]                 <- VSSM stage 1
        f3       : [B, 384,  14,  14]                 <- VSSM stage 2
        f4       : [B, 768,   7,   7]                 <- VSSM stage 3

    Param count for 24 output channels: ~15M (vs MambaDecoder ~10M).
    """

    def __init__(
        self,
        num_landmarks: int = 24,
        in_channels: Tuple[int, int, int, int] = (96, 192, 384, 768),
        stem_channels: int = 128,
        decoder_channels: Tuple[int, int, int, int] = (512, 256, 128, 64),
    ):
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError(
                f"LandmarkUNetHead expects 4 VSSM feature maps (stride 4/8/16/32), "
                f"got {len(in_channels)} channel widths"
            )
        c1, c2, c3, c4 = in_channels
        c0 = stem_channels
        d3, d2, d1, d0 = decoder_channels

        self.stem = _StemConv(in_ch=3, out_ch=stem_channels)

        # Same ladder as UNetEnhanceHead. Each up-step concatenates the
        # upsampled previous decoder feature with the corresponding skip.
        self.block4_to_3 = _ConvBlock(c4 + c3, d3)
        self.block3_to_2 = _ConvBlock(d3 + c2, d2)
        self.block2_to_1 = _ConvBlock(d2 + c1, d1)
        self.block1_to_0 = _ConvBlock(d1 + c0, d0)

        self.final_conv = nn.Conv2d(d0, num_landmarks, kernel_size=1)

    def forward(
        self,
        feats_list: List[torch.Tensor],
        orig_hw,
        lq: torch.Tensor = None,
    ) -> torch.Tensor:
        if lq is None:
            raise ValueError(
                "LandmarkUNetHead requires lq=... (the raw normalised input image) "
                "for the Conv stem; eval_landmark.py must pass `lq=imgs` to head(...)"
            )
        if len(feats_list) != 4:
            raise ValueError(
                f"LandmarkUNetHead expects 4 VSSM feature maps, got {len(feats_list)}"
            )
        f1, f2, f3, f4 = feats_list
        f0 = self.stem(lq)

        u3 = F.interpolate(f4, size=f3.shape[-2:],
                           mode="bilinear", align_corners=False)
        u3 = self.block4_to_3(torch.cat([u3, f3], dim=1))

        u2 = F.interpolate(u3, size=f2.shape[-2:],
                           mode="bilinear", align_corners=False)
        u2 = self.block3_to_2(torch.cat([u2, f2], dim=1))

        u1 = F.interpolate(u2, size=f1.shape[-2:],
                           mode="bilinear", align_corners=False)
        u1 = self.block2_to_1(torch.cat([u1, f1], dim=1))

        u0 = F.interpolate(u1, size=f0.shape[-2:],
                           mode="bilinear", align_corners=False)
        u0 = self.block1_to_0(torch.cat([u0, f0], dim=1))

        logits = self.final_conv(u0)
        # Return raw logits at orig_hw; no sigmoid (landmark soft-argmax
        # consumes raw logits per v4/v6-A convention).
        return F.interpolate(logits, size=tuple(orig_hw),
                             mode="bilinear", align_corners=False)


def build_head(head_type: str, num_landmarks: int, backbone_dims, img_size: int,
               patch_size: int, linear_head_stage: int = 0):
    """Factory used by eval_landmark.py.

    backbone_dims: tuple of 4 ints, e.g. backbone.dims = (96, 192, 384, 768).
    """
    if head_type == "mamba_decoder":
        return LandmarkHeatmapHead(
            num_landmarks=num_landmarks,
            in_channels=tuple(backbone_dims),
            img_size=img_size,
            patch_size=patch_size,
        )
    elif head_type == "linear":
        return LandmarkLinearHead(
            num_landmarks=num_landmarks,
            in_channels=backbone_dims[linear_head_stage],
            feature_index=linear_head_stage,
        )
    elif head_type == "unet":
        return LandmarkUNetHead(
            num_landmarks=num_landmarks,
            in_channels=tuple(backbone_dims),
        )
    else:
        raise ValueError(f"unknown head_type={head_type!r}; "
                         f"use 'mamba_decoder', 'linear', or 'unet'")
