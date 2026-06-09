"""Image-enhancement heads for the US-DINO pipeline.
"""

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from vmamba_models.MambaDecoder import MambaDecoder


class EnhanceHead(nn.Module):
    """MambaDecoder + Sigmoid — v1/v2 default head for US-DINO."""

    def __init__(
        self,
        image_size: int = 256,
        in_channels: Tuple[int, int, int, int] = (96, 192, 384, 768),
        embed_dim: int = 96,
        patch_size: int = 4,
        depths: Tuple[int, int, int, int] = (4, 4, 4, 4),
        out_channels: int = 3,
    ):
        super().__init__()
        self.decoder = MambaDecoder(
            img_size=[image_size, image_size],
            in_channels=list(in_channels),
            num_classes=out_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            depths=list(depths),
        )

    def forward(
        self,
        feats_list: List[torch.Tensor],
        orig_hw,
        lq: torch.Tensor = None,        # accepted for call-signature parity
    ) -> torch.Tensor:
        x = self.decoder(feats_list)
        if x.shape[-2:] != tuple(orig_hw):
            x = F.interpolate(x, size=tuple(orig_hw),
                              mode="bilinear", align_corners=False)
        return torch.sigmoid(x.float())


class _ConvBlock(nn.Module):
    """Two 3x3 Conv + InstanceNorm + LeakyReLU (matches EchoCare's _ConvBlock)."""

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
    """Conv stem producing a stride-2 feature from a raw 3-channel image."""

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


class UNetEnhanceHead(nn.Module):
    """5-stage U-Net decoder for US-DINO with a Conv stem for the stride-2 feature.

    Architecture-parity with ``EchoCareEnhanceHead``: identical decoder ladder
    (upsample → concat skip → 2×(Conv-IN-LeakyReLU)), identical block widths
    ``(512, 256, 128, 64)``, identical final ``1×1`` Conv + Sigmoid. Differs
    only in the encoder feature dims and the source of the f0 stride-2 feature
    (which the VSSM encoder doesn't natively expose).

    Feature contract (256×256 input, vmamba_small):
        f0_stem  : [B, stem_channels=128, 128, 128]   ← Conv stem on raw LQ
        f1       : [B,  96,  64,  64]                 ← VSSM stage 0
        f2       : [B, 192,  32,  32]                 ← VSSM stage 1
        f3       : [B, 384,  16,  16]                 ← VSSM stage 2
        f4       : [B, 768,   8,   8]                 ← VSSM stage 3
    """

    def __init__(
        self,
        in_channels: Tuple[int, int, int, int] = (96, 192, 384, 768),
        stem_channels: int = 128,
        decoder_channels: Tuple[int, int, int, int] = (512, 256, 128, 64),
        out_channels: int = 3,
    ):
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError(
                f"UNetEnhanceHead expects 4 VSSM feature maps (stride 4/8/16/32), "
                f"got {len(in_channels)} channel widths"
            )
        c1, c2, c3, c4 = in_channels
        c0 = stem_channels
        d3, d2, d1, d0 = decoder_channels

        self.stem = _StemConv(in_ch=3, out_ch=stem_channels)

        # Same ladder as EchoCareEnhanceHead. Each up-step concatenates the
        # upsampled previous decoder feature with the corresponding skip.
        self.block4_to_3 = _ConvBlock(c4 + c3, d3)
        self.block3_to_2 = _ConvBlock(d3 + c2, d2)
        self.block2_to_1 = _ConvBlock(d2 + c1, d1)
        self.block1_to_0 = _ConvBlock(d1 + c0, d0)

        self.final_conv = nn.Conv2d(d0, out_channels, kernel_size=1)

    def forward(
        self,
        feats_list: List[torch.Tensor],
        orig_hw,
        lq: torch.Tensor = None,
    ) -> torch.Tensor:
        if lq is None:
            raise ValueError("UNetEnhanceHead requires lq=... (the raw input)")
        if len(feats_list) != 4:
            raise ValueError(
                f"UNetEnhanceHead expects 4 VSSM feature maps, got {len(feats_list)}"
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
        logits = F.interpolate(logits, size=tuple(orig_hw),
                               mode="bilinear", align_corners=False)
        return torch.sigmoid(logits.float())


def build_head(
    head_type: str,
    image_size: int = 256,
    in_channels: Tuple[int, int, int, int] = (96, 192, 384, 768),
    embed_dim: int = 96,
    patch_size: int = 4,
    stem_channels: int = 128,
    out_channels: int = 3,
) -> nn.Module:
    """Factory used by eval_enhance.py to switch between heads."""
    head_type = head_type.lower()
    if head_type in ("mamba", "mamba_decoder"):
        return EnhanceHead(
            image_size=image_size, in_channels=in_channels,
            embed_dim=embed_dim, patch_size=patch_size,
            out_channels=out_channels,
        )
    if head_type == "unet":
        return UNetEnhanceHead(
            in_channels=in_channels,
            stem_channels=stem_channels,
            out_channels=out_channels,
        )
    raise ValueError(
        f"unknown head_type {head_type!r}; expected 'mamba' or 'unet'"
    )
