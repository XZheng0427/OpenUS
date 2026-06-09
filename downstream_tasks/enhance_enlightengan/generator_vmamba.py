"""OpenUS encoder inside the EnlightenGAN generator contract.
"""

import contextlib
import os
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from downstream_tasks.enhance.head_enhance import _ConvBlock, _StemConv
from downstream_tasks.enhance.backbone_wrapper import VSSMFeatureExtractor

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class _AttnDecoder(nn.Module):
    """EnlightenGAN-style attention decoder over a 4-stage VMamba pyramid.

    Takes the 4 VMamba features (strides 4/8/16/32), a stem feature (stride 2)
    derived from the raw input, the gray attention map, and the raw [-1,1]
    input; returns ``(output[-1,1], latent)``.
    """

    def __init__(
        self,
        in_channels: Tuple[int, int, int, int] = (96, 192, 384, 768),
        stem_channels: int = 128,
        decoder_channels: Tuple[int, int, int, int] = (512, 256, 128, 64),
        skip: float = 1.0,
    ):
        super().__init__()
        c1, c2, c3, c4 = in_channels
        c0 = stem_channels
        d3, d2, d1, d0 = decoder_channels
        self.skip = skip
        self.stem = _StemConv(in_ch=3, out_ch=stem_channels)
        self.block4_to_3 = _ConvBlock(c4 + c3, d3)
        self.block3_to_2 = _ConvBlock(d3 + c2, d2)
        self.block2_to_1 = _ConvBlock(d2 + c1, d1)
        self.block1_to_0 = _ConvBlock(d1 + c0, d0)
        self.final_conv = nn.Conv2d(d0, 3, kernel_size=1)

    @staticmethod
    def _g(gray, hw):
        # downsample the attention map to a feature size, preserving the
        # "max over region" (dark-region-weighted) semantics
        return F.adaptive_max_pool2d(gray, hw)

    def forward(self, feats: List[torch.Tensor], x_pm1: torch.Tensor,
                gray: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f1, f2, f3, f4 = feats               # strides 4/8/16/32
        f0 = self.stem(x_pm1)                # stride 2
        g0 = self._g(gray, f0.shape[-2:]); g1 = self._g(gray, f1.shape[-2:])
        g2 = self._g(gray, f2.shape[-2:]); g3 = self._g(gray, f3.shape[-2:])
        g4 = self._g(gray, f4.shape[-2:])

        u3 = F.interpolate(f4 * g4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        u3 = self.block4_to_3(torch.cat([u3, f3 * g3], dim=1))
        u2 = F.interpolate(u3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.block3_to_2(torch.cat([u2, f2 * g2], dim=1))
        u1 = F.interpolate(u2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.block2_to_1(torch.cat([u1, f1 * g1], dim=1))
        u0 = F.interpolate(u1, size=f0.shape[-2:], mode="bilinear", align_corners=False)
        u0 = self.block1_to_0(torch.cat([u0, f0 * g0], dim=1))

        latent = self.final_conv(u0)
        latent = F.interpolate(latent, size=x_pm1.shape[-2:], mode="bilinear", align_corners=False)
        latent = latent * gray                       # full-res attention
        output = latent + x_pm1 * self.skip          # residual, no tanh
        return output, latent


class VMambaAttentionGenerator(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        in_channels: Tuple[int, int, int, int] = (96, 192, 384, 768),
        stem_channels: int = 128,
        decoder_channels: Tuple[int, int, int, int] = (512, 256, 128, 64),
        skip: float = 1.0,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.extractor = VSSMFeatureExtractor(backbone)
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.extractor.parameters():
                p.requires_grad = False
            self.extractor.eval()
        self.decoder = _AttnDecoder(
            in_channels=in_channels, stem_channels=stem_channels,
            decoder_channels=decoder_channels, skip=skip,
        )
        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.extractor.eval()   # keep frozen encoder in eval regardless
        return self

    def _encode(self, x_pm1: torch.Tensor) -> List[torch.Tensor]:
        # [-1,1] -> [0,1] -> ImageNet-normalize for the VMamba encoder
        x01 = (x_pm1 + 1.0) / 2.0
        x_in = (x01 - self._mean) / self._std
        ctx = torch.no_grad() if self.freeze_encoder else contextlib.nullcontext()
        with ctx:
            feats = self.extractor(x_in)
        return list(feats)

    def forward(self, x_pm1: torch.Tensor, gray: torch.Tensor):
        feats = self._encode(x_pm1)
        return self.decoder(feats, x_pm1, gray)


def _init_decoder_weights(m: nn.Module):
    cn = m.__class__.__name__
    if "Conv" in cn and hasattr(m, "weight") and m.weight is not None:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif "InstanceNorm" in cn and getattr(m, "weight", None) is not None:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def build_vmamba_generator(backbone, freeze_encoder: bool = False, skip: float = 1.0,
                           stem_channels: int = 128):
    g = VMambaAttentionGenerator(
        backbone, freeze_encoder=freeze_encoder, skip=skip, stem_channels=stem_channels,
    )
    g.decoder.apply(_init_decoder_weights)   # init the NEW decoder only, not the encoder
    return g


# ===========================================================================
# Generalisation to EchoCare / USFM / SimMIM encoders (same EnlightenGAN
# recipe; only the encoder + decoder-arity change). The VMamba path above is
# left untouched so its completed runs stay reproducible.
# ===========================================================================

class _AttnDecoder5(nn.Module):
    """5-stage attention decoder for EchoCare.

    EchoCare's encoder natively provides the stride-2 stage (patch_size=2), so
    there is NO Conv stem — ``feats[0]`` is the stride-2 feature. Mirrors
    ``enhance_echocare/head_enhance.py:EchoCareEnhanceHead`` with two changes:
    (a) gray-attention multiply into each skip + the final latent; (b) a
    residual [-1,1] output (no sigmoid). Returns ``(output, latent)``.
    """

    def __init__(
        self,
        in_channels: Tuple[int, int, int, int, int] = (128, 256, 512, 1024, 2048),
        decoder_channels: Tuple[int, int, int, int] = (512, 256, 128, 64),
        skip: float = 1.0,
    ):
        super().__init__()
        c0, c1, c2, c3, c4 = in_channels
        d3, d2, d1, d0 = decoder_channels
        self.skip = skip
        self.block4_to_3 = _ConvBlock(c4 + c3, d3)
        self.block3_to_2 = _ConvBlock(d3 + c2, d2)
        self.block2_to_1 = _ConvBlock(d2 + c1, d1)
        self.block1_to_0 = _ConvBlock(d1 + c0, d0)
        self.final_conv = nn.Conv2d(d0, 3, kernel_size=1)

    @staticmethod
    def _g(gray, hw):
        return F.adaptive_max_pool2d(gray, hw)

    def forward(self, feats: List[torch.Tensor], x_pm1: torch.Tensor,
                gray: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f0, f1, f2, f3, f4 = feats               # strides 2/4/8/16/32
        g0 = self._g(gray, f0.shape[-2:]); g1 = self._g(gray, f1.shape[-2:])
        g2 = self._g(gray, f2.shape[-2:]); g3 = self._g(gray, f3.shape[-2:])
        g4 = self._g(gray, f4.shape[-2:])

        u3 = F.interpolate(f4 * g4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        u3 = self.block4_to_3(torch.cat([u3, f3 * g3], dim=1))
        u2 = F.interpolate(u3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.block3_to_2(torch.cat([u2, f2 * g2], dim=1))
        u1 = F.interpolate(u2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.block2_to_1(torch.cat([u1, f1 * g1], dim=1))
        u0 = F.interpolate(u1, size=f0.shape[-2:], mode="bilinear", align_corners=False)
        u0 = self.block1_to_0(torch.cat([u0, f0 * g0], dim=1))

        latent = self.final_conv(u0)
        latent = F.interpolate(latent, size=x_pm1.shape[-2:], mode="bilinear", align_corners=False)
        latent = latent * gray
        output = latent + x_pm1 * self.skip
        return output, latent


class EncoderAttnGenerator(nn.Module):
    """Encoder-agnostic generator: any feature extractor + an attention decoder.

    Same contract/recipe as ``VMambaAttentionGenerator`` (ImageNet bridge from
    [-1,1], no_grad when frozen, ``forward(x_pm1, gray) -> (output, latent)``),
    used for EchoCare (5-stage) and USFM/SimMIM (4-stage) encoders.
    """

    def __init__(self, extractor: nn.Module, decoder: nn.Module, freeze_encoder: bool = False):
        super().__init__()
        self.extractor = extractor
        self.decoder = decoder
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.extractor.parameters():
                p.requires_grad = False
            self.extractor.eval()
        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.extractor.eval()
        return self

    def _encode(self, x_pm1: torch.Tensor) -> List[torch.Tensor]:
        x01 = (x_pm1 + 1.0) / 2.0
        x_in = (x01 - self._mean) / self._std
        ctx = torch.no_grad() if self.freeze_encoder else contextlib.nullcontext()
        with ctx:
            feats = self.extractor(x_in)
        return list(feats)

    def forward(self, x_pm1: torch.Tensor, gray: torch.Tensor):
        return self.decoder(self._encode(x_pm1), x_pm1, gray)


def build_echocare_generator(ckpt_path: str, freeze_encoder: bool = True,
                             image_size: int = 256):
    from downstream_tasks.enhance_echocare.backbone_echocare import (
        ECHOCARE_DIMS, build_echocare_encoder, load_echocare_weights,
        EchoCareFeatureExtractor,
    )
    enc = build_echocare_encoder(use_checkpoint=(not freeze_encoder))
    if ckpt_path and os.path.isfile(ckpt_path):
        load_echocare_weights(enc, ckpt_path, strict=True)
        print(f"EchoCare encoder loaded (strict) from {ckpt_path}")
    else:
        print(f"WARNING: no EchoCare weights ({ckpt_path!r}); random init")
    extractor = EchoCareFeatureExtractor(enc)
    dec = _AttnDecoder5(in_channels=ECHOCARE_DIMS)
    g = EncoderAttnGenerator(extractor, dec, freeze_encoder=freeze_encoder)
    dec.apply(_init_decoder_weights)
    return g


def build_usfm_generator(ckpt_path: str, image_size: int = 224):
    from downstream_tasks.enhance_usfm.backbone_usfm import (
        USFM_DIMS, build_usfm_encoder, load_usfm_weights, USFMFeatureExtractor,
    )
    enc = build_usfm_encoder(img_size=image_size)
    if ckpt_path and os.path.isfile(ckpt_path):
        rep = load_usfm_weights(enc, ckpt_path)
        print(f"USFM load: loaded={rep['loaded']} dropped={rep['dropped']} "
              f"hard_missing={len(rep['hard_missing'])} hard_unexpected={len(rep['hard_unexpected'])}")
    else:
        print(f"WARNING: no USFM weights ({ckpt_path!r}); random init")
    extractor = USFMFeatureExtractor(enc)              # strict-frozen internally
    dec = _AttnDecoder(in_channels=(768, 768, 768, 768), stem_channels=128)
    g = EncoderAttnGenerator(extractor, dec, freeze_encoder=True)
    dec.apply(_init_decoder_weights)
    return g


def build_simmim_generator(ckpt_path: str, image_size: int = 224):
    from downstream_tasks.enhance_simmim.backbone_simmim import (
        SIMMIM_DIMS, build_simmim_encoder, load_simmim_weights, SimMIMFeatureExtractor,
    )
    enc = build_simmim_encoder(img_size=image_size)
    if ckpt_path and os.path.isfile(ckpt_path):
        rep = load_simmim_weights(enc, ckpt_path)
        print(f"SimMIM load: loaded={rep['loaded']} dropped={rep['dropped']} "
              f"hard_missing={len(rep['hard_missing'])} hard_unexpected={len(rep['hard_unexpected'])}")
    else:
        print(f"WARNING: no SimMIM weights ({ckpt_path!r}); random init")
    extractor = SimMIMFeatureExtractor(enc)            # strict-frozen internally
    dec = _AttnDecoder(in_channels=(768, 768, 768, 768), stem_channels=128)
    g = EncoderAttnGenerator(extractor, dec, freeze_encoder=True)
    dec.apply(_init_decoder_weights)
    return g
