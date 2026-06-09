"""EnlightenGAN networks — faithful reimplementation (USenhance-adapted).
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# attention map
# ---------------------------------------------------------------------------

def compute_attention_map(x: torch.Tensor) -> torch.Tensor:
    """`gray = 1 − normalized_luminance`, from a [-1,1] image batch.

    Mirrors EnlightenGAN's `data/unaligned_dataset.py`:
        r,g,b = in+1            # [-1,1] -> [0,2]
        gray  = 1 − (0.299r + 0.587g + 0.114b)/2
    Returns [B,1,H,W] in [0,1] (high weight on dark regions).
    """
    r = x[:, 0:1, :, :] + 1.0
    g = x[:, 1:2, :, :] + 1.0
    b = x[:, 2:3, :, :] + 1.0
    gray = 1.0 - (0.299 * r + 0.587 * g + 0.114 * b) / 2.0
    return gray


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------

class UnetResizeConv(nn.Module):
    """Self-regularized attention U-Net (EnlightenGAN `sid_unet_resize`).

    5 encoder stages (32/64/128/256/512), two convs each, MaxPool between
    stages; decoder mirrors via bilinear upsample + 1x1-ish deconv +
    skip-concat. The gray attention map is concatenated to the RGB input
    (conv1_1 is 4->32) and multiplied into the bottleneck, each skip, and the
    final latent. Output = latent*gray + input (residual, no tanh).

    Assumes H, W divisible by 16 (we run at 256).
    """

    def __init__(self, use_norm: bool = True, skip: float = 1.0):
        super().__init__()
        self.use_norm = use_norm
        self.skip = skip
        lrelu = lambda: nn.LeakyReLU(0.2, inplace=True)

        def bn(c):
            return nn.BatchNorm2d(c) if use_norm else nn.Identity()

        # ---- encoder ----
        self.conv1_1 = nn.Conv2d(4, 32, 3, padding=1);   self.lr1_1 = lrelu(); self.bn1_1 = bn(32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1);  self.lr1_2 = lrelu(); self.bn1_2 = bn(32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1);  self.lr2_1 = lrelu(); self.bn2_1 = bn(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1);  self.lr2_2 = lrelu(); self.bn2_2 = bn(64)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1);  self.lr3_1 = lrelu(); self.bn3_1 = bn(128)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1); self.lr3_2 = lrelu(); self.bn3_2 = bn(128)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1); self.lr4_1 = lrelu(); self.bn4_1 = bn(256)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1); self.lr4_2 = lrelu(); self.bn4_2 = bn(256)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5_1 = nn.Conv2d(256, 512, 3, padding=1); self.lr5_1 = lrelu(); self.bn5_1 = bn(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1); self.lr5_2 = lrelu(); self.bn5_2 = bn(512)

        # ---- decoder ----
        # each up-step: upsample -> deconv (halve ch) -> concat skip(*gray) -> 2 convs
        self.deconv5 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv6_1 = nn.Conv2d(512, 256, 3, padding=1); self.lr6_1 = lrelu(); self.bn6_1 = bn(256)
        self.conv6_2 = nn.Conv2d(256, 256, 3, padding=1); self.lr6_2 = lrelu(); self.bn6_2 = bn(256)

        self.deconv6 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=1); self.lr7_1 = lrelu(); self.bn7_1 = bn(128)
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=1); self.lr7_2 = lrelu(); self.bn7_2 = bn(128)

        self.deconv7 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding=1);  self.lr8_1 = lrelu(); self.bn8_1 = bn(64)
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=1);   self.lr8_2 = lrelu(); self.bn8_2 = bn(64)

        self.deconv8 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv9_1 = nn.Conv2d(64, 32, 3, padding=1);   self.lr9_1 = lrelu(); self.bn9_1 = bn(32)
        self.conv9_2 = nn.Conv2d(32, 32, 3, padding=1);   self.lr9_2 = lrelu()

        self.conv10 = nn.Conv2d(32, 3, 1)

    @staticmethod
    def _up(x):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor, gray: torch.Tensor):
        """x: [B,3,H,W] in [-1,1]; gray: [B,1,H,W] in [0,1]. H,W divisible by 16."""
        assert x.shape[-1] % 16 == 0 and x.shape[-2] % 16 == 0, \
            f"UnetResizeConv needs H,W divisible by 16, got {tuple(x.shape[-2:])}"
        # downsampled attention maps, one per scale
        gray2 = F.max_pool2d(gray, 2)
        gray3 = F.max_pool2d(gray2, 2)
        gray4 = F.max_pool2d(gray3, 2)
        gray5 = F.max_pool2d(gray4, 2)

        inp = torch.cat([x, gray], dim=1)                          # 4-ch

        c = self.bn1_1(self.lr1_1(self.conv1_1(inp)))
        conv1 = self.bn1_2(self.lr1_2(self.conv1_2(c)))
        x1 = self.pool1(conv1)

        c = self.bn2_1(self.lr2_1(self.conv2_1(x1)))
        conv2 = self.bn2_2(self.lr2_2(self.conv2_2(c)))
        x2 = self.pool2(conv2)

        c = self.bn3_1(self.lr3_1(self.conv3_1(x2)))
        conv3 = self.bn3_2(self.lr3_2(self.conv3_2(c)))
        x3 = self.pool3(conv3)

        c = self.bn4_1(self.lr4_1(self.conv4_1(x3)))
        conv4 = self.bn4_2(self.lr4_2(self.conv4_2(c)))
        x4 = self.pool4(conv4)

        c = self.bn5_1(self.lr5_1(self.conv5_1(x4)))
        conv5 = self.bn5_2(self.lr5_2(self.conv5_2(c)))
        conv5 = conv5 * gray5                                       # attention @ bottleneck

        up = self.deconv5(self._up(conv5))
        up6 = torch.cat([up, conv4 * gray4], dim=1)
        c = self.bn6_1(self.lr6_1(self.conv6_1(up6)))
        conv6 = self.bn6_2(self.lr6_2(self.conv6_2(c)))

        up = self.deconv6(self._up(conv6))
        up7 = torch.cat([up, conv3 * gray3], dim=1)
        c = self.bn7_1(self.lr7_1(self.conv7_1(up7)))
        conv7 = self.bn7_2(self.lr7_2(self.conv7_2(c)))

        up = self.deconv7(self._up(conv7))
        up8 = torch.cat([up, conv2 * gray2], dim=1)
        c = self.bn8_1(self.lr8_1(self.conv8_1(up8)))
        conv8 = self.bn8_2(self.lr8_2(self.conv8_2(c)))

        up = self.deconv8(self._up(conv8))
        up9 = torch.cat([up, conv1 * gray], dim=1)
        c = self.bn9_1(self.lr9_1(self.conv9_1(up9)))
        conv9 = self.lr9_2(self.conv9_2(c))

        latent = self.conv10(conv9)
        latent = latent * gray                                     # times_residual
        output = latent + x * self.skip                            # residual, no tanh
        return output, latent


# ---------------------------------------------------------------------------
# discriminator
# ---------------------------------------------------------------------------

class NoNormDiscriminator(nn.Module):
    """PatchGAN with no normalization (EnlightenGAN `no_norm_4`).

    Conv-stride2 + LeakyReLU only. Returns a [B,1,h,w] patch score map (no
    sigmoid — LSGAN uses raw logits).
    """

    def __init__(self, input_nc: int = 3, ndf: int = 64, n_layers: int = 5):
        super().__init__()
        kw, padw = 4, 2
        seq = [nn.Conv2d(input_nc, ndf, kw, stride=2, padding=padw),
               nn.LeakyReLU(0.2, inplace=True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2 ** n, 8)
            seq += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, inplace=True)]
        nf_mult_prev, nf_mult = nf_mult, min(2 ** n_layers, 8)
        seq += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, stride=1, padding=padw),
                nn.LeakyReLU(0.2, inplace=True)]
        seq += [nn.Conv2d(ndf * nf_mult, 1, kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------

class GANLoss(nn.Module):
    """LSGAN loss (MSE against 1/0 targets)."""

    def __init__(self, target_real: float = 1.0, target_fake: float = 0.0):
        super().__init__()
        self.real = target_real
        self.fake = target_fake
        self.loss = nn.MSELoss()

    def __call__(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        t = self.real if target_is_real else self.fake
        target = torch.full_like(pred, t)
        return self.loss(pred, target)


class VGGFeatureLoss(nn.Module):
    """Self-feature-preserving loss at torchvision VGG16 relu5_1.

    DEVIATION FROM UPSTREAM (documented): EnlightenGAN uses an authors'
    Caffe-converted `vgg16.weight` with BGR [0,255] input. We use torchvision
    VGG16 with its correct preprocessing (RGB, /255, ImageNet mean/std). The
    conceptual role — deep relu5_1 feature preservation with instance-normed
    features — is preserved.

    Inputs are [-1,1] image batches; the comparison is OUTPUT-vs-INPUT (the
    "self" in self-feature-preserving), not against a paired target.
    """

    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)
    _RELU5_1_SLICE = 26  # torchvision vgg16.features[:26] ends at relu5_1 (idx 25)

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        feats = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.vgg = nn.Sequential(*list(feats.children())[:self._RELU5_1_SLICE])
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.vgg.eval()
        self.register_buffer("_mean", torch.tensor(self._MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(self._STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        self.vgg.eval()
        return self

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        # [-1,1] -> [0,1] -> ImageNet-normalize
        x = (x.float() + 1.0) / 2.0
        return (x - self._mean) / self._std

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return self.vgg(self._prep(x))

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        f_out = self._features(output)
        with torch.no_grad():
            f_tgt = self._features(target)
        # instance-norm the feature maps before MSE
        f_out = F.instance_norm(f_out)
        f_tgt = F.instance_norm(f_tgt)
        return F.mse_loss(f_out, f_tgt)


# ---------------------------------------------------------------------------
# factories + init
# ---------------------------------------------------------------------------

def _init_weights(m: nn.Module):
    cn = m.__class__.__name__
    if hasattr(m, "weight") and ("Conv" in cn or "Linear" in cn):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif "BatchNorm2d" in cn and getattr(m, "weight", None) is not None:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def define_G(use_norm: bool = True, skip: float = 1.0) -> UnetResizeConv:
    g = UnetResizeConv(use_norm=use_norm, skip=skip)
    g.apply(_init_weights)
    return g


def define_D(input_nc: int = 3, ndf: int = 64, n_layers: int = 5) -> NoNormDiscriminator:
    d = NoNormDiscriminator(input_nc=input_nc, ndf=ndf, n_layers=n_layers)
    d.apply(_init_weights)
    return d
