"""Conditional 70x70 PatchGAN discriminator for OpenUS (pix2pix-style).
"""

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 6, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        kw, padw = 4, 1
        seq = [nn.Conv2d(in_channels, ndf, kw, stride=2, padding=padw),
               nn.LeakyReLU(0.2, inplace=True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2 ** n, 8)
            seq += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, stride=2, padding=padw, bias=False),
                nn.InstanceNorm2d(ndf * nf_mult, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        nf_mult_prev, nf_mult = nf_mult, min(2 ** n_layers, 8)
        seq += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, stride=1, padding=padw, bias=False),
            nn.InstanceNorm2d(ndf * nf_mult, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        seq += [nn.Conv2d(ndf * nf_mult, 1, kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _init_weights(m: nn.Module):
    cn = m.__class__.__name__
    if "Conv" in cn and hasattr(m, "weight") and m.weight is not None:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif "InstanceNorm" in cn and getattr(m, "weight", None) is not None:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def build_patchgan(in_channels: int = 6, ndf: int = 64, n_layers: int = 3) -> PatchDiscriminator:
    d = PatchDiscriminator(in_channels=in_channels, ndf=ndf, n_layers=n_layers)
    d.apply(_init_weights)
    return d
