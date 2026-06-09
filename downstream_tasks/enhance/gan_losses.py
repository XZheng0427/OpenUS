"""GAN losses for the v3 adversarial branch.
"""

import torch
import torch.nn.functional as F


class _LSGAN:
    """Least-squares GAN (MSE against 1/0 soft targets)."""
    def d(self, pred_real, pred_fake):
        pr, pf = pred_real.float(), pred_fake.float()
        return 0.5 * (F.mse_loss(pr, torch.ones_like(pr)) +
                      F.mse_loss(pf, torch.zeros_like(pf)))

    def g(self, pred_fake):
        pf = pred_fake.float()
        return F.mse_loss(pf, torch.ones_like(pf))


class _BCE:
    """Vanilla GAN with logits (BCE)."""
    def d(self, pred_real, pred_fake):
        pr, pf = pred_real.float(), pred_fake.float()
        return 0.5 * (F.binary_cross_entropy_with_logits(pr, torch.ones_like(pr)) +
                      F.binary_cross_entropy_with_logits(pf, torch.zeros_like(pf)))

    def g(self, pred_fake):
        pf = pred_fake.float()
        return F.binary_cross_entropy_with_logits(pf, torch.ones_like(pf))


class _Hinge:
    """Hinge GAN."""
    def d(self, pred_real, pred_fake):
        pr, pf = pred_real.float(), pred_fake.float()
        return (F.relu(1.0 - pr).mean() + F.relu(1.0 + pf).mean())

    def g(self, pred_fake):
        return -pred_fake.float().mean()


def build_gan_loss_fns(name: str = "lsgan"):
    name = name.lower()
    if name == "lsgan":
        return _LSGAN()
    if name == "bce":
        return _BCE()
    if name == "hinge":
        return _Hinge()
    raise ValueError(f"unknown gan_loss {name!r}; expected lsgan|bce|hinge")
