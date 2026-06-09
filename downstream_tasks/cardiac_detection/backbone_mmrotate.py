"""OpenUS backbone registered in mmrotate's MODELS registry.
"""

import logging
import os
from typing import Optional, Sequence

import torch
import torch.nn as nn


# Register in BOTH ``mmrotate.registry.MODELS`` AND ``mmdet.registry.MODELS``.
#   - The outer Runner build uses ``mmrotate.MODELS`` (default_scope="mmrotate")
#   - The inner ``mmdet.FasterRCNN.__init__`` does ``MODELS.build(backbone)``
#     with ``MODELS = mmdet.registry.MODELS`` (hardcoded in
#     ``mmdet/models/detectors/two_stage.py``).
# Registering in only one of the two leaves a gap depending on which path
# triggers the lookup.
try:
    from mmrotate.registry import MODELS  # type: ignore
    from mmdet.registry import MODELS as MMDET_MODELS  # type: ignore
    _HAS_MMDET = True
except Exception:  # pragma: no cover -- exercised only in mmrotate-free envs
    MODELS = None
    MMDET_MODELS = None
    _HAS_MMDET = False


from vmamba_models.dino_vmamba import Backbone_DINOv2_VSSM_2
from downstream_tasks._backbone_init import load_openus_backbone


# ---------- core module -----------------------------------------------------

class _OpenUSVMambaImpl(nn.Module):
    """Plain torch implementation of the OpenUS-VMamba wrapper.

    Kept as a separate class so it is importable in test environments where
    mmrotate / mmdet are not installed. The mmrotate-registered subclass below
    only adds the ``@MODELS.register_module()`` decoration.
    """

    #: channel counts for vmamba_small, exposed for FPN config wiring.
    OUT_CHANNELS = (96, 192, 384, 768)

    #: feature-map strides, exposed for FPN config wiring.
    STRIDES = (4, 8, 16, 32)

    def __init__(
        self,
        arch: str = "vmamba_small",
        out_indices: Sequence[int] = (0, 1, 2, 3),
        vmamba_imagenet_ckpt: Optional[str] = None,
        openus_ckpt: Optional[str] = None,
        openus_key: str = "teacher",
        frozen_stages: int = -1,
        freeze_encoder: bool = False,
        init_cfg=None,        # accepted for API compatibility; unused
    ):
        super().__init__()
        if arch != "vmamba_small":
            raise NotImplementedError(
                f"only vmamba_small wired; got arch={arch!r}"
            )
        self.out_indices = tuple(out_indices)
        self.vmamba_imagenet_ckpt = vmamba_imagenet_ckpt
        self.openus_ckpt = openus_ckpt
        self.openus_key = openus_key
        self.frozen_stages = frozen_stages
        # When True, **every** encoder parameter has ``requires_grad=False``
        # and the encoder stays in ``.eval()`` mode during training (so BN/LN
        # running stats are frozen too). This is the fair-comparison setting
        # against the EchoCare variant.
        self.freeze_encoder = bool(freeze_encoder)
        self._weights_initialised = False

        # Underlying backbone. Passing ``pretrained=<path>`` loads ImageNet
        # VMamba weights inside the constructor; ``pretrained=None`` skips
        # that and leaves the random init in place.
        self.backbone = Backbone_DINOv2_VSSM_2(
            pretrained=vmamba_imagenet_ckpt,
            out_indices=tuple(out_indices),
        )

    # --- mmdet/mmengine hooks ------------------------------------------------

    def init_weights(self):
        """Load OpenUS weights once.

        mmdet/mmengine calls this exactly once during ``Runner`` setup,
        AFTER ``__init__`` (which already loaded the ImageNet vmamba init).
        Re-entrant calls are no-ops so resuming training does not re-load.
        """
        if self._weights_initialised:
            return
        self._weights_initialised = True

        if self.openus_ckpt and os.path.isfile(self.openus_ckpt):
            load_openus_backbone(
                self.backbone, self.openus_ckpt, key=self.openus_key,
            )
        elif self.openus_ckpt:
            logging.warning(
                "[OpenUSVMamba] openus_ckpt=%r not found; falling back to "
                "ImageNet-vmamba-only init", self.openus_ckpt,
            )
        else:
            logging.info(
                "[OpenUSVMamba] no openus_ckpt given; using "
                "ImageNet-vmamba init only (if provided) or random init."
            )

        self._apply_frozen_stages()

    def _apply_frozen_stages(self):
        if self.freeze_encoder:
            # Full freeze: every parameter of the underlying VMamba is
            # excluded from the optimizer. The output norms (``outnorm{i}``)
            # are reachable via ``self.backbone.parameters()`` so they are
            # frozen too.
            for p in self.backbone.parameters():
                p.requires_grad = False
            return
        if self.frozen_stages < 0:
            return
        # patch_embed always frozen when any stage is frozen (mmdet convention).
        for p in self.backbone.patch_embed.parameters():
            p.requires_grad = False
        for i in range(min(self.frozen_stages + 1, len(self.backbone.layers))):
            for p in self.backbone.layers[i].parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):  # noqa: D401  (mmdet convention)
        super().train(mode)
        if self.freeze_encoder:
            # Whole encoder stays in eval mode — keeps any BN/LN running
            # stats fixed at the pretraining values so the frozen features
            # are deterministic across training batches.
            self.backbone.eval()
        elif mode and self.frozen_stages >= 0:
            # Keep partially-frozen submodules in eval mode.
            self.backbone.patch_embed.eval()
            for i in range(min(self.frozen_stages + 1, len(self.backbone.layers))):
                self.backbone.layers[i].eval()
        return self

    # --- forward ------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """Return a tuple of 4 spatial feature maps ``(P2, P3, P4, P5)``.

        Replicates ``downstream_tasks.landmark.backbone_wrapper.VSSMFeatureExtractor``
        — we cannot call ``self.backbone(x)`` directly because that returns
        the concatenated CLS+patch tokens used by DINO training.
        """
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
        return tuple(outs)


# ---------- mmrotate-registered subclass ------------------------------------

if _HAS_MMDET:

    @MMDET_MODELS.register_module()
    @MODELS.register_module()
    class OpenUSVMamba(_OpenUSVMambaImpl):
        """OpenUS-VMamba backbone for mmrotate detectors.

        Constructor kwargs (all optional except as noted):
            arch:                  ``"vmamba_small"`` (only choice for now).
            out_indices:           which VSSM stages to expose (default all 4).
            vmamba_imagenet_ckpt:  ImageNet-pretrained VMamba .pth path.
            openus_ckpt:           OpenUS DINO .pth checkpoint path.
            openus_key:            ``"teacher"`` (default) or ``"student"``.
            frozen_stages:         -1 for no freeze (default), or 0..3 to
                                   freeze patch_embed + stages 0..N.
            init_cfg:              ignored (mmdet API compatibility only).
        """
        pass

else:  # pragma: no cover -- exercised only in mmrotate-free envs
    # Expose the unregistered class under the canonical name so unit tests
    # can construct it without importing mmrotate.
    OpenUSVMamba = _OpenUSVMambaImpl
