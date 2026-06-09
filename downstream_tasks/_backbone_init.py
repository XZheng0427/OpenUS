"""Shared OpenUS-checkpoint loader for VSSM-based downstream tasks.

Factored out of `eval_landmark.py:_load_openus` so that new downstream tasks
(image enhancement, future evaluators) can call the same loader without
importing it across top-level eval scripts.

The function strips the DINO `backbone.` / `module.backbone.` prefix from
the per-key state dict before calling `load_state_dict(strict=False)`, then
asserts that no `patch_embed.*` / `layers.*` keys ended up missing (that
would mean the prefix-strip silently produced an empty state dict).
"""

import torch
import torch.nn as nn


def load_openus_backbone(
    backbone: nn.Module,
    ckpt_path: str,
    key: str = "teacher",
) -> None:
    """Load an OpenUS DINO checkpoint into `backbone` in place.

    Args:
        backbone: a `Backbone_DINOv2_VSSM_2` (or compatible) instance.
        ckpt_path: path to the OpenUS DINO checkpoint (.pth).
        key: which sub-dict to read from the checkpoint ("teacher" or "student").

    Raises:
        SystemExit if the checkpoint is missing `key`, or if the prefix-strip
        leaves the encoder backbone keys missing.
    """
    print(f"loading OpenUS weights from {ckpt_path} (key={key!r})")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if key not in ckpt:
        raise SystemExit(f"checkpoint missing key {key!r}; have {list(ckpt)}")
    raw = ckpt[key]
    sd = {}
    for k, v in raw.items():
        if k.startswith("backbone."):
            sd[k[len("backbone."):]] = v
        elif k.startswith("module.backbone."):
            sd[k[len("module.backbone."):]] = v
    if not sd:
        raise SystemExit(
            f"no 'backbone.*' keys after prefix-strip in {ckpt_path}; "
            f"first 3 raw keys: {list(raw)[:3]}"
        )
    msg = backbone.load_state_dict(sd, strict=False)
    bad = [k for k in msg.missing_keys
           if k.startswith("patch_embed.") or k.startswith("layers.")]
    if bad:
        raise SystemExit(
            f"OpenUS load left encoder backbone keys missing — prefix-strip wrong. "
            f"first 5: {bad[:5]}"
        )
    print(f"  raw keys: {len(raw)}, after strip: {len(sd)}")
    print(f"  missing_keys ({len(msg.missing_keys)}): "
          f"{msg.missing_keys[:5]}{'...' if len(msg.missing_keys) > 5 else ''}")
    print(f"  unexpected_keys ({len(msg.unexpected_keys)}): "
          f"{msg.unexpected_keys[:5]}{'...' if len(msg.unexpected_keys) > 5 else ''}")
