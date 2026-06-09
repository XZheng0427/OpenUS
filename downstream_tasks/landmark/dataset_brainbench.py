"""BrainBenchmark landmark dataset.
"""

import json
import os
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset


class BrainBenchmarkLandmarkDataset(Dataset):
    """Dataset for the BrainBenchmark 24-landmark task.

    Args:
        manifest_path: path to landmark_manifest.json (built by prepare_brainbench.py).
        images_root:   directory whose relative joins with ``record['image']`` resolve
                       to a readable image. Typically the BrainBenchmark root.
        split:         one of {'train', 'val', 'test'}.
        transform:     callable (PIL_image, coords_K2) ->
                       (image_tensor, heatmaps[K,H,W], coords_scaled[K,2], scale[2]).
    """

    def __init__(
        self,
        manifest_path: str,
        images_root: str,
        split: str,
        transform: Callable,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train|val|test, got {split!r}")

        with open(manifest_path) as f:
            all_records = json.load(f)

        self.records = [r for r in all_records if r["split"] == split]
        if not self.records:
            raise ValueError(
                f"manifest {manifest_path} has no records for split={split!r}"
            )

        self.images_root = images_root
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        image_path = os.path.join(self.images_root, rec["image"])
        image = Image.open(image_path).convert("RGB")

        orig_coords = torch.tensor(rec["landmarks"], dtype=torch.float32)  # [K, 2]
        if orig_coords.ndim != 2 or orig_coords.shape[1] != 2:
            raise ValueError(
                f"record {rec['subject']}: landmarks must be [K,2], got "
                f"shape {tuple(orig_coords.shape)}"
            )

        img_t, heatmaps, coords_scaled, scale = self.transform(image, orig_coords)

        meta = {
            "scale": scale,                       # tensor [2]
            "orig_coords": orig_coords,           # tensor [K, 2]
            "orig_hw": tuple(rec["image_hw"]),    # (H, W)
            "subject": rec["subject"],
            "image_path": rec["image"],
        }
        return img_t, heatmaps, coords_scaled, meta


def landmark_collate(batch):
    """Stack tensors; keep meta as a list of dicts.

    DataLoader's default_collate chokes on the (str, tuple) entries in meta.
    """
    imgs       = torch.stack([b[0] for b in batch], dim=0)
    heatmaps   = torch.stack([b[1] for b in batch], dim=0)
    coords     = torch.stack([b[2] for b in batch], dim=0)
    metas      = [b[3] for b in batch]
    # Convenience: stack the per-sample scale and orig_coords into batched tensors
    scale_b      = torch.stack([m["scale"]       for m in metas], dim=0)  # [B, 2]
    orig_coords  = torch.stack([m["orig_coords"] for m in metas], dim=0)  # [B, K, 2]
    return imgs, heatmaps, coords, {
        "scale": scale_b,
        "orig_coords": orig_coords,
        "orig_hw": [m["orig_hw"] for m in metas],
        "subject": [m["subject"] for m in metas],
        "image_path": [m["image_path"] for m in metas],
    }
