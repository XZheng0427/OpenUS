"""USenhance paired dataset and holdout (no-GT) dataset.
"""

import json
import os
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


class USenhancePairedDataset(Dataset):
    """Paired LQ/HQ ultrasound dataset for the USenhance Challenge 2023.

    Args:
        manifest_path: path to enhance_manifest_seed<seed>.json
        data_root:    path to OpenUS_datasets/image_enhancement (prepended to
                      each record's relative `lq`/`hq`).
        split:        'train' or 'test'. Records with other split values are
                      dropped.
        transform:    callable (lq_pil, hq_pil) -> (lq_tensor, hq_tensor).
    """

    def __init__(
        self,
        manifest_path: str,
        data_root: str,
        split: str,
        transform: Callable,
    ):
        if split not in ("train", "test"):
            raise ValueError(f"split must be train|test, got {split!r}")
        with open(manifest_path) as f:
            all_records = json.load(f)
        self.records = [r for r in all_records if r.get("split") == split]
        if not self.records:
            raise ValueError(
                f"manifest {manifest_path} has no records with split={split!r}"
            )
        self.data_root = data_root
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        lq = Image.open(os.path.join(self.data_root, rec["lq"])).convert("RGB")
        hq = Image.open(os.path.join(self.data_root, rec["hq"])).convert("RGB")
        lq_t, hq_t = self.transform(lq, hq)
        stem = os.path.splitext(os.path.basename(rec["lq"]))[0]
        meta = {
            "organ":   rec["organ"],
            "lq_path": rec["lq"],
            "hq_path": rec["hq"],
            "stem":    stem,
            "split":   self.split,
        }
        return lq_t, hq_t, meta


class USenhanceHoldoutDataset(Dataset):
    """The 364 unpaired challenge holdout images.

    Args:
        manifest_path: path to enhance_holdout.json
        data_root:    path to OpenUS_datasets/image_enhancement
        transform:    callable (lq_pil) -> lq_tensor.
    """

    def __init__(
        self,
        manifest_path: str,
        data_root: str,
        transform: Callable,
    ):
        with open(manifest_path) as f:
            self.records = json.load(f)
        if not self.records:
            raise ValueError(f"holdout manifest {manifest_path} is empty")
        self.data_root = data_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        lq = Image.open(os.path.join(self.data_root, rec["lq"])).convert("RGB")
        lq_t = self.transform(lq)
        stem = os.path.splitext(os.path.basename(rec["lq"]))[0]
        meta = {
            "lq_path": rec["lq"],
            "stem":    stem,
            "organ":   "holdout",
            "split":   "holdout",
        }
        return lq_t, meta


def enhance_collate(batch):
    """Stack LQ/HQ tensors; keep meta as a list of dicts."""
    lq = torch.stack([b[0] for b in batch], dim=0)
    hq = torch.stack([b[1] for b in batch], dim=0)
    metas = [b[2] for b in batch]
    return lq, hq, metas


def holdout_collate(batch):
    lq = torch.stack([b[0] for b in batch], dim=0)
    metas = [b[1] for b in batch]
    return lq, metas
