"""CAMUS LVEF dataset.
"""

import json
import os
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset


FRAME_KEYS = ("ch2_ed", "ch2_es", "ch4_ed", "ch4_es")


class CAMUSLVEFDataset(Dataset):
    """One sample = one patient = 4 frames + 1 scalar EF.

    Args:
        manifest_path: path to camus_lvef_manifest_seed{seed}.json.
        images_root:   directory whose relative joins with ``record[key]``
                       resolve to a readable PNG. Typically the CAMUS_2 root.
        split:         one of {'train', 'val', 'test'}.
        transform:     callable (PIL_image_list) -> tensor [4, 3, H, W].
                       Receives all 4 frames at once so train-time augs
                       (flip / jitter) can apply the same random params
                       across the 4 views of a single patient.
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
        frames = [
            Image.open(os.path.join(self.images_root, rec[k])).convert("RGB")
            for k in FRAME_KEYS
        ]
        frames_t = self.transform(frames)  # [4, 3, H, W]
        ef = torch.tensor(float(rec["ef"]), dtype=torch.float32)

        meta = {
            "patient": rec["patient"],
            "ef_orig": float(rec["ef"]),
        }
        return frames_t, ef, meta


def lvef_collate(batch):
    """Stack frames and EF; keep meta as a list of dicts.

    DataLoader's default_collate would coerce the per-sample meta string
    fields into tuples; we keep them as a list-of-dicts so per-patient CSV
    dumps at eval time are straightforward.
    """
    frames = torch.stack([b[0] for b in batch], dim=0)   # [B, 4, 3, H, W]
    efs    = torch.stack([b[1] for b in batch], dim=0)   # [B]
    metas  = [b[2] for b in batch]
    return frames, efs, {
        "patient": [m["patient"] for m in metas],
        "ef_orig": [m["ef_orig"] for m in metas],
    }
