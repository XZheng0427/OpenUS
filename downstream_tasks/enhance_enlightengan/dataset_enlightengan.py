"""Unpaired + test datasets for the USenhance-adapted EnlightenGAN pipeline.
"""

import json
import os
import random
from typing import Callable, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms_enlightengan import attention_for


def make_worker_init_fn(base_seed: int) -> Callable:
    def _init(worker_id: int):
        s = base_seed + worker_id
        random.seed(s)
        np.random.seed(s % (2 ** 32 - 1))
        torch.manual_seed(s)
    return _init


class EnlightenGANUnpairedDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        data_root: str,
        image_transform: Callable,
    ):
        with open(manifest_path) as f:
            all_records = json.load(f)
        train = [r for r in all_records if r.get("split") == "train"]
        if not train:
            raise ValueError(f"{manifest_path} has no train records")
        self.lq_paths: List[str] = [r["lq"] for r in train]
        self.hq_paths: List[str] = [r["hq"] for r in train]
        self.data_root = data_root
        self.transform = image_transform
        self.n_b = len(self.hq_paths)

    def __len__(self) -> int:
        return len(self.lq_paths)

    def __getitem__(self, idx: int):
        # Domain A: LQ by index.
        a_pil = Image.open(os.path.join(self.data_root, self.lq_paths[idx])).convert("RGB")
        A = self.transform(a_pil)                       # [-1,1]
        A_gray = attention_for(A)                       # [1,H,W]
        # Domain B: HQ uniformly at random (unpaired; breaks manifest pairing).
        b_idx = random.randrange(self.n_b)
        b_pil = Image.open(os.path.join(self.data_root, self.hq_paths[b_idx])).convert("RGB")
        B = self.transform(b_pil)                       # [-1,1]
        return A, A_gray, B


class EnlightenGANTestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        data_root: str,
        image_transform: Callable,
    ):
        with open(manifest_path) as f:
            all_records = json.load(f)
        self.records = [r for r in all_records if r.get("split") == "test"]
        if not self.records:
            raise ValueError(f"{manifest_path} has no test records")
        self.data_root = data_root
        self.transform = image_transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        a_pil = Image.open(os.path.join(self.data_root, rec["lq"])).convert("RGB")
        A = self.transform(a_pil)                       # [-1,1]
        A_gray = attention_for(A)
        stem = os.path.splitext(os.path.basename(rec["lq"]))[0]
        meta = {
            "organ":   rec["organ"],
            "lq_path": rec["lq"],
            "hq_path": rec["hq"],
            "stem":    stem,
            "split":   "test",
        }
        return A, A_gray, meta


def unpaired_collate(batch):
    A = torch.stack([b[0] for b in batch], dim=0)
    A_gray = torch.stack([b[1] for b in batch], dim=0)
    B = torch.stack([b[2] for b in batch], dim=0)
    return A, A_gray, B


def test_collate(batch):
    A = torch.stack([b[0] for b in batch], dim=0)
    A_gray = torch.stack([b[1] for b in batch], dim=0)
    metas = [b[2] for b in batch]
    return A, A_gray, metas
