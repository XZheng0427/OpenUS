"""IQA metrics + DDP-safe image dumping for the image-enhancement task.
"""

import os
import shutil
import statistics
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image


def _is_dist():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _is_main():
    return (not _is_dist()) or dist.get_rank() == 0


class IQAEvaluator:
    """Per-image NIQE / BRISQUE / PIQE via pyiqa.

    Lazily imports pyiqa so unit tests / smoke tests that don't need IQA can
    run without the dependency. Inputs are expected as [B, 3, H, W] tensors
    in [0, 1].
    """

    METRIC_NAMES = ("niqe", "brisque", "piqe")

    def __init__(self, device, metric_names=METRIC_NAMES):
        try:
            import pyiqa  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "pyiqa is required for IQAEvaluator. "
                "Install with: pip install pyiqa"
            ) from e
        import pyiqa
        self.device = device
        self.metric_names = tuple(metric_names)
        self._metrics = {
            name: pyiqa.create_metric(name, device=device, as_loss=False)
            for name in self.metric_names
        }

    @torch.no_grad()
    def __call__(self, img_01: torch.Tensor) -> Dict[str, List[float]]:
        img_01 = img_01.to(self.device).float().clamp(0.0, 1.0)
        out: Dict[str, List[float]] = {}
        for name, fn in self._metrics.items():
            try:
                vals = fn(img_01).flatten().detach().cpu().tolist()
            except Exception as e:
                # Some pyiqa metrics need [0, 255] uint8-equivalent floats or
                # min-size constraints. Fall back to per-image evaluation with
                # the same input so we still get a list of length B.
                vals = []
                for i in range(img_01.shape[0]):
                    try:
                        v = fn(img_01[i:i + 1]).item()
                    except Exception:
                        v = float("nan")
                    vals.append(v)
            out[name] = vals
        return out


def aggregate_iqa(rows: List[Dict]) -> Dict:
    """Aggregate a list of per-image dicts into per-organ + overall means.

    Each row: {"organ": str, "niqe": float, "brisque": float, "piqe": float}
    Returns: {"overall": {"niqe": mean, ...}, "by_organ": {organ: {...}}}
    """
    by_organ: Dict[str, Dict[str, List[float]]] = {}
    for r in rows:
        organ = r.get("organ", "unknown")
        by_organ.setdefault(organ, {})
        for k, v in r.items():
            if k == "organ":
                continue
            if not isinstance(v, (int, float)):
                continue
            if v != v:  # NaN
                continue
            by_organ[organ].setdefault(k, []).append(float(v))

    def _stats(values):
        if not values:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {
            "mean": float(statistics.mean(values)),
            "std":  float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
            "n":    len(values),
        }

    out = {"by_organ": {}, "overall": {}}
    metric_keys = set()
    for organ, metrics in by_organ.items():
        out["by_organ"][organ] = {k: _stats(v) for k, v in metrics.items()}
        metric_keys.update(metrics.keys())

    for k in metric_keys:
        all_values = []
        for organ_metrics in by_organ.values():
            all_values.extend(organ_metrics.get(k, []))
        out["overall"][k] = _stats(all_values)
    return out


# ---------- DDP-safe image dumping ----------------------------------------

def tensor_to_uint8_png(img_01: torch.Tensor) -> np.ndarray:
    """Convert a single [3, H, W] tensor in [0, 1] to a [H, W, 3] uint8 array."""
    x = img_01.detach().cpu().clamp(0.0, 1.0).float().numpy()
    x = (x * 255.0 + 0.5).astype(np.uint8)
    return x.transpose(1, 2, 0)


def write_png(img_01: torch.Tensor, path: str) -> None:
    arr = tensor_to_uint8_png(img_01)
    Image.fromarray(arr, mode="RGB").save(path)


def rank_scoped_dir(output_dir: str, subdir: str) -> str:
    rank = dist.get_rank() if _is_dist() else 0
    d = os.path.join(output_dir, f"{subdir}_rank{rank}")
    os.makedirs(d, exist_ok=True)
    return d


def merge_rank_dirs(output_dir: str, subdir: str, world_size: Optional[int] = None) -> str:
    """Rank 0 only: move files from <subdir>_rank* into a flat <subdir>/.

    Returns the path of the final merged dir.
    """
    final_dir = os.path.join(output_dir, subdir)
    if not _is_main():
        return final_dir
    os.makedirs(final_dir, exist_ok=True)
    if world_size is None:
        world_size = dist.get_world_size() if _is_dist() else 1
    seen = set()
    for r in range(world_size):
        src = os.path.join(output_dir, f"{subdir}_rank{r}")
        if not os.path.isdir(src):
            continue
        for fname in os.listdir(src):
            if fname in seen:
                raise RuntimeError(
                    f"filename collision while merging rank dirs into "
                    f"{final_dir}: {fname!r} appears in rank>={r}"
                )
            seen.add(fname)
            shutil.move(os.path.join(src, fname), os.path.join(final_dir, fname))
        try:
            os.rmdir(src)
        except OSError:
            pass
    return final_dir


def compute_fid(enhanced_dir: str, reference_dir: str, **kwargs) -> float:
    """Rank-0-only FID via cleanfid. Other ranks return NaN."""
    if not _is_main():
        return float("nan")
    try:
        from cleanfid import fid
    except ImportError as e:
        raise ImportError(
            "cleanfid is required for FID. Install with: pip install cleanfid"
        ) from e
    return float(fid.compute_fid(enhanced_dir, reference_dir,
                                 mode=kwargs.pop("mode", "clean"), **kwargs))
