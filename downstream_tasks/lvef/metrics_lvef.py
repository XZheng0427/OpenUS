"""Regression metrics for LVEF.

MAE, RMSE, R² and per-patient standard-error of the MAE. The accumulator
keeps sufficient statistics (sum |e|, sum e², sum y, sum y², n) so a final
all-reduce is enough to produce DDP-consistent numbers — same approach as
``evaluate`` in ``eval_landmark_echocare.py``.
"""

from typing import Dict, List

import math
import torch
import torch.distributed as dist


class RegressionAccumulator:
    """DDP-safe accumulator for MAE / RMSE / R²."""

    __slots__ = ("sum_abs", "sum_sq", "sum_y", "sum_y2", "n",
                 "per_patient_abs", "per_patient_patient", "per_patient_pred", "per_patient_gt")

    def __init__(self, keep_per_patient: bool = False):
        self.sum_abs = 0.0
        self.sum_sq  = 0.0
        self.sum_y   = 0.0
        self.sum_y2  = 0.0
        self.n       = 0
        self.per_patient_abs:     List[float] = []
        self.per_patient_patient: List[str]   = []
        self.per_patient_pred:    List[float] = []
        self.per_patient_gt:      List[float] = []
        if not keep_per_patient:
            # mark slots as None so callers can introspect easily
            self.per_patient_abs = None
            self.per_patient_patient = None
            self.per_patient_pred = None
            self.per_patient_gt = None

    def update(self, pred: torch.Tensor, gt: torch.Tensor,
               patients: List[str] = None) -> None:
        pred = pred.detach().float().view(-1)
        gt   = gt.detach().float().view(-1)
        diff = pred - gt
        self.sum_abs += diff.abs().sum().item()
        self.sum_sq  += (diff ** 2).sum().item()
        self.sum_y   += gt.sum().item()
        self.sum_y2  += (gt ** 2).sum().item()
        self.n       += int(gt.numel())
        if self.per_patient_abs is not None:
            abs_vals  = diff.abs().tolist()
            pred_vals = pred.tolist()
            gt_vals   = gt.tolist()
            self.per_patient_abs.extend(abs_vals)
            self.per_patient_pred.extend(pred_vals)
            self.per_patient_gt.extend(gt_vals)
            if patients is None:
                self.per_patient_patient.extend([""] * len(abs_vals))
            else:
                self.per_patient_patient.extend(patients)

    def all_reduce(self) -> None:
        if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
            return
        t = torch.tensor(
            [self.sum_abs, self.sum_sq, self.sum_y, self.sum_y2, float(self.n)],
            dtype=torch.float64, device=torch.cuda.current_device(),
        )
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        self.sum_abs, self.sum_sq, self.sum_y, self.sum_y2 = t[:4].tolist()
        self.n = int(t[4].item())

    def metrics(self) -> Dict[str, float]:
        if self.n == 0:
            return {"mae": float("inf"), "rmse": float("inf"),
                    "r2": float("-inf"), "n": 0, "mae_stderr": float("inf")}
        mae  = self.sum_abs / self.n
        mse  = self.sum_sq  / self.n
        rmse = math.sqrt(mse)
        # R^2 = 1 - SS_res / SS_tot. SS_tot = sum (y - mean_y)^2 = sum_y2 - n * mean_y^2.
        mean_y = self.sum_y / self.n
        ss_tot = max(self.sum_y2 - self.n * mean_y * mean_y, 1e-12)
        r2 = 1.0 - self.sum_sq / ss_tot

        # Standard error of the per-patient MAE estimator.
        # Computed from the per-patient abs-errors when available; otherwise
        # fall back to a Gaussian approximation from the second moment.
        if self.per_patient_abs is not None and len(self.per_patient_abs) > 1:
            errs = self.per_patient_abs
            m = sum(errs) / len(errs)
            var = sum((e - m) ** 2 for e in errs) / (len(errs) - 1)
            mae_stderr = math.sqrt(var / len(errs))
        else:
            # variance of |e| ≈ E[e²] - (E[|e|])² (rough; uses raw moments)
            var = max(mse - mae * mae, 0.0)
            mae_stderr = math.sqrt(var / max(self.n, 1))

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "n": self.n,
            "mae_stderr": mae_stderr,
        }


def regression_metrics(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    """One-shot non-DDP convenience wrapper."""
    acc = RegressionAccumulator(keep_per_patient=True)
    acc.update(pred, gt)
    return acc.metrics()
