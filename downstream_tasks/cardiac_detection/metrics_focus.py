"""Rotated-AP + CTR evaluator for the FOCUS cardiac-detection task.
"""

import json
import logging
import os
from typing import Optional, Sequence

import numpy as np

try:
    from mmrotate.registry import METRICS  # type: ignore
    from mmrotate.evaluation.metrics.dota_metric import DOTAMetric  # type: ignore
    _HAS_MMROTATE = True
except Exception:  # pragma: no cover
    METRICS = None
    DOTAMetric = object
    _HAS_MMROTATE = False

from .ctr_utils import compute_ctr, ctr_from_ellipses, inverse_transform_rbox


# ----------------------------------------------------------------------------
# CTR-only helper (works without mmrotate, used by unit tests)
# ----------------------------------------------------------------------------

def per_image_ctr_record(pred_rboxes_by_class, gt_ellipses_by_class, meta,
                          score_threshold=0.3):
    """Build one CTR record for the aggregate metric.

    Args:
        pred_rboxes_by_class:
            ``{"thorax": [(score, rbox), ...], "cardiac": [(score, rbox), ...]}``
            where each rbox is ``(cx, cy, w, h, theta)`` in network-input space.
        gt_ellipses_by_class:
            ``{"thorax": (cx, cy, a, b, theta_deg),
              "cardiac": (cx, cy, a, b, theta_deg)}``
            in original-image space.
        meta:
            dict with keys ``scale_factor``, ``pad_shape``, ``ori_shape``.
        score_threshold:
            ignore predictions with confidence below this value.

    Returns:
        dict with ``ctr_gt`` and ``ctr_pred`` (None on missing detection).
    """
    ctr_gt = ctr_from_ellipses(
        gt_ellipses_by_class["thorax"], gt_ellipses_by_class["cardiac"],
    )

    def _best(cls):
        cands = [(s, b) for s, b in pred_rboxes_by_class.get(cls, [])
                 if s >= score_threshold]
        if not cands:
            return None
        return max(cands, key=lambda sb: sb[0])[1]

    thorax_pred = _best("thorax")
    cardiac_pred = _best("cardiac")
    if thorax_pred is None or cardiac_pred is None:
        return {"ctr_gt": ctr_gt, "ctr_pred": None}

    thorax_rbox  = inverse_transform_rbox(
        thorax_pred,  meta["scale_factor"], meta["pad_shape"], meta["ori_shape"],
    )
    cardiac_rbox = inverse_transform_rbox(
        cardiac_pred, meta["scale_factor"], meta["pad_shape"], meta["ori_shape"],
    )
    ctr_pred = compute_ctr(thorax_rbox, cardiac_rbox)
    return {"ctr_gt": ctr_gt, "ctr_pred": ctr_pred}


# ----------------------------------------------------------------------------
# mmrotate-registered metric
# ----------------------------------------------------------------------------

if _HAS_MMROTATE:

    @METRICS.register_module()
    class FocusCTRMetric(DOTAMetric):
        """Rotated AP + CTR family on FOCUS.

        Inherits the rotated-AP path from ``DOTAMetric`` and appends CTR
        metrics computed per image. Metric names exposed to the runner:
            focus/AP50_thorax
            focus/AP50_cardiac
            focus/ap50_mean
            focus/ctr_mae_valid
            focus/ctr_missing_rate
            focus/ctr_valid_rate
            focus/ctr_acc_0.03
            focus/ctr_acc_0.05
            focus/ctr_acc_0.1
        """

        default_prefix = "focus"

        def __init__(
            self,
            meta_json_path: str = "",
            score_threshold: float = 0.3,
            tolerances: Sequence[float] = (0.03, 0.05, 0.10),
            iou_thrs: Sequence[float] = (0.5,),
            metric: str = "mAP",
            **kwargs,
        ):
            super().__init__(metric=metric, iou_thrs=list(iou_thrs), **kwargs)
            self.meta_json_path = meta_json_path
            self.score_threshold = float(score_threshold)
            self.tolerances = tuple(tolerances)
            self._meta_cache: Optional[dict] = None

        # --- meta loading ----------------------------------------------------

        def _load_meta(self):
            if self._meta_cache is not None:
                return self._meta_cache
            if not self.meta_json_path or not os.path.isfile(self.meta_json_path):
                logging.warning(
                    "[FocusCTRMetric] meta_json_path=%r not found -- CTR metrics "
                    "will be NaN. Set val_evaluator.meta_json_path in the config.",
                    self.meta_json_path,
                )
                self._meta_cache = {}
                return self._meta_cache
            with open(self.meta_json_path) as f:
                self._meta_cache = json.load(f)
            return self._meta_cache

        # --- per-batch hook --------------------------------------------------

        def process(self, data_batch, data_samples):
            """Defer AP collection to the parent class; cache CTR records here."""
            super().process(data_batch, data_samples)
            if not hasattr(self, "_ctr_records"):
                self._ctr_records = []
            meta = self._load_meta()
            if not meta:
                return

            classes = self.dataset_meta["classes"]
            cls_to_id = {c: i for i, c in enumerate(classes)}

            for sample in data_samples:
                img_path = sample.get("img_path", "")
                stem = os.path.splitext(os.path.basename(img_path))[0]
                m = meta.get(stem)
                if m is None:
                    continue

                # Build pred dict: class_name -> [(score, rbox), ...].
                pred_inst = sample["pred_instances"]
                bboxes = pred_inst["bboxes"]
                scores = pred_inst["scores"]
                labels = pred_inst["labels"]
                if hasattr(bboxes, "tensor"):
                    bboxes = bboxes.tensor
                bboxes = bboxes.detach().cpu().numpy()
                scores = scores.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()

                pred_by_class = {c: [] for c in classes}
                for s, b, l in zip(scores, bboxes, labels):
                    if 0 <= int(l) < len(classes):
                        pred_by_class[classes[int(l)]].append(
                            (float(s), b.astype(np.float64).tolist())
                        )

                # GT ellipses in original-image space.
                ellipses = {e["class"]: (
                    e["cx"], e["cy"], e["a"], e["b"], e["theta_deg"],
                ) for e in m["ellipses"]}
                if "thorax" not in ellipses or "cardiac" not in ellipses:
                    continue

                # mmrotate keep-ratio Resize sets ``scale_factor`` as
                # ``(scale_w, scale_h)`` in img_shape direction.
                meta_for_ctr = dict(
                    scale_factor=sample.get("scale_factor", (1.0, 1.0)),
                    pad_shape=sample.get("pad_shape",
                                          (m["orig_h"], m["orig_w"])),
                    ori_shape=sample.get("ori_shape",
                                          (m["orig_h"], m["orig_w"])),
                )
                rec = per_image_ctr_record(
                    pred_by_class, ellipses, meta_for_ctr,
                    score_threshold=self.score_threshold,
                )
                self._ctr_records.append(rec)

        # --- final aggregation -----------------------------------------------

        def compute_metrics(self, results):
            ap_metrics = super().compute_metrics(results)

            ctr_recs = getattr(self, "_ctr_records", [])
            self._ctr_records = []                 # reset for next .evaluate()

            from .ctr_utils import aggregate_ctr_metrics
            ctr_metrics = aggregate_ctr_metrics(ctr_recs, self.tolerances)

            # --- per-class AP extraction --------------------------------------
            # mmrotate's DOTAMetric emits only the across-class mean
            # (``mAP`` / ``AP50``); it computes per-class APs internally via
            # eval_rbbox_map but discards them. Re-run the computation here
            # to attach ``AP50_<class>`` keys, which we need for the paper-
            # comparable AP_thorax / AP_cardiac headline.
            try:
                from mmrotate.evaluation.functional import eval_rbbox_map  # type: ignore
                # Replay the parent's (ann, result) tuple unpack:
                # DOTAMetric.process appends ``(ann, result)`` to self.results,
                # so the first element is the GT and the second is the
                # prediction dict that owns ``pred_bbox_scores``.
                gts, preds = zip(*results)
                dataset_classes = self.dataset_meta["classes"]
                dets = [pred["pred_bbox_scores"] for pred in preds]
                # Call eval_rbbox_map at EVERY IoU threshold in self.iou_thrs.
                # The parent DOTAMetric only emits the across-class mean per
                # IoU; we re-run to also keep the per-class breakdown
                # (AP50_thorax, AP75_thorax, AP50_cardiac, ...).
                per_class_apx = {}        # iou_thr -> {class -> AP}
                for iou_thr in self.iou_thrs:
                    _, per_class = eval_rbbox_map(
                        dets, gts,
                        iou_thr=float(iou_thr),
                        use_07_metric=getattr(self, "use_07_metric", False),
                        box_type=self.predict_box_type,
                        dataset=dataset_classes,
                        logger="silent",
                    )
                    iou_tag = f"AP{int(round(iou_thr * 100)):02d}"
                    per_class_apx[iou_thr] = {}
                    for cls_name, cls_eval in zip(dataset_classes, per_class):
                        cls_ap = cls_eval.get("ap", 0.0)
                        if hasattr(cls_ap, "item"):
                            cls_ap = float(cls_ap.item())
                        else:
                            cls_ap = float(cls_ap)
                        ap_metrics[f"{iou_tag}_{cls_name}"] = cls_ap
                        per_class_apx[iou_thr][cls_name] = cls_ap
                # ap50_mean for backward compat (mean of per-class AP50).
                if 0.5 in [float(x) for x in self.iou_thrs] and per_class_apx:
                    ap_metrics["ap50_mean"] = float(np.mean(
                        [per_class_apx[0.5][c] for c in dataset_classes]
                    ))
                # COCO-style per-class mAP@[0.5:0.95] when the full sweep was run.
                if len(self.iou_thrs) > 1:
                    for cls_name in dataset_classes:
                        cls_means = [per_class_apx[t][cls_name] for t in self.iou_thrs]
                        ap_metrics[f"mAP_{cls_name}"] = float(np.mean(cls_means))
            except Exception as e:  # pragma: no cover - defensive
                logging.warning(
                    "[FocusCTRMetric] per-class AP extraction failed: %s. "
                    "Continuing with mean-only AP keys.", e,
                )

            # --- merge --------------------------------------------------------
            merged = {}
            for k, v in ap_metrics.items():
                merged[str(k)] = float(v) if not isinstance(v, str) else v
            for k, v in ctr_metrics.items():
                if isinstance(v, (int, float)):
                    merged[k] = float(v)
                else:
                    merged[k] = v
            return merged

else:  # pragma: no cover

    class FocusCTRMetric:  # type: ignore[no-redef]
        """Placeholder for environments without mmrotate."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "FocusCTRMetric requires mmrotate. Install via "
                "`mim install \"mmrotate>=1.0.0rc1\"` inside conda env "
                "`openus-mmrot`."
            )
