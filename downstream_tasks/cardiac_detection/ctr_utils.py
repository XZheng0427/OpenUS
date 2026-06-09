"""Cardiothoracic-ratio (CTR) helpers.
"""

import math

import numpy as np


# ---------- CTR -------------------------------------------------------------

def compute_ctr(thorax_rbox, heart_rbox):
    """CTR from two rotated boxes.

    Args:
        thorax_rbox: array-like ``(cx, cy, w, h, theta)``.
        heart_rbox:  array-like ``(cx, cy, w, h, theta)``.

    Returns:
        float: ``min(w_heart, h_heart) / min(w_thorax, h_thorax)``.
    """
    th = np.asarray(thorax_rbox, dtype=np.float64)
    he = np.asarray(heart_rbox, dtype=np.float64)
    short_thorax = min(th[2], th[3])
    short_heart  = min(he[2], he[3])
    if short_thorax <= 0:
        return float("nan")
    return float(short_heart / short_thorax)


def ctr_from_ellipses(thorax_ell, heart_ell):
    """CTR from two FOCUS ellipses ``(cx, cy, a, b, theta_deg)``.

    Uses the ellipse short DIAMETER (``2 * min(a, b)``); CTR is dimensionless
    so the factor of 2 cancels, but we keep it explicit for clarity.
    """
    th = np.asarray(thorax_ell, dtype=np.float64)
    he = np.asarray(heart_ell, dtype=np.float64)
    short_thorax = 2.0 * min(th[2], th[3])
    short_heart  = 2.0 * min(he[2], he[3])
    if short_thorax <= 0:
        return float("nan")
    return float(short_heart / short_thorax)


# ---------- rotated-box inverse-transform -----------------------------------

def inverse_transform_rbox(rbox, scale_factor, pad_shape, ori_shape):
    """Map an rbox from network-input coords back to original-image coords.

    Matches the mmrotate keep-ratio resize + letterbox-pad pipeline used in
    the FOCUS config: image is first resized by ``scale_factor`` (the same
    factor for x and y under keep_ratio=True), then padded up to
    ``pad_shape`` on the right/bottom only. Inversion is therefore:

        (cx, cy)  ->  (cx, cy) / scale_factor   # padding is on the bottom-right,
                                                 # so it never shifts the
                                                 # top-left origin
        (w,  h)   ->  (w,  h)  / scale_factor   # uniform scale -> w, h shrink
                                                 # by the same factor
        theta     ->  theta                     # rotation is invariant under
                                                 # uniform scale + translation

    Args:
        rbox:          ``(cx, cy, w, h, theta)`` in network-input space.
        scale_factor:  scalar or 2-tuple ``(sx, sy)``. Under keep_ratio=True
                       these are equal; we average defensively if they differ
                       by < 1e-3 and otherwise raise.
        pad_shape:     ``(H_pad, W_pad)`` of the padded canvas (unused for the
                       inversion itself; kept in the signature so callers can
                       sanity-check that ``cx, cy`` fall inside the un-padded
                       region).
        ori_shape:     ``(H_ori, W_ori)`` of the original image (used only for
                       clipping `cx, cy` into the original frame).

    Returns:
        np.ndarray shape (5,) — the rbox in original-image space.
    """
    rbox = np.asarray(rbox, dtype=np.float64).reshape(5)
    if np.isscalar(scale_factor):
        sx = sy = float(scale_factor)
    else:
        sf = np.asarray(scale_factor, dtype=np.float64).reshape(-1)
        if sf.size == 1:
            sx = sy = float(sf[0])
        else:
            sx, sy = float(sf[0]), float(sf[1])
            if abs(sx - sy) > 1e-3:
                raise ValueError(
                    f"inverse_transform_rbox expects keep_ratio scale, got "
                    f"sx={sx} sy={sy}"
                )
    s = 0.5 * (sx + sy)

    cx, cy, w, h, theta = rbox
    out = np.array([cx / s, cy / s, w / s, h / s, theta], dtype=np.float64)

    H_ori, W_ori = float(ori_shape[0]), float(ori_shape[1])
    out[0] = float(np.clip(out[0], 0.0, W_ori))
    out[1] = float(np.clip(out[1], 0.0, H_ori))
    return out


# ---------- aggregation -----------------------------------------------------

def aggregate_ctr_metrics(per_image_records, tolerances=(0.03, 0.05, 0.10)):
    """Aggregate per-image CTR records into the headline metrics.

    Each record is a dict ``{"ctr_pred": float-or-None, "ctr_gt": float}``.
    ``ctr_pred`` is None when either the heart or the thorax detection was
    missing for that image (per review finding #9).

    Returns dict with:
        ctr_n_total:       int
        ctr_n_missing:     int
        ctr_missing_rate:  float  (n_missing / n_total)
        ctr_valid_rate:    float  (1 - ctr_missing_rate)
        ctr_mae_valid:     float  mean |ctr_pred - ctr_gt| over valid images;
                                   NaN if no valid images.
        ctr_acc_<eps>:     float  fraction of ALL images (including missing
                                   counted as failures) with
                                   |ctr_pred - ctr_gt| <= eps.
    """
    n = len(per_image_records)
    if n == 0:
        out = {"ctr_n_total": 0, "ctr_n_missing": 0,
               "ctr_missing_rate": float("nan"),
               "ctr_valid_rate":   float("nan"),
               "ctr_mae_valid":    float("nan")}
        for eps in tolerances:
            out[f"ctr_acc_{eps:g}"] = float("nan")
        return out

    n_missing = 0
    abs_errs = []
    correct_by_eps = {eps: 0 for eps in tolerances}

    for rec in per_image_records:
        gt = float(rec["ctr_gt"])
        pred = rec.get("ctr_pred")
        if pred is None or (isinstance(pred, float) and math.isnan(pred)):
            n_missing += 1
            continue
        err = abs(float(pred) - gt)
        abs_errs.append(err)
        for eps in tolerances:
            if err <= eps:
                correct_by_eps[eps] += 1

    n_valid = n - n_missing
    out = {
        "ctr_n_total":      n,
        "ctr_n_missing":    n_missing,
        "ctr_missing_rate": n_missing / n,
        "ctr_valid_rate":   n_valid / n,
        "ctr_mae_valid":    float(np.mean(abs_errs)) if abs_errs else float("nan"),
    }
    for eps in tolerances:
        out[f"ctr_acc_{eps:g}"] = correct_by_eps[eps] / n
    return out
