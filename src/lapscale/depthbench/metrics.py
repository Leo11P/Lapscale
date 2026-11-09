"""Depth metrics (standard monocular-depth set, in mm) under three alignments of the prediction:

* ``native``: the backend's own metric scale (only if it claims one): is its metric output usable?
* ``scale``: one scale per frame (median ratio, in the backend's output space): quality of the SHAPE,
  what remains once a correct scale is known;
* ``affine``: scale and shift per frame (least squares): the usual protocol for affine-invariant models.

For backends that predict inverse depth, alignment is done in inverse space. Errors are also reported per
ground-truth distance band, because laparoscopic depth spans 2-20 cm and relative errors are not uniform.
"""
from __future__ import annotations

import numpy as np

ALIGNMENTS = ("native", "scale", "affine")
BANDS_MM = ((0, 50), (50, 100), (100, 150), (150, np.inf))


def align(pred: np.ndarray, gt_mm: np.ndarray, mask: np.ndarray, mode: str, kind: str = "depth",
          native_unit_mm: float | None = None) -> tuple[np.ndarray | None, dict]:
    """Prediction (backend units) -> mm under an alignment; returns (pred_mm, parameters)."""
    p, g = pred[mask].astype(np.float64), gt_mm[mask].astype(np.float64)
    ok = np.isfinite(p) & (p > 0)
    p, g = p[ok], g[ok]
    if len(p) < 10:
        return None, {}
    if mode == "native":
        if native_unit_mm is None or kind != "depth":
            return None, {}
        return pred * native_unit_mm, dict(scale=native_unit_mm)
    t = g if kind == "depth" else 1.0 / g
    if mode == "scale":
        a, b = float(np.median(t / p)), 0.0
    elif mode == "affine":
        A = np.column_stack([p, np.ones_like(p)])
        (a, b), *_ = np.linalg.lstsq(A, t, rcond=None)
    else:
        raise ValueError(mode)
    v = a * pred + b
    if kind == "depth":
        return v, dict(scale=float(a), shift=float(b))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > 0, 1.0 / v, np.nan), dict(scale=float(a), shift=float(b))


def depth_metrics(pred_mm: np.ndarray, gt_mm: np.ndarray, mask: np.ndarray) -> dict:
    p, g = pred_mm[mask].astype(np.float64), gt_mm[mask].astype(np.float64)
    ok = np.isfinite(p) & (p > 0) & (g > 0)
    p, g = p[ok], g[ok]
    if len(p) == 0:
        return {}
    r = np.maximum(p / g, g / p)
    return dict(abs_rel=float(np.mean(np.abs(p - g) / g)), sq_rel=float(np.mean((p - g) ** 2 / g)),
                rmse_mm=float(np.sqrt(np.mean((p - g) ** 2))), mae_mm=float(np.mean(np.abs(p - g))),
                rmse_log=float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))),
                delta_1_05=float(np.mean(r < 1.05)), delta_1_25=float(np.mean(r < 1.25)),
                median_ratio=float(np.median(g / p)), n=int(len(p)))


def band_metrics(pred_mm: np.ndarray, gt_mm: np.ndarray, mask: np.ndarray, bands=BANDS_MM) -> dict:
    out = {}
    for lo, hi in bands:
        m = mask & (gt_mm >= lo) & (gt_mm < hi)
        if m.sum() >= 50:
            d = depth_metrics(pred_mm, gt_mm, m)
            name = f"{lo:g}-{hi:g}mm" if np.isfinite(hi) else f">{lo:g}mm"
            out[f"abs_rel_{name}"] = d.get("abs_rel")
    return out
