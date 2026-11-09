"""Detection of the visible optical disc of a laparoscope.

Laparoscopic frames usually show a bright disc on a black background, often clipped by the top and
bottom of a 16:9 frame. Everything outside the disc must be ignored by every stage (no corners, no
depth, no instrument edges there).

Method: per-pixel maximum over many frames (the disc is bright in at least some frame, the rim never
is), threshold, keep the largest component, then fit a circle by least squares to its boundary
points that do NOT lie on the image border (the clipped parts are not on the circle).
"""
from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from ..camera import OpticalCircle


def brightness_envelope(frames: Iterable[np.ndarray]) -> np.ndarray:
    env = None
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
        env = g.copy() if env is None else np.maximum(env, g)
    if env is None:
        raise ValueError("no frames")
    return env


def fit_circle(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Algebraic (Kasa) least-squares circle: x^2 + y^2 + D x + E y + F = 0."""
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x**2 + y**2)
    (D, E, F), *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = -D / 2, -E / 2
    return float(cx), float(cy), float(np.sqrt(max(cx**2 + cy**2 - F, 0.0)))


def detect_optical_circle(frames: Iterable[np.ndarray], dark_threshold: float | None = None,
                          border_px: int = 3, min_dark_fraction: float = 0.03,
                          margin_px: float = 2.0) -> tuple[OpticalCircle | None, dict]:
    """Return (circle or None, diagnostics). None means "no black rim": the whole frame is valid."""
    env = cv2.GaussianBlur(brightness_envelope(frames), (5, 5), 0)
    h, w = env.shape
    thr = dark_threshold if dark_threshold is not None else max(
        15.0, float(cv2.threshold(env, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]) * 0.5)
    bright = (env > thr).astype(np.uint8)
    diag = dict(threshold=round(thr, 1), dark_fraction=round(float(1 - bright.mean()), 4))
    if diag["dark_fraction"] < min_dark_fraction:
        diag["result"] = "no dark rim: full frame valid"
        return None, diag

    n, lab, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    if n < 2:
        diag["result"] = "no bright region"
        return None, diag
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (lab == k).astype(np.uint8)
    comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = max(contours, key=len).reshape(-1, 2).astype(np.float64)
    inner = ((pts[:, 0] > border_px) & (pts[:, 0] < w - 1 - border_px)
             & (pts[:, 1] > border_px) & (pts[:, 1] < h - 1 - border_px))
    pts = pts[inner]
    if len(pts) < 50:
        diag["result"] = "bright region touches the borders almost everywhere: full frame valid"
        return None, diag

    # robust: two passes, dropping boundary points far from the first fit (vignetting notches, text)
    cx, cy, r = fit_circle(pts[:, 0], pts[:, 1])
    res = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r
    keep = np.abs(res) < max(3.0, 3 * np.median(np.abs(res)))
    cx, cy, r = fit_circle(pts[keep, 0], pts[keep, 1])
    res = np.hypot(pts[keep, 0] - cx, pts[keep, 1] - cy) - r
    diag.update(result="circle", n_boundary_points=int(keep.sum()),
                fit_rms_px=round(float(np.sqrt(np.mean(res**2))), 3),
                arc_fraction=round(_arc_fraction(pts[keep], cx, cy), 3))
    return OpticalCircle(cx, cy, r - margin_px), diag


def _arc_fraction(pts: np.ndarray, cx: float, cy: float, bins: int = 72) -> float:
    """Fraction of the circumference supported by boundary points (low = clipped, less reliable fit)."""
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    hist, _ = np.histogram(ang, bins=bins, range=(-np.pi, np.pi))
    return float((hist > 0).mean())
