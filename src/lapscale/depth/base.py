"""Depth stage contract (stage 2).

A backend turns one undistorted BGR frame into a depth map in ITS OWN units: the scale stage
(stage 4) maps it to millimetres with the instrument. Backends that also estimate the focal
length report it, which the calibration fallback (``lapscale calib moge``) relies on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class DepthPrediction:
    depth: np.ndarray            # (H, W) float32, backend units (z along the optical axis)
    valid: np.ndarray            # (H, W) bool
    fx_px: float | None = None   # focal length estimated by the backend (None if it was given)
    fy_px: float | None = None
    metric: bool = False         # True if the backend claims metres; still rescaled downstream


class DepthBackend(Protocol):
    name: str
    output_kind: str            # "depth" (z, any scale) or "inverse_depth" (affine-invariant disparity)
    native_unit_mm: float | None  # millimetres per output unit if the backend claims metric depth, else None

    def predict(self, image_bgr: np.ndarray, fov_x_deg: float | None = None) -> DepthPrediction:
        """``fov_x_deg``: known horizontal FOV of the (undistorted) image, or None to let the
        backend estimate it."""
        ...
