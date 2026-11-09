"""Calibration fallback: focal length estimated by a depth backend (MoGe) from surgical frames.

Use only when no calibration is possible. Limits, stated in the output file:
* no distortion model (laparoscopes usually have strong barrel distortion);
* principal point assumed at the image centre;
* f comes from a network trained on everyday scenes; in monocular metric reconstruction its
  error becomes the scale error, one to one.
The per-frame spread of the estimate is reported as a (lower bound of the) scale uncertainty.
"""
from __future__ import annotations

import numpy as np

from ..camera import Camera, OpticalCircle
from ..depth.base import DepthBackend
from ..io.video import FrameSource


def estimate_focal(source: FrameSource, backend: DepthBackend, indices: list[int],
                   circle: OpticalCircle | None = None, log=print) -> Camera:
    fx, fy = [], []
    for i, frame in source.iter(indices):
        if circle is not None:
            frame = frame * circle.mask(source.size)[..., None].astype(frame.dtype)
        p = backend.predict(frame, fov_x_deg=None)
        if p.fx_px and np.isfinite(p.fx_px):
            fx.append(p.fx_px)
            fy.append(p.fy_px if p.fy_px else p.fx_px)
            log(f"  frame {i}: fx {p.fx_px:.1f} px")
    if len(fx) < 3:
        raise RuntimeError(f"focal length estimated on only {len(fx)} frames: need at least 3")
    fx, fy = np.array(fx), np.array(fy)
    f = float(np.median(np.concatenate([fx, fy])))
    mad_pct = float(100 * 1.4826 * np.median(np.abs(fx - np.median(fx))) / np.median(fx))
    w, h = source.size
    K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]])
    q = dict(n_frames=len(fx), fx_median_px=round(float(np.median(fx)), 3),
             fy_median_px=round(float(np.median(fy)), 3), focal_spread_pct=round(mad_pct, 3),
             backend=getattr(backend, "name", type(backend).__name__),
             warnings=["focal length from a depth network, no distortion model, principal point at the "
                       "centre: scale error inherits the focal error one to one; calibrate when possible",
                       f"frame-to-frame spread of f: {mad_pct:.1f}% (a lower bound of the scale uncertainty)"])
    return Camera(K, np.zeros(0), (w, h), source="moge", optical_circle=circle, quality=q,
                  notes="fallback: focal length estimated by the depth backend")
