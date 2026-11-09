"""Depth Anything 3 backend (ByteDance-Seed/Depth-Anything-3).

    git clone https://github.com/ByteDance-Seed/Depth-Anything-3 && pip install -e Depth-Anything-3

Checkpoints (Hugging Face, licences differ: DA3-SMALL/BASE, DA3MONO-LARGE, DA3METRIC-LARGE are Apache 2.0;
LARGE/GIANT/NESTED are CC BY-NC 4.0):
  depth-anything/DA3MONO-LARGE           relative monocular depth (z, unknown scale)
  depth-anything/DA3METRIC-LARGE         metric: metres = focal_px * output / 300 (focal of the processed image)
  depth-anything/DA3NESTED-GIANT-LARGE-1.1  metric, already in metres
  depth-anything/DA3-LARGE-1.1, -GIANT-1.1, -BASE, -SMALL  any-view models, relative depth
The network works at ``process_res`` (longest side, default 504); the map is resized back to the frame.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .base import DepthPrediction


class DA3Backend:
    output_kind = "depth"

    def __init__(self, checkpoint: str = "depth-anything/DA3MONO-LARGE", device: str = "cuda",
                 process_res: int = 504):
        try:
            import torch
            from depth_anything_3.api import DepthAnything3
            from PIL import Image
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError(f"Depth Anything 3 not available ({e}); see the module docstring") from e
        self.torch, self.Image = torch, Image
        self.model = DepthAnything3.from_pretrained(checkpoint).to(torch.device(device)).eval()
        self.process_res = process_res
        up = checkpoint.upper()
        self.kind = "metric_focal" if "METRIC" in up else "metres" if "NESTED" in up else "relative"
        self.native_unit_mm = None if self.kind == "relative" else 1000.0
        self.name = f"da3:{checkpoint.split('/')[-1]}"

    def predict(self, image_bgr: np.ndarray, fov_x_deg: float | None = None) -> DepthPrediction:  # pragma: no cover
        h, w = image_bgr.shape[:2]
        img = self.Image.fromarray(np.ascontiguousarray(image_bgr[:, :, ::-1]))
        with self.torch.inference_mode():
            pred = self.model.inference([img], process_res=self.process_res)
        d = np.asarray(pred.depth[0], np.float32)
        hp, wp = d.shape
        if self.kind == "metric_focal":
            if fov_x_deg is not None:
                f_proc = (wp / 2) / math.tan(math.radians(fov_x_deg) / 2)
            elif getattr(pred, "intrinsics", None) is not None:
                K = np.asarray(pred.intrinsics[0])
                f_proc = 0.5 * (K[0, 0] + K[1, 1])
            else:
                raise RuntimeError("DA3METRIC needs the focal length: give a calibrated camera")
            d = f_proc * d / 300.0                                  # metres (DA3 FAQ)
        depth = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
        valid = np.isfinite(depth) & (depth > 0)
        conf = getattr(pred, "conf", None)
        if conf is not None:
            c = cv2.resize(np.asarray(conf[0], np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            valid &= c > 0
        fx = None
        if fov_x_deg is None and getattr(pred, "intrinsics", None) is not None:
            K = np.asarray(pred.intrinsics[0])
            fx = float(K[0, 0] * w / wp)
        return DepthPrediction(depth, valid, fx, fx, metric=self.kind != "relative")
