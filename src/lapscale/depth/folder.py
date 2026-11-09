"""Precomputed depth maps from a folder: any model run elsewhere (Depth Pro, DA2, an endoscopic
network ...), or ground truth to check the pipeline.

Files per frame id (``set_frame_id``): ``<id>.npy`` / ``<id>.npz`` (key ``depth``), or 16-bit
``<id>.png`` divided by ``png_scale``. Maps must have the size of the (undistorted) frame.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .base import DepthPrediction


class FolderBackend:
    def __init__(self, depth_dir: str | Path, output_kind: str = "depth", native_unit_mm: float | None = None,
                 png_scale: float = 1.0, name: str | None = None):
        self.dir = Path(depth_dir)
        if not self.dir.is_dir():
            raise FileNotFoundError(self.dir)
        if output_kind not in ("depth", "inverse_depth"):
            raise ValueError("output_kind must be depth or inverse_depth")
        self.output_kind, self.native_unit_mm, self.png_scale = output_kind, native_unit_mm, png_scale
        self.name = name or f"folder:{self.dir.name}"
        self._id: str | None = None

    def set_frame_id(self, frame_id: str) -> None:
        self._id = frame_id

    def _candidates(self):
        ids = [self._id]
        if self._id and self._id.rsplit("_", 1)[-1].isdigit():
            ids.append(self._id.rsplit("_", 1)[-1])           # "<video>_000123" -> "000123"
        return ids

    def predict(self, image_bgr: np.ndarray, fov_x_deg: float | None = None) -> DepthPrediction:
        if self._id is None:
            raise RuntimeError("FolderBackend needs set_frame_id() before predict()")
        for fid in self._candidates():
            for ext in (".npy", ".npz", ".png"):
                p = self.dir / f"{fid}{ext}"
                if p.exists():
                    d = _read(p, self.png_scale)
                    h, w = image_bgr.shape[:2]
                    if d.shape != (h, w):
                        raise ValueError(f"{p}: {d.shape[::-1]} vs frame {w}x{h}")
                    return DepthPrediction(d, np.isfinite(d) & (d > 0), None, None, self.native_unit_mm is not None)
        raise FileNotFoundError(f"no depth file for frame {self._id} in {self.dir}")


def _read(p: Path, png_scale: float) -> np.ndarray:
    if p.suffix == ".npy":
        return np.load(p).astype(np.float32)
    if p.suffix == ".npz":
        return np.load(p)["depth"].astype(np.float32)
    m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    return m.astype(np.float32) / png_scale
