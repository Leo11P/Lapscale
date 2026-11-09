"""Precomputed masks from a folder: any external model (e.g. a SurgeNetXL-based network trained
elsewhere), or ground-truth masks to check the benchmark itself.

Accepted per frame id (the name of the frame in the benchmark, without extension):
  <dir>/<id>.png        one semantic map (0-255 = probability), split into connected components
  <dir>/<id>_<k>.png    one file per instance
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..base import InstrumentInstance, split_instances


class MasksBackend:
    def __init__(self, masks_dir: str | Path, name: str | None = None):
        self.dir = Path(masks_dir)
        if not self.dir.is_dir():
            raise FileNotFoundError(self.dir)
        self.name = name or f"masks:{self.dir.name}"
        self._frame_id: str | None = None

    def set_frame_id(self, frame_id: str) -> None:
        self._frame_id = frame_id

    def predict(self, frame_bgr: np.ndarray) -> list[InstrumentInstance]:
        if self._frame_id is None:
            raise RuntimeError("MasksBackend needs set_frame_id() before predict()")
        fid = self._frame_id
        single = self.dir / f"{fid}.png"
        if single.exists():
            return split_instances(_read(single))
        return [InstrumentInstance(_read(p)) for p in sorted(self.dir.glob(f"{fid}_*.png"))]


def _read(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise ValueError(f"cannot read {path}")
    if m.ndim == 3:
        m = m[..., 0]
    scale = 65535.0 if m.dtype == np.uint16 else 255.0 if m.dtype == np.uint8 else 1.0
    return (m.astype(np.float32) / scale).clip(0, 1)
