"""Ultralytics YOLO segmentation (AGPL-3.0). Needs weights trained on surgical instruments: the COCO
weights do not know them. ``retina_masks=True`` returns masks at the frame resolution (binary): the
edge extractor then puts each edge midway between inside and outside pixels.

    pip install ultralytics
"""
from __future__ import annotations

import numpy as np

from ..base import InstrumentInstance


class YoloBackend:
    def __init__(self, weights: str, confidence: float = 0.25, classes: list[int] | None = None,
                 device: str | None = None, imgsz: int | None = None):
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError(f"ultralytics not available ({e}): pip install ultralytics") from e
        self.model = YOLO(weights)
        self.kw = dict(conf=confidence, classes=classes, retina_masks=True, verbose=False)
        if device:
            self.kw["device"] = device
        if imgsz:
            self.kw["imgsz"] = imgsz
        self.name = f"yolo:{weights.replace('/', '_').replace(chr(92), '_')}"

    def predict(self, frame_bgr: np.ndarray) -> list[InstrumentInstance]:  # pragma: no cover - needs weights
        r = self.model.predict(frame_bgr, **self.kw)[0]               # numpy input is read as BGR
        if r.masks is None:
            return []
        masks = r.masks.data.float().cpu().numpy()
        scores = r.boxes.conf.float().cpu().numpy()
        names = r.names
        cls = r.boxes.cls.int().cpu().numpy()
        return [InstrumentInstance(m.astype(np.float32), float(s), names.get(int(c), str(c)))
                for m, s, c in zip(masks, scores, cls, strict=True)]
