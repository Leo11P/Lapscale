"""Segmentation contract of stage 3.

A backend turns one RAW frame (BGR, as recorded, distortion included) into instrument instances, each
with a soft mask in [0, 1] at the full frame resolution. Soft masks matter: the shaft edges are located
with sub-pixel precision at the 0.5 crossing, and one pixel of edge error is ~2% of depth on a
100 px wide shaft.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class InstrumentInstance:
    prob: np.ndarray          # (H, W) float32 in [0, 1], full frame resolution
    score: float = 1.0        # detection confidence (backend specific)
    label: str = "instrument"


class SegBackend(Protocol):
    name: str

    def predict(self, frame_bgr: np.ndarray) -> list[InstrumentInstance]:
        ...


def split_instances(prob: np.ndarray, min_area_px: int = 200, score: float = 1.0) -> list[InstrumentInstance]:
    """Semantic (one-channel) output -> one instance per connected component of prob > 0.5. Touching
    instruments stay merged: the edge extractor then finds more than two lines and rejects or picks."""
    import cv2
    binary = (prob > 0.5).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < min_area_px:
            continue
        sel = lab == k
        # keep the soft values around this component (dilated support), zero elsewhere
        support = cv2.dilate(sel.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        out.append(InstrumentInstance(np.where(support, prob, 0).astype(np.float32), score))
    return out
