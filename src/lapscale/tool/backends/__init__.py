"""Segmentation backends. Heavy ones are imported only when requested."""
from __future__ import annotations

from ..base import SegBackend

NAMES = ("sam3", "yolo", "masks")


def make_seg_backend(name: str, **kw) -> SegBackend:
    if name == "sam3":
        from .sam3 import Sam3Backend
        return Sam3Backend(**kw)
    if name == "yolo":
        from .yolo import YoloBackend
        return YoloBackend(**kw)
    if name == "masks":
        from .masks import MasksBackend
        return MasksBackend(**kw)
    raise ValueError(f"unknown segmentation backend {name!r}; available: {NAMES}")
