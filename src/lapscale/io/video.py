"""Frame sources: a video file or a folder of images, behind one interface.

Frames are BGR uint8 (OpenCV convention) and always identified by their index in the source, so
that every stage's outputs can be joined on ``frame``.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class SourceInfo:
    path: str
    n_frames: int
    fps: float | None
    size: tuple[int, int]   # (width, height)


class FrameSource:
    """``FrameSource(path)`` opens a video file or a folder of images."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if self.path.is_dir():
            self._images = sorted(p for p in self.path.iterdir() if p.suffix.lower() in IMAGE_EXT)
            if not self._images:
                raise ValueError(f"no images in {self.path}")
            first = cv2.imread(str(self._images[0]))
            if first is None:
                raise ValueError(f"cannot read {self._images[0]}")
            self.info = SourceInfo(str(self.path), len(self._images), None, (first.shape[1], first.shape[0]))
        else:
            self._images = None
            cap = cv2.VideoCapture(str(self.path))
            if not cap.isOpened():
                raise ValueError(f"cannot open video {self.path}")
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or None
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            cap.release()
            self.info = SourceInfo(str(self.path), n, fps, size)

    @property
    def size(self) -> tuple[int, int]:
        return self.info.size

    def __len__(self) -> int:
        return self.info.n_frames

    def iter(self, indices: list[int] | None = None) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (index, frame). With ``indices`` only those frames, read sequentially (grab() is
        much cheaper than seeking, and seeking is not frame-exact with many codecs)."""
        if self._images is not None:
            for i in (range(len(self._images)) if indices is None else sorted(set(indices))):
                img = cv2.imread(str(self._images[i]))
                if img is not None:
                    yield i, img
            return
        want = None if indices is None else sorted({int(i) for i in indices if i >= 0})
        cap = cv2.VideoCapture(str(self.path))
        try:
            i, k = 0, 0
            while want is None or k < len(want):
                if want is None or i == want[k]:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    yield i, frame
                    if want is not None:
                        k += 1
                elif not cap.grab():
                    break
                i += 1
        finally:
            cap.release()

    def sample_indices(self, n: int | None = None, stride: int | None = None,
                       start: int = 0, stop: int | None = None) -> list[int]:
        """Evenly spread indices: either every ``stride`` frames or ``n`` frames in [start, stop)."""
        stop = len(self) if stop is None else min(stop, len(self))
        if stride:
            return list(range(start, stop, stride))
        if n is None or n >= stop - start:
            return list(range(start, stop))
        return sorted({int(round(x)) for x in np.linspace(start, stop - 1, n)})
