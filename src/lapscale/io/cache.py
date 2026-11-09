"""Per-run cache shared by the stages.

Layout of a run directory (``runs/<video_stem>/``)::

    depth/<tag>/meta.json        backend, view intrinsics, FOV, configuration
    depth/<tag>/frames.csv       one row per processed frame: latency, estimated focal, valid fraction
    depth/<tag>/<frame>.npz      depth (float16, ~0.05% relative precision), valid, K of the view
    tool/<backend>/frames.csv    (stage 3, planned) edges, depth of the shaft, uncertainty, reject reason
    scale/<variant>/frames.csv   (stage 4, planned) per-frame scale parameters and residuals

A stage skips frames already in its cache: interrupted runs resume, and changing a later stage does not
recompute an earlier one. A different configuration must use a different ``tag``.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..camera import _plain


@dataclass
class DepthFrame:
    """Depth map of one frame as the network produced it, addressable by normalised coordinates."""

    index: int
    depth: np.ndarray        # (H, W) float32, backend units, along the optical axis
    valid: np.ndarray        # (H, W) bool
    K: np.ndarray            # 3x3 of the depth map (undistorted view)

    def sample(self, xy_norm: np.ndarray) -> np.ndarray:
        """Bilinear depth at normalised undistorted coordinates (N, 2); NaN where any of the four
        neighbours is invalid (no blending across the instrument boundary or the optical rim)."""
        xy = np.asarray(xy_norm, np.float64).reshape(-1, 2)
        u = (xy[:, 0] * self.K[0, 0] + self.K[0, 2]).astype(np.float32).reshape(-1, 1)
        v = (xy[:, 1] * self.K[1, 1] + self.K[1, 2]).astype(np.float32).reshape(-1, 1)
        d = cv2.remap(self.depth.astype(np.float32), u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                      borderValue=np.nan).ravel()
        ok = cv2.remap(self.valid.astype(np.float32), u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                       borderValue=0).ravel()
        d[ok < 0.999] = np.nan
        return d


class DepthStore:
    FIELDS = ["frame", "latency_ms", "fx_px", "fy_px", "valid_fraction", "depth_median"]

    def __init__(self, run_dir: str | Path, tag: str):
        self.dir = Path(run_dir) / "depth" / tag
        self.tag = tag

    # ------------------------------------------------------------------ metadata
    def write_meta(self, meta: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.dir / "meta.json"
        if p.exists():
            old = json.loads(p.read_text())
            keys = ("backend", "K_view", "size", "fov_x_deg", "config")
            changed = [k for k in keys if old.get(k) != _plain(meta.get(k))]
            if changed:
                raise ValueError(f"{self.dir} was produced with a different {changed}: use another --tag")
        p.write_text(json.dumps(_plain(meta), indent=2))

    def meta(self) -> dict:
        return json.loads((self.dir / "meta.json").read_text())

    # ------------------------------------------------------------------ frames
    def _path(self, index: int) -> Path:
        return self.dir / f"{index:06d}.npz"

    def has(self, index: int) -> bool:
        return self._path(index).exists()

    def frames(self) -> list[int]:
        return sorted(int(p.stem) for p in self.dir.glob("*.npz"))

    def save(self, index: int, depth: np.ndarray, valid: np.ndarray, K: np.ndarray, latency_ms: float,
             fx_px: float | None = None, fy_px: float | None = None) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        d = np.where(valid, depth, 0).astype(np.float16)
        tmp = self.dir / f"{index:06d}.tmp.npz"
        np.savez_compressed(tmp, depth=d, valid=valid.astype(bool), K=np.asarray(K, np.float64))
        tmp.replace(self._path(index))
        log = self.dir / "frames.csv"
        new = not log.exists()
        with open(log, "a", newline="") as fh:
            wr = csv.writer(fh)
            if new:
                wr.writerow(self.FIELDS)
            vals = depth[valid]
            wr.writerow([index, round(latency_ms, 2), "" if fx_px is None else round(fx_px, 3),
                         "" if fy_px is None else round(fy_px, 3), round(float(valid.mean()), 4),
                         round(float(np.median(vals)), 5) if vals.size else ""])

    def load(self, index: int) -> DepthFrame:
        z = np.load(self._path(index))
        return DepthFrame(index, z["depth"].astype(np.float32), z["valid"], z["K"])
