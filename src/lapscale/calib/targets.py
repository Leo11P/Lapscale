"""Calibration targets and their detection.

Two targets are supported:

* ``checkerboard``: classic chessboard, ``cols x rows`` INNER corners. Must be fully visible, which
  inside the circular field of a laparoscope means holding it small in the centre.
* ``charuco``: chessboard with ArUco markers in the white squares. Every corner is identified, so
  partial views count: the board can fill the image and reach the edge of the optical disc, which is
  exactly where distortion has to be measured. Recommended for laparoscopes.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    image_points: np.ndarray    # (N, 2) float32, pixels
    object_points: np.ndarray   # (N, 3) float32, mm, z = 0
    ids: np.ndarray             # (N,) corner ids (stable across views, used for de-duplication)


@dataclass
class TargetSpec:
    kind: str                   # "checkerboard" | "charuco"
    cols: int                   # checkerboard: inner corners per row; charuco: squares per row
    rows: int
    square_mm: float
    marker_mm: float | None = None      # charuco only
    dictionary: str = "DICT_5X5_100"    # charuco only
    legacy: bool = False                # charuco boards printed with the pre-4.6 OpenCV layout

    def validate(self) -> None:
        if self.kind not in ("checkerboard", "charuco"):
            raise ValueError(f"unknown target {self.kind!r}")
        if self.square_mm <= 0:
            raise ValueError("square size must be > 0 mm")
        if self.kind == "charuco":
            if not self.marker_mm or not 0 < self.marker_mm < self.square_mm:
                raise ValueError("charuco needs 0 < marker_mm < square_mm")
            if not hasattr(cv2.aruco, self.dictionary):
                raise ValueError(f"unknown ArUco dictionary {self.dictionary}")
        if self.cols < 2 or self.rows < 2:
            raise ValueError("board too small")

    @property
    def min_corners(self) -> int:
        """Below this a view constrains the intrinsics too weakly to be worth keeping."""
        return (self.cols * self.rows) if self.kind == "checkerboard" else 8


class TargetDetector:
    def __init__(self, spec: TargetSpec):
        spec.validate()
        self.spec = spec
        if spec.kind == "checkerboard":
            g = np.mgrid[0:spec.cols, 0:spec.rows].T.reshape(-1, 2)
            self._obj = np.hstack([g * spec.square_mm, np.zeros((len(g), 1))]).astype(np.float32)
        else:
            dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dictionary))
            self.board = cv2.aruco.CharucoBoard((spec.cols, spec.rows), spec.square_mm, spec.marker_mm, dic)
            if spec.legacy:
                self.board.setLegacyPattern(True)
            self._obj = np.asarray(self.board.getChessboardCorners(), np.float32).reshape(-1, 3)
            self._det = cv2.aruco.CharucoDetector(self.board)

    def detect(self, image: np.ndarray, mask: np.ndarray | None = None) -> Detection | None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if mask is not None:                       # black outside the optical disc: no false corners
            gray = np.where(mask, gray, 0).astype(np.uint8)
        if self.spec.kind == "checkerboard":
            flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
            ok, corners = cv2.findChessboardCornersSB(gray, (self.spec.cols, self.spec.rows), flags=flags)
            if not ok:
                return None
            pts = corners.reshape(-1, 2).astype(np.float32)
            ids = np.arange(len(pts))
        else:
            corners, ids, _, _ = self._det.detectBoard(gray)
            if ids is None or len(ids) < self.spec.min_corners:
                return None
            pts = corners.reshape(-1, 2).astype(np.float32)
            ids = ids.ravel().astype(int)
        if mask is not None:                       # corners on the dark rim are unreliable
            h, w = mask.shape
            xi = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
            yi = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
            keep = mask[yi, xi]
            pts, ids = pts[keep], ids[keep]
        if len(pts) < self.spec.min_corners:
            return None
        if self.spec.kind == "charuco" and _collinear(self._obj[ids]):
            return None                            # a single row of corners does not constrain a homography
        return Detection(pts, self._obj[ids], ids)


def _collinear(obj: np.ndarray, tol: float = 1e-6) -> bool:
    xy = obj[:, :2] - obj[:, :2].mean(0)
    s = np.linalg.svd(xy, compute_uv=False)
    return s[-1] <= tol * max(s[0], 1e-12)


def render_board(spec: TargetSpec, px_per_square: int = 80, margin_squares: float = 0.5) -> np.ndarray:
    """Image of the target (for printing, and for the synthetic tests).

    For a checkerboard the image has ``cols+1 x rows+1`` squares (``cols x rows`` inner corners).
    """
    spec.validate()
    if spec.kind == "charuco":
        det = TargetDetector(spec)
        m = int(margin_squares * px_per_square)
        size = (spec.cols * px_per_square + 2 * m, spec.rows * px_per_square + 2 * m)
        return det.board.generateImage(size, marginSize=m)
    nx, ny = spec.cols + 1, spec.rows + 1
    m = int(margin_squares * px_per_square)
    img = np.full((ny * px_per_square + 2 * m, nx * px_per_square + 2 * m), 255, np.uint8)
    for j in range(ny):
        for i in range(nx):
            if (i + j) % 2 == 0:
                y, x = m + j * px_per_square, m + i * px_per_square
                img[y:y + px_per_square, x:x + px_per_square] = 0
    return img
