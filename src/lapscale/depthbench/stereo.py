"""Depth ground truth from a calibrated stereo pair (e.g. the da Vinci videos already validated with
phase0_check: sync, calibration choice, refinement, shaft-diameter test).

The benchmark image is the RECTIFIED LEFT image: an exact pinhole view (no distortion, K = P1), so the
monocular networks see a normal camera, and the ground-truth depth lives on the same pixel grid with no
reprojection. Depth = f * B / disparity, with a left-right consistency check, saturated (specular) pixels
removed and a depth range.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class StereoCalib:
    K1: np.ndarray
    D1: np.ndarray
    K2: np.ndarray
    D2: np.ndarray
    R: np.ndarray            # X_right = R X_left + T
    T: np.ndarray            # mm
    size: tuple[int, int]    # (w, h) of the images the intrinsics refer to


def load_stereo_calib(path: str | Path, size: tuple[int, int], crop_offset: tuple[float, float] = (0.0, 0.0)
                      ) -> StereoCalib:
    """OpenCV file with M_l, D_l, M_r, D_r, R, T (the da Vinci format). ``crop_offset`` = (x0, y0) of the crop
    applied to the frames when the intrinsics refer to the full frame (phase0_check ':full' convention)."""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise ValueError(f"cannot read {path}")

    def get(*names):
        for n in names:
            m = fs.getNode(n).mat()
            if m is not None:
                return np.asarray(m, np.float64)
        raise ValueError(f"{path}: none of {names}")

    K1, K2 = get("M_l", "M1", "K1").copy(), get("M_r", "M2", "K2").copy()
    D1, D2 = get("D_l", "D1").ravel(), get("D_r", "D2").ravel()
    R, T = get("R"), get("T").reshape(3, 1)
    fs.release()
    for K in (K1, K2):
        K[0, 2] -= crop_offset[0]
        K[1, 2] -= crop_offset[1]
    if np.abs(R @ R.T - np.eye(3)).max() > 1e-6:          # nearest rotation
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
    return StereoCalib(K1, D1, K2, D2, R, T, (int(size[0]), int(size[1])))


@dataclass
class Rectifier:
    maps_l: tuple
    maps_r: tuple
    K: np.ndarray            # intrinsics of the rectified left image
    f: float
    B: float                 # baseline, mm

    @classmethod
    def from_calib(cls, c: StereoCalib) -> Rectifier:
        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(c.K1, c.D1, c.K2, c.D2, c.size, c.R, c.T,
                                                    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        if abs(P2[1, 3]) > abs(P2[0, 3]):
            raise ValueError("vertical stereo rectification is not supported")
        m1 = cv2.initUndistortRectifyMap(c.K1, c.D1, R1, P1, c.size, cv2.CV_32FC1)
        m2 = cv2.initUndistortRectifyMap(c.K2, c.D2, R2, P2, c.size, cv2.CV_32FC1)
        return cls(m1, m2, P1[:3, :3].copy(), float(P1[0, 0]), float(abs(P2[0, 3] / P2[0, 0])))

    def rectify(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (cv2.remap(left, *self.maps_l, cv2.INTER_LINEAR), cv2.remap(right, *self.maps_r, cv2.INTER_LINEAR))


@dataclass
class StereoConfig:
    z_min_mm: float = 20.0
    z_max_mm: float = 300.0
    block: int = 5
    lr_tol_px: float = 1.0
    saturation: int = 245            # brighter than this in either image: specular, no GT
    uniqueness: int = 10
    speckle_window: int = 100
    speckle_range: int = 2
    erode_px: int = 2                # shrink the valid area (disparity edges are unreliable)


def stereo_depth(left_r: np.ndarray, right_r: np.ndarray, f: float, B: float, cfg: StereoConfig | None = None
                 ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Rectified pair -> (depth mm, valid, diagnostics). Left-right consistent SGBM disparities only."""
    cfg = cfg or StereoConfig()
    gl = cv2.cvtColor(left_r, cv2.COLOR_BGR2GRAY) if left_r.ndim == 3 else left_r
    gr = cv2.cvtColor(right_r, cv2.COLOR_BGR2GRAY) if right_r.ndim == 3 else right_r
    d_min = int(np.floor(f * B / cfg.z_max_mm))
    d_max = int(np.ceil(f * B / cfg.z_min_mm))
    num = int(np.ceil(max(16, d_max - d_min) / 16) * 16)

    def sgbm(min_d):
        return cv2.StereoSGBM_create(minDisparity=min_d, numDisparities=num, blockSize=cfg.block,
                                     P1=8 * cfg.block**2, P2=32 * cfg.block**2, disp12MaxDiff=-1,
                                     uniquenessRatio=cfg.uniqueness, speckleWindowSize=cfg.speckle_window,
                                     speckleRange=cfg.speckle_range, mode=cv2.STEREO_SGBM_MODE_HH)

    dl = sgbm(d_min).compute(gl, gr).astype(np.float32) / 16.0
    # right-view disparity: same matcher on the horizontally flipped pair
    dr = sgbm(d_min).compute(gr[:, ::-1].copy(), gl[:, ::-1].copy()).astype(np.float32)[:, ::-1] / 16.0
    h, w = gl.shape
    valid = (dl > max(d_min, 0)) & (dl < d_min + num)
    xs = np.arange(w)[None].repeat(h, 0)
    xr = np.round(xs - dl).astype(int)
    inside = valid & (xr >= 0) & (xr < w)
    yy = np.arange(h)[:, None].repeat(w, 1)
    lr = np.full((h, w), np.inf, np.float32)
    lr[inside] = np.abs(dl[inside] - dr[yy[inside], xr[inside]])
    valid &= lr <= cfg.lr_tol_px
    valid &= (gl < cfg.saturation) & (cv2.dilate(gr, np.ones((3, 3), np.uint8)) < cfg.saturation)
    if cfg.erode_px:
        valid = cv2.erode(valid.astype(np.uint8), np.ones((2 * cfg.erode_px + 1,) * 2, np.uint8)).astype(bool)
    with np.errstate(divide="ignore"):
        depth = np.where(valid, f * B / np.where(dl > 0, dl, np.nan), 0).astype(np.float32)
    valid &= (depth >= cfg.z_min_mm) & (depth <= cfg.z_max_mm)
    diag = dict(valid_fraction=round(float(valid.mean()), 4), disparity_range=[d_min, d_min + num],
                depth_median_mm=round(float(np.median(depth[valid])), 2) if valid.any() else None)
    return np.where(valid, depth, 0).astype(np.float32), valid, diag
