"""Synthetic laparoscopic video with exact depth: a tilted, textured tissue plane and a 5 mm shaft in front of
it, seen by a distorted camera with an optical disc. Fake backends: oracle segmentation (exact soft mask)
and a depth "network" that returns the true depth of whatever pinhole view it is given, times an unknown
factor (and optionally plus an offset). Used to test the whole pipeline end to end."""
from __future__ import annotations

import cv2
import numpy as np
from synthetic_scene import SS, _hit_cylinder, _rays
from synthetic_shaft import CAMERA as CAM_NO_DISC

from lapscale.camera import Camera, OpticalCircle
from lapscale.depth.base import DepthPrediction
from lapscale.tool.base import InstrumentInstance

CAMERA = Camera(CAM_NO_DISC.K, CAM_NO_DISC.dist, CAM_NO_DISC.image_size, source="manual",
                optical_circle=OpticalCircle(481.0, 270.0, 330.0))
PLANE_N = np.array([0.12, -0.08, 1.0])
PLANE_D = 120.0
RADIUS = 2.5
NET_SCALE = 0.0123          # network units per mm
NET_OFFSET = 0.0


def shaft_pose(k: int):
    a = np.array([10.0 - 1.5 * k, 3.0 + 0.5 * k, 75.0 + 2.0 * k])
    u = np.array([0.8, 0.35, 0.35])
    return a, u / np.linalg.norm(u)


def _cyl_depth(xy, a, u, r, t_range=(-80.0, 8.0)):
    """Depth of the first hit of the finite cylinder along normalised rays (NaN = miss)."""
    d = np.column_stack([xy, np.ones(len(xy))])
    dp = d - np.outer(d @ u, u)
    ap = -a + (a @ u) * u
    A = np.einsum("ij,ij->i", dp, dp)
    B = 2 * dp @ ap
    C = ap @ ap - r**2
    disc = B**2 - 4 * A * C
    z = np.full(len(xy), np.nan)
    ok = disc >= 0
    for sgn in (1, -1):                          # far root first, then near root overwrites
        lam = np.where(ok, (-B + sgn * np.sqrt(np.where(ok, disc, 0))) / (2 * A), np.nan)
        t = (lam[:, None] * d - a) @ u
        good = ok & (lam > 0) & (t >= t_range[0]) & (t <= t_range[1])
        z[good] = lam[good]
    return z


def plane_depth(xy):
    return PLANE_D / (np.column_stack([xy, np.ones(len(xy))]) @ PLANE_N)


def true_depth(xy, k):
    a, u = shaft_pose(k)
    zc = _cyl_depth(xy, a, u, RADIUS)
    zp = plane_depth(xy)
    return np.where(np.isfinite(zc) & (zc < zp), zc, zp)


def instrument_mask(k: int) -> np.ndarray:
    a, u = shaft_pose(k)
    n, (H, W) = _rays()
    hit = _hit_cylinder(n, a, u, RADIUS, (-80.0, 8.0))
    cov = hit.reshape(H, W).astype(np.float32)
    h, w = H // SS, W // SS
    return cov.reshape(h, SS, w, SS).mean(axis=(1, 3))


def frame_image(k: int, rng) -> np.ndarray:
    w, h = CAMERA.image_size
    yy, xx = np.mgrid[0:h, 0:w]
    xy = cv2.undistortPoints(np.stack([xx.ravel(), yy.ravel()], 1).reshape(-1, 1, 2).astype(np.float64),
                             CAMERA.K, CAMERA.dist).reshape(-1, 2)
    z = plane_depth(xy)
    X = xy * z[:, None]
    tex = 0.5 + 0.25 * np.sin(X[:, 0] * 1.3) * np.cos(X[:, 1] * 1.1) + 0.15 * np.sin(X[:, 0] * 0.37 + X[:, 1] * 0.5)
    tissue = np.stack([60 + 60 * tex, 50 + 40 * tex, 140 + 90 * tex], 1).reshape(h, w, 3)
    m = instrument_mask(k)[..., None]
    img = tissue * (1 - m) + np.array([170.0, 170.0, 175.0]) * m + rng.normal(0, 2, (h, w, 3))
    img *= CAMERA.mask()[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def write_video(path, n_frames=6, seed=0):
    rng = np.random.default_rng(seed)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, CAMERA.image_size)
    for k in range(n_frames):
        vw.write(frame_image(k, rng))
    vw.release()
    return path


class OracleSeg:
    name = "oracle-seg"

    def __init__(self):
        self.k = 0

    def set_frame_id(self, fid: str):
        self.k = int(fid.rsplit("_", 1)[-1])

    def predict(self, frame):
        return [InstrumentInstance(instrument_mask(self.k), 1.0)]


class TrueDepthNet:
    """Depth of the scene for each pixel of the pinhole view it receives (intrinsics ``K_view``)."""

    name = "fake:true-depth"
    output_kind = "depth"
    native_unit_mm = 1000.0      # pretends to be metric (in metres); the scale stage reports how wrong

    def __init__(self, K_view, scale=NET_SCALE, offset=NET_OFFSET):
        self.K, self.scale, self.offset, self.k = K_view, scale, offset, 0

    def set_frame_id(self, fid: str):
        self.k = int(fid.rsplit("_", 1)[-1])

    def predict(self, img, fov_x_deg=None):
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        xy = np.column_stack([(xx.ravel() - self.K[0, 2]) / self.K[0, 0], (yy.ravel() - self.K[1, 2]) / self.K[1, 1]])
        z = true_depth(xy, self.k).reshape(h, w)
        return DepthPrediction((self.scale * z + self.offset).astype(np.float32), np.ones((h, w), bool))
