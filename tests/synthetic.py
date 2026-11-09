"""Synthetic laparoscope: renders a calibration target seen by a camera with KNOWN intrinsics and
distortion, inside a circular optical disc clipped by a 16:9 frame. Ground truth for the tests."""
from __future__ import annotations

import cv2
import numpy as np

from lapscale.calib.targets import TargetDetector, TargetSpec, render_board

SIZE = (960, 540)
K_TRUE = np.array([[700.0, 0, 482.0], [0, 700.0, 268.0], [0, 0, 1]])
D_TRUE = np.array([-0.28, 0.09, 0.0005, -0.0003, 0.0])
CIRCLE_TRUE = (481.0, 270.0, 330.0)      # cx, cy, r: clipped by top and bottom (h = 540)

CHARUCO = TargetSpec("charuco", 11, 8, 5.0, 3.75, "DICT_5X5_100")
CHECKER = TargetSpec("checkerboard", 7, 5, 5.0)


def board_to_pixel_affine(spec: TargetSpec, board_img: np.ndarray) -> np.ndarray:
    """2x3 affine, board mm -> board-image pixel, measured by detecting the target on its own image
    (no assumption about OpenCV's board axis conventions)."""
    det = TargetDetector(spec).detect(cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR))
    assert det is not None, "target not detected on its own rendering"
    obj = np.ascontiguousarray(det.object_points[:, :2])
    A, _ = cv2.estimateAffine2D(obj, det.image_points, ransacReprojThreshold=0.5)
    return A


def random_pose(rng, depth=(90.0, 150.0), tilt_deg=(15.0, 40.0), spread=0.45, board_mm=(55.0, 40.0)):
    tilt = np.radians(rng.uniform(*tilt_deg))
    axis = rng.normal(size=3)
    axis[2] = 0.0
    axis /= np.linalg.norm(axis)
    R, _ = cv2.Rodrigues(axis * tilt)
    R = R @ cv2.Rodrigues(np.array([0, 0, rng.uniform(-0.6, 0.6)]))[0]
    z = rng.uniform(*depth)
    centre = np.array([rng.uniform(-spread, spread) * z, rng.uniform(-spread, spread) * z * 0.6, z])
    t = centre - R @ np.array([board_mm[0] / 2, board_mm[1] / 2, 0.0])   # board centre at `centre`
    return R, t


def render_view(spec, board_img, A, R, t, rng, K=K_TRUE, D=D_TRUE, size=SIZE, circle=CIRCLE_TRUE,
                noise=2.0):
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    pix = np.stack([xx.ravel(), yy.ravel()], 1).reshape(-1, 1, 2)
    n = cv2.undistortPoints(pix, K, D).reshape(-1, 2)
    d = np.column_stack([n, np.ones(len(n))])
    Rt = R.T
    db = d @ Rt.T                    # ray direction in board frame
    tb = Rt @ t
    lam = tb[2] / db[:, 2]
    Xb = lam[:, None] * db - tb      # point on the board plane (z = 0)
    u = A[0, 0] * Xb[:, 0] + A[0, 1] * Xb[:, 1] + A[0, 2]
    v = A[1, 0] * Xb[:, 0] + A[1, 1] * Xb[:, 1] + A[1, 2]
    mapx = u.reshape(h, w).astype(np.float32)
    mapy = v.reshape(h, w).astype(np.float32)
    bg = np.full((h, w), 110, np.uint8)
    img = cv2.remap(board_img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    inside = (mapx >= 0) & (mapx < board_img.shape[1] - 1) & (mapy >= 0) & (mapy < board_img.shape[0] - 1)
    img = np.where(inside & (lam.reshape(h, w) > 0), img, bg).astype(np.float32)
    img = cv2.GaussianBlur(img, (3, 3), 0.7) + rng.normal(0, noise, img.shape)
    cx, cy, r = circle
    img[(xx - cx) ** 2 + (yy - cy) ** 2 > r**2] = 0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def make_views(spec, n, seed=0, **kw):
    rng = np.random.default_rng(seed)
    board = render_board(spec, px_per_square=120)
    A = board_to_pixel_affine(spec, board)
    bw = (spec.cols + (1 if spec.kind == "checkerboard" else 0)) * spec.square_mm
    bh = (spec.rows + (1 if spec.kind == "checkerboard" else 0)) * spec.square_mm
    frames = []
    for _ in range(n):
        R, t = random_pose(rng, board_mm=(bw, bh), **kw)
        frames.append(render_view(spec, board, A, R, t, rng))
    return frames
