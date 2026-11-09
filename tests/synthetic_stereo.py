"""Synthetic stereo laparoscope (da Vinci-like: 4.1 mm baseline, f ~ 700 px, mild distortion) looking at a
tilted, textured plane. Exact depth is known in any pinhole view."""
from __future__ import annotations

import cv2
import numpy as np

SIZE = (640, 480)
K1 = np.array([[700.0, 0, 322.0], [0, 700.0, 238.0], [0, 0, 1]])
K2 = np.array([[702.0, 0, 318.0], [0, 701.0, 241.0], [0, 0, 1]])
D1 = np.array([-0.12, 0.05, 0.0, 0.0, 0.0])
D2 = np.array([-0.11, 0.04, 0.0, 0.0, 0.0])
R = cv2.Rodrigues(np.array([0.002, -0.004, 0.001]))[0]
T = np.array([[-4.1], [0.05], [0.02]])          # X_right = R X_left + T, mm
PLANE_N = np.array([0.15, -0.1, 1.0])          # n . X = D in the LEFT camera frame
PLANE_D = 90.0


def texture(X):
    return (0.5 + 0.2 * np.sin(1.7 * X[:, 0]) * np.cos(1.3 * X[:, 1]) + 0.15 * np.sin(0.45 * X[:, 0] + 0.8 * X[:, 1])
            + 0.1 * np.sin(5.1 * X[:, 0] + 0.3) * np.sin(4.3 * X[:, 1]))


def render(K, D, Rc, tc):
    """Camera with X_cam = Rc X_left + tc."""
    w, h = SIZE
    yy, xx = np.mgrid[0:h, 0:w]
    n = cv2.undistortPoints(np.stack([xx.ravel(), yy.ravel()], 1).reshape(-1, 1, 2).astype(np.float64), K, D)
    d_cam = np.column_stack([n.reshape(-1, 2), np.ones(w * h)])
    d_left = d_cam @ Rc                          # R^T d (row vectors)
    c_left = -Rc.T @ tc.ravel()                  # camera centre in the left frame
    lam = (PLANE_D - PLANE_N @ c_left) / (d_left @ PLANE_N)
    X = c_left + lam[:, None] * d_left
    v = texture(X)
    img = np.clip(np.stack([60 + 120 * v, 50 + 100 * v, 120 + 120 * v], 1), 0, 255).reshape(h, w, 3)
    return img.astype(np.uint8)


def write_pair(folder, n=3):
    L = render(K1, D1, np.eye(3), np.zeros((3, 1)))
    Rr = render(K2, D2, R, T)
    paths = []
    for name, img in (("left.avi", L), ("right.avi", Rr)):
        vw = cv2.VideoWriter(str(folder / name), cv2.VideoWriter_fourcc(*"MJPG"), 25, SIZE)
        for _ in range(n):
            vw.write(img)
        vw.release()
        paths.append(folder / name)
    calib = folder / "stereo.xml"
    fs = cv2.FileStorage(str(calib), cv2.FILE_STORAGE_WRITE)
    for k, v in (("M_l", K1), ("D_l", D1[None]), ("M_r", K2), ("D_r", D2[None]), ("R", R), ("T", T)):
        fs.write(k, v)
    fs.release()
    return paths[0], paths[1], calib


def plane_depth_pinhole(K, size=SIZE, R_left_to_view=None):
    """Exact depth map of the plane in a pinhole view (K) whose axes are R_left_to_view applied to the left
    camera axes (rectified left: R1)."""
    R_left_to_view = np.eye(3) if R_left_to_view is None else R_left_to_view
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    d_view = np.column_stack([(xx.ravel() - K[0, 2]) / K[0, 0], (yy.ravel() - K[1, 2]) / K[1, 1], np.ones(w * h)])
    d_left = d_view @ R_left_to_view            # view -> left: R^T
    lam = PLANE_D / (d_left @ PLANE_N)
    return lam.reshape(h, w)                    # z in the view frame (d_view has z = 1)
