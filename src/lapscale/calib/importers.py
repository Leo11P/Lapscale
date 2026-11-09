"""Import intrinsics you already have (the calibration bypass).

Reads OpenCV FileStorage files (XML / YAML / JSON) with any of the usual key names, including one
side of a stereo calibration (``M_l``/``D_l`` or ``M_r``/``D_r``, as in the da Vinci files), and
writes a ``lapscale.camera/1`` YAML.

The image size is part of the intrinsics: if the file does not store it, it must be given. If the
intrinsics were computed on a crop of the frame, give the crop offset so the principal point is moved
back to full-frame pixels.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..camera import Camera

KEY_SETS = {
    "left": [("M_l", "D_l"), ("M1", "D1"), ("K1", "D1"), ("cameraMatrixL", "distCoeffsL")],
    "right": [("M_r", "D_r"), ("M2", "D2"), ("K2", "D2"), ("cameraMatrixR", "distCoeffsR")],
    "mono": [("camera_matrix", "distortion_coefficients"), ("cameraMatrix", "distCoeffs"),
             ("K", "D"), ("K", "dist"), ("M", "D")],
}
SIZE_KEYS = [("image_width", "image_height"), ("width", "height"), ("imageWidth", "imageHeight")]


def _mat(fs, name):
    node = fs.getNode(name)
    if node.empty():
        return None
    m = node.mat()
    if m is None and node.isSeq():
        m = np.array([node.at(i).real() for i in range(node.size())])
    return None if m is None else np.asarray(m, np.float64)


def _size(fs):
    for kw, kh in SIZE_KEYS:
        nw, nh = fs.getNode(kw), fs.getNode(kh)
        if not nw.empty() and not nh.empty():
            return int(nw.real()), int(nh.real())
    node = fs.getNode("image_size")
    if not node.empty():
        m = node.mat()
        v = m.ravel() if m is not None else [node.at(i).real() for i in range(node.size())]
        return int(v[0]), int(v[1])
    return None


def import_opencv(path: str | Path, side: str = "mono", image_size: tuple[int, int] | None = None,
                  crop_offset: tuple[float, float] = (0.0, 0.0), key_K: str | None = None,
                  key_D: str | None = None) -> Camera:
    """``side``: mono | left | right. ``crop_offset`` = (x0, y0) of the crop the intrinsics refer to."""
    path = Path(path)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise ValueError(f"cannot read {path}")
    try:
        if key_K:
            candidates = [(key_K, key_D)]
        else:
            candidates = KEY_SETS.get(side, []) + (KEY_SETS["mono"] if side != "mono" else [])
        K = D = None
        used = None
        for kK, kD in candidates:
            K = _mat(fs, kK)
            if K is not None:
                D = _mat(fs, kD) if kD else None
                used = (kK, kD)
                break
        if K is None:
            raise ValueError(f"{path}: no intrinsic matrix found for side={side!r} "
                             f"(tried {[c[0] for c in candidates]}); pass key names explicitly")
        size = image_size or _size(fs)
    finally:
        fs.release()
    if size is None:
        raise ValueError(f"{path} does not store the image size: pass it (e.g. --image-size 1920x1080)")
    K = K.reshape(3, 3).copy()
    K[0, 2] += crop_offset[0]
    K[1, 2] += crop_offset[1]
    D = np.zeros(0) if D is None else D.ravel()
    if D.size not in (0, 4, 5, 8):
        D = D[:8] if D.size > 8 else np.pad(D, (0, 5 - D.size)) if D.size < 5 else D[:5]
    notes = f"imported from {path.name} keys {used}"
    if any(crop_offset):
        notes += f", crop offset {crop_offset}"
    cam = Camera(K, D, size, source="import", notes=notes)
    w, h = size
    if not (0.35 * w < cam.cx < 0.65 * w and 0.35 * h < cam.cy < 0.65 * h):
        cam.quality["warnings"] = [f"principal point ({cam.cx:.0f}, {cam.cy:.0f}) far from the image centre: "
                                   "wrong image size or missing crop offset?"]
    return cam
